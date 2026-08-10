// The sun shadow bake: casters rasterized through the fit's orthographic matrix
// into a depth map that the main pass compares its shaded points against.
//
// Two vertex entry points, one per vertex variant, mirroring scene.slang's pair
// and reading the same instance and joint rings. One fragment entry point, used
// by masked casters alone: an opaque caster is drawn with no fragment shader at
// all, and the rasterizer's depth write is the whole output.
//
// Set 0 is the frame set the main pass also binds. Set 2 is one material's
// texture set, of which only the base colour is read. Set 1 is named by the
// pipeline layout so that the texture set keeps index 2, and nothing here
// declares anything out of it.
//
// The matrix carries no Vulkan Y flip. The bake rasterizes through it and the
// lookup turns its own product with it into texture coordinates, so the mirror
// cancels only if neither side applies one.

// Mirrors `frame_set.Instance`, as scene.slang does. The whole record and not
// the model matrix alone: a StructuredBuffer steps by the element's own size, so
// a shortened mirror reads every instance after the first from the wrong offset.
struct Instance
{
    float4x4 model;
    uint joint_base;
    // Not read here. The cutoff a masked caster tests against arrives as a push
    // constant instead, so this pass never indexes the material array.
    uint material_index;
    uint padding_0;
    uint padding_1;
};

// Mirrors `uniforms.SunShadow`. Only the matrix is read here; the strength and
// the normal offset belong to the lookup, which is scene.slang's half.
struct SunShadowUbo
{
    float4x4 view_projection;
    float strength;
    float normal_offset;
    uint light;
    uint padding;
};

[[vk::binding(1, 0)]] StructuredBuffer<Instance> instances;
// One array for the whole frame, indexed from each instance's own base, exactly
// as the main pass reads it.
[[vk::binding(3, 0)]] StructuredBuffer<float4x4> joints;
[[vk::binding(4, 0)]] ConstantBuffer<SunShadowUbo> sun_shadow;
[[vk::binding(0, 2)]] Sampler2D base_colour;

// What a masked caster needs to run the cutoff test, pushed per material rather
// than read out of the material array. Two floats, against binding that array
// and carrying an index through as a varying, and it leaves the bake
// independent of the array's layout.
struct MaskPushConstants
{
    // glTF 2.0 section 3.9.4: below the cutoff the fragment is fully
    // transparent, at or above it fully opaque.
    float alpha_cutoff;

    // The fourth component of the material's base colour factor. Section 3.9.2
    // makes the factor a linear multiplier of the texture, alpha included.
    float factor_alpha;
};

[[vk::push_constant]]
ConstantBuffer<MaskPushConstants> mask;

// The locations are stated rather than left to declaration order. This struct is
// shorter than scene.slang's, so positional numbering would put the skinning
// stream at 2 and 3 while `GpuSkinVertex` declares it at 4 and 5.
struct Vertex
{
    [[vk::location(0)]] float3 position : POSITION;
    // Read only by the masked fragment entry point. It is fetched for every
    // caster because both fragment configurations share these vertex stages;
    // splitting them would duplicate the skinning body to save one interpolant
    // on a draw that has no fragment shader to receive it.
    [[vk::location(1)]] float2 uv : TEXCOORD0;
};

struct Varying
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// `SV_InstanceID` counts from the start of the draw rather than the start of the
// buffer, so the base has to be added back for a non-zero `firstInstance` to
// index the ring. The same reasoning, and the same `shaderDrawParameters`
// requirement, as scene.slang's.
uint instanceIndex(uint local_instance, uint first_instance)
{
    return first_instance + local_instance;
}

// Row vectors: the vector multiplies from the left, which is the convention
// zmath builds both of these matrices with.
Varying place(float3 local_position, float2 uv, Instance instance)
{
    const float4 world = mul(float4(local_position, 1.0), instance.model);

    Varying varying;
    varying.position = mul(world, sun_shadow.view_projection);
    varying.uv = uv;
    return varying;
}

[shader("vertex")]
Varying vertexMain(
    Vertex vertex,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    return place(vertex.position, vertex.uv, instance);
}

// The skinned variant. Position only: a depth map records where a surface is and
// nothing about how it faces, so the normal and the tangent the main pass skins
// are not computed here.
[shader("vertex")]
Varying skinnedVertexMain(
    Vertex vertex,
    [[vk::location(4)]] uint4 joint_indices : JOINTS,
    [[vk::location(5)]] float4 joint_weights : WEIGHTS,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];

    // glTF 2.0 section 3.7.3.3, and the same summation as scene.slang: the
    // weighted sum of the joint matrices, applied once. The two shaders have to
    // agree here exactly, or a skinned caster's shadow is cast from a pose the
    // main pass never draws.
    const uint base = instance.joint_base;
    const float4x4 skin =
        joint_weights.x * joints[base + joint_indices.x] +
        joint_weights.y * joints[base + joint_indices.y] +
        joint_weights.z * joints[base + joint_indices.z] +
        joint_weights.w * joints[base + joint_indices.w];

    const float3 skinned_position = mul(float4(vertex.position, 1.0), skin).xyz;
    return place(skinned_position, vertex.uv, instance);
}

// The masked caster's fragment stage, which writes nothing and exists to drop
// fragments the cutoff rejects. Without it a cutout casts the shadow of the
// quad it is drawn on rather than of the shape it shows.
//
// No output and no colour attachment: depth is written by the rasterizer after
// this stage, so a dropped fragment leaves the map alone.
[shader("fragment")]
void fragmentMain(Varying varying)
{
    const float alpha = base_colour.Sample(varying.uv).a * mask.factor_alpha;
    if (alpha < mask.alpha_cutoff) discard;
}
