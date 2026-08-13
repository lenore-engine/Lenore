// The main pass: one instanced mesh, transformed, textured and lit.
//
// Four descriptor sets, split by how often they are rewritten. Set 0 holds what
// the frame supplies and is bound once with a dynamic offset per ring; set 1
// holds the scene's material array and is bound once; set 2 holds one material's
// textures and is bound when the ordered list reaches a new material; set 3
// holds the sun shadow map and is bound once.
//
// The map has a set to itself rather than a slot in set 1. The bake that fills
// it names sets 0 to 2 in its own pipeline layout, so a slot in any of them
// would leave the map bound as a texture while it is the attachment being
// written into.

struct CameraUbo
{
    float4x4 view_projection;
    // xyz is the eye in world space, which the fragment stage needs for the
    // view-dependent half of the BRDF. The fourth lane is not read and exists
    // because the block is aligned to sixteen bytes either way.
    float4 position;
    // The view ray in device coordinates, read by sky.slang and by nothing
    // here. The block is one buffer, so both shaders declare all of it: a
    // shorter declaration on this side would still be laid out identically, and
    // the mirror check compares the whole block against `uniforms.Camera`.
    float4 ray_right;
    float4 ray_up;
    float4 ray_front;
};

// Mirrors `uniforms.Light`. Each kind reads only its own fields and the rest
// are zero, so no lane means one thing here and another there.
struct Light
{
    float3 position; // point and spot
    uint kind;       // 0 = directional, 1 = point, 2 = spot
    float3 colour;
    float intensity;
    float3 direction; // unit, the way the light travels; directional and spot
    float range;      // point and spot
    float cos_inner;  // spot
    float cos_outer;  // spot
    float2 padding;
};

// Mirrors `uniforms.max_lights`. The array bounds the loop; `count.x` is how
// much of it holds a light, and the rest is not written by anyone.
static const uint max_lights = 16;

struct LightsUbo
{
    uint4 count; // x = live lights, yzw are the array's alignment
    Light lights[max_lights];
};

// Mirrors `frame_set.Instance`. The padding is declared rather than left to the
// rules, because the stride this struct is stepped through by has to be the one
// the host writes: a disagreement reads the wrong instance from the second one
// on, and the first still looks right.
struct Instance
{
    float4x4 model;
    // Where this instance's joint matrices begin in `joints`. Read by
    // `skinnedVertexMain` and by nothing else.
    uint joint_base;
    // The material and two padding words are scalars and not a `uint3`. A
    // three-component vector aligns to sixteen bytes, which pushes the tail to
    // offset eighty and rounds the element to ninety-six; measured, against the
    // host's eighty.
    uint material_index;
    uint padding_0;
    uint padding_1;
};

// Mirrors `uniforms.SunShadow`, and shadow.slang reads the same block for the
// matrix alone.
struct SunShadowUbo
{
    float4x4 view_projection;
    float strength;
    float normal_offset;
    // Which entry of `lights` this map was baked for. A fit is built from one
    // direction, so one light and no other is shadowed through it.
    uint light;
    uint padding;
};

// Mirrors `material_storage.TexTransform`. One slot's KHR_texture_transform:
// `rs` is the row-major 2x2 of rotation times scale, `params.xy` is the offset,
// and `params.z` is the UV set the slot reads. `transformedUv` is the only
// reader.
struct TexTransform
{
    float4 rs;
    float4 params;
};

// Mirrors `material_storage.MaterialData`. One record per material, indexed by
// what the instance carries rather than by which material set is bound.
struct MaterialData
{
    float4 base_colour_factor;
    // Emissive in xyz, occlusion strength in w.
    float4 emissive_factor;
    // Metallic, roughness, alpha cutoff, normal scale.
    float4 metallic_roughness_cutoff;
    // Alpha mode, double sided, the texture-presence mask, unlit.
    uint4 flags;
    TexTransform tex[5];
};

[[vk::binding(0, 0)]] ConstantBuffer<CameraUbo> camera;
[[vk::binding(1, 0)]] StructuredBuffer<Instance> instances;
[[vk::binding(2, 0)]] ConstantBuffer<LightsUbo> lights;
// One array for the whole frame. Each instance owns a contiguous run in it,
// starting at its own `joint_base`, so several skeletons can be drawn as
// instances of one draw.
[[vk::binding(3, 0)]] StructuredBuffer<float4x4> joints;
[[vk::binding(4, 0)]] ConstantBuffer<SunShadowUbo> sun_shadow;
[[vk::binding(0, 1)]] StructuredBuffer<MaterialData> materials;
// The environment, prefiltered before the engine saw it. Cosine-convolved
// irradiance along the normal, the GGX integral per roughness step along the
// reflection, and the environment BRDF tabulated against (N.V, roughness).
//
// With no environment loaded all three are bound to black, and every term below
// is linear in the two cubemap samples, so the image-based contribution is
// exactly zero without a branch or a flag saying so.
[[vk::binding(1, 1)]] SamplerCube lambertian_environment;
[[vk::binding(2, 1)]] SamplerCube ggx_environment;
[[vk::binding(3, 1)]] Sampler2D ggx_lut;
[[vk::binding(0, 2)]] Sampler2D base_colour;
[[vk::binding(1, 2)]] Sampler2D metallic_roughness;
[[vk::binding(2, 2)]] Sampler2D normal_texture;
[[vk::binding(3, 2)]] Sampler2D emissive_texture;
[[vk::binding(4, 2)]] Sampler2D occlusion_texture;
// The sun shadow map, sampled through a comparison sampler: a fetch returns the
// result of the depth test rather than a depth, and with a linear filter the
// four texels of the footprint are tested and averaged, which is one tap of
// percentage-closer filtering for one fetch. The comparison and the border are
// the sampler's, in shadow.zig.
[[vk::binding(0, 3)]] Sampler2DShadow sun_shadow_map;

struct Vertex
{
    float3 position : POSITION;
    float2 uv : TEXCOORD0;
    // Delivered as signed normalized floats: the attribute is packed 10-10-10-2
    // and the vertex input stage expands it, so nothing is unpacked here.
    float4 normal : NORMAL;
    float4 tangent : TANGENT;
};

struct Varying
{
    float4 position : SV_Position;
    float3 world_position : POSITION;
    float3 normal : NORMAL;
    float4 tangent : TANGENT;
    float2 uv : TEXCOORD0;
    // The second UV set, zero on the entry points whose vertex input has no
    // binding 3. glTF 2.0 specification, 5.30.2: a mesh primitive must carry the
    // texture coordinate attribute a material's texCoord names for that material
    // to apply to it, so on a valid asset the zero is never what a slot reads.
    float2 uv1 : TEXCOORD1;
    // Uniform over a primitive, so it is flat rather than interpolated: an
    // interpolated index is a fractional index between two records at every
    // fragment but the vertices. `nointerpolation` lowers to the SPIR-V `Flat`
    // decoration on both the output and the input, which the disassembly shows.
    nointerpolation uint material_index : MATERIAL;
};

// The same set with COLOR_0 on it, for the meshes that declare the stream.
//
// A second structure rather than a lane on the first: section 3.9.2 makes the
// vertex colour a multiplier of the base colour, so a mesh without one would
// have to interpolate a constant white across every fragment to be shaded by
// the same code. It is nested rather than restated so that the two cannot
// describe different varyings.
struct VaryingColour
{
    Varying base;
    // Delivered as unorm8 and expanded by the vertex input stage, which is the
    // range section 3.7.2 clamps COLOR_0 to.
    float4 colour : COLOR;
};

// `SV_InstanceID` is the index within the draw, not within the buffer: Slang
// declares both `InstanceIndex` and `BaseInstance` and subtracts them, which is
// the D3D semantic. Adding `SV_StartInstanceLocation` back is what makes a
// non-zero `firstInstance` index the ring correctly, and it is why this pass
// requires `shaderDrawParameters`.
uint instanceIndex(uint local_instance, uint first_instance)
{
    return first_instance + local_instance;
}

// A normal carried by the linear part of a transform that need not be rigid.
//
// A normal is defined by staying perpendicular to its surface, and only the
// inverse transpose preserves that: a non-uniform scale stretches the surface
// one way and tilts the normal the other. The instance matrix carries whatever
// scale the asset's node hierarchy declares, and the corpus declares both
// non-uniform and negative ones.
//
// The cofactor matrix is the inverse transpose times the determinant, and the
// fragment stage normalizes what it receives, so only the determinant's sign
// has to be put back. Two things follow from writing it this way. There is no
// division, so a collapsed transform gives a zero normal rather than an
// infinity, and no matrix inverse is needed: this version of Slang has none.
float3 carryNormal(float3 local_normal, float3x3 linear)
{
    const float3x3 cofactor = float3x3(
        cross(linear[1], linear[2]),
        cross(linear[2], linear[0]),
        cross(linear[0], linear[1]));
    const float determinant = dot(linear[0], cofactor[0]);
    return mul(local_normal, cofactor) * (determinant < 0.0 ? -1.0 : 1.0);
}

// The half both entry points share, from a position and a normal already in the
// mesh's local space. Row vectors: the vector is on the left of the matrix,
// which is the convention zmath builds these with and the layout they are
// stored in.
Varying place(
    float3 local_position,
    float3 local_normal,
    float4 local_tangent,
    float2 uv,
    float2 uv1,
    Instance instance)
{
    const float4 world = mul(float4(local_position, 1.0), instance.model);

    Varying varying;
    varying.position = mul(world, camera.view_projection);
    varying.world_position = world.xyz;
    varying.normal = carryNormal(local_normal, (float3x3)instance.model);
    // A tangent follows an edge of the surface rather than standing off it, so
    // it takes the plain matrix and not the one above.
    varying.tangent = float4(
        mul(local_tangent.xyz, (float3x3)instance.model),
        local_tangent.w);
    varying.uv = uv;
    varying.uv1 = uv1;
    varying.material_index = instance.material_index;
    return varying;
}

// The skinned half, shared by the two entry points that read binding 1.
Varying placeSkinned(
    Vertex vertex,
    uint4 joint_indices,
    float4 joint_weights,
    float2 uv1,
    Instance instance)
{
    // glTF 2.0 specification, 3.7.3.3: the per-vertex transform is the weighted
    // linear sum of the joint matrices. Summing the matrices and transforming
    // once is not an optimisation of transforming four times and summing; it is
    // what the section states, and the two differ for a normal.
    const uint base = instance.joint_base;
    const float4x4 skin =
        joint_weights.x * joints[base + joint_indices.x] +
        joint_weights.y * joints[base + joint_indices.y] +
        joint_weights.z * joints[base + joint_indices.z] +
        joint_weights.w * joints[base + joint_indices.w];

    // Section 3.7.3.2: only the joint transforms are applied to a skinned mesh,
    // and the transform of the mesh's own node is ignored. The importer leaves
    // these vertices in local space for that reason, so the skin matrix is what
    // places them, and `model` carries the engine's placement of the whole model
    // and nothing from the asset's mesh node.
    const float3 skinned_position = mul(float4(vertex.position, 1.0), skin).xyz;
    // The weighted sum is a per-vertex matrix, so the normal's is per vertex
    // too. A joint carrying a non-uniform scale makes the two differ, and a
    // blend of two joints can be non-uniform where neither of them is.
    const float3 skinned_normal = carryNormal(vertex.normal.xyz, (float3x3)skin);
    const float4 skinned_tangent = float4(
        mul(vertex.tangent.xyz, (float3x3)skin),
        vertex.tangent.w);

    return place(skinned_position, skinned_normal, skinned_tangent, vertex.uv, uv1, instance);
}

// Four vertex entry points over two independent axes, selected by name at
// pipeline creation: one module and one set of bindings rather than four builds
// of the file. Skinning reads binding 1 at locations 4 and 5, where the weights
// arrive as floats because the attribute is unorm16 and the vertex input stage
// expands it. The second UV set reads binding 3 at location 7.
//
// The location is stated rather than left to declaration order, which would put
// this attribute at 4 on the unskinned variant and 6 on the skinned one. Binding
// 3 is `GpuUv1Vertex` and it names location 7 for both.
//
// The variant without the stream passes zero and not a copy of the first set: a
// slot reading set 1 on a mesh with no TEXCOORD_1 is an asset error, and a
// silent fall back to set 0 would render it as though it were not.
[shader("vertex")]
Varying vertexMain(
    Vertex vertex,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    return place(
        vertex.position,
        vertex.normal.xyz,
        vertex.tangent,
        vertex.uv,
        float2(0.0),
        instance);
}

[shader("vertex")]
Varying uv1VertexMain(
    Vertex vertex,
    [[vk::location(7)]] float2 uv1 : TEXCOORD1,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    return place(
        vertex.position,
        vertex.normal.xyz,
        vertex.tangent,
        vertex.uv,
        uv1,
        instance);
}

// The four above with COLOR_0 on binding 2 at location 6. The stream is an axis
// of the pipeline variant rather than something every mesh carries: a mesh
// without it would otherwise interpolate a constant white to every fragment,
// and the parameter that carries it is not free.
[shader("vertex")]
VaryingColour colourVertexMain(
    Vertex vertex,
    [[vk::location(6)]] float4 colour : COLOR,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    VaryingColour varying;
    varying.base = place(
        vertex.position,
        vertex.normal.xyz,
        vertex.tangent,
        vertex.uv,
        float2(0.0),
        instance);
    varying.colour = colour;
    return varying;
}

[shader("vertex")]
VaryingColour uv1ColourVertexMain(
    Vertex vertex,
    [[vk::location(6)]] float4 colour : COLOR,
    [[vk::location(7)]] float2 uv1 : TEXCOORD1,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    VaryingColour varying;
    varying.base = place(
        vertex.position,
        vertex.normal.xyz,
        vertex.tangent,
        vertex.uv,
        uv1,
        instance);
    varying.colour = colour;
    return varying;
}

[shader("vertex")]
Varying skinnedVertexMain(
    Vertex vertex,
    uint4 joint_indices : JOINTS,
    float4 joint_weights : WEIGHTS,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    return placeSkinned(vertex, joint_indices, joint_weights, float2(0.0), instance);
}

[shader("vertex")]
Varying skinnedUv1VertexMain(
    Vertex vertex,
    uint4 joint_indices : JOINTS,
    float4 joint_weights : WEIGHTS,
    [[vk::location(7)]] float2 uv1 : TEXCOORD1,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    return placeSkinned(vertex, joint_indices, joint_weights, uv1, instance);
}

[shader("vertex")]
VaryingColour skinnedColourVertexMain(
    Vertex vertex,
    uint4 joint_indices : JOINTS,
    float4 joint_weights : WEIGHTS,
    [[vk::location(6)]] float4 colour : COLOR,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    VaryingColour varying;
    varying.base = placeSkinned(vertex, joint_indices, joint_weights, float2(0.0), instance);
    varying.colour = colour;
    return varying;
}

[shader("vertex")]
VaryingColour skinnedUv1ColourVertexMain(
    Vertex vertex,
    uint4 joint_indices : JOINTS,
    float4 joint_weights : WEIGHTS,
    [[vk::location(6)]] float4 colour : COLOR,
    [[vk::location(7)]] float2 uv1 : TEXCOORD1,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    VaryingColour varying;
    varying.base = placeSkinned(vertex, joint_indices, joint_weights, uv1, instance);
    varying.colour = colour;
    return varying;
}

// Windowed inverse-square falloff: pure 1/d^2 over most of the radius, reaching
// exactly zero at `range` instead of being cut there.
//
// KHR_lights_punctual, "Range Property", recommends exactly
// `max(min(1 - (d/range)^4, 1), 0) / d^2`, and that is what this is. Frostbite
// squares the window on top of it for a vanishing derivative at the cutoff; the
// extension is the authority for the assets this reads, and the kink it leaves
// is in a term that is already zero there.
//
// Squared distance in, so a caller can cull before taking a root. `d^4` is the
// square of that, so no root is needed here either. The floor on the divisor is
// what a surface at the light's own position lands on.
float rangeAttenuation(float distance_squared, float range)
{
    const float ratio = distance_squared / (range * range);
    return saturate(1.0 - ratio * ratio) / max(distance_squared, 1.0e-4);
}

// The cone, between the two cosines. `cos_inner > cos_outer` is established
// where the cone is built, in `lenore-scene`'s `Light.spot`, which is what keeps
// smoothstep off its undefined equal-edge case.
float spotAttenuation(float3 to_light, Light light)
{
    return smoothstep(light.cos_outer, light.cos_inner, dot(to_light, -light.direction));
}

// How much of the sun reaches a point: 1 is lit, 0 is fully shadowed, and the
// runtime strength interpolates between the two.
//
// The geometric normal and not the shaded one. The offset exists to move the
// lookup off the surface the depth was recorded from, and a normal map perturbs
// how a surface faces without moving where it is, so offsetting along it would
// make the shadow follow texture detail that has no depth behind it.
//
// Two things fall outside the fit and both read as lit. Laterally the sampler's
// border does it, being opaque white and so the far plane. In depth the test
// below does it: a point past the far plane would otherwise be compared against
// a cleared texel and come back shadowed, which is how geometry that has
// animated out of the fitted bounds would go dark.
float sunVisibility(float3 world_position, float3 geometric_normal)
{
    if (sun_shadow.strength <= 0.0) return 1.0;

    const float3 lookup = world_position + geometric_normal * sun_shadow.normal_offset;
    const float4 clip = mul(float4(lookup, 1.0), sun_shadow.view_projection);
    const float3 ndc = clip.xyz / clip.w;
    if (ndc.z > 1.0) return 1.0;

    // No Y flip on either side. The bake rasterized through this same matrix
    // with a positive viewport height, so the row a clip position lands on there
    // is the row this coordinate addresses here.
    const float2 uv = ndc.xy * 0.5 + 0.5;
    // An explicit level and not the implicit one. This is reached under
    // per-fragment control flow, both from the facing test at the call site and
    // from the depth test above, and an implicit level is derived from
    // neighbouring fragments that may not have taken the same path. The map has
    // one level in any case, so naming it costs nothing; it lowers to
    // `OpImageSampleDrefExplicitLod` with a level of zero, which the
    // disassembly shows.
    const float visible = sun_shadow_map.SampleCmpLevelZero(uv, ndc.z);
    return 1.0 - (1.0 - visible) * sun_shadow.strength;
}

static const float pi = 3.14159265;

// Mirrors the bits of `material_storage.texture_present` used by this pass.
// Base colour has no bit: a material without one is bound to white.
static const uint texture_present_normal = 1u << 0;
static const uint texture_present_metallic_roughness = 1u << 1;
static const uint texture_present_emissive = 1u << 2;
static const uint texture_present_occlusion = 1u << 3;

// Mirrors the ordinals of `material_storage.TextureSlot`, which are the indices
// of the packed transform array and are pinned by asserts on that side.
static const uint texture_slot_base_colour = 0;
static const uint texture_slot_metallic_roughness = 1;
static const uint texture_slot_normal = 2;
static const uint texture_slot_emissive = 3;
static const uint texture_slot_occlusion = 4;

// KHR_texture_transform for one slot, and the choice of UV set with it. The
// matrix is applied as row vectors on the left, matching the row-major pair the
// host packs and the convention the rest of this file uses for `model`.
//
// The set arrives as a float because the record is four float lanes and a
// separate integer lane would cost sixteen bytes of padding per slot. It is
// written by `@floatFromInt` from a value the importer has already rejected
// above one, so the halfway test reads a 0.0 or a 1.0 and nothing between.
float2 transformedUv(TexTransform transform, float2 uv0, float2 uv1)
{
    const float2 uv = transform.params.z < 0.5 ? uv0 : uv1;
    return float2(dot(transform.rs.xy, uv), dot(transform.rs.zw, uv)) + transform.params.xy;
}

// Mirrors the ordinals of `MaterialInfo.Rendering.AlphaMode`, which
// material_storage.zig packs into the first flag lane and pins with asserts.
// OPAQUE is zero and is never compared against, because it is what every branch
// below falls through to.
static const uint alpha_mode_mask = 1u;
static const uint alpha_mode_blend = 2u;

// The floor the roughness is held above. A punctual light is a delta, so at the
// mirror direction the distribution below is 1/(pi * a^2) with a = roughness^2,
// and the whole specular term approaches intensity / (4 * pi * a^2). The HDR
// attachment's narrowest channel is an unsigned 10-bit float, whose largest
// finite value is 2^15 * (1 + 31/32) = 63488 (Khronos Data Format
// Specification 1.3, section "Unsigned 10-bit floating-point numbers"). At this
// floor a light of unit-ish intensity peaks near 6e3, an order of magnitude
// under that, which leaves room for several lights before a highlight saturates
// the format rather than the tone mapper.
static const float min_roughness = 0.08;

// Specular antialiasing. A GGX lobe narrower than the pixel it is sampled at
// falls between shading points, so a highlight appears and disappears as the
// camera turns by a fraction of a degree. The filter widens the lobe by how
// much the shading normal varies across the pixel, which is what makes it hold
// still. Tokuyoshi and Kaplanyan, "Improved Geometric Specular Antialiasing",
// I3D '19, section 4.5 equation 4 and Listing 2: the conservative isotropic
// form, whose kernel is the trace of the filter covariance. Section 4.1 is why
// a normal-based kernel is used in a forward renderer as well, its cost being
// independent of the number of lights.
//
// The paper's sigma squared. Sigma is 0.5, the standard deviation of the pixel
// filter kernel in image space measured in pixels, so this is a property of
// what a pixel is and not a setting.
static const float specular_aa_variance = 0.25;

// The paper's kappa, the widest the kernel may push the lobe. Set by looking,
// because the whole of what it sets is how much sharpness is traded for how much
// stability, and nothing offline ranks that trade.
//
// The paper's own 0.18 is a clamp meant to be reached rarely, and at the floor
// roughness it is not usable here. The filter adds to alpha squared, which at
// `min_roughness` is 4.1e-5, so any threshold above about 1e-4 swamps the
// material entirely: 0.18 takes a perceptual roughness of 0.08 to 0.65, which
// is a mirror rendered as a matte surface. A sharp ridge in a normal map
// saturates the clamp at every pixel it covers, so the clamp is what the ridge
// gets, not the variance.
//
// Deliberately below what fully stabilises that ridge. A lobe wide enough to
// never fall between shading points there is a lobe wide enough to have no
// highlight in it, so this is set to keep the highlight and accept some
// shimmer. Measured against a material at the floor roughness: 1e-3 gives 0.18,
// 3e-3 gives 0.24, 1e-2 gives 0.32.
//
// Judged in motion against builds an order of magnitude either side, which is
// where the trade is visible and a still frame says nothing.
static const float specular_aa_threshold = 0.002;

// The Trowbridge-Reitz/GGX distribution, glTF 2.0 Appendix B.3.2. The Heaviside
// term is the caller's: this is reached only where the light is above the
// horizon.
float distribution(float n_dot_h, float alpha)
{
    const float alpha_squared = alpha * alpha;
    const float denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    return alpha_squared / (pi * denominator * denominator);
}

// The Smith joint masking-shadowing function in its visibility form, from the
// same section: V = G / (4 |N.L| |N.V|), so the 1/(4 |N.L| |N.V|) of the
// microfacet BRDF is already inside this and is not applied again.
float visibility(float n_dot_l, float n_dot_v, float alpha)
{
    const float alpha_squared = alpha * alpha;
    const float masking = n_dot_l + sqrt(alpha_squared + (1.0 - alpha_squared) * n_dot_l * n_dot_l);
    const float shadowing = n_dot_v + sqrt(alpha_squared + (1.0 - alpha_squared) * n_dot_v * n_dot_v);
    return 1.0 / (masking * shadowing);
}

// Schlick's approximation, glTF 2.0 Appendix B.3.4. f90 is 1.0: the grazing
// reflectance of any material approaches white.
float3 fresnel(float3 f0, float v_dot_h)
{
    const float grazing = 1.0 - abs(v_dot_h);
    const float weight = grazing * grazing * grazing * grazing * grazing;
    return f0 + (1.0 - f0) * weight;
}

// The metallic-roughness BRDF, glTF 2.0 Appendix B.3.5, in the form that
// section reaches after folding the metal and dielectric mixes together:
//
//   c_diff = base colour * (1 - metallic)
//   f0     = lerp(0.04, base colour, metallic)
//   F      = f0 + (1 - f0) * (1 - |V.H|)^5
//   diffuse  = (1 - F) * (1/pi) * c_diff
//   specular = F * D(a) * V(a)
//
// The dielectric f0 of 0.04 is the fixed index of refraction 1.5 the same
// section states, and it is why a non-metal cannot be given a reflectance here.
float3 brdf(
    float3 base,
    float metallic,
    float alpha,
    float3 normal,
    float3 to_light,
    float3 to_eye)
{
    const float3 half_vector = normalize(to_light + to_eye);
    const float n_dot_l = max(dot(normal, to_light), 0.0);
    const float n_dot_v = max(dot(normal, to_eye), 0.0);
    const float n_dot_h = max(dot(normal, half_vector), 0.0);
    const float v_dot_h = max(dot(to_eye, half_vector), 0.0);

    const float3 f0 = lerp(float3(0.04), base, metallic);
    const float3 f = fresnel(f0, v_dot_h);

    const float3 diffuse = (1.0 - f) * (1.0 / pi) * (base * (1.0 - metallic));
    const float3 specular = f * distribution(n_dot_h, alpha) * visibility(n_dot_l, n_dot_v, alpha);
    return diffuse + specular;
}

// The Fresnel-weighted environment BRDF, with the energy a single-scatter
// microfacet model loses added back. Fdez-Aguera, "A Multiple-Scattering
// Microfacet Model for Real-Time Image-based Lighting", Journal of Computer
// Graphics Techniques 8(1), 2019. Mirrored by `iblFresnel` in shading.zig.
//
// The reflectance is raised toward 1 - roughness before Schlick is applied,
// because a rough surface reflects less at grazing angles than a smooth one and
// plain Schlick does not know that. The second term redistributes the energy the
// single-scatter integral drops, which is what stops a rough metal darkening as
// roughness grows.
float3 iblFresnel(float3 f0, float roughness, float n_dot_v, float2 f_ab)
{
    const float grazing = 1.0 - n_dot_v;
    const float weight = grazing * grazing * grazing * grazing * grazing;

    const float3 reflectance = max(float3(1.0 - roughness), f0) - f0;
    const float3 k_s = f0 + reflectance * weight;
    const float3 single = k_s * f_ab.x + f_ab.y;

    // The 21 is the analytic average of the Schlick term over the hemisphere,
    // from the same paper.
    const float missing = 1.0 - (f_ab.x + f_ab.y);
    const float3 average = f0 + (1.0 - f0) / 21.0;
    // A perfect reflector has an average of exactly one, so a lookup table
    // reading zero at that texel drives this denominator to zero while the
    // numerator is also zero. The quotient has a finite limit there and the
    // hardware produces a NaN, which survives being multiplied by a zero
    // interpolation weight and reaches the attachment as black. Measured on a
    // white base colour with a table whose channels were empty.
    const float3 multiple =
        missing * single * average / max(1.0 - average * missing, 1.0e-4);

    return single + multiple;
}

// The environment's contribution, composed the way glTF 2.0 Appendix B.1
// composes a material: a linear interpolation of a metal and a dielectric BRDF
// by metalness. The metal reflects the environment tinted by its base colour and
// has no diffuse lobe; the dielectric mixes irradiance and reflection by the
// same Fresnel term.
//
// The diffuse term multiplies the base colour itself and not a base colour
// already scaled by 1 - metallic. The mixture is what removes a metal's diffuse
// lobe here, and scaling first would remove it twice.
//
// Left as the mixture on purpose, where the direct BRDF uses the form B.3.5
// folds the same mix into. That fold does not carry over. `iblFresnel` is the
// multiple-scattering term and not a plain Schlick, and it is not linear in f0,
// so one blended f0 is not the mix of two. Folding this one moves the picture
// visibly at intermediate metalness, not marginally. The two paths therefore
// attenuate diffuse differently there, and each follows the construction its
// Fresnel term was written for.
float3 imageBasedLight(
    float3 base,
    float metallic,
    float roughness,
    float3 normal,
    float3 to_eye)
{
    const float n_dot_v = max(dot(normal, to_eye), 0.0);

    // The prefilter spread one roughness per mip level linearly over the chain,
    // so the level is the roughness times the last index. The count is asked of
    // the descriptor rather than carried as a constant beside it: a constant
    // here and a chain of another depth in the file is a silent mismatch that
    // returns a correctly filtered value of the wrong roughness. What it costs
    // to ask was measured rather than assumed: forty bytes of code and no
    // register, against a mismatch nothing would report.
    uint width, height, levels;
    ggx_environment.GetDimensions(0, width, height, levels);
    const float lod = roughness * float(max(levels, 1u) - 1u);

    const float3 reflection = normalize(reflect(-to_eye, normal));
    const float3 specular = ggx_environment.SampleLevel(reflection, lod).rgb;
    const float3 irradiance = lambertian_environment.Sample(normal).rgb;
    const float2 f_ab = ggx_lut.Sample(float2(n_dot_v, roughness)).rg;

    const float3 diffuse = irradiance * base;
    const float3 metal = iblFresnel(base, roughness, n_dot_v, f_ab) * specular;
    const float3 dielectric = lerp(
        diffuse,
        specular,
        iblFresnel(float3(0.04), roughness, n_dot_v, f_ab));

    return lerp(dielectric, metal, metallic);
}

// Direct lighting plus the environment. A surface facing away from every light
// is no longer black: it still receives whatever the environment sends it, which
// is also what makes a metal read as metal, since a metal has no diffuse lobe
// and the reflection is all it has.
//
// Alpha modes are not sampled: this pass has a pipeline for OPAQUE alone.
// The shading both fragment entry points below run, with the vertex colour as a
// parameter rather than as a field of what is interpolated. A mesh without the
// stream passes white, which is the identity of the product section 3.9.2
// defines and costs nothing to write here.
float4 shade(Varying varying, float4 vertex_colour, bool front_facing)
{
    const MaterialData material = materials[varying.material_index];

    // Section 3.9.2: where both are present the factor is a linear multiplier of
    // the texture. A material with no base-colour image is bound to the neutral
    // white fallback, which is that same rule with the texture at 1.0.
    // The vertex colour is the third term of that product, and section 3.9.2
    // names all three: it is a linear multiplier like the factor, not a tint
    // applied afterwards.
    const float4 sampled = base_colour.Sample(
        transformedUv(material.tex[texture_slot_base_colour], varying.uv, varying.uv1));
    const float3 base = sampled.rgb * material.base_colour_factor.rgb * vertex_colour.rgb;

    // Section 3.9.4: the alpha is the fourth component of the base colour, so
    // the factor multiplies the texture here as it does for the other three
    // channels. MASK is fully opaque at or above the cutoff and fully
    // transparent below it, and the test is made here rather than after the
    // lighting that a dropped fragment would never have needed.
    // Named for the base colour it comes from: `alpha` below is the GGX
    // distribution's parameter and the two are unrelated.
    const uint alpha_mode = material.flags.x;
    const float base_alpha = sampled.a * material.base_colour_factor.a * vertex_colour.a;
    if (alpha_mode == alpha_mode_mask && base_alpha < material.metallic_roughness_cutoff.z)
        discard;

    // KHR_materials_unlit, Definition: the colour is the base-colour term alone
    // and every property of the PBR model bearing on lighting is ignored. So the
    // return is here, above the normal map and the light loop, rather than as a
    // select at the end: the extension exists so that a surface can cost less,
    // and shading it first would keep the cost and throw the result away.
    //
    // Alpha coverage and doubleSided still apply, in the extension's own words.
    // Coverage was decided by the cutoff test above and the alpha is carried out
    // by the same rule as below; doubleSided is a pipeline state and a face that
    // reaches here was not culled. What the reversed normal would have been for
    // is the lighting this material does not have.
    if (material.flags.w != 0)
        return float4(base, alpha_mode == alpha_mode_blend ? base_alpha : 1.0);

    // Section 3.9.2: roughness is green, metalness is blue, and each multiplies
    // its factor. An absent texture has value 1.0. Select that value from the
    // presence mask rather than assigning material meaning to a fallback texel.
    float metallic_sample = 1.0;
    float roughness_sample = 1.0;
    if ((material.flags.z & texture_present_metallic_roughness) != 0)
    {
        const float4 sampled_mr = metallic_roughness.Sample(
            transformedUv(material.tex[texture_slot_metallic_roughness], varying.uv, varying.uv1));
        metallic_sample = sampled_mr.b;
        roughness_sample = sampled_mr.g;
    }

    const float metallic = saturate(metallic_sample * material.metallic_roughness_cutoff.x);
    const float material_roughness = clamp(
        roughness_sample * material.metallic_roughness_cutoff.y,
        min_roughness,
        1.0);

    // Interpolating unit directions gives shorter vectors between vertices.
    // glTF 2.0 section 3.7.2.1 defines B = cross(N,T) * tangent.w. Section
    // 3.9.3 maps all three sampled channels to tangent-space XYZ, applies scale
    // to X and Y, then requires normalization before lighting.
    // Kept separately from the shaded normal below: the sun shadow lookup
    // offsets along the surface's own direction, and neither the map nor the
    // reversal below turns that direction. Where the offset has to push a sample
    // is a question about the light, and the two things that follow are answers
    // about the material and the viewer.
    const float3 geometric_normal = normalize(varying.normal);

    // Section 3.9.5: with doubleSided true, back-face culling is disabled,
    // double sided lighting is enabled, and the back face MUST have its normals
    // reversed before the lighting equation is evaluated. The tangent basis
    // below is built from this vector, so it turns with it and a tangent-space
    // map is read in the frame the face actually presents. The specification
    // says nothing about the tangent itself, and nothing here reverses it.
    //
    // Conditioned on the material because the specification conditions it
    // there: at doubleSided false the back faces are culled, so how one of them
    // would shade is undefined, and reversing unconditionally would be
    // answering a question the specification does not ask.
    const bool reversed = !front_facing && material.flags.y != 0;
    float3 normal = reversed ? -geometric_normal : geometric_normal;
    if ((material.flags.z & texture_present_normal) != 0)
    {
        // Sampled before the basis is tested, so the implicit level of detail is
        // still derived under control flow that is uniform across the quad.
        float3 tangent_normal = normal_texture.Sample(
            transformedUv(material.tex[texture_slot_normal], varying.uv, varying.uv1)).rgb * 2.0 - 1.0;
        tangent_normal.xy *= material.metallic_roughness_cutoff.w;
        tangent_normal = normalize(tangent_normal);

        // A tangent pointing along the normal projects to exactly zero, and
        // normalizing that is a NaN. The importer repairs the vertices it is
        // given, but interpolation can still cancel a tangent between two that
        // were each usable, so the fragment stage checks as well. The cost of
        // missing one is not local: a NaN passes the facing test below, because
        // every comparison against a NaN is false, and the bloom chain spreads
        // one fragment of it over a rectangle of the screen.
        //
        // The geometric normal stands where the basis is gone. There is nothing
        // left to interpret a tangent space map in, and inventing an axis would
        // rotate the map by an arbitrary amount instead of declining to apply it.
        const float3 projected = varying.tangent.xyz - normal * dot(normal, varying.tangent.xyz);
        const float projected_squared = dot(projected, projected);
        if (projected_squared > 1e-12)
        {
            const float3 tangent = projected * rsqrt(projected_squared);
            const float3 bitangent = cross(normal, tangent) * varying.tangent.w;
            normal = normalize(
                tangent * tangent_normal.x +
                bitangent * tangent_normal.y +
                normal * tangent_normal.z);
        }
    }
    // The shading normal, not the geometric one: it is the normal map's ridges
    // that vary fastest inside a pixel, and they are what aliases. Taken here
    // rather than beside the material roughness because the normal is only
    // final once the map has been applied.
    const float3 dndu = ddx(normal);
    const float3 dndv = ddy(normal);
    const float variance = specular_aa_variance * (dot(dndu, dndu) + dot(dndv, dndv));
    const float kernel = min(2.0 * variance, specular_aa_threshold);

    // Equation 4 filters the square of the NDF parameter, and `alpha` is that
    // parameter: glTF 2.0 Appendix B.3.2 defines it as the square of the
    // material roughness, and `distribution` squares it once more. Adding the
    // kernel to the roughness instead would widen the lobe by one power too
    // many.
    const float material_alpha = material_roughness * material_roughness;
    const float alpha = sqrt(saturate(material_alpha * material_alpha + kernel));
    // What the image-based term takes, which reads a mip level and a lookup
    // table from the material's own roughness rather than from the NDF's.
    const float roughness = sqrt(alpha);

    const float3 to_eye = normalize(camera.position.xyz - varying.world_position);

    float3 radiance = float3(0.0);
    const uint live = min(lights.count.x, max_lights);
    for (uint index = 0; index < live; ++index)
    {
        const Light light = lights.lights[index];

        // Toward the source, which is the direction a dot against the normal
        // wants. The light's own direction is the way it travels.
        float3 to_light;
        float attenuation = 1.0;

        if (light.kind == 0)
        {
            to_light = -light.direction;
        }
        else
        {
            const float3 offset = light.position - varying.world_position;
            const float distance_squared = dot(offset, offset);
            // Out of range before any root or divide.
            if (distance_squared > light.range * light.range) continue;
            to_light = offset * rsqrt(max(distance_squared, 1.0e-8));
            attenuation = rangeAttenuation(distance_squared, light.range);
            if (light.kind == 2) attenuation *= spotAttenuation(to_light, light);
        }

        const float lambert = dot(normal, to_light);
        if (lambert <= 0.0) continue;

        // After the facing test, so a surface turned away from the sun does not
        // pay for a lookup whose answer it discards. One light and no other:
        // the map was fitted to a single direction, and the block says which.
        if (index == sun_shadow.light)
            attenuation *= sunVisibility(varying.world_position, geometric_normal);

        const float3 reflected = brdf(base, metallic, alpha, normal, to_light, to_eye);
        radiance += reflected * light.colour * (light.intensity * attenuation * lambert);
    }

    // Section 3.9.3: occlusion attenuates indirect light and leaves direct
    // lighting untouched, so it scales the environment term alone. The loop
    // above and the emissive below are outside it by that rule, and emissive is
    // radiated rather than received in the first place.
    //
    // Section 5.19.6 carries the value in red and requires the other channels to
    // be ignored. Section 5.21.3 gives the strength form, and the strength rides
    // in the emissive vector's fourth lane.
    float occlusion = 1.0;
    if ((material.flags.z & texture_present_occlusion) != 0)
    {
        const float sampled_occlusion = occlusion_texture.Sample(
            transformedUv(material.tex[texture_slot_occlusion], varying.uv, varying.uv1)).r;
        occlusion = 1.0 + material.emissive_factor.w * (sampled_occlusion - 1.0);
    }
    radiance += occlusion * imageBasedLight(base, metallic, roughness, normal, to_eye);

    // Radiance the surface adds on its own, so it is outside the light loop and
    // unlit by construction. Section 3.9.3 makes the decoded linear sample a
    // multiplier of the factor; an absent texture has value 1.0.
    float3 emissive_sample = float3(1.0);
    if ((material.flags.z & texture_present_emissive) != 0)
        emissive_sample = emissive_texture.Sample(
            transformedUv(material.tex[texture_slot_emissive], varying.uv, varying.uv1)).rgb;
    radiance += emissive_sample * material.emissive_factor.rgb;

    // Section 3.9.4: OPAQUE ignores the alpha entirely, and a MASK fragment that
    // reached here passed the cutoff and is fully opaque. Only BLEND carries the
    // value out, and only a blended pipeline enables blending, so what the other
    // two write into this lane is never read.
    return float4(radiance, alpha_mode == alpha_mode_blend ? base_alpha : 1.0);
}

// White for the meshes with no COLOR_0, which is the product's identity. The
// alternative is a constant varying interpolated across every fragment of every
// mesh in the scene, and a packed parameter is not free.
[shader("fragment")]
float4 fragmentMain(Varying varying, bool front_facing : SV_IsFrontFace) : SV_Target
{
    return shade(varying, float4(1.0), front_facing);
}

[shader("fragment")]
float4 colourFragmentMain(VaryingColour varying, bool front_facing : SV_IsFrontFace) : SV_Target
{
    return shade(varying.base, varying.colour, front_facing);
}
