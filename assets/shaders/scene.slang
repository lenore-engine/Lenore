// The main pass: one instanced mesh, transformed, textured and lit.
//
// Four descriptor sets, split by how often they are rewritten. Set 0 holds what
// the frame supplies and is bound once with a dynamic offset per ring; set 1
// holds the scene's material array and is bound once; set 2 holds one material's
// textures and is bound when the ordered list reaches a new material; set 3
// holds the shadow maps, the sun's and the local lights' array, and is bound
// once.
//
// The maps have a set to themselves rather than slots in set 1. The bake that
// fills them names sets 0 to 2 in its own pipeline layout, so a slot in any of
// them would leave a map bound as a texture while it is the attachment being
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
    float4x4 previous_view_projection;
    uint4 motion;
    // The sub-pixel offset `view_projection` carries this frame, in device
    // coordinates, zero when unjittered.
    float4 jitter;
};

#include "include/lights_block.slang"

// The indirect-light probe field, written by probes.slang and read by
// `sampleProbeField`. Declared before the include, which reads them by name.
[[vk::binding(7, 1)]] StructuredBuffer<float4> probe_sh;
[[vk::binding(8, 1)]] StructuredBuffer<float2> probe_moments;
#include "include/probe_field.slang"

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
    // three-component vector would move these fields and the previous matrix
    // away from the offsets the host's Instance declares.
    uint material_index;
    uint padding_0;
    uint padding_1;
    float4x4 previous_model;
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
// The lighting cache: irradiance arriving at each patch of static surface,
// addressed by the second UV set.
//
// A mesh with no second set has a zero coordinate in it, and the sampler is
// clamped, so it reads the texel at the origin. That texel is black by the
// binding's own contract, which is what lets one shader serve both without a
// flag deciding between them.
[[vk::binding(4, 1)]] Sampler2D lightmap;
// Reserved beside irradiance for application-owned surface shading. Lenore's
// default glTF path does not reinterpret its normal-mapped PBR indirect term as
// Backrooms' achromatic directional cache, but both shaders share one scene-set
// layout and every slot in that layout is declared here for reflection.
[[vk::binding(5, 1)]] Sampler2D lightmap_direction;
// How much ambient reaches each pixel, estimated from the camera depth between
// the prepass and this pass at half the target's resolution, and one wherever
// the effect is off. Read by screen position, so it is the same for every
// surface drawn at a pixel; a blended surface in front of an occluded one
// takes the darkness meant for what is behind it.
[[vk::binding(6, 1)]] Sampler2D ambient_occlusion;
[[vk::binding(9, 1)]] Sampler2D motion_vectors;

static const uint debug_view_motion = 4;

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
// The local lights' layers, through the same comparison sampler: spots and
// point light faces alike, each layer read through its own matrix.
[[vk::binding(1, 3)]] Sampler2DArrayShadow local_shadow_maps;
// The same layers through a nearest sampler that returns depths, for the
// search that finds how far the casters over a point stand from the light.
[[vk::binding(2, 3)]] Sampler2DArray local_depths;

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

// Motion follows the shaded interface so its existing field locations stay fixed.
struct Motion
{
    float4 current : MOTION0;
    float4 previous : MOTION1;
};

struct VaryingMotion
{
    Varying base;
    Motion motion;
};

struct VaryingColourMotion
{
    Varying base;
    float4 colour : COLOR;
    Motion motion;
};

Motion vertexMotion(float4 current, float3 previous_local, Instance instance)
{
    Motion motion;
    motion.current = current;
    motion.previous = mul(mul(float4(previous_local, 1.0), instance.previous_model), camera.previous_view_projection);
    return motion;
}

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

VaryingMotion placeMotion(
    float3 local_position,
    float3 local_normal,
    float4 local_tangent,
    float2 uv,
    float2 uv1,
    Instance instance)
{
    VaryingMotion varying;
    varying.base = place(local_position, local_normal, local_tangent, uv, uv1, instance);
    varying.motion = vertexMotion(varying.base.position, local_position, instance);
    return varying;
}

// The skinned half, shared by the two entry points that read binding 1.
VaryingMotion placeSkinned(
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

    const uint previous_base = camera.motion.x + base;
    const float4x4 previous_skin =
        joint_weights.x * joints[previous_base + joint_indices.x] +
        joint_weights.y * joints[previous_base + joint_indices.y] +
        joint_weights.z * joints[previous_base + joint_indices.z] +
        joint_weights.w * joints[previous_base + joint_indices.w];
    const float3 previous_position = mul(float4(vertex.position, 1.0), previous_skin).xyz;
    VaryingMotion varying;
    varying.base = place(skinned_position, skinned_normal, skinned_tangent, vertex.uv, uv1, instance);
    varying.motion = vertexMotion(varying.base.position, previous_position, instance);
    return varying;
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
VaryingMotion vertexMain(
    Vertex vertex,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    return placeMotion(
        vertex.position,
        vertex.normal.xyz,
        vertex.tangent,
        vertex.uv,
        float2(0.0),
        instance);
}

[shader("vertex")]
VaryingMotion uv1VertexMain(
    Vertex vertex,
    [[vk::location(7)]] float2 uv1 : TEXCOORD1,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    return placeMotion(
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
VaryingColourMotion colourVertexMain(
    Vertex vertex,
    [[vk::location(6)]] float4 colour : COLOR,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    VaryingColourMotion varying;
    const VaryingMotion placed = placeMotion(
        vertex.position,
        vertex.normal.xyz,
        vertex.tangent,
        vertex.uv,
        float2(0.0),
        instance);
    varying.base = placed.base;
    varying.motion = placed.motion;
    varying.colour = colour;
    return varying;
}

[shader("vertex")]
VaryingColourMotion uv1ColourVertexMain(
    Vertex vertex,
    [[vk::location(6)]] float4 colour : COLOR,
    [[vk::location(7)]] float2 uv1 : TEXCOORD1,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    VaryingColourMotion varying;
    const VaryingMotion placed = placeMotion(
        vertex.position,
        vertex.normal.xyz,
        vertex.tangent,
        vertex.uv,
        uv1,
        instance);
    varying.base = placed.base;
    varying.motion = placed.motion;
    varying.colour = colour;
    return varying;
}

[shader("vertex")]
VaryingMotion skinnedVertexMain(
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
VaryingMotion skinnedUv1VertexMain(
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
VaryingColourMotion skinnedColourVertexMain(
    Vertex vertex,
    uint4 joint_indices : JOINTS,
    float4 joint_weights : WEIGHTS,
    [[vk::location(6)]] float4 colour : COLOR,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    VaryingColourMotion varying;
    const VaryingMotion placed = placeSkinned(vertex, joint_indices, joint_weights, float2(0.0), instance);
    varying.base = placed.base;
    varying.motion = placed.motion;
    varying.colour = colour;
    return varying;
}

[shader("vertex")]
VaryingColourMotion skinnedUv1ColourVertexMain(
    Vertex vertex,
    uint4 joint_indices : JOINTS,
    float4 joint_weights : WEIGHTS,
    [[vk::location(6)]] float4 colour : COLOR,
    [[vk::location(7)]] float2 uv1 : TEXCOORD1,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    VaryingColourMotion varying;
    const VaryingMotion placed = placeSkinned(vertex, joint_indices, joint_weights, uv1, instance);
    varying.base = placed.base;
    varying.motion = placed.motion;
    varying.colour = colour;
    return varying;
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

// Which of a point light's six layers a direction from the light falls in: the
// largest component's axis, the negative face second. The face order of
// `LocalShadowFit` in lenore-scene, whose margin keeps a direction on the edge
// between two faces inside whichever this picks.
uint localFace(float3 from_light)
{
    const float3 magnitude = abs(from_light);
    uint axis = 0;
    float largest = magnitude.x;
    if (magnitude.y > largest) { axis = 1; largest = magnitude.y; }
    if (magnitude.z > largest) axis = 2;
    return 2 * axis + (from_light[axis] < 0.0 ? 1 : 0);
}

// Taps in the filter, and gathers (four texels each) in the blocker search.
static const uint soft_taps = 16;
static const uint search_gathers = 8;

// DECIDE: below this irradiance from a light at a point, in the light block's
// units, its shadow there is taken hard. The ambient's sky in Reclaim gives
// about 0.16; a lamp of intensity 5 falls under 0.5 past about 3.2 m. Closed by
// looking at a soft edge at the boundary, and by the frame time.
static const float soft_shadow_least_irradiance = 0.5;
// The widest a penumbra or a search may spread, in texels of the layer. Past
// it the taps are too sparse to average into an edge, and on a point light's
// face they would cross more of the two-texel margin than it holds.
static const float max_soft_texels = 24.0;
static const float golden_angle = 2.39996323;

// Tap `index` of `count` on a unit disk, spread evenly by area along the golden
// angle (Vogel 1979) and turned by `spin`.
float2 vogelTap(uint index, uint count, float spin)
{
    const float radius = sqrt((float(index) + 0.5) / float(count));
    const float angle = float(index) * golden_angle + spin;
    return radius * float2(cos(angle), sin(angle));
}

// Jimenez 2014 interleaved gradient noise, as in ao.slang: a per-pixel turn of
// the tap pattern, so neighbouring pixels sample different points of the disk.
float shadowNoise(float2 pixel)
{
    return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
}

// The distance along a layer's axis a stored depth stands for. The layer is
// zmath's `perspectiveFovRh` from the shared near plane to the light's range,
// so stored = f (z - n) / (z (f - n)) and this is its inverse.
float layerDistance(float stored, float range)
{
    const float near = lights.local_shadow.near;
    return near * range / (range - stored * (range - near));
}

// How much of a local light reaches a point, as `sunVisibility` for the sun.
//
// The normal offset is authored in texels like the sun's, but a perspective
// texel grows with distance, so it is scaled by the distance from the light.
// The face is picked from the offset point, the one projected, so the face and
// the projection cannot disagree about which side of an edge the point is on.
//
// Behind a spot's projection, and past its far plane, read as lit: the cone's
// own attenuation is zero there in the first case, and in the second the
// light's range test already rejected the point before this was called.
//
// With a source radius the edge is soft, by percentage-closer soft shadows
// (Fernando, "Percentage-Closer Soft Shadows", SIGGRAPH 2005 sketch): the
// casters over the point are found and averaged, the penumbra is the width
// similar triangles give a light of that radius through a caster at that
// distance, and the comparison is filtered over it. Taps are clamped into the
// layer, so a wide filter on a point light's face repeats the edge rather than
// reading the border, which reads as lit.
float localVisibility(Light light, float3 world_position, float3 geometric_normal, float2 pixel, float arriving)
{
    const float strength = lights.local_shadow.strength;
    if (strength <= 0.0 || light.shadow_layer == no_shadow_layer) return 1.0;

    const float distance = length(world_position - light.position);
    const float offset = lights.local_shadow.normal_offset_texels * light.shadow_texel_slope * distance;
    const float3 lookup = world_position + geometric_normal * offset;

    uint layer = light.shadow_layer;
    if (light.kind == 1) layer += localFace(lookup - light.position);

    const float4 clip = mul(float4(lookup, 1.0), lights.shadow_view_projection[layer]);
    if (clip.w <= 0.0) return 1.0;
    const float3 ndc = clip.xyz / clip.w;
    if (ndc.z > 1.0) return 1.0;

    // No Y flip on either side, as for the sun, and an explicit level for the
    // same reason as there.
    const float2 uv = ndc.xy * 0.5 + 0.5;
    const float source = lights.local_shadow.source_radius;
    // A light this dim at the point gets the hard edge: the penumbra's width
    // is invisible under the rest of the light there, and the soft search is
    // most of a shadowed light's cost.
    if (source <= 0.0 || arriving < soft_shadow_least_irradiance)
    {
        const float visible = local_shadow_maps.SampleCmpLevelZero(float3(uv, float(layer)), ndc.z);
        return 1.0 - (1.0 - visible) * strength;
    }

    uint width, height, layers;
    local_depths.GetDimensions(width, height, layers);
    const float texel = 1.0 / float(width);
    const float2 lowest = float2(0.5 * texel, 0.5 * texel);
    const float2 highest = float2(1.0 - 0.5 * texel, 1.0 - 0.5 * texel);

    // Along the layer's axis, which is what its depth measures, and the
    // world width of one texel at that distance.
    const float receiver = clip.w;
    const float texel_world = light.shadow_texel_slope * receiver;
    const float spin = shadowNoise(pixel) * 6.28318530718;

    // A caster at distance b from the light shades the point if it lies in the
    // cone from the point to the light's disk, whose radius there is
    // r (receiver - b) / receiver. Searched for casters down to a quarter of
    // the receiver's distance, where that cone is three radii wide at the
    // point's own distance in texels.
    //
    // Gathered: each fetch returns the 2x2 texels around its point, so the
    // search reads four times the texels it makes fetches.
    const float search = min(3.0 * source / texel_world, max_soft_texels);
    float blockers = 0.0;
    uint found = 0;
    for (uint i = 0; i < search_gathers; ++i)
    {
        const float2 tap = clamp(uv + vogelTap(i, search_gathers, spin) * (search * texel), lowest, highest);
        const float4 stored = local_depths.Gather(float3(tap, float(layer)));
        for (uint k = 0; k < 4; ++k)
        {
            if (stored[k] < ndc.z)
            {
                blockers += layerDistance(stored[k], light.range);
                found += 1;
            }
        }
    }
    if (found == 0) return 1.0;
    // Every texel searched holds a caster in front: the search reaches as far
    // as any penumbra that a caster at a quarter of the distance could cast,
    // so the filter would find nothing lit either.
    if (found == 4 * search_gathers) return 1.0 - strength;
    const float blocker = blockers / float(found);

    // Similar triangles: the penumbra's radius on the receiver is
    // r (receiver - blocker) / blocker, in texels at the receiver's distance.
    // At least one texel, the hard edge the map resolves.
    const float penumbra = clamp(source * (receiver - blocker) / (blocker * texel_world), 1.0, max_soft_texels);
    float visible = 0.0;
    for (uint i = 0; i < soft_taps; ++i)
    {
        const float2 tap = clamp(uv + vogelTap(i, soft_taps, spin) * (penumbra * texel), lowest, highest);
        visible += local_shadow_maps.SampleCmpLevelZero(float3(tap, float(layer)), ndc.z);
    }
    visible /= float(soft_taps);
    return 1.0 - (1.0 - visible) * strength;
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

// The ambient over the cosine lobe around `normal`, over pi.
float3 ambientIrradiance(float3 normal)
{
    return hemisphereIrradiance(lights.ambient.sky, lights.ambient.ground, normal);
}

// What the surroundings send a diffuse surface, over pi: the probe field where
// there is one, fading into the ambient past its edge, and the ambient alone
// where there is none. The ambient is what an open point receives, which is
// what a point the probes do not reach is taken to be.
float3 surroundingIrradiance(float3 position, float3 normal, float3 to_eye)
{
    const float3 open = ambientIrradiance(normal);
    if (lights.probe_grid.enabled == 0) return open;
    const float pi = 3.14159265358979;
    return sampleProbeField(lights.probe_grid, position, normal, to_eye, open * pi) * (1.0 / pi);
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
    float3 position,
    float3 base,
    float metallic,
    float roughness,
    float3 normal,
    float3 to_eye,
    float2 uv1)
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
    // The ambient is added to what the environment sends rather than being a
    // term of its own, so it takes the same split-sum Fresnel and the same
    // occlusion. Its specular half is read along the reflection unblurred:
    // the hemisphere is already smooth enough that roughness changes it little.
    float3 specular = ggx_environment.SampleLevel(reflection, lod).rgb + lights.ambient.reflected * ambientRadiance(reflection);
    // With a probe field the ambient's diffuse half comes from it instead: the
    // probes carry the sky through the openings they see it through and the
    // light bounced off everything else, where the hemisphere carries the sky
    // everywhere, walls or not. The field returns irradiance; this term holds
    // it over pi, the convention the lambertian map arrives in.
    float3 surroundings;
    if (lights.probe_grid.enabled == 0)
    {
        surroundings = ambientIrradiance(normal);
    }
    else
    {
        // The environment is the world outside, seen from nowhere in
        // particular, and a reflection indoors must not show it through the
        // walls. So the field also says, along the reflection, how much of
        // the environment the probes see there and what else arrives: the
        // reflection is the environment by that share, plus the rest of
        // what the probes carry along it, which is the room's own light
        // (its lamps, its walls), as a fully rough surface would see it. The
        // rest is the field's light less the environment's share of the
        // ambient the probes were lit by, never below zero. Outdoors the
        // share is near one and the environment stays sharp; in a closed
        // room it is near zero and the room reflects itself, blurred to the
        // probes' second-order harmonics.
        const float pi = 3.14159265358979;
        const float3 open = ambientIrradiance(normal);
        const float3 open_along = ambientIrradiance(reflection);
        const ProbeLookup lookup = sampleProbeReflection(lights.probe_grid, position, normal, to_eye, reflection, open * pi, open_along * pi);
        surroundings = lookup.irradiance * (1.0 / pi);
        const float3 room = max(lookup.along * (1.0 / pi) - lookup.visible * open_along, float3(0.0, 0.0, 0.0));
        specular = specular * lookup.visible + room;
    }
    const float3 irradiance = lambertian_environment.Sample(normal).rgb + surroundings;
    const float2 f_ab = ggx_lut.Sample(float2(n_dot_v, roughness)).rg;

    // Two sources of the same quantity, summed before the albedo multiplies
    // them: what the world outside sends this normal, and what was computed for
    // this patch of surface and stored. The cache holds irradiance, and what
    // leaves a diffuse surface is that over pi, which is the convention the
    // environment's prefiltered map already arrives in.
    const float3 stored = lightmap.Sample(uv1).rgb * (1.0 / 3.14159265358979);
    const float3 diffuse = (irradiance + stored) * base;
    const float3 metal = iblFresnel(base, roughness, n_dot_v, f_ab) * specular;
    const float3 dielectric = lerp(
        diffuse,
        specular,
        iblFresnel(float3(0.04), roughness, n_dot_v, f_ab));

    return lerp(dielectric, metal, metallic);
}

// The alpha arithmetic shared by the depth prepass and the shaded pass. Keeping
// the cutoff in one function makes a MASK surface write depth for exactly the
// fragments the colour pass later accepts.
float baseAlpha(MaterialData material, float sampled_alpha, float vertex_alpha)
{
    return sampled_alpha * material.base_colour_factor.a * vertex_alpha;
}

void applyAlphaCutoff(MaterialData material, float alpha)
{
    if (material.flags.x == alpha_mode_mask && alpha < material.metallic_roughness_cutoff.z)
        discard;
}

void maskDepth(Varying varying, float vertex_alpha)
{
    const MaterialData material = materials[varying.material_index];
    const float sampled_alpha = base_colour.Sample(
        transformedUv(material.tex[texture_slot_base_colour], varying.uv, varying.uv1)).a;
    applyAlphaCutoff(material, baseAlpha(material, sampled_alpha, vertex_alpha));
}

// Direct lighting plus the environment. A surface facing away from every light
// is no longer black: it still receives whatever the environment sends it, which
// is also what makes a metal read as metal, since a metal has no diffuse lobe
// and the reflection is all it has.
//
// Both shaded fragment entry points run this body, with the vertex colour as a
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
    // channels. MASK coverage was already applied by the depth prepass. Keeping
    // `discard` out of these shaded entry points lets their early depth test
    // reject hidden surfaces before any lighting runs.
    // Named for the base colour it comes from: `alpha` below is the GGX
    // distribution's parameter and the two are unrelated.
    const uint alpha_mode = material.flags.x;
    const float base_alpha = baseAlpha(material, sampled.a, vertex_colour.a);

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
        // Only a point or a spot is ever given a layer; the frame refuses
        // anything else.
        if (light.shadow_layer != no_shadow_layer)
            attenuation *= localVisibility(
                light,
                varying.world_position,
                geometric_normal,
                varying.position.xy,
                light.intensity * attenuation * lambert * dot(light.colour, float3(0.2126, 0.7152, 0.0722)));

        const float3 reflected = brdf(base, metallic, alpha, normal, to_light, to_eye);
        radiance += reflected * light.colour * (light.intensity * attenuation * lambert);
    }
    // The views of `uniforms.DebugView` that need nothing past the loop.
    const uint view = lights.debug.x;
    if (view == debug_view_motion)
    {
        const float2 m = motion_vectors.Load(int3(int2(varying.position.xy), 0)).xy;
        return float4(float3(0.5 + 20 * m.x, 0.5 + 20 * m.y, 0.5), 1.0);
    }
    if (view == 2) return float4(radiance, 1.0);
    if (view == 1)
    {
        const float3 surroundings = surroundingIrradiance(varying.world_position, normal, to_eye);
        return float4(surroundings, 1.0);
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
    // The screen-space estimate multiplies the authored occlusion: both say how
    // much of the ambient arrives, one from the asset and one from the scene
    // around it. A target texel maps to texel floor(x / 2) of the half-size
    // image, which an odd side rounds up, so the coordinate is the position
    // halved over the image's own size rather than over the target's.
    {
        uint width, height;
        ambient_occlusion.GetDimensions(width, height);
        const float2 uv = varying.position.xy * 0.5 / float2(width, height);
        occlusion *= ambient_occlusion.SampleLevel(uv, 0.0).r;
    }
    if (view == 3) return float4(occlusion, occlusion, occlusion, 1.0);
    radiance += occlusion * imageBasedLight(
        varying.world_position,
        base,
        metallic,
        roughness,
        normal,
        to_eye,
        varying.uv1);

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

// The jitter is taken out of the current position and never was in the
// previous one, so a still scene reads no motion while the samples move.
float4 fragmentMotion(Motion motion)
{
    const float2 uv_motion = (motion.current.xy / motion.current.w - camera.jitter.xy -
        motion.previous.xy / motion.previous.w) * 0.5;
    return float4(uv_motion, 0.0, 0.0);
}

[shader("fragment")]
float4 motionFragmentMain(VaryingMotion varying) : SV_Target
{
    return fragmentMotion(varying.motion);
}

[shader("fragment")]
float4 colourMotionFragmentMain(VaryingColourMotion varying) : SV_Target
{
    return fragmentMotion(varying.motion);
}

[shader("fragment")]
float4 maskFragmentMain(VaryingMotion varying) : SV_Target
{
    maskDepth(varying.base, 1.0);
    return fragmentMotion(varying.motion);
}

[shader("fragment")]
float4 colourMaskFragmentMain(VaryingColourMotion varying) : SV_Target
{
    maskDepth(varying.base, varying.colour.a);
    return fragmentMotion(varying.motion);
}

// White for the meshes with no COLOR_0, which is the product's identity. The
// alternative is a constant varying interpolated across every fragment of every
// mesh in the scene, and a packed parameter is not free.
[earlydepthstencil]
[shader("fragment")]
float4 fragmentMain(Varying varying, bool front_facing : SV_IsFrontFace) : SV_Target
{
    return shade(varying, float4(1.0), front_facing);
}

[earlydepthstencil]
[shader("fragment")]
float4 colourFragmentMain(VaryingColour varying, bool front_facing : SV_IsFrontFace) : SV_Target
{
    return shade(varying.base, varying.colour, front_facing);
}
