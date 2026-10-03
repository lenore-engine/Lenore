// The foliage: planted in compute, counted into an indirect command, and drawn
// without ever being a vertex on the host.
//
// Nothing here is stored between frames. Every blade is derived from the integer
// coordinates of the cell it grows in, so the field is a pure function of the
// camera's position and the seed: no upload, no residency, no rebuild when the
// level of detail changes, and no host memory proportional to how much grass
// there is. What crosses the bus is the camera and the clock.
//
// The grid is snapped to a multiple of its own cell size in world space rather
// than centred on the camera exactly. That is what stops the field from swimming
// as the camera walks: a cell keeps its integer coordinates, so it keeps its
// hash, so its blade keeps its position, its lean and its height until it leaves
// the ring entirely.
//
// Rings are how density falls with distance. Ring k has the same cell count as
// ring zero and twice its spacing, so it covers four times the area with the
// same work, and the blades per square metre fall by four each ring out. The
// alternative, one grid at the near spacing over the whole radius, is sixty-four
// times the threads for grass that is under a pixel wide.

// Mirrors `grass.Blade` on the host. Sixteen-byte alignment puts `yaw` in the
// same slot as the position, which is why it is there and not beside the other
// scalars.
struct Blade
{
    float3 position;
    float yaw;
    float height;
    float width;
    float tint;
    uint kind; // 0 = grass, 1 = wheat
};

// The baked terrain map, two words per texel: the low half of `x` is the height
// and its high half the grass density, and `y` carries the wheat density and the
// cover in the same order. Unorm16 throughout.
//
// A buffer and not an image, because every fetch here is a gather the shader
// filters itself and there is no mip chain to select from. It costs a descriptor
// and no sampler, and the host uploads it by writing bytes.
[[vk::binding(0, 0)]] StructuredBuffer<uint2> map;

[[vk::binding(1, 0)]] RWStructuredBuffer<Blade> blades;

// `VkDrawIndirectCommand` as four words: vertex count, instance count, first
// vertex, first instance. The planter increments the second and the draw reads
// all four without the host ever learning what the number was.
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> command;

struct PlantPush
{
    float3 camera_position;
    float time;

    float3 camera_forward;
    float base_spacing;

    float extent;
    float relief;
    uint map_resolution;
    uint capacity;

    float grass_height;
    float wheat_height;
    float height_spread;
    uint ring_side;

    float cull_cos;
    float density_scale;
    uint blade_vertices;
    uint padding;
};

struct DrawPush
{
    float4x4 view_projection;

    float3 camera_position;
    float time;

    float3 sun_direction; // the way the light travels
    // How far a blade leans at rest, before the wind is added. It belongs here
    // and not in the planting block because it is a property of the shape being
    // drawn, and the vertex stage is the only thing that reads it.
    float lean;

    float3 sun_colour; // already multiplied by the intensity
    float ambient;
};

[[vk::push_constant]] PlantPush plant_push;
[[vk::push_constant]] DrawPush draw_push;

// Wind, in world units. One travelling wave, so neighbouring blades lean
// together and the field moves in gusts rather than shimmering per blade. The
// per-blade part is the deviation from it and lives in the hash below.
static const float2 wind_direction = float2(0.80, 0.60);
static const float wind_wavelength = 9.0;
static const float wind_strength = 0.42;
// Radians per second of the travelling wave. A constant because nothing outside
// this file has an opinion about it, and a push constant that never varies is a
// byte the host has to keep in step for no reason.
static const float wind_speed = 1.7;

// How a blade is built. Four segments up its length: three quads and a triangle
// closing the tip, as a triangle list, because the pipelines this example builds
// against are created with that topology and no other.
//
// Two triangles per quad plus the tip, so `3 * 6 + 3`. The tip is written out
// rather than made a fourth quad whose top edge has zero width: that would cost
// three more vertices per blade in what is the heaviest vertex stage in the
// frame, to save four lines here.
static const uint blade_segments = 4;
static const uint blade_quads = blade_segments - 1;
static const uint blade_vertex_count = blade_quads * 6 + 3;

// How far a blade turns to face the camera, from nothing to fully. A blade seen
// exactly edge-on covers no pixels and vanishes, and a field of them flickers as
// the view turns. Turning them all the way removes the variety that random yaw
// is there to give, so this is partial and deliberately so.
static const float view_facing = 0.55;

// Chris Wellons, "Prospecting for Hash Functions", 2018, the `lowbias32`
// function: https://nullprogram.com/blog/2018/07/31/. The same hash the host
// side of this example plants with, so that a value derived here and a value
// derived there from one cell agree.
uint mix32(uint value)
{
    uint x = value;
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

uint hashCell(int2 cell, uint channel)
{
    return mix32(uint(cell.x) * 0x27d4eb2d ^ uint(cell.y) * 0x165667b1 ^ channel);
}

// The top 24 bits, which is every bit an f32 mantissa holds, mapped to [0, 1).
float unitFloat(uint bits)
{
    return float(bits >> 8) * 0x1p-24;
}

float2 unpackUnorm(uint word)
{
    return float2(word & 0xffffu, word >> 16) * (1.0 / 65535.0);
}

// One texel of the map, as height in the unorm encoding plus the two densities.
float3 fetchMap(int2 texel, uint resolution)
{
    const int2 clamped = clamp(texel, int2(0, 0), int2(int(resolution) - 1, int(resolution) - 1));
    const uint index = uint(clamped.y) * resolution + uint(clamped.x);
    const uint2 word = map[index];
    const float2 low = unpackUnorm(word.x);
    const float2 high = unpackUnorm(word.y);
    return float3(low.x, low.y, high.x);
}

// The map filtered at a world position. Bilinear and by hand: the values vary
// over metres while blades are planted every few centimetres, so the nearest
// texel would plant a whole neighbourhood at one height and the field would
// terrace.
float3 sampleMap(float2 world_xz, float extent, uint resolution)
{
    const float step = 2.0 * extent / float(resolution);
    // The map's texels sit at cell centres, which is the half this subtracts.
    const float2 coordinate = (world_xz + extent) / step - 0.5;
    const int2 base = int2(floor(coordinate));
    const float2 fraction = coordinate - float2(base);

    const float3 c00 = fetchMap(base, resolution);
    const float3 c10 = fetchMap(base + int2(1, 0), resolution);
    const float3 c01 = fetchMap(base + int2(0, 1), resolution);
    const float3 c11 = fetchMap(base + int2(1, 1), resolution);

    return lerp(lerp(c00, c10, fraction.x), lerp(c01, c11, fraction.x), fraction.y);
}

// Sets the indirect command to an empty draw. One thread, because the whole of
// the state is four words and a dispatch that clears it in parallel would need a
// barrier to say when it was done.
[shader("compute")]
[numthreads(1, 1, 1)]
void resetMain()
{
    command[0] = plant_push.blade_vertices;
    command[1] = 0; // the instance count the planter fills
    command[2] = 0;
    command[3] = 0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void plantMain(uint3 id: SV_DispatchThreadID)
{
    const uint ring = id.z;
    const uint side = plant_push.ring_side;
    if (id.x >= side || id.y >= side) return;

    // Ring k has twice the spacing of ring k-1 and the same cell count, so it
    // covers four times the area for the same threads.
    const float spacing = plant_push.base_spacing * float(1u << ring);
    const float half_side = 0.5 * float(side);

    // Snapped to the grid's own pitch, which is what keeps a cell's integer
    // coordinates fixed while the camera moves through it.
    const float2 anchor = floor(plant_push.camera_position.xz / spacing);
    const int2 cell = int2(anchor) + int2(int(id.x), int(id.y)) - int2(int(side / 2), int(side / 2));

    // The inner half of a ring is covered by the ring before it at twice the
    // density, so it is skipped here rather than planted twice. Ring zero has
    // nothing inside it and keeps all of its cells.
    if (ring > 0)
    {
        const int2 inner = abs(cell - int2(anchor));
        if (inner.x < int(side / 4) && inner.y < int(side / 4)) return;
    }

    const float2 centre = (float2(cell) + 0.5) * spacing;

    // A cell's identity is its world coordinates at its own spacing, so the
    // channels below are stable for as long as the cell exists. The ring is
    // folded in because two rings can name the same cell at different pitches.
    const int2 key = cell + int2(int(ring) * 7919, int(ring) * 104729);

    const float2 jitter = float2(
        unitFloat(hashCell(key, 1)),
        unitFloat(hashCell(key, 2))
    ) - 0.5;
    const float2 position = centre + jitter * (spacing * 0.9);

    // Outside the domain the map says nothing, and the ground out there is the
    // skirt rather than the field.
    if (abs(position.x) >= plant_push.extent || abs(position.y) >= plant_push.extent) return;

    // A cone about the view direction rather than six frustum planes. The cone
    // is cheap, has no aspect ratio in it, and is wide enough that a blade
    // entering from the side is already planted; what it is for is discarding
    // the ring behind the camera.
    //
    // Before the map is read, and that order is the point. The cull is a dot
    // product on values already in registers; the fetch below is four loads with
    // no locality between neighbouring threads. Rejecting first is the
    // difference between sampling the map for every cell of every ring and
    // sampling it for the ones that might be drawn.
    const float2 toward = position - plant_push.camera_position.xz;
    const float distance = length(toward);
    // The view direction flattened and renormalised. Taking `xz` of a unit
    // vector leaves something of length cos(pitch), so a camera looking down
    // would shrink one side of this test and plant a lopsided field.
    const float2 heading = plant_push.camera_forward.xz;
    const float heading_length = length(heading);
    if (distance > 1.0 && heading_length > 1e-4 &&
        dot(toward / distance, heading / heading_length) < plant_push.cull_cos) return;

    const float3 sampled = sampleMap(position, plant_push.extent, plant_push.map_resolution);
    const float density = (sampled.y + sampled.z) * plant_push.density_scale;
    if (unitFloat(hashCell(key, 3)) >= density) return;

    const bool wheat = sampled.z > sampled.y;
    const float nominal = wheat ? plant_push.wheat_height : plant_push.grass_height;
    const float roll = unitFloat(hashCell(key, 4)) * 2.0 - 1.0;

    uint slot;
    InterlockedAdd(command[1], 1, slot);
    // The count keeps climbing past the capacity and the draw clamps it, which
    // is the one place this can be wrong without being visible: a field that
    // wanted more blades than the buffer holds draws the ones that fit. The host
    // reads the same number back and reports it.
    if (slot >= plant_push.capacity) return;

    Blade blade;
    blade.position = float3(position.x, plant_push.relief * (2.0 * sampled.x - 1.0), position.y);
    blade.yaw = unitFloat(hashCell(key, 5)) * 6.2831853;
    blade.height = nominal * (1.0 + plant_push.height_spread * roll);
    // Wider further out, so a blade that is under a pixel across does not
    // shimmer in and out as it crosses a sample point. It is the cheap half of
    // what a coverage-preserving fade does and it costs nothing here.
    blade.width = 0.012 * spacing / plant_push.base_spacing;
    blade.tint = unitFloat(hashCell(key, 6));
    blade.kind = wheat ? 1u : 0u;
    blades[slot] = blade;
}

struct Varying
{
    float4 position : SV_Position;
    // Along the blade, zero at the root and one at the tip. Everything the
    // fragment stage does with shape comes from this.
    float height : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float tint : TEXCOORD2;
    nointerpolation uint kind : TEXCOORD3;
    // World space. The fragment stage needs the direction the blade is seen
    // from, and `SV_Position` is in framebuffer coordinates by the time it
    // arrives, not a place in the world.
    float3 world : TEXCOORD4;
};

[shader("vertex")]
Varying bladeVertex(uint vertex_id: SV_VertexID, uint instance_id: SV_InstanceID)
{
    const Blade blade = blades[instance_id];

    // Decoded from the vertex index, so the blade needs no index buffer and no
    // vertex buffer. The first three quads are six vertices each, wound so that
    // the two triangles of a quad share its diagonal, and the last three close
    // the tip.
    uint level;
    float side;
    if (vertex_id < blade_quads * 6)
    {
        const uint quad = vertex_id / 6;
        const uint corner = vertex_id % 6;
        const uint lift = (corner == 1 || corner == 4 || corner == 5) ? 1u : 0u;
        level = quad + lift;
        side = (corner == 2 || corner == 3 || corner == 5) ? 1.0 : -1.0;
    }
    else
    {
        const uint corner = vertex_id - blade_quads * 6;
        level = corner == 2 ? blade_segments : blade_quads;
        side = corner == 1 ? 1.0 : -1.0;
    }
    const float along = float(level) / float(blade_segments);

    const float2 facing = float2(cos(blade.yaw), sin(blade.yaw));
    float3 across = float3(-facing.y, 0.0, facing.x);

    // Turned partly toward the camera. The sign follows the blade's own axis so
    // that the width direction does not flip as the camera crosses the plane the
    // blade stands in, which would swap its two faces and invert the normal.
    const float3 to_camera = draw_push.camera_position - blade.position;
    const float3 level_across = cross(float3(0.0, 1.0, 0.0), to_camera);
    const float level_length = length(level_across);
    if (level_length > 1e-4)
    {
        const float3 view_across = level_across / level_length;
        const float3 aligned = dot(across, view_across) < 0.0 ? -view_across : view_across;
        across = normalize(lerp(across, aligned, view_facing));
    }

    // One travelling wave over the field plus the blade's own offset. The
    // coherent part has to come from world position or the field shimmers
    // instead of bending.
    const float phase = dot(blade.position.xz, wind_direction) / wind_wavelength -
        draw_push.time * wind_speed;
    const float gust = sin(phase) + 0.35 * sin(2.7 * phase + blade.tint * 6.28);
    // How far the tip is displaced, as a fraction of the blade's height. It
    // multiplies `along` squared, so the root stays put and the curvature is
    // constant along the blade.
    const float curvature = draw_push.lean + wind_strength * gust;

    // A quadratic arc, and the tangent is the derivative of that same curve.
    // Taking it rather than assuming the blade is upright is what makes the
    // normal below the surface's own: a bent blade catches a low sun on its back
    // and an upright normal has that face flat.
    const float3 spine = blade.position +
        float3(0.0, blade.height * along, 0.0) +
        float3(facing.x, 0.0, facing.y) * (blade.height * curvature * along * along);
    const float3 tangent = normalize(float3(
        facing.x * 2.0 * curvature * along,
        1.0,
        facing.y * 2.0 * curvature * along
    ));

    // Tapered to a point, so the tip is one vertex and the strip closes itself.
    const float width = blade.width * (1.0 - along * along);

    const float3 world = spine + across * (side * width);

    Varying varying;
    // Row vector times matrix, which is the order every shader the engine ships
    // multiplies in. The other order is the transpose, it compiles, and it draws
    // the field as a fan radiating out of the eye.
    varying.position = mul(float4(world, 1.0), draw_push.view_projection);
    varying.height = along;
    varying.normal = normalize(cross(tangent, across));
    varying.tint = blade.tint;
    varying.kind = blade.kind;
    varying.world = world;
    return varying;
}

// The two ends of each kind's colour ramp, linear. Darker than a photograph of
// grass looks, because this is albedo and not a picture of lit grass.
static const float3 grass_low = float3(0.035, 0.075, 0.022);
static const float3 grass_high = float3(0.085, 0.130, 0.038);
static const float3 wheat_low = float3(0.150, 0.115, 0.040);
static const float3 wheat_high = float3(0.290, 0.225, 0.075);

[shader("fragment")]
float4 bladeFragment(Varying varying, bool front : SV_IsFrontFace) : SV_Target
{
    const float3 low = varying.kind == 1u ? wheat_low : grass_low;
    const float3 high = varying.kind == 1u ? wheat_high : grass_high;
    float3 albedo = lerp(low, high, varying.tint);

    // Darker toward the root. A canopy shadows itself, and without this the
    // field reads as a carpet of separate blades rather than as something with
    // depth in it. The exponent is what the picture wanted rather than a
    // measurement.
    albedo *= 0.30 + 0.70 * varying.height * varying.height;

    // Nothing culls a blade's back, because a blade has two sides and both are
    // seen. The interpolated normal describes one of them, so the other takes
    // its opposite; without this the far half of every blade is lit as though it
    // faced the way its front does.
    const float3 normal = normalize(varying.normal) * (front ? 1.0 : -1.0);
    const float3 to_sun = -draw_push.sun_direction;

    // Wrapped rather than clamped, because a blade is thin enough to light from
    // behind: the half of the field facing away from a low sun is where the
    // reference pictures are brightest, and a clamped Lambert has it black.
    const float wrapped = saturate((dot(normal, to_sun) + 0.4) / 1.4);

    // Forward scattering through the blade, which is the rest of that effect:
    // strongest looking straight into the sun through the leaf. It rises toward
    // the tip because that is where a blade is thin enough to pass light.
    const float3 view = normalize(varying.world - draw_push.camera_position);
    const float through = pow(saturate(dot(draw_push.sun_direction, view)), 4.0) * varying.height;

    const float3 lit = albedo * (wrapped + draw_push.ambient) * draw_push.sun_colour +
        albedo * through * draw_push.sun_colour * 0.6;

    return float4(lit, 1.0);
}
