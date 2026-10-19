// Screen-space ambient occlusion: how much of the ambient a point receives,
// estimated from the camera depth the prepass stored, at half the target's
// resolution, and then blurred along each axis with the depth as the guide.
//
// The estimator is the obscurance of McGuire, Mara and Luebke, "Scalable
// Ambient Obscurance", HPG 2012: samples on a spiral around the point, each
// counted by how much it lies above the point's tangent plane and falling off
// to nothing at the radius. So is the depth it reads, their section 2.2: the
// distance in front of the eye at half resolution, and a chain of levels
// below it, so that a tap far out on the disk reads a coarser level and the
// taps of one pixel stay close in memory. Without the chain a wall close to
// the eye spreads twelve taps over hundreds of texels.
//
// Five entry points share one vertex stage, one set and one push block:
// `distanceMain` writes level zero of the chain from the camera depth,
// `downsampleMain` each level below from the one above, `occlusionMain` the
// raw estimate, and `blurMain` runs once along X into a second image and once
// along Y back into the first.

// Mirrors `ao.PushConstants`. Both fragment entry points read it whole.
struct AoPushConstants
{
    // One over the extent being written, in texels.
    float2 texel;

    // `AmbientOcclusionReconstruction` in lenore-scene: the view's half extents
    // at unit distance, and the two projection terms that turn a stored depth
    // into a distance.
    float half_width;
    float half_height;
    float depth_scale;
    float depth_offset;

    // World units, and the pixels one of them spans at unit distance at the
    // resolution being written.
    float radius;
    float pixels_per_unit;

    float intensity;
    float bias;

    // The blur's step, one texel along X or along Y.
    float2 direction;

    // The sub-pixel offset the frame's projection carries, in the framebuffer's
    // device coordinates, zero when unjittered.
    float2 jitter;
};

[[vk::push_constant]]
ConstantBuffer<AoPushConstants> ao;

// The camera depth, through a nearest sampler: a filtered depth between a near
// and a far surface is a point on neither. Read by `distanceMain` alone.
[[vk::binding(0, 0)]] Sampler2D depth_map;
// The estimate the blur reads. Unbound by nothing: every set writes all three.
[[vk::binding(1, 0)]] Sampler2D source;
// The distance chain, read by texel and level and never filtered, for the
// same reason as the depth. `downsampleMain` sees one level, the one above
// the level it writes; the estimate and the blur see the whole chain.
[[vk::binding(2, 0)]] Sampler2D distances;

struct Varying
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
Varying vertexMain(uint index : SV_VertexID)
{
    // The covering triangle of fullscreen.slang and bloom.slang.
    const float2 uv = float2((index << 1) & 2, index & 2);

    Varying varying;
    varying.uv = uv;
    varying.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return varying;
}

// The distance in front of the eye a stored depth stands for; see
// `Reconstruction.distance`.
float distanceAt(float depth)
{
    return ao.depth_offset / (depth + ao.depth_scale);
}

// What the chain holds where nothing was drawn. Far enough that a tap there
// lies outside any radius, and small enough that its square, which the
// estimate takes, stays finite in single precision.
static const float background_distance = 1.0e18;

// The view-space point under `uv` at distance `d`. Row zero is the top of the
// target, and the camera's matrix carries the Vulkan flip, so the device
// coordinate the projection produced is up-positive at the top: `1 - 2 v`.
// A jittered frame drew each point `ao.jitter` away from where it is, in the
// framebuffer's coordinates, whose y runs the other way; both are undone.
float3 viewPoint(float2 uv, float d)
{
    return float3(
        (uv.x * 2.0 - 1.0 - ao.jitter.x) * d * ao.half_width,
        (1.0 - uv.y * 2.0 + ao.jitter.y) * d * ao.half_height,
        -d);
}

// Jimenez, "Next Generation Post Processing in Call of Duty: Advanced
// Warfare", SIGGRAPH 2014: interleaved gradient noise, a per-pixel angle that
// the blur after this averages out without the pattern of a small tiled
// texture.
float interleavedGradientNoise(float2 pixel)
{
    return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
}

// Level zero of the chain: the distance at each half-resolution texel, from
// the one camera depth sample its centre falls on.
[shader("fragment")]
float4 distanceMain(Varying varying) : SV_Target
{
    const float stored = depth_map.SampleLevel(varying.uv, 0.0).r;
    const float d = stored >= 1.0 ? background_distance : distanceAt(stored);
    return float4(d, 0.0, 0.0, 1.0);
}

// One level below another, by the paper's rotated grid subsample (its
// table 1): one of the four texels above, alternating with the position, so
// every value in the chain is a distance something in the scene stands at.
// An average of a near and a far surface is a point on neither.
[shader("fragment")]
float4 downsampleMain(Varying varying) : SV_Target
{
    uint width;
    uint height;
    uint levels;
    distances.GetDimensions(0, width, height, levels);
    const int2 p = int2(varying.position.xy);
    const int2 above = int2(2 * p.x + ((p.y & 1) ^ 1), 2 * p.y + ((p.x & 1) ^ 1));
    const int2 last = int2(int(width), int(height)) - 1;
    return float4(distances.Load(int3(clamp(above, int2(0, 0), last), 0)).r, 0.0, 0.0, 1.0);
}

// How far out on the disk, in texels, a tap moves one level down: the
// paper's q' of equation 9, which it found equivalent anywhere from 2^3 to
// 2^5. As a power of two, so the level is the tap's distance's exponent less
// this.
static const int log_level_step = 3;

// The distance a tap reads, `pixels` out from the point at `uv`, from the
// level of the chain that spacing calls for (equation 10). `base` is level
// zero's extent and `levels` how many the chain has.
float tapDistance(float2 uv, float pixels, int2 base, int levels)
{
    const int level = clamp(int(floor(log2(max(pixels, 1.0)))) - log_level_step, 0, levels - 1);
    const int2 texel = int2(floor(uv * float2(base))) >> level;
    const int2 last = max(base >> level, int2(1, 1)) - 1;
    return distances.Load(int3(clamp(texel, int2(0, 0), last), level)).r;
}

static const uint sample_count = 12;
// Turns of the spiral the samples are spread over, so neighbouring samples are
// not neighbours in angle. Coprime with the count.
static const float spiral_turns = 7.0;
static const float two_pi = 6.28318530718;

// Below this many pixels the whole disk sits inside the point's own texel and
// the samples would only read the point itself back.
static const float min_disk_pixels = 1.0;

[shader("fragment")]
float4 occlusionMain(Varying varying) : SV_Target
{
    uint width;
    uint height;
    uint levels;
    distances.GetDimensions(0, width, height, levels);
    const int2 base = int2(int(width), int(height));

    const float distance = distances.Load(int3(int2(varying.position.xy), 0)).r;
    // The background, unoccluded.
    if (distance >= background_distance) return float4(1.0, 1.0, 1.0, 1.0);
    const float3 centre = viewPoint(varying.uv, distance);

    // Flat: the normal of the depth itself rather than of the shading, which
    // is what the samples are compared against. Across one of these texels
    // `ddx` runs right (+x) and `ddy` down (-y), so their cross product in
    // this order faces the eye; the flip covers the triangle-pair edges where
    // a derivative straddles two surfaces.
    float3 normal = normalize(cross(ddy(centre), ddx(centre)));
    if (dot(normal, -centre) < 0.0) normal = -normal;

    const float disk = ao.radius * ao.pixels_per_unit / distance;
    if (disk < min_disk_pixels) return float4(1.0, 1.0, 1.0, 1.0);

    const float radius_squared = ao.radius * ao.radius;
    // In world units squared, a hundredth of the radius's square, so a sample
    // almost on the point cannot divide by almost nothing.
    const float epsilon = 0.01 * radius_squared;
    const float bias = ao.bias * distance;
    const float spin = interleavedGradientNoise(varying.position.xy) * two_pi;

    float sum = 0.0;
    for (uint i = 0; i < sample_count; ++i)
    {
        const float alpha = (float(i) + 0.5) / float(sample_count);
        const float angle = alpha * spiral_turns * two_pi + spin;
        const float2 offset = float2(cos(angle), sin(angle)) * (alpha * disk);
        const float2 uv = varying.uv + offset * ao.texel;
        const float3 v = viewPoint(uv, tapDistance(uv, alpha * disk, base, int(levels))) - centre;
        const float vv = dot(v, v);
        const float vn = dot(v, normal);
        // Zero at the radius and beyond, so a far surface behind the point
        // does not darken it.
        const float falloff = max(radius_squared - vv, 0.0);
        sum += falloff * falloff * falloff * max((vn - bias) / (vv + epsilon), 0.0);
    }

    // The falloff's cube is up to the radius to the sixth, so dividing by it
    // keeps the scale of `intensity` independent of the radius; five is the
    // normalization the paper's estimator carries.
    const float scale = 5.0 * ao.intensity / (radius_squared * radius_squared * radius_squared * float(sample_count));
    const float visibility = saturate(1.0 - sum * scale);
    return float4(visibility, visibility, visibility, 1.0);
}

// A Gaussian of nine taps, the centre and four each side, normalized.
static const float blur_weights[5] = { 0.153170, 0.144893, 0.122649, 0.092902, 0.062970 };

// How much farther than the centre, as a share of its distance, a tap may lie
// and still be averaged in. Past it the tap is on another surface, and
// averaging across the edge is what bleeds a crease's darkness onto the wall
// in front of it.
static const float edge_share = 0.1;

[shader("fragment")]
float4 blurMain(Varying varying) : SV_Target
{
    uint width;
    uint height;
    uint levels;
    distances.GetDimensions(0, width, height, levels);
    const int2 last = int2(int(width), int(height)) - 1;
    const int2 pixel = int2(varying.position.xy);

    const float centre_distance = distances.Load(int3(pixel, 0)).r;
    float total = source.SampleLevel(varying.uv, 0.0).r * blur_weights[0];
    float weight = blur_weights[0];

    for (int i = 1; i < 5; ++i)
    {
        for (int side = -1; side <= 1; side += 2)
        {
            const float2 uv = varying.uv + ao.direction * ao.texel * float(i * side);
            const int2 tap = clamp(pixel + int2(ao.direction) * (i * side), int2(0, 0), last);
            const float tap_distance = distances.Load(int3(tap, 0)).r;
            const float similar = saturate(1.0 - abs(tap_distance - centre_distance) / (edge_share * centre_distance));
            const float w = blur_weights[i] * similar;
            total += source.SampleLevel(uv, 0.0).r * w;
            weight += w;
        }
    }

    const float visibility = total / weight;
    return float4(visibility, visibility, visibility, 1.0);
}
