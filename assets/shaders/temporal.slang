// The temporal resolve: this frame's jittered picture at the render resolution
// gathered with the resolved picture of the frames before it, at the output
// resolution. One covering triangle; `temporal.zig` owns the images and says
// which history each frame reads.
//
// Per output pixel:
// - the new frame's colour, reconstructed from the 3x3 render texels around
//   the pixel, each weighted by how near its jittered sample lies;
// - the same 3x3 as a box in YCoCg, mean and spread, that last frame's colour
//   is clipped into, so history that no longer matches the scene here is
//   pulled back toward it rather than kept;
// - last frame's colour read where this surface was, `uv - motion`, with the
//   motion of the nearest surface in the 3x3 so that an edge takes the motion
//   of what is in front; where the whole 3x3 is background, which the depth
//   prepass leaves without motion, where the camera's turn last frame put it;
// - the two blended, the new frame weighing more where one of its samples
//   lands near the pixel's centre.
//
// Everything is gathered with each colour divided by one plus its luma and
// undone at the end, so a single very bright texel counts no more than white
// does and cannot smear across the history.

// Mirrors `temporal.PushConstants`.
struct TemporalPushConstants
{
    float2 render_size;
    float2 output_size;
    // The frame's sub-pixel offset in render pixels, x right and y down: a
    // render texel's sample sits at its centre less this.
    float2 jitter;
    float blend;
    // Nonzero takes the frame alone.
    uint reset;
    // Last frame's clip position of the background at device (x, y) is
    // `x * sky_across + y * sky_down + sky_ahead`: the view direction through
    // last frame's view-projection with w = 0, so only the camera's turning
    // moves it.
    float4 sky_across;
    float4 sky_down;
    float4 sky_ahead;
};

[[vk::push_constant]]
ConstantBuffer<TemporalPushConstants> temporal;

// The main pass's colour, motion and depth at the render resolution, fetched
// by texel.
[[vk::binding(0, 0)]] Sampler2D current;
[[vk::binding(1, 0)]] Sampler2D motion_vectors;
[[vk::binding(2, 0)]] Sampler2D depth_map;
// Last frame's resolved picture at the output resolution, through a linear
// sampler for the bicubic read below.
[[vk::binding(3, 0)]] Sampler2D history;

struct Varying
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
Varying vertexMain(uint index : SV_VertexID)
{
    // The covering triangle of fullscreen.slang.
    const float2 uv = float2((index << 1) & 2, index & 2);

    Varying varying;
    varying.uv = uv;
    varying.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return varying;
}

struct Resolved
{
    // What the passes after this one read.
    float4 picture : SV_Target0;
    // The same value, kept for the next frame.
    float4 history : SV_Target1;
};

// ITU-R BT.709 luma weights, which are what "brightness" means for the
// compression below; any fixed positive weights would keep it invertible.
float luma(float3 colour)
{
    return dot(colour, float3(0.2126, 0.7152, 0.0722));
}

float3 compress(float3 colour)
{
    return colour / (1.0 + luma(colour));
}

// The inverse of `compress`: a compressed colour's luma is below one.
float3 expand(float3 colour)
{
    return colour / max(1.0 - luma(colour), 1.0e-4);
}

float3 toYCoCg(float3 c)
{
    return float3(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

float3 fromYCoCg(float3 y)
{
    return float3(y.x + y.y - y.z, y.x + y.z, y.x - y.y - y.z);
}

// Last frame's picture at `uv`, by the Catmull-Rom spline over its 4x4 texels,
// sharper than bilinear so that reading the history every frame does not blur
// it a little more each time. Five bilinear fetches: the spline's middle two
// weights on each axis are merged into one fetch between their texels, and
// the four corner texels, whose weights are the smallest, are left out and the
// rest renormalized.
float3 sampleHistory(float2 uv)
{
    const float2 size = temporal.output_size;
    const float2 position = uv * size;
    const float2 centre = floor(position - 0.5) + 0.5;
    const float2 f = position - centre;

    const float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    const float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    const float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    const float2 w3 = f * f * (-0.5 + 0.5 * f);

    const float2 w12 = w1 + w2;
    const float2 middle = (centre + w2 / w12) / size;
    const float2 before = (centre - 1.0) / size;
    const float2 after = (centre + 2.0) / size;

    float3 total = float3(0.0, 0.0, 0.0);
    float weight = 0.0;
    float w;
    w = w12.x * w0.y;
    total += history.SampleLevel(float2(middle.x, before.y), 0.0).rgb * w;
    weight += w;
    w = w0.x * w12.y;
    total += history.SampleLevel(float2(before.x, middle.y), 0.0).rgb * w;
    weight += w;
    w = w12.x * w12.y;
    total += history.SampleLevel(middle, 0.0).rgb * w;
    weight += w;
    w = w3.x * w12.y;
    total += history.SampleLevel(float2(after.x, middle.y), 0.0).rgb * w;
    weight += w;
    w = w12.x * w3.y;
    total += history.SampleLevel(float2(middle.x, after.y), 0.0).rgb * w;
    weight += w;
    // The spline overshoots beside a hard edge.
    return max(total / weight, float3(0.0, 0.0, 0.0));
}

// How much a sample `offset` render pixels from the output pixel's centre
// counts toward it: a Gaussian whose weight falls to a tenth at a pixel.
float sampleWeight(float2 offset)
{
    return exp(-2.3 * dot(offset, offset));
}

// How far past its spread the neighbourhood's box reaches, in standard
// deviations. Wider keeps more history and ghosts more.
static const float box_spread = 1.25;

// `point` moved toward the box's centre until it lies inside, along the line
// to the centre, so its hue is kept where clamping each axis would change it.
float3 clipToBox(float3 point, float3 low, float3 high)
{
    const float3 centre = 0.5 * (high + low);
    const float3 extent = 0.5 * (high - low) + 1.0e-5;
    const float3 offset = point - centre;
    const float3 reach = abs(offset / extent);
    const float most = max(reach.x, max(reach.y, reach.z));
    return most > 1.0 ? centre + offset / most : point;
}

[shader("fragment")]
Resolved fragmentMain(Varying varying)
{
    // Where this output pixel's centre falls among the render texels' samples,
    // in render pixels: texel i's sample sits at i + 0.5 - jitter, so offsetting
    // the point by the jitter instead compares it with plain texel centres.
    const float2 point = varying.uv * temporal.render_size + temporal.jitter;
    const int2 nearest = int2(floor(point));
    const int2 last = int2(temporal.render_size) - 1;

    float3 gathered = float3(0.0, 0.0, 0.0);
    float gathered_weight = 0.0;
    float nearest_weight = 0.0;
    float3 first_moment = float3(0.0, 0.0, 0.0);
    float3 second_moment = float3(0.0, 0.0, 0.0);
    float closest_depth = 2.0;
    int2 closest = nearest;

    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            const int2 texel = clamp(nearest + int2(x, y), int2(0, 0), last);
            const float3 colour = compress(current.Load(int3(texel, 0)).rgb);
            const float w = sampleWeight(float2(texel) + 0.5 - point);
            gathered += colour * w;
            gathered_weight += w;
            nearest_weight = max(nearest_weight, w);

            const float3 ycocg = toYCoCg(colour);
            first_moment += ycocg;
            second_moment += ycocg * ycocg;

            // Depth runs from 0 at the near plane, so the nearest surface is the
            // smallest value.
            const float depth = depth_map.Load(int3(texel, 0)).r;
            if (depth < closest_depth)
            {
                closest_depth = depth;
                closest = texel;
            }
        }
    }

    const float3 now = gathered / gathered_weight;
    float2 history_uv;
    bool behind = false;
    // The depth clear is the far plane, and the background is drawn there.
    if (closest_depth >= 1.0)
    {
        // The output pixel's own centre, which no jitter moves.
        const float2 device = varying.uv * 2.0 - 1.0;
        const float4 clip = device.x * temporal.sky_across + device.y * temporal.sky_down + temporal.sky_ahead;
        // A turn of more than a right angle puts the direction behind last
        // frame's camera, where nothing of it was seen.
        behind = clip.w <= 0.0;
        history_uv = clip.xy / clip.w * 0.5 + 0.5;
    }
    else
    {
        history_uv = varying.uv - motion_vectors.Load(int3(closest, 0)).xy;
    }
    const bool outside = behind || any(history_uv < 0.0) || any(history_uv > 1.0);

    float3 result = now;
    if (temporal.reset == 0 && !outside)
    {
        const float3 mean = first_moment / 9.0;
        const float3 spread = sqrt(max(second_moment / 9.0 - mean * mean, float3(0.0, 0.0, 0.0)));
        const float3 low = mean - box_spread * spread;
        const float3 high = mean + box_spread * spread;
        const float3 before = compress(sampleHistory(history_uv));
        const float3 kept = fromYCoCg(clipToBox(toYCoCg(before), low, high));
        result = lerp(kept, now, temporal.blend * nearest_weight);
    }

    const float4 picture = float4(expand(result), 1.0);
    Resolved resolved;
    resolved.picture = picture;
    resolved.history = picture;
    return resolved;
}
