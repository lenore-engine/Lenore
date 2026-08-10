// The bloom chain: the HDR target reduced through a half-resolution mip chain
// and walked back up, so that a bright surface spills light into the pixels
// around it the way a lens and a sensor do.
//
// Two directions, three vertices each, one covering triangle per level. Down
// takes thirteen bilinear taps of the level above and writes the level below.
// Up takes a three by three tent of the level below and adds it onto the level
// above, which the blend unit does: the pipeline's additive mode is what makes
// this shader's output a summand rather than a replacement.
//
// The whole chain runs on linear radiance. Nothing here tone maps, and the post
// pass adds level zero into the scene before its operator, so the glow is
// compressed by the same curve the surface under it is.
//
// One set with one binding, rebound per step: the source is the HDR target for
// the first downsample and a level of the chain for every step after. The level
// being written is never the level being read, and the two are separate views
// of one image, so no step samples its own attachment.

// Shared by both fragment entry points, because both are created against one
// pipeline layout. Each reads a subset: the downsample ignores the weight and
// the upsample ignores everything but the weight and the texel size.
struct BloomPushConstants
{
    // One over the source level's extent, in texels. Pushed rather than queried:
    // the host creates every level and so knows the number exactly, and a query
    // would put an image operation in the inner loop to recover it.
    float2 source_texel;

    // Applied on the extract step alone, so that the threshold below is a
    // statement about what will be displayed as white rather than about the
    // scene's own units. The post pass exposes the surface term with the same
    // value and does not expose this chain a second time.
    float exposure;

    // Where a pixel starts contributing, and over what range it reaches full
    // contribution. The ramp is smooth in value and slope, so an emitter
    // drifting across the threshold fades rather than switching.
    float threshold;
    float knee;

    // The contribution floor, applied to every pixel whatever its brightness.
    // Zero is the thresholded picture. One removes the threshold entirely, which
    // is the physically honest case: real glare applies to the whole image and
    // is dominated by bright pixels because the response scales, not because
    // dim ones are excluded.
    float minimum;

    // The upsample's level weight. Level k of the chain therefore reaches level
    // zero multiplied by weight to the k, and the post pass divides by the sum
    // of that series so the total is one whatever depth the window allowed.
    float weight;

    // Whether this is the first downsample, the one reading the HDR target.
    // Uniform across the draw.
    //
    // A branch rather than a second entry point, because the thirteen tap
    // positions below are exactly what the two paths must not disagree about,
    // and a separate entry point would hold a second copy of them.
    uint extract;
};

[[vk::push_constant]]
ConstantBuffer<BloomPushConstants> bloom;

// The level being read. Sampled with a linear filter and clamped to the edge, so
// a tap that leaves the level repeats its border texel rather than wrapping to
// the far side of the image.
[[vk::binding(0, 0)]]
Sampler2D source;

struct Varying
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
Varying vertexMain(uint index : SV_VertexID)
{
    // The three vertices fullscreen.slang generates, for the reason stated
    // there: one oversized triangle covers the visible square exactly once,
    // where a quad's diagonal would rasterize its seam twice.
    const float2 uv = float2((index << 1) & 2, index & 2);

    Varying varying;
    varying.uv = uv;
    varying.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return varying;
}

// Linear luminance from linear BT.709 primaries. Khronos Data Format
// Specification 1.3, section 14.1: the middle row of the RGB to XYZ matrix for
// those primaries is the Y row, and Y is luminance.
//
// Not the more familiar 0.2126, 0.7152, 0.0722. Those are section 15.1.1's
// K values, which that section defines for components carrying the ITU transfer
// function, and the chain holds linear radiance. The two agree to four decimal
// places and only one of them describes this quantity.
static const float3 luminance_709 = float3(0.212639, 0.715169, 0.072192);

// The weight that turns the sum below into a luminance-weighted average.
// Dividing by one plus luminance bounds what any single sample can contribute,
// so a sample twice as bright as its neighbour does not carry twice the weight.
//
// It is what keeps one pixel of an aliased highlight from flickering as it
// crosses a texel boundary between frames. A single hot sample in an otherwise
// dim neighbourhood is what it suppresses, and it is applied to the four-sample
// quads rather than to individual taps for that reason.
float karisWeight(float3 colour)
{
    return 1.0 / (1.0 + dot(colour, luminance_709));
}

// How much of a pixel enters the chain.
//
// A multiplier and not a subtraction. The result cannot go negative and so
// cannot need a clamp, which matters because the chain's format is unsigned:
// b10g11r11_ufloat_pack32 has no sign bit to hold one.
//
// The maximum channel rather than luminance decides it, so that a saturated
// colour is judged by how bright it is in its own channel rather than by how
// much the eye weights that channel.
float3 threshold(float3 colour)
{
    const float brightest = max(colour.r, max(colour.g, colour.b));
    const float ramp = smoothstep(bloom.threshold, bloom.threshold + bloom.knee, brightest);
    return colour * max(ramp, bloom.minimum);
}

[shader("fragment")]
float4 downsampleMain(Varying varying) : SV_Target
{
    const float2 texel = bloom.source_texel;
    const float x = texel.x;
    const float y = texel.y;

    // The thirteen taps: an outer ring and centre at two texels, and an inner
    // quad at one. Each is bilinear, so the pattern stands in for a much wider
    // kernel at thirteen fetches. Jimenez, "Next Generation Post Processing in
    // Call of Duty: Advanced Warfare", SIGGRAPH 2014.
    const float3 a = source.Sample(varying.uv + float2(-2.0 * x, 2.0 * y)).rgb;
    const float3 b = source.Sample(varying.uv + float2(0.0, 2.0 * y)).rgb;
    const float3 c = source.Sample(varying.uv + float2(2.0 * x, 2.0 * y)).rgb;
    const float3 d = source.Sample(varying.uv + float2(-2.0 * x, 0.0)).rgb;
    const float3 e = source.Sample(varying.uv).rgb;
    const float3 f = source.Sample(varying.uv + float2(2.0 * x, 0.0)).rgb;
    const float3 g = source.Sample(varying.uv + float2(-2.0 * x, -2.0 * y)).rgb;
    const float3 h = source.Sample(varying.uv + float2(0.0, -2.0 * y)).rgb;
    const float3 i = source.Sample(varying.uv + float2(2.0 * x, -2.0 * y)).rgb;
    const float3 j = source.Sample(varying.uv + float2(-x, y)).rgb;
    const float3 k = source.Sample(varying.uv + float2(x, y)).rgb;
    const float3 l = source.Sample(varying.uv + float2(-x, -y)).rgb;
    const float3 m = source.Sample(varying.uv + float2(x, -y)).rgb;

    if (bloom.extract == 0)
    {
        // The plain weights, summing to one: the centre and the inner quad carry
        // half the kernel between them, the edges a quarter, the corners an
        // eighth.
        const float3 reduced =
            e * 0.125 +
            (a + c + g + i) * 0.03125 +
            (b + d + f + h) * 0.0625 +
            (j + k + l + m) * 0.125;
        return float4(reduced, 1.0);
    }

    // The five overlapping quads the same weights describe, each averaged before
    // it is weighted. Exposure is applied to the taps rather than to the result
    // because everything after it is nonlinear in the sample's magnitude.
    const float scale = bloom.exposure * 0.25;
    const float3 q0 = (a + b + d + e) * scale;
    const float3 q1 = (b + c + e + f) * scale;
    const float3 q2 = (d + e + g + h) * scale;
    const float3 q3 = (e + f + h + i) * scale;
    const float3 q4 = (j + k + l + m) * scale;

    const float w0 = 0.125 * karisWeight(q0);
    const float w1 = 0.125 * karisWeight(q1);
    const float w2 = 0.125 * karisWeight(q2);
    const float w3 = 0.125 * karisWeight(q3);
    const float w4 = 0.5 * karisWeight(q4);

    // Divided by the weight sum rather than summed raw. Without the division a
    // uniformly bright region is darkened by its own luminance, because every
    // quad in it carries the same reduced weight, and the threshold below would
    // then be applied to a value the filter had already pulled down.
    const float3 averaged =
        (q0 * w0 + q1 * w1 + q2 * w2 + q3 * w3 + q4 * w4) /
        max(w0 + w1 + w2 + w3 + w4, 1e-4);

    return float4(threshold(averaged), 1.0);
}

[shader("fragment")]
float4 upsampleMain(Varying varying) : SV_Target
{
    const float x = bloom.source_texel.x;
    const float y = bloom.source_texel.y;

    // A three by three tent, weights one two one in each axis over sixteen.
    // Repeated across the chain it approximates a Gaussian far wider than any
    // one step, which is what gives the halo a tail rather than an edge.
    const float3 a = source.Sample(varying.uv + float2(-x, y)).rgb;
    const float3 b = source.Sample(varying.uv + float2(0.0, y)).rgb;
    const float3 c = source.Sample(varying.uv + float2(x, y)).rgb;
    const float3 d = source.Sample(varying.uv + float2(-x, 0.0)).rgb;
    const float3 e = source.Sample(varying.uv).rgb;
    const float3 f = source.Sample(varying.uv + float2(x, 0.0)).rgb;
    const float3 g = source.Sample(varying.uv + float2(-x, -y)).rgb;
    const float3 h = source.Sample(varying.uv + float2(0.0, -y)).rgb;
    const float3 i = source.Sample(varying.uv + float2(x, -y)).rgb;

    const float3 tent =
        e * 0.25 +
        (b + d + f + h) * 0.125 +
        (a + c + g + i) * 0.0625;

    // The blend unit adds this to what the downsample left in the attachment,
    // so the weight belongs here: it is the only place the level's share of the
    // total is expressed, and the host computes the reciprocal of the series it
    // generates.
    return float4(tent * bloom.weight, 1.0);
}
