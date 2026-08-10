// The post pass geometry: one triangle covering the screen, generated from the
// vertex index with no buffer bound.
//
// Three vertices rather than a quad's four. A quad's diagonal splits the screen
// into two triangles, and every pixel along it is rasterized by both, so the
// fragment stage runs twice on that seam. One oversized triangle has no seam.

struct Varying
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
Varying vertexMain(uint index : SV_VertexID)
{
    // 0 -> (0,0), 1 -> (2,0), 2 -> (0,2) in UV, which is (-1,-1), (3,-1),
    // (-1,3) in clip space: the triangle reaches one screen width past the
    // right edge and one height past the bottom, so the visible square is
    // covered exactly once.
    const float2 uv = float2((index << 1) & 2, index & 2);

    Varying varying;
    varying.uv = uv;
    varying.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return varying;
}

// The HDR scene target, written by the main pass and read here.
[[vk::binding(0, 0)]]
Sampler2D scene;

// The finest level of the bloom chain, at half this pass's resolution. Read by
// one of the two fragment entry points below; the other does not name it, which
// is what makes a recording with no chain behind it a different pipeline rather
// than a multiplication by zero.
[[vk::binding(1, 0)]]
Sampler2D bloom_chain;

struct PostPushConstants
{
    float exposure;

    // What the chain's finest level is multiplied by. It carries the intensity
    // and the reciprocal of the series the chain's upsample generates, so that
    // a window admitting one level fewer does not dim the glow.
    float bloom;
};

[[vk::push_constant]]
ConstantBuffer<PostPushConstants> post;

// KhronosGroup/ToneMapping, PBR_Neutral specification.
float3 toneMapPbrNeutral(float3 colour)
{
    const float start_compression = 0.8 - 0.04;
    const float darkest = min(colour.r, min(colour.g, colour.b));
    const float offset = darkest < 0.08
        ? darkest - 6.25 * darkest * darkest
        : 0.04;
    colour -= offset;

    const float peak = max(colour.r, max(colour.g, colour.b));
    if (peak < start_compression)
        return colour;

    const float distance_to_white = 1.0 - start_compression;
    const float compressed_peak = 1.0 - distance_to_white * distance_to_white /
        (peak + distance_to_white - start_compression);
    colour *= compressed_peak / peak;

    const float desaturation = 1.0 - 1.0 / (0.15 * (peak - compressed_peak) + 1.0);
    return lerp(colour, float3(compressed_peak), desaturation);
}

// The operator's input is non-negative linear radiance, so the clamp is where
// the two entry points below meet rather than something either repeats.
float4 present(float3 radiance)
{
    return float4(toneMapPbrNeutral(max(radiance, 0.0)), 1.0);
}

[shader("fragment")]
float4 fragmentMain(Varying varying) : SV_Target
{
    return present(scene.Sample(varying.uv).rgb * post.exposure);
}

// The same, with the bloom chain composited in.
//
// Added before the operator and not after. The core of a bright emitter and the
// glow around it are then compressed by one curve, which is what leaves the
// core white and lets the tail keep its hue; adding afterwards would lay a flat
// wash of the emitter's colour over an already mapped picture.
//
// The chain is not exposed again. Its first step multiplied by this same value,
// which is what puts the threshold it applied in the units that will be
// displayed as white.
[shader("fragment")]
float4 bloomFragmentMain(Varying varying) : SV_Target
{
    const float3 radiance = scene.Sample(varying.uv).rgb * post.exposure;
    const float3 glow = bloom_chain.Sample(varying.uv).rgb * post.bloom;
    return present(radiance + glow);
}
