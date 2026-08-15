// The overlay: a two-dimensional draw list composited onto the finished
// picture, in display colour rather than in scene radiance.
//
// It is drawn after the tone operator, so nothing here is mapped, exposed or
// bloomed. A colour that arrives at 1.0 is the white the display shows.
//
// One vertex stage and one fragment stage, with no variants. The fragment stage
// multiplies the interpolated colour by the sampled texel unconditionally: a
// solid fill samples a white texel and is the colour, and a tinted glyph
// samples coverage and is the colour scaled by it. There is no branch between
// the two cases and no second pipeline.
//
// The image holds four channels of coverage per texel. For a glyph rasterised
// per colour stripe they differ, RGB being the coverage of each stripe and A
// that of the whole pixel; for everything else all four are the same number.
// That is what lets one expression serve both.

struct Varying
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;

    // Linear and already multiplied by its own alpha, which is what the blend
    // state is built for: source factor one, destination one minus source
    // alpha. Interpolating a premultiplied colour is correct where
    // interpolating a straight one is not, because the weighted sum of
    // premultiplied pairs is the premultiplied form of their composite.
    float4 colour : COLOR0;
};

struct UiPushConstants
{
    // The reciprocal of the target extent in pixels. The reciprocal rather than
    // the extent, so the position below costs a multiply rather than a divide
    // at every vertex.
    float2 inverse_extent;
};

[[vk::push_constant]]
ConstantBuffer<UiPushConstants> ui;

[shader("vertex")]
Varying vertexMain(
    float2 position : POSITION,
    float2 uv : TEXCOORD0,
    float4 colour : COLOR0)
{
    Varying varying;
    varying.uv = uv;
    varying.colour = colour;

    // Pixels to clip space, with no flip on either axis.
    //
    // The draw list is written with the origin at the top left and Y
    // increasing downward. The viewport this is recorded under has a positive
    // height, which maps clip Y of -1 to framebuffer row zero and +1 to the
    // last row, so clip Y already increases downward and the same expression
    // serves both axes. The scene's own shaders negate Y instead, because the
    // world they come from has it pointing up; this list is authored in the
    // raster's own orientation and has nothing to undo.
    varying.position = float4(position * ui.inverse_extent * 2.0 - 1.0, 0.0, 1.0);
    return varying;
}

// The image a draw names, resolved by whoever registered it. A solid fill binds
// a single white texel, so this is sampled for every draw rather than branched
// around.
[[vk::binding(0, 0)]]
Sampler2D image;

// The two source outputs the blend state reads.
//
// `colour` is what is added to the target and `coverage` is what the target is
// scaled down by, and they are separate because a subpixel glyph covers each
// channel by a different amount. A single alpha cannot say three things, which
// is the whole reason this pipeline asks for the dual-source blend feature.
// Both are the same attachment: dual-source blending is two sources for one
// target, which SPIR-V spells as one location and two indices. Slang decorates
// `SV_Target1` as location 1 on its own, which would name a second attachment
// this pipeline does not have, so the location and the index are both stated.
// Measured by disassembling the module, and `tests/shader_reflection.zig` holds
// it.
struct Fragment
{
    [[vk::location(0)]] [[vk::index(0)]] float4 colour : SV_Target0;
    [[vk::location(0)]] [[vk::index(1)]] float4 coverage : SV_Target1;
};

[shader("fragment")]
Fragment fragmentMain(Varying varying)
{
    const float4 sampled = image.Sample(varying.uv);

    Fragment fragment;
    // Both operands are premultiplied, and a premultiplied colour scaled per
    // channel by a coverage stays premultiplied, so the product is in the same
    // encoding the blend state expects.
    fragment.colour = varying.colour * sampled;
    // The alpha alone, times the same coverage. For a solid fill the sample is
    // one in every channel and this is the alpha in all four, which makes the
    // blend arithmetic identical to a single-source premultiplied one.
    fragment.coverage = varying.colour.a * sampled;
    return fragment;
}
