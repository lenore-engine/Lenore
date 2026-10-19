// The background: the environment the scene lights itself from, drawn wherever
// no opaque surface stands.
//
// One screen-covering triangle pinned to the far plane, recorded inside the
// main pass after the opaque draws and before the blended ones. Depth is tested
// and not written. The test is what leaves the geometry alone, since the
// opaques have already filled their pixels with something nearer, and drawing
// before the blended set is what lets a transparent surface composite over the
// sky instead of over the clear colour.
//
// Nothing here declares a set of its own. The camera and the lights come from
// the frame set and the cube from the scene set, all of which the main pass has
// already bound for the draws around this one, so the whole pass is a pipeline
// bind and three vertices.
//
// What it draws is what the surfaces are lit by: the cube, the ambient they
// add to it, and each directional light as the disc it would be in the sky.

// Mirrors `uniforms.Camera`, as scene.slang does. The fields before the basis
// are what place it, so they are declared rather than replaced by padding of
// the same width: the block is one text in both files or it is two accounts of
// one buffer.
struct CameraUbo
{
    float4x4 view_projection;
    float4 position;
    // For a device coordinate (x, y) the ray leaving the eye is
    // `ray_front + ray_right * x + ray_up * y`, already in the coordinates the
    // framebuffer counts in.
    float4 ray_right;
    float4 ray_up;
    float4 ray_front;
    float4x4 previous_view_projection;
    uint4 motion;
    // The sub-pixel offset `view_projection` carries this frame, in device
    // coordinates, zero when unjittered.
    float4 jitter;
};

[[vk::binding(0, 0)]] ConstantBuffer<CameraUbo> camera;

// The same prefiltered cube the surface shading samples for its specular term.
// One image for the background and for the reflections is what puts them on one
// radiometric scale by construction, with no second copy and no intensity to
// keep in step between the two.
[[vk::binding(2, 1)]] SamplerCube ggx_environment;

#include "include/lights_block.slang"

static const float pi = 3.14159265358979;

// The sine of the sun's angular radius, 0.267 degrees, which is the disc every
// directional light is drawn as. A light has no size of its own, so they all
// share the sun's.
static const float sun_sin_radius = 0.0046600;

// The disc is drawn no brighter than this many times a white surface facing
// its light. The light itself is unchanged: what a surface receives comes from
// the light block, not from the disc. Uncapped, the disc is about 4.6e4 times
// that, and a disc seven pixels across at such a value is binary at its edge;
// the temporal resolve's jittered copies and the bloom's energy-linear chain
// then turn it into a lumpy, pulsing square. Fifty is still far past any
// exposure's white, so the disc stays the brightest thing in the sky and
// glows through the bloom, and its edge has a range the resolve can carry.
static const float disc_over_white = 50.0;

// The largest value the HDR attachment holds in its narrowest channel, the
// unsigned 10-bit float's 63488 (see `min_roughness` in scene.slang). The disc
// is held under it rather than left to whatever the conversion makes of a
// larger one.
static const float hdr_largest = 63488.0;

struct Varying
{
    float4 position : SV_Position;

    // The ray leaving the eye, in world space and unnormalized. It is affine in
    // the device coordinate and the triangle is drawn at w = 1, so
    // interpolation across it is exact and the normalization is per pixel,
    // which is where it is free.
    float3 ray : TEXCOORD0;
};

[shader("vertex")]
Varying vertexMain(uint index : SV_VertexID)
{
    // The three vertices fullscreen.slang generates, for the reason stated
    // there: one oversized triangle covers the visible square exactly once,
    // where a quad's diagonal would rasterize its seam twice.
    const float2 uv = float2((index << 1) & 2, index & 2);
    const float2 device = uv * 2.0 - 1.0;

    Varying varying;
    // Depth one is the far plane and what the main pass clears to, so a
    // less-or-equal test passes exactly where no opaque surface wrote.
    varying.position = float4(device, 1.0, 1.0);
    varying.ray = camera.ray_front.xyz +
        camera.ray_right.xyz * device.x +
        camera.ray_up.xyz * device.y;
    return varying;
}

[shader("fragment")]
float4 fragmentMain(Varying varying) : SV_Target
{
    // Level zero explicitly. The chain on this cube is a roughness axis rather
    // than an anti-aliasing chain, so a level picked from the screen-space
    // footprint would return a correctly filtered sample of the wrong
    // roughness, which reads as a blurred sky and not as a sampling error.
    const float3 direction = normalize(varying.ray);
    float3 sky = ggx_environment.SampleLevel(direction, 0.0).rgb + lights.ambient.reflected * ambientRadiance(direction);

    // A disc of uniform radiance L and angular radius r gives a surface facing
    // it L pi sin^2 r, so the light's irradiance E over that is the radiance
    // that makes the drawn disc and the light the same light: about 4.6e4 for
    // E = pi. Past E of about 4.3 the disc is clamped to the attachment's
    // largest value and is dimmer than its light.
    //
    // The angle is measured by its sine, the length of the cross product,
    // because within a quarter of a degree its cosine is too near one for a
    // float to resolve the edge. The edge is covered over one pixel's width of
    // that sine, so the disc, seven or so pixels across, does not shimmer as the
    // view turns.
    const float disc = 1.0 / (pi * sun_sin_radius * sun_sin_radius);
    for (uint i = 0; i < lights.count.x; ++i)
    {
        const Light light = lights.lights[i];
        if (light.kind != 0) continue;
        const float3 to_light = -light.direction;
        const float off_centre = length(cross(direction, to_light));
        const float edge = max(fwidth(off_centre), 1.0e-7);
        const float covered = dot(direction, to_light) > 0.0
            ? saturate((sun_sin_radius - off_centre) / edge + 0.5)
            : 0.0;
        // A white Lambertian surface facing the light returns E / pi.
        const float drawn = min(disc, disc_over_white / pi);
        sky += min(light.colour * (light.intensity * drawn * covered), hdr_largest);

        // The halo, by the angle from the light's centre. Atan2 of the sine
        // and cosine keeps it smooth through the centre and past a right
        // angle, where either alone loses precision.
        const float degrees = atan2(off_centre, dot(direction, to_light)) * (180.0 / pi);
        const float glow = lights.halo.narrow * exp(-degrees / lights.halo.narrow_degrees) +
            lights.halo.wide * exp(-degrees / lights.halo.wide_degrees);
        sky += min(light.colour * (light.intensity * glow / pi), hdr_largest);
    }

    // Opaque. The target's alpha is composited against nothing and the post
    // pass reads the colour alone.
    return float4(sky, 1.0);
}
