// The frame's lights block at the frame set's slot 2: every light, the local
// shadow layers, the ambient and the probe grid. Included by scene.slang,
// which shades surfaces by all of it, and sky.slang, which draws the ambient
// and the directional lights behind them; one declaration, so the two read one
// buffer at the same offsets.

#include "lights.slang"
#include "probe_grid.slang"

// Mirrors `uniforms.max_lights`. The array bounds the loop; `count.x` is how
// much of it holds a light, and the rest is not written by anyone.
static const uint max_lights = 16;

// Mirrors `uniforms.max_shadow_layers` and `uniforms.no_shadow_layer`.
static const uint max_shadow_layers = 24;
static const uint no_shadow_layer = 0xffffffff;

// Mirrors `uniforms.LocalShadow`.
struct LocalShadow
{
    // Zero switches every local lookup off.
    float strength;
    // In texels of the light's layer at the fragment's distance.
    float normal_offset_texels;
    // World units; zero is a point light's hard edge.
    float source_radius;
    // Every layer's near plane, for turning a stored depth into a distance.
    float near;
};

// Mirrors `uniforms.Ambient`: radiance from above and from below.
struct Ambient
{
    float3 sky;
    float reflected;
    float3 ground;
    float padding_1;
};

// Mirrors `uniforms.Halo`.
struct Halo
{
    float narrow;
    float narrow_degrees;
    float wide;
    float wide_degrees;
};

struct LightsUbo
{
    uint4 count; // x = live lights, yzw are the array's alignment
    Light lights[max_lights];
    LocalShadow local_shadow;
    // World space to each local layer's clip space, from `LocalShadowFit`.
    float4x4 shadow_view_projection[max_shadow_layers];
    Ambient ambient;
    // Where the probes stand; `enabled` zero is no field.
    ProbeGridParams probe_grid;
    // Mirrors `uniforms.Debug`: x is the `DebugView`, zero the shaded frame.
    uint4 debug;
    Halo halo;
};

[[vk::binding(2, 0)]] ConstantBuffer<LightsUbo> lights;

// The ambient's radiance along `direction`: ground straight down, sky straight
// up, linear in the direction's height between.
float3 ambientRadiance(float3 direction)
{
    return lerp(lights.ambient.ground, lights.ambient.sky, direction.y * 0.5 + 0.5);
}
