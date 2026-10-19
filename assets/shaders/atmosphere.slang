// The physical sky into the environment cube: one thread a texel of level zero,
// marching the atmosphere along the texel's direction. `pass/atmosphere.zig`
// dispatches it when the sun moves and blits the mips after it; the transport
// itself is `include/atmosphere.slang`.

#include "include/atmosphere.slang"

// Level zero's six faces as an array, written once per texel. The format is
// stated rather than left to the image, so the device needs no support for
// storage images written without one.
[[vk::binding(3, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2DArray<float4> sky_cube;

// Mirrors `atmosphere.Sun`: toward the sun, and its irradiance above the
// atmosphere, which the march is linear in.
struct Sun
{
    float3 toward_sun;
    float top;
};

[[vk::push_constant]]
ConstantBuffer<Sun> sun;

// `atmosphere.extent`.
static const float extent = 128.0;

[shader("compute")]
[numthreads(8, 8, 1)]
void atmosphereMain(uint3 id : SV_DispatchThreadID)
{
    const float3 direction = cubeDirection(id.z, (float2(id.xy) + 0.5) / extent);
    sky_cube[id] = float4(integrate(model[0], direction, sun.toward_sun) * sun.top, 1.0);
}
