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
// Nothing here declares a set of its own. The camera comes from the frame set
// and the cube from the scene set, both of which the main pass has already
// bound for the draws around this one, so the whole pass is a pipeline bind and
// three vertices.

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
};

[[vk::binding(0, 0)]] ConstantBuffer<CameraUbo> camera;

// The same prefiltered cube the surface shading samples for its specular term.
// One image for the background and for the reflections is what puts them on one
// radiometric scale by construction, with no second copy and no intensity to
// keep in step between the two.
[[vk::binding(2, 1)]] SamplerCube ggx_environment;

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
    const float3 sky = ggx_environment.SampleLevel(normalize(varying.ray), 0.0).rgb;

    // Opaque. The target's alpha is composited against nothing and the post
    // pass reads the colour alone.
    return float4(sky, 1.0);
}
