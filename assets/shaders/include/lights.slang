// The light record, the falloffs every pass that shades by it applies, and the
// ambient's convolution.
// Included by lights_block.slang for scene.slang, which lights surfaces, and
// by probes.slang, which lights the points its rays hit; one copy, so the two
// cannot disagree about how bright a lamp is at a distance.

// Mirrors `uniforms.Light`. Each kind reads only its own fields and the rest
// are zero, so no lane means one thing here and another there.
struct Light
{
    float3 position; // the centre for rect, the source for point and spot
    uint kind;       // 0 = directional, 1 = point, 2 = spot, 3 = rect
    float3 colour;
    float intensity;
    // Unit, the way the light travels. For rect it is the face the rectangle
    // emits from. Directional, spot and rect.
    float3 direction;
    float range;      // point, spot and rect
    float cos_inner;  // spot
    float cos_outer;  // spot
    // The first layer in `local_shadow_maps`, or `no_shadow_layer`. A spot
    // reads that layer, a point light the six from it in `localFace` order.
    uint shadow_layer;
    // World units across one texel of the light's layers at unit distance.
    float shadow_texel_slope;
    // The rectangle, for rect. `half_edge` is half of one edge as a vector from
    // the centre; the other edge is perpendicular to it and to `direction`, with
    // `half_extent` for its half length.
    float3 half_edge;
    float half_extent;
};

// Windowed inverse-square falloff: pure 1/d^2 over most of the radius, reaching
// exactly zero at `range` instead of being cut there.
//
// KHR_lights_punctual, "Range Property", recommends exactly
// `max(min(1 - (d/range)^4, 1), 0) / d^2`, and that is what this is. Frostbite
// squares the window on top of it for a vanishing derivative at the cutoff; the
// extension is the authority for the assets this reads, and the kink it leaves
// is in a term that is already zero there.
//
// Squared distance in, so a caller can cull before taking a root. `d^4` is the
// square of that, so no root is needed here either. The floor on the divisor is
// what a surface at the light's own position lands on.
float rangeAttenuation(float distance_squared, float range)
{
    const float ratio = distance_squared / (range * range);
    return saturate(1.0 - ratio * ratio) / max(distance_squared, 1.0e-4);
}

// The cone, between the two cosines. `cos_inner > cos_outer` is established
// where the cone is built, in `lenore-scene`'s `Light.spot`, which is what keeps
// smoothstep off its undefined equal-edge case.
float spotAttenuation(float3 to_light, Light light)
{
    return smoothstep(light.cos_outer, light.cos_inner, dot(to_light, -light.direction));
}

// What a diffuse surface facing `normal` receives, over pi, from a sky whose
// radiance is `ground` straight down, `sky` straight up and linear in the
// height between: radiance linear in the height convolves to its mean plus two
// thirds of its slope. Ramamoorthi and Hanrahan, "An Efficient Representation
// for Irradiance Environment Maps", SIGGRAPH 2001, section 3: the first band's
// clamped cosine coefficient is 2 pi / 3 against pi for the constant one, and
// over pi (the convention the lambertian map arrives in) that leaves 2 / 3.
// Without it a wall would read the sky at full strength where it sees half of
// it.
float3 hemisphereIrradiance(float3 sky, float3 ground, float3 normal)
{
    const float3 mean = 0.5 * (sky + ground);
    const float3 slope = 0.5 * (sky - ground);
    return mean + slope * ((2.0 / 3.0) * normal.y);
}
