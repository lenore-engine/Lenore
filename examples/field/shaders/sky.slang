// The sky over the field: single-scattering Rayleigh and Mie, marched along the
// view ray.
//
// This replaces the engine's background, which samples the environment cube. A
// cube is the right answer for an asset viewer, where the environment is the
// asset's own and the picture should be the one the author lit it under. It is
// the wrong answer here, because the whole subject of this example is a sun that
// moves: a cube would have to be re-baked at every elevation, and the sky, the
// sunlight and the ambient would then be three things kept in step by hand.
//
// The sun is not a constant here and not a second copy of one. It is read from
// the same light block the surfaces are shaded from, so the direction the field
// is lit from and the direction the sun is drawn in cannot disagree, and the
// irradiance that scatters in the air is the irradiance that lands on the
// ground. That is the one property worth having before any of the rest: a sky
// and a scene on separate scales look wrong in a way no amount of grading fixes.
//
// What this is not: a precomputed transmittance and multiple-scattering pair of
// lookup tables. Those are what this should become, and they are what make the
// twilight sky right, because the light that reaches the eye after the sun has
// set has bounced more than once and single scattering has it black. This model
// goes dark too early and too suddenly for that reason, and the elevation slider
// reaches below the horizon so that the failure is visible rather than avoided.

// Mirrors `uniforms.Camera`, as the engine's own background and scene shaders
// do. The block is one text in all three files or it is three accounts of one
// buffer.
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

// Mirrors `uniforms.Light`, as scene.slang does.
struct Light
{
    float3 position;
    uint kind; // 0 = directional, 1 = point, 2 = spot
    float3 colour;
    float intensity;
    float3 direction; // unit, the way the light travels
    float range;
    float cos_inner;
    float cos_outer;
    float2 padding;
};

static const uint max_lights = 16;

struct LightsUbo
{
    uint4 count;
    Light lights[max_lights];
};

[[vk::binding(0, 0)]] ConstantBuffer<CameraUbo> camera;
[[vk::binding(2, 0)]] ConstantBuffer<LightsUbo> lights;

// Metres. Earth, and an atmosphere thick enough that the top of it contributes
// nothing: at 100 km the density of both species is under a millionth of the
// surface value.
static const float planet_radius = 6371000.0;
static const float atmosphere_radius = 6471000.0;

// Metres. The height at which each species falls to 1/e of its surface density.
static const float rayleigh_scale_height = 8000.0;
static const float mie_scale_height = 1200.0;

// Per metre, at sea level, for the sRGB primaries near 680, 550 and 440 nm.
//
// Rayleigh's own expression is beta = 8 pi^3 (n^2 - 1)^2 / (3 N lambda^4), and
// evaluating it at 680 nm with n = 1.00029 and N = 2.545e25 per cubic metre
// gives 5.1e-6 against the 5.8e-6 below. The difference is the King correction
// for the depolarisation of air, which the bare expression omits; these are the
// corrected values in common use rather than the formula's own.
static const float3 rayleigh_coefficient = float3(5.802e-6, 13.558e-6, 33.100e-6);

// Mie scattering is very nearly wavelength independent over the visible range,
// which is why haze is white and the sky is not.
static const float mie_coefficient = 21.0e-6;
// Aerosols absorb as well as scatter. The ratio is the usual one for a clear
// continental atmosphere.
static const float mie_extinction = mie_coefficient / 0.9;
// Forward scattering anisotropy. This is what puts the bright halo around the
// sun and what makes a low sun light the haze along the horizon rather than
// overhead.
static const float mie_g = 0.76;

// Radians. The sun subtends about half a degree from Earth.
static const float sun_angular_radius = 0.004675;

// The cost knob, and the only one. Sixteen steps along the view ray, each of
// which marches four more toward the sun for the light that reaches it.
//
// Sixty-four samples a pixel, over whatever fraction of the screen no surface
// covered. The background is recorded inside the main pass, so what this costs
// shows up in the panel's `main` row and not in a row of its own: to read it,
// change these and compare, rather than expecting a toggle.
//
// The inner loop is the honest part and the expensive part. Replacing it with a
// transmittance lookup is the first thing the table work removes.
static const uint primary_steps = 16;
static const uint light_steps = 4;

struct Varying
{
    float4 position : SV_Position;
    // Unnormalized, in world space. It is affine in the device coordinate and
    // the triangle is drawn at w = 1, so interpolation is exact and the
    // normalization is per pixel, which is where it is free.
    float3 ray : TEXCOORD0;
};

[shader("vertex")]
Varying vertexMain(uint index : SV_VertexID)
{
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

// The Rayleigh phase function, normalized so that its integral over the sphere
// is one. Symmetric forward and back, which is why the sky is nearly as bright
// opposite the sun as toward it.
float rayleighPhase(float cos_theta)
{
    return 0.05968310365 * (1.0 + cos_theta * cos_theta); // 3 / (16 pi)
}

// Henyey-Greenstein, the standard one-parameter fit for aerosol scattering.
float miePhase(float cos_theta, float g)
{
    const float gg = g * g;
    const float denominator = 1.0 + gg - 2.0 * g * cos_theta;
    return 0.07957747155 * (1.0 - gg) / max(denominator * sqrt(denominator), 1e-6); // 1 / (4 pi)
}

// The far intersection of a ray with a sphere centred on the origin, or a
// negative number when there is none. The origin is always inside the
// atmosphere here, so only the far root is ever wanted.
float raySphereFar(float3 origin, float3 direction, float radius)
{
    const float b = dot(origin, direction);
    const float c = dot(origin, origin) - radius * radius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0) return -1.0;
    return -b + sqrt(discriminant);
}

// The near intersection, for the planet: a ray that hits the ground stops there
// rather than marching through the Earth.
float raySphereNear(float3 origin, float3 direction, float radius)
{
    const float b = dot(origin, direction);
    const float c = dot(origin, origin) - radius * radius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0) return -1.0;
    const float near = -b - sqrt(discriminant);
    return near;
}

// Where along a path the i-th of n segment boundaries falls, as a fraction.
//
// Squared and not linear, which is the difference between a sky and a black
// screen. Both species fall off exponentially with a scale height of kilometres,
// while a ray aimed near the horizon crosses hundreds of kilometres of
// atmosphere: sixteen even steps over that put the first sample tens of
// kilometres up, where the air is already gone, and the integral returns almost
// nothing. Squaring packs the samples toward the eye, which is where the mass
// is, and the same sixteen steps then resolve the bottom of the atmosphere at
// every angle.
//
// A quadratic rather than an exponential distribution because it is exact at
// both ends, needs no parameter tied to the scale heights, and the segment
// boundaries below are what carry the length, so no sample is weighted by a step
// it does not stand in the middle of.
float quadratic(float fraction)
{
    return fraction * fraction;
}

// Densities of the two species at a point, relative to their sea level values.
float2 densitiesAt(float3 point)
{
    const float altitude = length(point) - planet_radius;
    return float2(
        exp(-altitude / rayleigh_scale_height),
        exp(-altitude / mie_scale_height)
    );
}

// The column density between a point and the top of the atmosphere along a
// direction, as the pair of species integrals. Returns false when the path is
// blocked by the planet, which is what puts the Earth's shadow on the air below
// the horizon at sunset.
bool opticalDepthToSun(float3 point, float3 to_sun, out float2 depth)
{
    depth = float2(0.0, 0.0);

    // A path that enters the ground carries no sunlight, whatever the air along
    // it would have done.
    if (raySphereNear(point, to_sun, planet_radius) > 0.0) return false;

    const float distance = raySphereFar(point, to_sun, atmosphere_radius);
    if (distance <= 0.0) return false;

    for (uint i = 0; i < light_steps; ++i)
    {
        const float near = quadratic(float(i) / float(light_steps)) * distance;
        const float far = quadratic(float(i + 1) / float(light_steps)) * distance;
        depth += densitiesAt(point + to_sun * (0.5 * (near + far))) * (far - near);
    }
    return true;
}

// Beer's law over both species, plus Mie's absorption.
float3 transmittance(float2 depth)
{
    return exp(-(rayleigh_coefficient * depth.x + mie_extinction * depth.y));
}

// The radiance arriving along a ray, from the air alone. `irradiance` is the
// sunlight above the atmosphere, and it is the same value the surfaces below are
// lit by.
// `view_depth` comes back with the column the ray crossed, so the sun's disc is
// attenuated by the integral this already walked rather than by a second one.
float3 scatter(
    float3 origin,
    float3 direction,
    float3 to_sun,
    float3 irradiance,
    out float2 view_depth)
{
    view_depth = float2(0.0, 0.0);

    float end = raySphereFar(origin, direction, atmosphere_radius);
    if (end <= 0.0) return float3(0.0, 0.0, 0.0);

    // Stop at the ground where the ray meets it. Without this a view aimed
    // downward marches through the planet and comes back with the column of a
    // path no photon takes.
    const float ground = raySphereNear(origin, direction, planet_radius);
    if (ground > 0.0) end = min(end, ground);

    const float cos_theta = dot(direction, to_sun);
    const float rayleigh_weight = rayleighPhase(cos_theta);
    const float mie_weight = miePhase(cos_theta, mie_g);

    float3 accumulated = float3(0.0, 0.0, 0.0);

    for (uint i = 0; i < primary_steps; ++i)
    {
        const float near = quadratic(float(i) / float(primary_steps)) * end;
        const float far = quadratic(float(i + 1) / float(primary_steps)) * end;
        const float3 sample = origin + direction * (0.5 * (near + far));
        const float2 densities = densitiesAt(sample) * (far - near);
        view_depth += densities;

        float2 sun_depth;
        if (!opticalDepthToSun(sample, to_sun, sun_depth)) continue;

        // What survives the trip in from space and back out to the eye.
        const float3 weight = transmittance(sun_depth + view_depth);
        accumulated += weight * (
            rayleigh_coefficient * densities.x * rayleigh_weight +
            mie_coefficient * densities.y * mie_weight
        );
    }

    return accumulated * irradiance;
}

// The sun's own disc, with a soft edge one pixel wide rather than a hard one.
//
// Drawn as a multiple of the irradiance and attenuated by the same optical depth
// the sky is, so it reddens as it sets for the same reason and by the same
// arithmetic. The solid angle divides it because irradiance is power per area
// and what a pixel wants is radiance.
float3 sunDisc(float3 direction, float3 to_sun, float3 irradiance, float2 view_depth)
{
    const float cos_theta = dot(direction, to_sun);
    const float cos_edge = cos(sun_angular_radius);
    // Angular width of a pixel is not known here, so the softening is a small
    // fixed fraction of the disc. It exists so the edge is not a staircase.
    const float softness = 0.15 * (1.0 - cos_edge);
    const float coverage = smoothstep(cos_edge - softness, cos_edge + softness, cos_theta);
    if (coverage <= 0.0) return float3(0.0, 0.0, 0.0);

    const float solid_angle = 6.283185307 * (1.0 - cos_edge);
    return irradiance * (coverage / solid_angle) * transmittance(view_depth);
}

[shader("fragment")]
float4 fragmentMain(Varying varying) : SV_Target
{
    const float3 direction = normalize(varying.ray);

    // The first directional light is the sun. Searched rather than assumed at
    // index zero: the block's order is the application's, and a scene that puts
    // a lamp first should not become a world lit by a lamp.
    float3 to_sun = float3(0.0, 1.0, 0.0);
    float3 irradiance = float3(0.0, 0.0, 0.0);
    const uint count = min(lights.count.x, max_lights);
    for (uint i = 0; i < count; ++i)
    {
        if (lights.lights[i].kind != 0) continue;
        // The block stores the direction light travels, and every geometry here
        // is written from the surface toward the source.
        to_sun = -lights.lights[i].direction;
        irradiance = lights.lights[i].colour * lights.lights[i].intensity;
        break;
    }

    // The eye, placed on a planet rather than at the origin: the world is flat
    // and a few metres tall, so the altitude is the camera's own height and the
    // rest is the radius. This is what puts the horizon where a horizon goes
    // instead of at the eye's own level.
    const float3 origin = float3(0.0, planet_radius + camera.position.y, 0.0);

    float2 view_depth;
    float3 radiance = scatter(origin, direction, to_sun, irradiance, view_depth);

    // The disc is drawn only where the view ray reaches space, and attenuated by
    // the column that ray crossed, which is the integral `scatter` already
    // walked. A ray that meets the ground first has the planet between the eye
    // and the sun, whatever the angle says.
    if (raySphereNear(origin, direction, planet_radius) <= 0.0)
    {
        radiance += sunDisc(direction, to_sun, irradiance, view_depth);
    }

    // Opaque. The target's alpha is composited against nothing and the post pass
    // reads the colour alone.
    return float4(radiance, 1.0);
}
