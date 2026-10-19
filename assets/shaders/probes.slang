// The probe field, write side: one workgroup per probe, one ray per thread,
// traced inline against the scene's acceleration structure. Each ray returns
// the radiance leaving whatever it hits toward the probe, and the probe's
// spherical harmonics and hit-distance moments are rebuilt from the 64 rays
// and blended into what the probe held.
//
// What a hit sends back is diffuse: the hit's albedo over pi times the
// irradiance arriving there, direct from every light (shadowed by a second
// ray query toward it) plus indirect read back out of the previous field.
// That read is what makes the second and later bounces: each update carries
// light one bounce further, and a settled field holds all of them.
//
// The previous field is read and the next written, two buffers each, so no
// workgroup reads a probe another is in the middle of rewriting. The host
// swaps the pair between updates.

#include "include/lights.slang"
#include "include/probe_field.slang"

// Mirrors `probes.Triangle`: one world-space triangle, its albedo in the
// fourth lanes. Indexed by the primitive index the bottom-level structure was
// built with, which is the order the host wrote them in.
struct Triangle
{
    float4 a;
    float4 b;
    float4 c;
};

// Mirrors `probes.PushConstants`.
struct UpdatePushConstants
{
    ProbeGridParams grid;
    // Radiance arriving from where a ray leaves the scene without a hit: the
    // ambient's sky above and ground below, blended by the ray's height.
    float3 sky;
    uint light_count;
    float3 ground;
    // How much of what a probe held survives an update, in [0, 1). Zero
    // replaces it outright, which is what the first update after a change
    // wants.
    float hysteresis;
    // A rotation, as a unit quaternion, applied to this update's rays so
    // successive updates sample different directions and their blend converges
    // over the sphere rather than over 64 fixed points.
    float4 rotation;
    uint first_probe;
    uint probe_count;
    // How far a ray looks before it counts as leaving the scene.
    float max_distance;
    float padding;
};

[[vk::push_constant]] ConstantBuffer<UpdatePushConstants> update;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure scene_acceleration;
[[vk::binding(1, 0)]] StructuredBuffer<Triangle> triangles;
[[vk::binding(2, 0)]] StructuredBuffer<Light> probe_lights;
[[vk::binding(3, 0)]] StructuredBuffer<float4> probe_sh;
[[vk::binding(4, 0)]] StructuredBuffer<float2> probe_moments;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> next_sh;
[[vk::binding(6, 0)]] RWStructuredBuffer<float2> next_moments;

static const uint rays_per_probe = 64;
static const float pi = 3.14159265359;
static const float golden_angle = 2.39996322973;

// How far off a surface a shadow ray starts, in metres. The geometry is
// world-space floats a few metres from the origin, where a micron is well
// above their precision.
static const float surface_offset = 1.0e-3;

// The sharpness of the moments' weighting: a ray contributes to a texel by
// the cosine between them to this power, so each texel averages the rays in a
// narrow cone around its direction. DDGI's value, from memory of the paper.
static const float moment_sharpness = 50.0;

// `probe_grid.fibonacciDirection`.
float3 fibonacciDirection(uint i, uint n)
{
    const float y = 1.0 - (2.0 * float(i) + 1.0) / float(n);
    const float radius = sqrt(max(0.0, 1.0 - y * y));
    const float angle = float(i) * golden_angle;
    return float3(radius * cos(angle), y, radius * sin(angle));
}

float3 rotate(float4 q, float3 v)
{
    const float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

bool occluded(float3 origin, float3 direction, float limit)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = limit;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> query;
    query.TraceRayInline(scene_acceleration, RAY_FLAG_NONE, 0xff, ray);
    while (query.Proceed()) {}
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

// Irradiance from every light at a point facing `normal`, each shadowed by a
// ray toward it. A rect light is taken as a point at its centre: the field is
// coarse enough that its extent changes nothing a probe can resolve.
float3 directIrradiance(float3 position, float3 normal)
{
    const float3 start = position + normal * surface_offset;
    float3 total = float3(0.0, 0.0, 0.0);
    for (uint i = 0; i < update.light_count; ++i)
    {
        const Light light = probe_lights[i];
        float3 to_light;
        float limit;
        float attenuation = 1.0;
        if (light.kind == 0)
        {
            to_light = -light.direction;
            limit = update.max_distance;
        }
        else
        {
            const float3 offset = light.position - position;
            const float distance_squared = dot(offset, offset);
            if (distance_squared > light.range * light.range) continue;
            limit = sqrt(distance_squared);
            to_light = offset / max(limit, 1.0e-4);
            attenuation = rangeAttenuation(distance_squared, light.range);
            if (light.kind == 2) attenuation *= spotAttenuation(to_light, light);
            if (light.kind == 3 && dot(-to_light, light.direction) <= 0.0) continue;
        }
        const float lambert = dot(normal, to_light);
        if (lambert <= 0.0 || attenuation <= 0.0) continue;
        if (occluded(start, to_light, limit)) continue;
        total += light.colour * (light.intensity * attenuation * lambert);
    }
    return total;
}

groupshared uint ray_backface[rays_per_probe];
groupshared float3 ray_direction[rays_per_probe];
groupshared float ray_distance[rays_per_probe];
groupshared float3 ray_radiance[rays_per_probe];

// Stated rather than left for slangc to infer: the module is compiled with
// the same profile as every other, and the ray query capability is this entry
// point's alone. The device enables the extension when it reports it, and the
// pass is only built then.
[shader("compute")]
[require(spvRayQueryKHR)]
[numthreads(64, 1, 1)]
void updateMain(uint3 group : SV_GroupID, uint thread : SV_GroupIndex)
{
    // Uniform across the group, so the barrier below is reached by all of it
    // or by none.
    if (group.x >= update.probe_count) return;
    const uint probe = update.first_probe + group.x;
    const ProbeGridParams grid = update.grid;
    const uint3 cell = uint3(
        probe % grid.counts.x,
        (probe / grid.counts.x) % grid.counts.y,
        probe / (grid.counts.x * grid.counts.y));
    const float3 origin = grid.origin + float3(cell) * grid.spacing;

    const float3 direction = rotate(update.rotation, fibonacciDirection(thread, rays_per_probe));
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = update.max_distance;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene_acceleration, RAY_FLAG_NONE, 0xff, ray);
    while (query.Proceed()) {}

    float3 radiance;
    float distance;
    uint backface = 0;
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        distance = query.CommittedRayT();
        const Triangle hit = triangles[query.CommittedPrimitiveIndex()];
        const float3 normal = normalize(cross(hit.b.xyz - hit.a.xyz, hit.c.xyz - hit.a.xyz));
        if (dot(normal, direction) > 0.0)
        {
            // The back of a surface: the probe is inside something, or sees
            // past an open edge. It sends nothing back, and the short distance
            // is what keeps the field from reading this probe through it.
            radiance = float3(0.0, 0.0, 0.0);
            backface = 1;
        }
        else
        {
            const float3 position = origin + direction * distance;
            const float3 albedo = float3(hit.a.w, hit.b.w, hit.c.w);
            const float3 open = pi * hemisphereIrradiance(update.sky, update.ground, normal);
            const float3 irradiance = directIrradiance(position, normal)
                + sampleProbeField(grid, position, normal, -direction, open);
            radiance = albedo / pi * irradiance;
        }
    }
    else
    {
        distance = update.max_distance;
        radiance = lerp(update.ground, update.sky, direction.y * 0.5 + 0.5);
    }

    ray_backface[thread] = backface;
    ray_direction[thread] = direction;
    ray_distance[thread] = distance;
    ray_radiance[thread] = radiance;
    GroupMemoryBarrierWithGroupSync();

    // Nine threads project the radiance onto the harmonics, one coefficient
    // each: the Monte Carlo estimate over the sphere is 4 pi / N times the sum.
    if (thread < probe_sh_count)
    {
        float3 sum = float3(0.0, 0.0, 0.0);
        for (uint r = 0; r < rays_per_probe; ++r)
        {
            float basis[9];
            shBasis(ray_direction[r], basis);
            sum += ray_radiance[r] * basis[thread];
        }
        const float3 estimate = sum * (4.0 * pi / float(rays_per_probe));
        const uint slot = probe * probe_sh_count + thread;
        // The first coefficient's fourth lane carries the share of this
        // probe's rays that met the back of a surface. Past a quarter the probe
        // stands inside a wall or a prop, sees nothing a point outside it
        // would, and the field passes it over (`probe_invalid_share`). Not
        // blended: it is a property of where the probe stands, and one update
        // states it as well as many.
        float backface_share = 0.0;
        if (thread == 0)
        {
            uint count = 0;
            for (uint r = 0; r < rays_per_probe; ++r) count += ray_backface[r];
            backface_share = float(count) / float(rays_per_probe);
        }
        next_sh[slot] = float4(lerp(estimate, probe_sh[slot].xyz, update.hysteresis), backface_share);
    }

    // Every thread one moment texel: the mean and mean square of the hit
    // distances of the rays near its direction.
    {
        const float3 texel_direction = momentTexelDirection(thread);
        float weight_sum = 0.0;
        float2 sum = float2(0.0, 0.0);
        for (uint r = 0; r < rays_per_probe; ++r)
        {
            const float weight = pow(saturate(dot(texel_direction, ray_direction[r])), moment_sharpness);
            const float d = ray_distance[r];
            sum += weight * float2(d, d * d);
            weight_sum += weight;
        }
        const uint slot = probe * probe_moment_texels + thread;
        // No ray near enough this update: keep what the texel held.
        const float2 estimate = weight_sum > 1.0e-6 ? sum / weight_sum : probe_moments[slot];
        next_moments[slot] = lerp(estimate, probe_moments[slot], update.hysteresis);
    }
}
