#pragma once
// The atmosphere's transport, device side: a port of `lenore-scene`'s
// atmosphere (Hillaire, "A Scalable and Production Ready Sky and Atmosphere
// Rendering Technique", EGSR 2020), itself a port of Epic's reference,
// UnrealEngineSkyAtmosphere, whose licence follows. The function names are
// the Zig file's, and the arithmetic is the same step for step; what differs
// is how positions are held (below). `examples/atmosphere_check.zig` compares
// the two on the device.
//
// MIT License
//
// Copyright (c) 2020 Epic Games, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Positions are relative to the point on the planet's surface under the
// viewer, with the planet's centre at (0, -bottom, 0). Absolute positions at
// 6360 km step about half a metre in a float, which is coarse against the
// reference's 10 m offset and against rays that end on the ground; relative
// ones keep the squared radii out of every subtraction, so no 64-bit floats
// are needed. The host does the same in f64.

// Mirrors `atmosphere.Parameters`, row for row.
struct Atmosphere
{
    float4 radii;      // bottom, top, Rayleigh scale height, Mie scale height
    float4 ray;        // Rayleigh scattering
    float4 mie;        // Mie scattering
    float4 extinction; // Mie extinction
    float4 absorption; // ozone absorption
    float4 ozone_tent; // layer width, lower slope, lower intercept, upper slope
    float4 knobs;      // upper intercept, aerosol, ozone, Mie g
    float4 samples;    // least and most samples, the viewer's altitude
    float4 ground;     // ground albedo
};

[[vk::binding(0, 0)]]
Sampler2D transmittance_lut;

[[vk::binding(1, 0)]]
Sampler2D multiple_lut;

[[vk::binding(2, 0)]]
StructuredBuffer<Atmosphere> model;

static const float pi = 3.141592653589793;

// Altitude above the bottom sphere: |p + bottom| - bottom, written so the two
// large terms cancel exactly.
float height(Atmosphere a, float3 p)
{
    const float r = length(p + float3(0.0, a.radii.x, 0.0));
    return (dot(p, p) + 2.0 * a.radii.x * p.y) / (r + a.radii.x);
}

// raySphereIntersectNearest for a sphere of `radius` about the planet's
// centre, from a surface-relative `p`: the nearest non-negative distance, or
// -1. The roots are taken in the numerically stable form, so neither
// subtracts two nearly equal lengths.
float sphere(Atmosphere a, float3 p, float3 d, float radius)
{
    const float b = dot(p, d) + a.radii.x * d.y;
    const float c = dot(p, p) + 2.0 * a.radii.x * p.y + (a.radii.x - radius) * (a.radii.x + radius);
    const float aa = dot(d, d);
    const float delta = b * b - aa * c;
    if (delta < 0.0) return -1.0;
    const float root = sqrt(delta);
    const float q = -b - (b >= 0.0 ? root : -root);
    const float t0 = q / aa;
    const float t1 = q == 0.0 ? -b / aa : c / q;
    const float lo = min(t0, t1);
    const float hi = max(t0, t1);
    return lo >= 0.0 ? lo : (hi >= 0.0 ? hi : -1.0);
}

// fromUnitToSubUvs.
float unitToSub(float u, float resolution)
{
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

// LutTransmittanceParamsToUv and the lookup, at altitude `h` with the sun at
// zenith cosine `mu`.
float3 sampleTransmittance(Atmosphere a, float h, float mu)
{
    const float r = a.radii.x + h;
    const float H = sqrt((a.radii.y - a.radii.x) * (a.radii.y + a.radii.x));
    const float rho = sqrt(max(0.0, h * (r + a.radii.x)));
    const float delta = r * r * mu * mu + (a.radii.y - r) * (a.radii.y + r);
    const float d = max(0.0, -r * mu + sqrt(max(0.0, delta)));
    const float d_min = a.radii.y - r;
    return transmittance_lut.SampleLevel(float2((d - d_min) / (rho + H - d_min), rho / H), 0.0).rgb;
}

// GetMultipleScattering's lookup.
float3 sampleMultiple(Atmosphere a, float h, float mu)
{
    const float2 uv = float2(
        unitToSub(saturate(mu * 0.5 + 0.5), 32.0),
        unitToSub(saturate(h / (a.radii.y - a.radii.x)), 32.0));
    return multiple_lut.SampleLevel(uv, 0.0).rgb;
}

struct Medium
{
    float3 ray;
    float3 mie;
    float3 scattering;
    float3 extinction;
};

// sampleMediumRGB, with the aerosol and ozone knobs.
Medium medium(Atmosphere a, float3 p)
{
    const float h = height(a, p);
    Medium m;
    m.ray = a.ray.rgb * exp(-h / a.radii.z);
    const float density = a.knobs.y * exp(-h / a.radii.w);
    m.mie = a.mie.rgb * density;
    m.scattering = m.ray + m.mie;
    const float ozone = saturate(h < a.ozone_tent.x
        ? a.ozone_tent.y * h + a.ozone_tent.z
        : a.ozone_tent.w * h + a.knobs.x);
    m.extinction = m.ray + a.extinction.rgb * density + a.absorption.rgb * (a.knobs.z * ozone);
    return m;
}

// IntegrateScatteredLuminance as SkyViewLutPS calls it, for a sun of unit
// irradiance above the atmosphere: a variable count of samples spaced as the
// square of their index, Cornette-Shanks Mie and Rayleigh phases, multiple
// scattering from its table, the earth's shadow, and the sunlit ground where
// the ray ends on it. No sun disc.
float3 integrate(Atmosphere a, float3 d, float3 sun)
{
    const float3 p = float3(0.0, a.samples.z, 0.0);
    const float bottom = sphere(a, p, d, a.radii.x);
    const float top = sphere(a, p, d, a.radii.y);
    const float end = bottom >= 0.0 ? min(bottom, top) : top;
    if (end <= 0.0) return float3(0.0, 0.0, 0.0);

    const float count = lerp(a.samples.x, a.samples.y, saturate(end * 0.01));
    const float count_floor = floor(count);
    const float end_floor = end * count_floor / count;

    // hgPhase(g, -cosine) with USE_CornetteShanks. The denominator
    // 1 + g^2 - 2 g cosine is written so it does not cancel near |g| = 1.
    const float cosine = clamp(dot(d, sun), -1.0, 1.0);
    const float g = a.knobs.w;
    const float denom = g >= 0.0
        ? (1.0 - g) * (1.0 - g) + 2.0 * g * (1.0 - cosine)
        : (1.0 + g) * (1.0 + g) - 2.0 * g * (1.0 + cosine);
    const float mie_phase = 3.0 / (8.0 * pi) * (1.0 - g * g) / (2.0 + g * g) * (1.0 + cosine * cosine) / (denom * sqrt(denom));
    const float ray_phase = 3.0 / (16.0 * pi) * (1.0 + cosine * cosine);

    float3 throughput = float3(1.0, 1.0, 1.0);
    float3 luminance = float3(0.0, 0.0, 0.0);
    for (float s = 0.0; s < count; s += 1.0)
    {
        const float f0 = s / count_floor;
        const float f1 = (s + 1.0) / count_floor;
        const float t0 = end_floor * f0 * f0;
        const float t1 = f1 > 1.0 ? end : end_floor * f1 * f1;
        const float dt = t1 - t0;
        const float3 point = p + d * (t0 + dt * 0.3);
        const float h = height(a, point);
        const float3 up = normalize(point + float3(0.0, a.radii.x, 0.0));
        const float mu = dot(sun, up);
        const Medium m = medium(a, point);
        const float3 segment = exp(-m.extinction * dt);
        // The earth's shadow against a sphere raised by the reference's
        // PLANET_RADIUS_OFFSET, 10 m.
        const float shadow = sphere(a, point - up * 0.01, sun, a.radii.x) >= 0.0 ? 0.0 : 1.0;
        const float3 source = shadow * sampleTransmittance(a, h, mu) * (m.mie * mie_phase + m.ray * ray_phase)
            + sampleMultiple(a, h, mu) * m.scattering;
        // The segment's analytic integral, (1 - e^-tau) / extinction, which
        // tends to dt in a vacuum; a short series there avoids 0 / 0 and the
        // cancellation just above it.
        float3 weight;
        for (int c = 0; c < 3; ++c)
        {
            const float tau = m.extinction[c] * dt;
            weight[c] = tau < 0.001 ? dt * (1.0 - tau * 0.5 + tau * tau / 6.0) : (1.0 - segment[c]) / m.extinction[c];
        }
        luminance += throughput * source * weight;
        throughput *= segment;
    }

    // The reference adds the ground only for its multiple-scattering table;
    // here the sky has it too, or below the horizon it is black.
    if (bottom > 0.0 && end == bottom)
    {
        const float3 point = p + d * bottom;
        const float3 up = normalize(point + float3(0.0, a.radii.x, 0.0));
        const float mu = dot(sun, up);
        luminance += sampleTransmittance(a, height(a, point), mu) * throughput * (saturate(mu) / pi) * a.ground.rgb;
    }
    return luminance;
}

// `atmosphere.cubeDirection`: Vulkan specification, Cube Map Face Selection
// and Transformations, inverted.
float3 cubeDirection(uint face, float2 uv)
{
    const float s = 2.0 * uv.x - 1.0;
    const float t = 2.0 * uv.y - 1.0;
    float3 d;
    switch (face)
    {
    case 0: d = float3(1.0, -t, -s); break;
    case 1: d = float3(-1.0, -t, s); break;
    case 2: d = float3(s, 1.0, t); break;
    case 3: d = float3(s, -1.0, -t); break;
    case 4: d = float3(s, -t, 1.0); break;
    default: d = float3(-s, -t, -1.0); break;
    }
    return normalize(d);
}
