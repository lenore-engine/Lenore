// The probe field, read side: second-order spherical harmonics of radiance at
// each probe, an octahedral map of hit-distance moments for visibility, and
// the interpolation of eight probes around a shaded point. Included by
// scene.slang, which lights surfaces from it, and probes.slang, which feeds the
// field back into its own update for the second and later bounces.
//
// The includer declares the two buffers this reads, read-only, under these
// names:
//
//   StructuredBuffer<float4> probe_sh;       // probe * 9 + i, rgb in xyz
//   StructuredBuffer<float2> probe_moments;  // probe * 64 + texel
//
// The arithmetic mirrors `probe_grid.zig` in lenore-scene, which is where it
// is tested: the basis, the clamped-cosine kernel, the octahedral mapping and
// the grid's indexing.

#include "probe_grid.slang"

static const uint probe_sh_count = 9;
static const uint probe_moment_side = 8;
static const uint probe_moment_texels = probe_moment_side * probe_moment_side;

// The share of a probe's rays that may meet the back of a surface before the
// probe is taken to stand inside something. DDGI classifies its probes the
// same way; a quarter is from memory of its later work, to be checked.
static const float probe_invalid_share = 0.25;

// `probe_grid.shBasis`: Y00; Y1-1, Y10, Y11; Y2-2, Y2-1, Y20, Y21, Y22.
static const float sh_c0 = 0.28209479177;  // 1/2 sqrt(1/pi)
static const float sh_c1 = 0.48860251190;  // sqrt(3/(4 pi))
static const float sh_c2 = 1.09254843059;  // 1/2 sqrt(15/pi)
static const float sh_c3 = 0.31539156525;  // 1/4 sqrt(5/pi)
static const float sh_c4 = 0.54627421529;  // 1/4 sqrt(15/pi)

void shBasis(float3 d, out float basis[9])
{
    basis[0] = sh_c0;
    basis[1] = sh_c1 * d.y;
    basis[2] = sh_c1 * d.z;
    basis[3] = sh_c1 * d.x;
    basis[4] = sh_c2 * d.x * d.y;
    basis[5] = sh_c2 * d.y * d.z;
    basis[6] = sh_c3 * (3.0 * d.z * d.z - 1.0);
    basis[7] = sh_c2 * d.x * d.z;
    basis[8] = sh_c4 * (d.x * d.x - d.y * d.y);
}

// `probe_grid.irradiance_kernel` per coefficient: pi, then 2 pi / 3 for the
// three of band one, then pi / 4 for the five of band two.
static const float sh_kernel[9] = {
    3.14159265359,
    2.09439510239, 2.09439510239, 2.09439510239,
    0.78539816340, 0.78539816340, 0.78539816340, 0.78539816340, 0.78539816340,
};

// Irradiance at a surface facing `normal`, from one probe's radiance
// harmonics. Clamped at zero: a truncated series can ring below it where the
// light is strongly one-sided.
float3 probeIrradiance(uint probe, float3 normal)
{
    float basis[9];
    shBasis(normal, basis);
    float3 total = float3(0.0, 0.0, 0.0);
    for (uint i = 0; i < probe_sh_count; ++i)
        total += sh_kernel[i] * probe_sh[probe * probe_sh_count + i].xyz * basis[i];
    return max(total, float3(0.0, 0.0, 0.0));
}

float2 signNotZero(float2 v)
{
    return float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

// `probe_grid.octEncode` / `octDecode`, Y the pole.
float2 octEncode(float3 d)
{
    const float3 n = d / (abs(d.x) + abs(d.y) + abs(d.z));
    float2 uv = n.xz;
    if (n.y < 0.0) uv = (1.0 - abs(uv.yx)) * signNotZero(uv);
    return uv;
}

float3 octDecode(float2 uv)
{
    float3 d = float3(uv.x, 1.0 - abs(uv.x) - abs(uv.y), uv.y);
    if (d.y < 0.0) d.xz = (1.0 - abs(d.zx)) * signNotZero(d.xz);
    return normalize(d);
}

// The direction a moment texel's centre stands for.
float3 momentTexelDirection(uint texel)
{
    const float2 cell = float2(texel % probe_moment_side, texel / probe_moment_side) + 0.5;
    return octDecode(cell / float(probe_moment_side) * 2.0 - 1.0);
}

// The probe's mean and mean square of hit distance along `direction`,
// bilinear within its own tile. No border texels: at the tile's edge the
// filter clamps rather than wrapping onto the octahedron's other side, which
// coarsens visibility along the fold and nowhere else.
float2 probeMoments(uint probe, float3 direction)
{
    const float2 texel = (octEncode(direction) * 0.5 + 0.5) * float(probe_moment_side) - 0.5;
    const float2 lower = clamp(floor(texel), 0.0, float(probe_moment_side - 1));
    const float2 upper = min(lower + 1.0, float(probe_moment_side - 1));
    const float2 t = saturate(texel - lower);
    const uint base = probe * probe_moment_texels;
    const float2 a = probe_moments[base + uint(lower.y) * probe_moment_side + uint(lower.x)];
    const float2 b = probe_moments[base + uint(lower.y) * probe_moment_side + uint(upper.x)];
    const float2 c = probe_moments[base + uint(upper.y) * probe_moment_side + uint(lower.x)];
    const float2 e = probe_moments[base + uint(upper.y) * probe_moment_side + uint(upper.x)];
    return lerp(lerp(a, b, t.x), lerp(c, e, t.x), t.y);
}

uint probeIndex(ProbeGridParams grid, uint3 cell)
{
    return cell.x + grid.counts.x * (cell.y + grid.counts.y * cell.z);
}

// Irradiance at `position` facing `normal`, interpolated from the eight probes
// around it and weighted so that a probe on the far side of a wall, or behind
// the surface, contributes little or nothing. The weights follow DDGI
// (Majercik et al., "Dynamic Diffuse Global Illumination with Ray-Traced
// Irradiance Fields", JCGT 8(2) 2019): trilinear; a wrapped backface term; a
// Chebyshev bound on the probability that the probe sees the point, from the
// moments; small weights crushed so the bound's tail does not leak. The bias
// that moves the lookup off the surface, mostly toward the eye, is theirs in
// shape; its constants were written from memory of the paper and are to be
// checked against it.
float3 interpolateProbes(ProbeGridParams grid, float3 position, float3 normal, float3 to_eye)
{
    const float3 biased = position + (normal * 0.2 + to_eye * 0.8) * (0.3 * grid.spacing);

    const float3 steps = (biased - grid.origin) / grid.spacing;
    const float3 last = float3(grid.counts) - 2.0;
    const float3 clamped = clamp(steps, float3(0.0, 0.0, 0.0), last + 1.0);
    const float3 cell = min(floor(clamped), last);
    const float3 fraction = clamped - cell;
    const uint3 base = uint3(cell);

    float3 total = float3(0.0, 0.0, 0.0);
    float weight_sum = 0.0;
    for (uint corner = 0; corner < 8; ++corner)
    {
        const uint3 offset = uint3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        const uint probe = probeIndex(grid, base + offset);
        // Inside a wall or a prop: it sees backfaces, and its light and its
        // distances describe nothing a point outside it receives.
        if (probe_sh[probe * probe_sh_count].w > probe_invalid_share) continue;
        const float3 probe_position = grid.origin + float3(base + offset) * grid.spacing;

        const float3 trilinear = lerp(1.0 - fraction, fraction, float3(offset));
        float weight = trilinear.x * trilinear.y * trilinear.z;

        // A probe behind the surface lights it less, smoothly, rather than
        // being cut off where the normal turns away.
        const float3 to_probe = normalize(probe_position - position);
        const float wrap = (dot(to_probe, normal) + 1.0) * 0.5;
        weight *= wrap * wrap + 0.2;

        // How likely the probe sees the lookup point: past its mean hit
        // distance in that direction, by Chebyshev's inequality on the
        // variance.
        const float3 from_probe = biased - probe_position;
        const float distance = length(from_probe);
        const float2 moments = probeMoments(probe, from_probe / max(distance, 1.0e-4));
        if (distance > moments.x)
        {
            const float variance = abs(moments.y - moments.x * moments.x);
            const float excess = distance - moments.x;
            const float chebyshev = variance / (variance + excess * excess);
            weight *= max(chebyshev * chebyshev * chebyshev, 0.0);
        }

        // Below a fifth, fall off as the cube, so a nearly invisible probe is
        // almost absent rather than contributing its last few percent.
        const float threshold = 0.2;
        if (weight < threshold) weight *= weight * weight / (threshold * threshold);

        total += probeIrradiance(probe, normal) * weight;
        weight_sum += weight;
    }
    return weight_sum > 0.0 ? total / weight_sum : float3(0.0, 0.0, 0.0);
}

// Irradiance at `position` facing `normal`: the probes' light inside their box,
// and `open`, what the caller says an unoccluded point there receives, outside
// it.
//
// The lookup clamps to the outermost probes, so without this every point past
// them would read their light, drawn out across the ground as stripes. Within
// half a spacing of them the clamped lookup stands: that is the margin a grid
// is laid out with so its outer probes stand off the surfaces it bounds, and a
// floor or wall there belongs to the probes. Over the next spacing the field
// gives way to `open`, which is right for open ground and is the only estimate
// there is anywhere else the probes do not reach.
float3 sampleProbeField(ProbeGridParams grid, float3 position, float3 normal, float3 to_eye, float3 open)
{
    const float3 lowest = grid.origin;
    const float3 highest = grid.origin + float3(grid.counts - 1) * grid.spacing;
    const float outside = length(max(max(lowest - position, position - highest), float3(0.0, 0.0, 0.0)));
    const float share = saturate(outside / grid.spacing - 0.5);
    if (share >= 1.0) return open;
    return lerp(interpolateProbes(grid, position, normal, to_eye), open, share);
}
