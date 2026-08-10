// A Schwarzschild black hole with a thin accretion disk, integrated per pixel.
//
// Geometric units throughout: G = M = c = 1. The horizon is then at r = 2, the
// photon sphere at r = 3, the innermost stable circular orbit at r = 6, and the
// critical impact parameter at b = 3*sqrt(3) = 5.196. Nothing here is fitted to
// make a picture; every one of those numbers is a consequence of the metric and
// they are what the shadow's size and the ring's radius come out to.
//
// This is not sphere tracing. A distance-field march walks along a straight
// line and only the step length adapts; here the ray's direction changes at
// every step, because what is being integrated is the trajectory itself.
//
// It reads no descriptor set. Everything it needs arrives in one push block,
// which is what lets it be recorded inside the main pass with nothing bound.

struct PushConstants
{
    // The ray leaving the eye for device coordinate (x, y) is
    // `front + right * x + up * y`, unnormalized, in world space.
    float4 ray_right;
    float4 ray_up;
    float4 ray_front;
    // xyz is the eye. The hole sits at the origin.
    float4 eye;

    // The integration budget. A ray that has not escaped or fallen in by here
    // is one winding near the photon sphere; see `fragmentMain` for what it is
    // given.
    uint max_steps;

    // The largest angle, in radians, the ray may turn through in one step. The
    // step length is derived from it rather than fixed: dphi/dlambda = h / r^2,
    // so dlambda = turn * r^2 / h holds the turn constant while the step grows
    // quadratically with distance. Far from the hole a ray then crosses in a
    // handful of steps and near it the trajectory stays resolved.
    float max_turn;

    // Where the disk begins and ends. The inner edge is the ISCO by default:
    // inside it there is no circular orbit for matter to be on.
    float disk_inner;
    float disk_outer;

    // The temperature at the inner edge, in kelvin.
    float disk_temperature;

    // Which way the disk turns, +1 or -1. It decides which side of the picture
    // is beamed towards the observer, which is the most visible single fact in
    // the image.
    float disk_spin;

    // What the disk's radiance is multiplied by before it leaves this shader.
    // The output is scene radiance and the engine's chain and tone operator run
    // after it, so this sets where the disk sits against the value the operator
    // will render as white.
    float exposure;

    // How bright the stars are in the same units.
    float star_intensity;

    // Seconds since the run began. The disk's structure is carried around by
    // the orbital motion and nothing else here moves, so this is the whole of
    // what makes the picture change when the camera does not.
    float time;

    // How far the turbulence swings the local temperature, as a fraction. Small
    // by design: a band's response is exponential in 1/T, so a fifth of the
    // temperature is several times the brightness, and a fluctuation large
    // enough to look right in temperature would be white noise in the picture.
    float turbulence;

    // The disk's half thickness as a fraction of the radius. A thin disk is
    // one whose pressure scale height is small against its radius, and a few
    // hundredths is the standard shape; it is written as a ratio and not a
    // length because that is what makes the disk flare rather than stay a slab.
    float disk_thickness;

    // How much the fluid absorbs, per unit of the density integrated along the
    // path. Zero renders an optically thin disk where the far side shows
    // through the near one; the dark lanes across a bright disk are what this
    // buys.
    float disk_opacity;

    // What the fragment stage returns instead of radiance. Zero is the picture;
    // the others are for finding out what a feature in it is, which is cheaper
    // than reasoning about one. See `debugColour`.
    uint debug;
};

// How a ray ended. Read only by the debug output.
static const uint ended_captured = 0;
static const uint ended_escaped = 1;
static const uint ended_budget = 2;

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

struct Varying
{
    float4 position : SV_Position;
    // The ray leaving the eye, unnormalized. Affine in the device coordinate
    // and the triangle is drawn at w = 1, so interpolating it is exact.
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
    varying.ray = push.ray_front.xyz +
        push.ray_right.xyz * device.x +
        push.ray_up.xyz * device.y;
    return varying;
}

static const float horizon = 2.0;

// How far above the plane the medium reaches, in scale heights.
//
// The Gaussian has no end, so something has to stop the sampling, and what that
// something is matters more than where it sits. Cutting the density at a height
// gives the medium a surface: the boundary is the cone y = reach * thickness *
// r, whose half angle here is a few degrees, and an eye a few degrees above the
// plane sees that cone almost along its own surface, where a cone's silhouette
// is a straight line. It drew straight edges and rectangular bites out of the
// shadow, in a picture whose every other boundary is a circle.
//
// So the density is taken to zero over the last third of the reach instead,
// with `disk_falloff` where the taper starts. The medium then ends without
// having a surface anywhere.
static const float disk_reach = 3.5;
static const float disk_falloff = 2.1;

// Where a ray is far enough for the remaining bending to be invisible. The
// deflection outstanding beyond radius R is of order 4/R, so at 1000 it is
// four milliradians spread over the rest of an infinite path.
static const float escape_radius = 1000.0;

// The acceleration of a null geodesic in Schwarzschild geometry.
//
// Derivation, because the coefficient is the whole physics and a wrong one
// still draws a plausible picture. Light in this metric obeys the Binet
// equation d2u/dphi2 + u = 3*M*u^2 with u = 1/r. For a central acceleration
// a = -C * h^2 * r_hat / r^4, the standard identity
// r'' - r*phi'^2 = -h^2*u^2*(u'' + u) turns that into u'' + u = C*u^2, so
// C = 3*M, and with M = 1 the coefficient below is 3.
//
// Written with r^5 in the denominator so the numerator can be the position
// vector itself rather than a normalized copy of it.
//
// `h2` is the squared angular momentum |r x v|^2, which is conserved. It is
// computed once from the initial conditions and passed in: recomputing it each
// step would let integration error feed back into the constant that error is
// measured against.
float3 acceleration(float3 position, float h2)
{
    const float r2 = dot(position, position);
    const float r5 = r2 * r2 * sqrt(r2);
    return -3.0 * h2 * position / r5;
}

// Planck's law at one wavelength, in nanometres, up to a constant factor.
//
// The constant is dropped because only ratios between the three samples below
// survive: the absolute scale of the disk is set by the Stefan-Boltzmann factor
// in `diskRadiance` and by the exposure the caller pushes.
//
// hc/k = 1.4388e7 nm*K (the second radiation constant, in these units).
float planck(float wavelength_nm, float temperature)
{
    const float w = wavelength_nm;
    const float w5 = w * w * w * w * w;
    return 1.0 / (w5 * (exp(1.4388e7 / (w * temperature)) - 1.0));
}

// A blackbody sampled in three bands, carrying hue and brightness together.
//
// Together and not separately, because for a target whose channels are bands
// those are one quantity. From the invariance of I_nu / nu^3 the observed
// spectrum is B_nu(nu, g*T) exactly, so a band at a fixed observed wavelength
// receives that and nothing else: no factor of g is left to apply. The familiar
// g^4 is this integrated over all frequencies, and it belongs to a bolometric
// renderer rather than to a channel.
//
// It is still an approximation, and worth being exact about which one: the
// spectrum is sampled at three wavelengths rather than integrated against the
// CIE colour matching functions, so the hue progression is right and the
// saturation is not. Replacing it needs the matching function data, which is
// not in this repository.
float3 blackbodyBands(float temperature)
{
    return float3(
        planck(630.0, temperature),
        planck(532.0, temperature),
        planck(465.0, temperature));
}

// The authored parameter is retained at the scale established by the original
// zero-torque profile: its hottest point was 0.488 of the parameter. Keeping
// that scale makes exposure and the runtime temperature control stable while
// the radial shape below changes to a finite-temperature inner edge.
static const float peak_fraction = 0.488;

// One hash of a lattice point, in [0, 1).
//
// Value noise and not gradient noise. What this modulates is a scalar field
// with no directional structure of its own, and the axis-aligned artefacts that
// make value noise a poor choice for surfaces are hidden here by the coordinate
// the disk is sampled in, which is neither axis-aligned nor stationary.
float hash(float3 p)
{
    p = frac(p * 0.3183099 + 0.1);
    p *= 17.0;
    return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float valueNoise(float3 x)
{
    const float3 cell = floor(x);
    float3 t = frac(x);
    // Smoothstep on each axis, so the field is continuous in value and in slope
    // across a cell boundary. Trilinear alone leaves creases the shear below
    // would draw attention to.
    t = t * t * (3.0 - 2.0 * t);

    const float c000 = hash(cell + float3(0, 0, 0));
    const float c100 = hash(cell + float3(1, 0, 0));
    const float c010 = hash(cell + float3(0, 1, 0));
    const float c110 = hash(cell + float3(1, 1, 0));
    const float c001 = hash(cell + float3(0, 0, 1));
    const float c101 = hash(cell + float3(1, 0, 1));
    const float c011 = hash(cell + float3(0, 1, 1));
    const float c111 = hash(cell + float3(1, 1, 1));

    return lerp(
        lerp(lerp(c000, c100, t.x), lerp(c010, c110, t.x), t.y),
        lerp(lerp(c001, c101, t.x), lerp(c011, c111, t.x), t.y),
        t.z);
}

// The fluid's own structure at a point on the disk, in [0, 1].
//
// The point is turned backwards through the angle its matter has covered since
// the run began, so the noise is attached to the fluid rather than to space. A
// ring at radius r has covered Omega(r) * t, and Omega goes as r^(-3/2), so the
// inner rings unwind further than the outer ones and a feature that started
// radial is drawn out into a trailing spiral. Nothing here draws a spiral; the
// differential rotation is what makes one.
//
// The visual evolution is accelerated without changing the orbital velocity in
// `redshift`. Multiplying disk_spin instead would also multiply the Doppler
// term, making a request for faster animation change the rendered physics.
static const float disk_animation_rate = 12.0;

// Two octaves. The second is finer and weaker, which is what gives an edge
// inside a filament without turning the field into noise.
float turbulence(float3 hit, float r)
{
    const float angle = -push.disk_spin * disk_animation_rate * pow(r, -1.5) * push.time;
    const float c = cos(angle);
    const float sn = sin(angle);
    const float3 unwound = float3(
        hit.x * c - hit.z * sn,
        hit.y,
        hit.x * sn + hit.z * c);

    // Coarse. The first pass at 1.6 and 5.1 put several cells across the
    // visible disk and read as grain on a surface; a fluid's structure is broad
    // with occasional fine lanes, and the volume integration below softens what
    // is left.
    return valueNoise(unwound * 0.34) * 0.62 +
        valueNoise(unwound * 0.95) * 0.26 +
        valueNoise(unwound * 3.1) * 0.12;
}

// The disk's display temperature at radius r.
//
// The earlier zero-torque profile fell back to zero at the ISCO. Its hottest
// region was therefore an annulus outside the inner edge, and any temperature
// colour inevitably drew a blue ring. The cinematic target instead wants heat
// to rise continuously inward, so this visible-light mapping has a finite hot
// inner edge and a gentler r^(-0.45) falloff. The exponent keeps the outer disk
// near its previous temperature while moving the maximum to the centre.
float diskTemperature(float r)
{
    const float profile = peak_fraction * push.disk_temperature *
        pow(push.disk_inner / r, 0.45);

    // The outer edge taken to zero over the last fifth of the disk. The rim was
    // a step before, and a step in the source is a hard circle in the image:
    // the lensed picture of the edge itself, drawn wherever a ray crosses just
    // inside it beside one that crosses just outside.
    const float fade = smoothstep(push.disk_outer, push.disk_outer * 0.8, r);
    return profile * fade;
}

// The frequency ratio between what the disk emits and what the eye receives.
//
// g = nu_obs / nu_emit for an emitter on a circular geodesic orbit, from
// g = E / (p.u), where u is the emitter's four-velocity and E the photon's
// conserved energy. For a circular orbit u^t = 1/sqrt(1 - 3/r), and the only
// other term is the photon's angular momentum about the disk's axis, so
//
//     g = sqrt(1 - 3/r) / (1 - Omega * L_z)
//
// with Omega = spin * r^(-3/2) the coordinate angular velocity. Both effects
// are in that one expression: the numerator is the gravitational shift and the
// denominator is the Doppler shift of the orbital motion.
//
// The square root is real for every r the disk occupies, since it begins at the
// ISCO at r = 6 and the root only fails inside the photon sphere at r = 3.
//
// The angular momentum is taken from the photon's own state. It is conserved,
// so its value at the crossing is its value anywhere.
//
// What the expression wants is L/E, and this passes L with the affine parameter
// fixed by |v| = 1 at the eye. The two agree exactly only for an eye at
// infinity; at a finite radius the difference is of order 1/r_eye, which is why
// the caller places the camera far from the hole and not beside it.
float redshift(float r, float axial_momentum)
{
    const float omega = push.disk_spin * pow(r, -1.5);
    return sqrt(1.0 - 3.0 / r) / (1.0 - omega * axial_momentum);
}

// The disk's half thickness at radius r.
//
// Proportional to the radius, which is what a hydrostatic disk does: the
// pressure scale height grows with the orbital radius, so the disk flares
// outward rather than staying a slab of constant thickness.
float diskHalfThickness(float r)
{
    return push.disk_thickness * r;
}

// Shorten a trajectory step before it crosses the disk.
//
// Bending is a poor step control for a nearly radial ray: its angular momentum
// is small, so the unconstrained first step can run from the eye to the hole.
// Testing only the starting point against the disk then misses the near side
// completely. Worse, a midpoint that happens to land in the medium attributes
// that whole eye-to-hole segment to one sample.
//
// The quadratic below is the same constant-acceleration trajectory the
// integrator is about to take. There are three ways a segment can enter the
// medium: through the outer rim, through either face of the flared disk, or
// through one face and out the other in a single step. Each case stops the ray
// just before entry, where the normal disk step limit in fragmentMain takes
// over. Looking only for a midplane crossing misses the second case; the long
// segment then becomes one oversized absorption sample, whose projected cone
// is the dark rectangle this limiter exists to prevent.
float limitDiskApproach(
    float3 position,
    float3 velocity,
    float3 acceleration_now,
    float dt)
{
    const float3 candidate = position + velocity * dt + 0.5 * acceleration_now * dt * dt;
    const float radius = length(position);
    const float candidate_radius = length(candidate);
    float limited = dt;

    // Entry through the outer rim while already between the disk's faces.
    if (radius > push.disk_outer && candidate_radius <= push.disk_outer)
    {
        const float fraction = (radius - push.disk_outer) /
            max(radius - candidate_radius, 1e-6);
        const float entry_dt = dt * fraction;
        const float3 entry_position = position + velocity * entry_dt +
            0.5 * acceleration_now * entry_dt * entry_dt;
        const float entry_radius = length(entry_position);
        const float thickness = diskHalfThickness(entry_radius);

        if (abs(entry_position.y) <= thickness * disk_reach)
        {
            const float3 entry_velocity = velocity + acceleration_now * entry_dt;
            const float sample_dt = thickness * 0.35 /
                max(length(entry_velocity), 1e-6);
            limited = min(limited, max(entry_dt - sample_dt, sample_dt));
        }
    }

    // Entry through a face without crossing the midplane in this step. The
    // signed quantity is positive outside the flared cone and non-positive
    // inside it.
    const float face_now = abs(position.y) - diskHalfThickness(radius) * disk_reach;
    const float face_candidate = abs(candidate.y) -
        diskHalfThickness(candidate_radius) * disk_reach;
    if (face_now > 0.0 && face_candidate <= 0.0)
    {
        const float fraction = face_now / max(face_now - face_candidate, 1e-6);
        const float entry_dt = dt * fraction;
        const float3 entry_position = position + velocity * entry_dt +
            0.5 * acceleration_now * entry_dt * entry_dt;
        const float entry_radius = length(entry_position);

        if (entry_radius >= push.disk_inner && entry_radius <= push.disk_outer)
        {
            const float3 entry_velocity = velocity + acceleration_now * entry_dt;
            const float sample_dt = diskHalfThickness(entry_radius) * 0.35 /
                max(length(entry_velocity), 1e-6);
            limited = min(limited, max(entry_dt - sample_dt, sample_dt));
        }
    }

    // Both endpoints can lie outside opposite faces while the segment passes
    // through the whole medium. Locate its midplane crossing, then step only as
    // far as the near face.
    if (position.y * candidate.y <= 0.0)
    {
        const float y_span = abs(position.y) + abs(candidate.y);
        const float crossing_fraction = y_span > 1e-6 ? abs(position.y) / y_span : 0.0;
        const float crossing_dt = dt * crossing_fraction;
        const float3 crossing_position = position + velocity * crossing_dt +
            0.5 * acceleration_now * crossing_dt * crossing_dt;
        const float crossing_radius = length(crossing_position);

        if (crossing_radius >= push.disk_inner && crossing_radius <= push.disk_outer)
        {
            const float3 crossing_velocity = velocity + acceleration_now * crossing_dt;
            const float speed = max(length(crossing_velocity), 1e-6);
            const float vertical_speed = max(abs(crossing_velocity.y), 1e-6);
            const float thickness = diskHalfThickness(crossing_radius);
            const float entry_dt = crossing_dt - disk_reach * thickness / vertical_speed;
            const float sample_dt = thickness * 0.35 / speed;
            limited = min(limited, max(entry_dt, sample_dt));
        }
    }

    return limited;
}

// How much fluid is at a point, in [0, 1].
//
// Gaussian across the plane, which is the hydrostatic profile: the vertical
// support is thermal pressure against the tidal component of gravity, and both
// are linear near the midplane.
//
// Radially the material begins and ends over a width rather than at a radius,
// and both ends are smooth in slope as well as in value. That is not a
// softening for its own sake. Every gradient in this field is magnified by the
// lensing that carries it to the eye, without bound near the photon sphere, so
// a profile whose derivative diverges anywhere puts a hard arc in the picture
// far from the radius it came from. The temperature is free to keep the
// square-root shape its own boundary condition gives it, because what reaches
// the eye from it is exponentially small where that shape is steep.
//
// The turbulence multiplies here as well as shifting the temperature. A
// filament is denser and hotter together, which is one fluctuation seen twice
// rather than two independent ones.
float diskDensity(float3 p, float r, float structure)
{
    if (r < push.disk_inner || r > push.disk_outer) return 0.0;

    const float h = diskHalfThickness(r);
    const float u = abs(p.y) / h;
    const float vertical = exp(-0.5 * u * u) * smoothstep(disk_reach, disk_falloff, u);

    // The two ends, each over a width. The inner transition is narrow enough
    // that the finite inner-edge temperature remains visible, while the outer
    // quarter stays broad because its projected rim spans much more of the
    // picture. Both are widths and not hard radii: a discontinuity here would
    // be magnified into an arc by the lens.
    const float radial = smoothstep(push.disk_inner, push.disk_inner * 1.2, r) *
        smoothstep(push.disk_outer, push.disk_outer * 0.75, r);

    return vertical * radial * structure;
}

// The vertical column of a unit-density disk, which is what the volume
// emissivity below is divided by.
//
// A Gaussian of standard deviation h integrates to sqrt(2*pi) * h across the
// plane. Dividing by it is what makes the quantity this shader carries the
// disk's surface brightness rather than its emissivity: without it the disk
// gets brighter simply where it is thicker, and since the thickness grows with
// the radius, the outer disk outshines the inner one for no physical reason at
// all.
float diskColumn(float r)
{
    return 2.5066283 * diskHalfThickness(r);
}

// Chromaticity of the visible-light mapping.
//
// The broad disk follows a restrained warm-to-ivory palette. Blue follows the
// monotonic inner temperature through a broad smooth transition; unlike the
// former 0.985 threshold over an annular temperature maximum, it has neither a
// hard edge nor a second warm side toward the hole. The base radial temperature,
// not its turbulent fluctuation, selects the palette so isolated hot noise
// cannot scatter blue patches across the outer disk.
//
// This is an authored display mapping in the spirit of the reference pictures.
// A physical conversion of the complete spectrum would need CIE
// colour-matching data, which this repository does not carry.
float3 diskTint(float temperature)
{
    const float peak_temperature = peak_fraction * push.disk_temperature;
    const float heat = temperature / peak_temperature;

    const float3 warm = float3(1.0, 0.52, 0.28);
    const float3 ivory = float3(1.0, 0.86, 0.72);
    const float3 core_blue = float3(0.32, 0.66, 1.0);

    const float warm_to_ivory = smoothstep(0.35, 0.9, heat);
    const float blue_core = smoothstep(0.65, 1.0, heat);
    return lerp(lerp(warm, ivory, warm_to_ivory), core_blue, blue_core);
}

// What one sample of the disk emits, before it is weighted by density.
//
// The observed blackbody bands carry radial falloff, gravitational shift and
// Doppler beaming together. Their largest component becomes intensity while
// `diskTint` supplies a deliberate display chromaticity. Raising normalized
// intensity to 0.86 compresses the visible-band range enough to retain the cool
// annulus without lifting most of it into the tone mapper's white compression;
// a value of one at the unshifted thermal peak remains one.
//
// The shift is taken from the midplane orbit at this radius. A parcel above the
// plane is not on a circular geodesic and its own velocity differs, but the
// disk is a few hundredths of a radius thick and the difference is smaller than
// the turbulence beside it.
float3 diskEmission(float r, float structure, float axial_momentum)
{
    const float temperature = diskTemperature(r);
    if (temperature <= 0.0) return float3(0.0);

    // The fluctuation is in temperature and not in brightness, so a hot
    // filament moves toward the inner colour as well as becoming brighter.
    const float local = temperature * (1.0 + push.turbulence * (2.0 * structure - 1.0));
    const float observed = redshift(r, axial_momentum) * local;
    const float3 bands = blackbodyBands(observed);

    const float3 reference = blackbodyBands(peak_fraction * push.disk_temperature);
    const float reference_peak = max(max(reference.r, reference.g), max(reference.b, 1e-30));
    const float relative_intensity = max(max(bands.r, bands.g), bands.b) / reference_peak;
    const float visible_intensity = pow(max(relative_intensity, 0.0), 0.86);

    // Per unit length, and normalized so that integrating it across the disk
    // returns the surface brightness rather than something proportional to the
    // thickness.
    return diskTint(temperature) * visible_intensity * push.exposure / diskColumn(r);
}

// A field of stars in the direction a ray escaped along.
//
// It is not decoration. What makes gravitational lensing legible is a field of
// point sources: each star behind the hole arrives as two separate images, and
// the ring is the locus where they merge. A smooth background would bend into a
// swirl and show none of it.
//
// One star per occupied cell of a direction lattice, placed at a hashed point
// inside its cell rather than filling it. That is what makes a star round and
// gives it an edge the ring can stretch: a cell-shaped star aliases into
// rectangles exactly where the deflection is strongest.
static const float star_lattice = 260.0;
static const float star_occupancy = 0.11;

float3 stars(float3 direction)
{
    const float3 scaled = direction * star_lattice;
    const float3 cell = floor(scaled);

    const float presence = hash(cell);
    if (presence > star_occupancy) return float3(0.0);

    // Two more hashes off the same cell: where in it the star sits, and how
    // bright it is. Offsetting the lattice point keeps the three independent.
    const float3 offset = float3(
        hash(cell + float3(11.3, 0, 0)),
        hash(cell + float3(0, 7.7, 0)),
        hash(cell + float3(0, 0, 5.1)));
    const float magnitude = hash(cell + float3(3.3, 3.3, 3.3));

    // A round profile falling to nothing at the cell's own scale.
    const float distance = length(frac(scaled) - offset);
    const float profile = max(1.0 - distance * 2.2, 0.0);

    // Raised so the field is mostly faint with a few bright ones, which is what
    // a magnitude distribution looks like. Real stellar colour would come from
    // the same blackbody path the disk uses; a background is not the subject.
    const float3 tint = lerp(float3(1.0, 0.93, 0.84), float3(0.78, 0.86, 1.0), magnitude);
    return tint * push.star_intensity * profile * profile * magnitude * magnitude;
}

// What the debug modes return in place of radiance.
//
// Flat colours and not a gradient: what these answer is which of a few
// categories a pixel fell into, and a ramp would invite the eye to read a
// magnitude that is not there.
float3 debugColour(uint ending, uint samples, uint steps)
{
    switch (push.debug)
    {
    // How the ray ended: red captured, green escaped, blue out of budget.
    case 1:
        if (ending == ended_captured) return float3(0.6, 0.1, 0.1);
        if (ending == ended_escaped) return float3(0.1, 0.5, 0.1);
        return float3(0.15, 0.25, 0.8);
    // How many segments of the ray lay inside the medium. It is a cost as much
    // as a picture: this is where the budget goes on a ray that crosses the
    // disk at a shallow angle.
    case 2:
        return float3(float(samples) / 64.0);
    // How much of the budget it spent, as a ramp. The one mode that is a
    // magnitude, and it is the only question here that has one.
    default:
        return float3(float(steps) / float(max(push.max_steps, 1u)));
    }
}

[shader("fragment")]
float4 fragmentMain(Varying varying) : SV_Target
{
    float3 position = push.eye.xyz;
    float3 velocity = normalize(varying.ray);

    // Conserved along the whole trajectory, so both are taken once. `h2` sets
    // the bending and the step length; the axial component sets the Doppler
    // shift at every crossing.
    const float3 angular_momentum = cross(position, velocity);
    const float h2 = dot(angular_momentum, angular_momentum);
    // The disk lies in the plane y = 0, so its axis is y and this is the
    // component about that axis.
    const float axial_momentum = angular_momentum.y;

    // The disk is optically thin, so a ray crossing it twice picks up both.
    // That is what puts the far side of the disk above and below the hole: the
    // image over the top is the underside of the disk behind it, bent over.
    float3 radiance = float3(0.0);

    float3 acceleration_now = acceleration(position, h2);

    // What survives of what lies further along the ray. The disk is a medium
    // and not a surface, so the far side shows through the near one in
    // proportion to this, and a lane thick enough to take it to nothing is what
    // draws a dark band across a bright disk.
    float transmittance = 1.0;

    uint samples = 0;
    uint step = 0;
    for (; step < push.max_steps; step += 1)
    {
        const float r = length(position);
        if (r < horizon)
        {
            if (push.debug != 0) return float4(debugColour(ended_captured, samples, step), 1.0);
            return float4(radiance, 1.0);
        }
        if (r > escape_radius && dot(position, velocity) > 0.0)
        {
            if (push.debug != 0) return float4(debugColour(ended_escaped, samples, step), 1.0);
            return float4(radiance + stars(normalize(velocity)) * transmittance, 1.0);
        }
        // Nothing further along the ray can reach the eye. Stopping here is not
        // an approximation: what is left is multiplied by a number that has
        // already fallen below what the target can hold.
        if (transmittance < 0.003)
        {
            if (push.debug != 0) return float4(debugColour(ended_budget, samples, step), 1.0);
            return float4(radiance, 1.0);
        }

        // The turn per step held constant, from dphi/dlambda = h / r^2. A
        // nearly radial ray has little bending to constrain it, so also limit
        // one step to half the current radius. Besides resolving the changing
        // field, this prevents the exact central ray from landing at r = 0,
        // where evaluating the next acceleration would divide by zero.
        float dt = min(push.max_turn * r * r / max(sqrt(h2), 1e-6), r * 0.5);
        dt = limitDiskApproach(position, velocity, acceleration_now, dt);

        // Near the disk the turn is the wrong control. A step sized by how much
        // the ray bends says nothing about how thick the medium it is crossing
        // is, and one step across the whole slab is a surface again. Once the
        // approach limiter has brought the ray to the medium, cut the spatial
        // step to a fraction of its local thickness. `dt` is affine time, not
        // distance, so the limit is divided by the ray's current speed.
        const float thickness = diskHalfThickness(max(r, push.disk_inner));
        if (r < push.disk_outer * 1.2 && abs(position.y) < thickness * disk_reach)
            dt = min(dt, max(abs(position.y) * 0.5, thickness * 0.35) /
                max(length(velocity), 1e-6));

        const float3 previous = position;
        position += velocity * dt + 0.5 * acceleration_now * dt * dt;
        const float segment_length = length(position - previous);
        const float3 acceleration_next = acceleration(position, h2);
        velocity += 0.5 * (acceleration_now + acceleration_next) * dt;
        acceleration_now = acceleration_next;

        // The segment's midpoint, which is where the medium along it is
        // sampled. Taking an endpoint instead biases every step by half its
        // length in whichever direction the ray happens to be going.
        const float3 middle = (previous + position) * 0.5;
        const float middle_radius = length(middle);
        if (middle_radius < push.disk_inner || middle_radius > push.disk_outer) continue;
        if (abs(middle.y) > diskHalfThickness(middle_radius) * disk_reach) continue;

        const float structure = turbulence(middle, middle_radius);
        const float density = diskDensity(middle, middle_radius, structure);
        if (density <= 0.0) continue;

        samples += 1;

        // Emission first and then the attenuation it suffers, in that order:
        // what this segment emits is not absorbed by itself. Radiative transfer
        // integrates over spatial distance; `dt` is the geodesic's affine
        // interval and the two differ as the ray accelerates near the hole.
        const float3 emission = diskEmission(middle_radius, structure, axial_momentum);
        radiance += emission * density * segment_length * transmittance;
        transmittance *= exp(-push.disk_opacity * density * segment_length);
    }

    // Out of budget. A ray that reaches here is winding near the photon sphere
    // and its remaining path is dominated by orbits it will not escape, so what
    // it has already gathered is the answer.
    if (push.debug != 0) return float4(debugColour(ended_budget, samples, step), 1.0);
    return float4(radiance, 1.0);
}
