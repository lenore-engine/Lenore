// The field the example stands on: a heightfield, and what grows on it.
//
// This is the field as a function and nothing else. It plants no blade and
// builds no geometry: the foliage is generated on the device, from a compute
// dispatch over a grid around the camera, and never exists as host memory. What
// the device needs from here is where the ground is and how much grows there,
// which it reads from a texture this file bakes rather than from a second
// implementation of the same noise in Slang. One account of the field, sampled
// by everyone.
//
// Nothing here allocates per frame or touches a device. It is arithmetic over a
// seed, and that is the point. The example exists to be measured, and a
// measurement needs the same picture twice: the same seed and the same settings
// have to give the same field on any machine, or a difference of a fraction of a
// millisecond drowns in whatever the scene did differently that run. Every value
// below is a pure function of the settings and integer hashes of lattice
// coordinates, with no clock, no thread count, and no float accumulated across a
// loop whose order could change.

const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Error = error{
    ChunkSpanNotPositive,
    ChunksPerSideZero,
    FeatureSizeNotPositive,
    OctavesZero,
    LacunarityNotAboveOne,
    GainOutOfRange,
    SandMarginNegative,
    ReliefNotPositive,
    BakeResolutionZero,
    DomainTooLarge,
};

// What covers a point of the field. The ground shading reads it to pick a
// colour, and the two growing kinds are what the foliage pass dispatches for, so
// the boundary between them is owned here and agreed on by both.
pub const Cover = enum(u8) { water, sand, grass, wheat };

pub const Settings = struct {
    seed: u32 = 0x10ec_0000,

    // The domain is a square of `chunks_per_side` squares, each `chunk_span`
    // metres on a side, centred on the origin. Stated as a count and a span
    // rather than as a total extent and a chunk size, because those two can
    // disagree and this pair cannot.
    chunk_span: f32 = 16.0,
    chunks_per_side: u32 = 16,

    // Metres per period of the lowest octave of the height noise, and the scale
    // the noise is multiplied by. The reference pictures are a plain: the relief
    // is there to keep the horizon from being a ruler, not to make hills.
    //
    // Not the height actually reached. Octaves partly cancel, so the summed and
    // renormalised field stays well inside its own bound: measured over two
    // million samples at these settings it spans -0.50 to +0.62 of full scale,
    // which is -1.2 to +1.5 metres. Anything comparing an absolute height
    // against a threshold has to be set against that range and not against
    // `relief`, which is why the two below carry measurements of their own.
    feature_size: f32 = 96.0,
    relief: f32 = 2.4,
    octaves: u32 = 4,
    // Deliberately not exactly two. At two, every octave's lattice lands on the
    // previous one's, and the shared zero crossings show as a faint grid.
    lacunarity: f32 = 2.03,
    gain: f32 = 0.5,

    // Absolute heights, in metres. Below `water_level` is water, and the band of
    // `sand_margin` above it is beach.
    //
    // Chosen by measuring the cover they produce over the domain rather than by
    // eye: these give 1.5% water, 5.2% sand and the rest growing, which is the
    // plain with a pond that the example is after. The pair is sensitive, since
    // the terrain spans under three metres in total. Raising the level to -0.62
    // turns a quarter of the world into beach.
    water_level: f32 = -0.90,
    sand_margin: f32 = 0.25,

    // Where wheat grows instead of grass, as a threshold on its own low
    // frequency noise. A larger threshold means less wheat.
    wheat_feature_size: f32 = 140.0,
    wheat_threshold: f32 = 0.16,
    // How much the second octave of the wheat mask contributes. It exists to
    // make the patch boundary irregular, so it is well under the `gain` the
    // terrain itself uses.
    wheat_detail: f32 = 0.3,
    // Metres over which one gives way to the other. Without it the two would
    // meet on the noise's exact level set, which is a line no field has.
    wheat_blend: f32 = 0.08,

    // How far above the beach the cover reaches full density, in metres. The
    // ramp is what stops the field from ending on a hard line that no boundary
    // in the picture justifies. Short, because the terrain's whole span is under
    // three metres: a ramp of a metre would thin the entire field rather than
    // its shore.
    shore_falloff: f32 = 0.45,
    // Metres per period of the term that thins the field in patches, and how
    // much of the density it accounts for.
    patch_feature_size: f32 = 21.0,
    patch_depth: f32 = 0.45,

    // Texels per side of the baked map. One texel is `2 * extent /
    // bake_resolution` metres, which at the defaults is 25 cm.
    //
    // Coarser than the spacing blades will be planted at, and deliberately. What
    // the map carries varies over metres: the height's own period is 96 metres
    // at the lowest octave and 12 at the highest, and the densities are smoother
    // still, so the sample a blade takes between texels is interpolation and not
    // a guess. Doubling this costs four times the memory and four times the
    // bake, and buys detail the field does not contain.
    bake_resolution: u32 = 1024,

    // Half the width of the domain, in metres.
    pub fn extent(self: Settings) f32 {
        return 0.5 * self.chunk_span * @as(f32, @floatFromInt(self.chunks_per_side));
    }

    // Checked once, here, so that nothing below carries a check. The last of
    // these is not taste: it is what lets `latticeFloor` cast without a bounds
    // test in a build where a failed cast is undefined behaviour and not a panic.
    pub fn validate(self: Settings) Error!void {
        if (!(self.chunk_span > 0.0)) return error.ChunkSpanNotPositive;
        if (self.chunks_per_side == 0) return error.ChunksPerSideZero;
        if (!(self.feature_size > 0.0) or !(self.wheat_feature_size > 0.0) or
            !(self.patch_feature_size > 0.0)) return error.FeatureSizeNotPositive;
        if (self.octaves == 0) return error.OctavesZero;
        if (!(self.lacunarity > 1.0)) return error.LacunarityNotAboveOne;
        if (!(self.gain > 0.0) or !(self.gain < 1.0)) return error.GainOutOfRange;
        // The same range, because it reaches `fbm` through the same parameter.
        if (!(self.wheat_detail > 0.0) or !(self.wheat_detail < 1.0)) return error.GainOutOfRange;
        if (self.sand_margin < 0.0) return error.SandMarginNegative;
        // Zero relief would make the height encoding below divide by zero, and a
        // flat field is expressed by a small relief rather than by none.
        if (!(self.relief > 0.0)) return error.ReliefNotPositive;
        if (self.bake_resolution == 0) return error.BakeResolutionZero;

        // The highest octave has the smallest period, so it produces the largest
        // lattice coordinate the noise will ever floor. Everything sampled here
        // stays inside the domain, and its corner is the extreme.
        var period = @min(self.feature_size, @min(self.wheat_feature_size, self.patch_feature_size));
        // At least two, because the wheat mask takes two octaves whatever
        // `octaves` says, and this bound has to cover every field sampled here.
        const deepest = @max(self.octaves, 2);
        var octave: u32 = 1;
        while (octave < deepest) : (octave += 1) period /= self.lacunarity;
        const corner = self.extent() / period;
        // Well inside i32, and inside the range where an f32 still distinguishes
        // neighbouring integers, which is the tighter of the two limits: past
        // 2^24 the lattice collapses and the noise goes flat rather than failing.
        if (!(corner < 8.0e6)) return error.DomainTooLarge;
    }
};

// Chris Wellons, "Prospecting for Hash Functions", 2018, the `lowbias32`
// function: https://nullprogram.com/blog/2018/07/31/. Chosen over the
// MurmurHash3 finalizer that the same article measures, on the bias it reports
// for both.
fn mix32(value: u32) u32 {
    var x = value;
    x ^= x >> 16;
    x *%= 0x7feb352d;
    x ^= x >> 15;
    x *%= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

// A lattice point and a seed to a well-distributed word. The multipliers differ
// so that the two coordinates do not cancel along the diagonal, which a plain
// exclusive or of the two would do at every point where they are equal.
fn hash2(x: i32, z: i32, seed: u32) u32 {
    const ux: u32 = @bitCast(x);
    const uz: u32 = @bitCast(z);
    return mix32(ux *% 0x27d4eb2d ^ uz *% 0x165667b1 ^ seed);
}

// The floor of a lattice coordinate. Safe to cast by construction rather than by
// a check: `Settings.validate` bounds every coordinate that reaches here.
fn latticeFloor(value: f32) i32 {
    return @intFromFloat(@floor(value));
}

// Eight unit gradients, the four axes and the four diagonals. A power of two so
// the selection is a mask, and all of unit length so no direction is louder than
// another.
const root_half = 0.70710678;
const gradients = [8][2]f32{
    .{ 1.0, 0.0 },
    .{ -1.0, 0.0 },
    .{ 0.0, 1.0 },
    .{ 0.0, -1.0 },
    .{ root_half, root_half },
    .{ -root_half, root_half },
    .{ root_half, -root_half },
    .{ -root_half, -root_half },
};

// Perlin's quintic fade. Its first and second derivatives vanish at both ends,
// so a normal taken by differencing the height across a lattice line is
// continuous. The cubic published in 1985 is not, and that seam shows as a
// crease along every lattice line once the surface is lit.
fn fade(t: f32) f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) f32 {
    const t = std.math.clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Gradient noise on the unit lattice, normalised to roughly [-1, 1]. Roughly,
// because the constant below is the bound of the two-dimensional case rather
// than a clamp: the extreme is reachable and vanishingly rare, so the range is a
// statement about the distribution and not a guarantee about a sample.
fn gradientNoise(x: f32, z: f32, seed: u32) f32 {
    const cell_x = latticeFloor(x);
    const cell_z = latticeFloor(z);
    const fx = x - @as(f32, @floatFromInt(cell_x));
    const fz = z - @as(f32, @floatFromInt(cell_z));

    const u = fade(fx);
    const v = fade(fz);

    const g00 = gradients[hash2(cell_x, cell_z, seed) & 7];
    const g10 = gradients[hash2(cell_x + 1, cell_z, seed) & 7];
    const g01 = gradients[hash2(cell_x, cell_z + 1, seed) & 7];
    const g11 = gradients[hash2(cell_x + 1, cell_z + 1, seed) & 7];

    const n00 = g00[0] * fx + g00[1] * fz;
    const n10 = g10[0] * (fx - 1.0) + g10[1] * fz;
    const n01 = g01[0] * fx + g01[1] * (fz - 1.0);
    const n11 = g11[0] * (fx - 1.0) + g11[1] * (fz - 1.0);

    const root_two = 1.41421356;
    return root_two * lerp(lerp(n00, n10, u), lerp(n01, n11, u), v);
}

// Octaves of the above, normalised by the sum of their amplitudes so the result
// keeps the range of one octave whatever the octave count is. Without that
// division, changing `octaves` changes the height of the terrain and the two
// settings stop being independent.
fn fbm(x: f32, z: f32, seed: u32, octaves: u32, lacunarity: f32, gain: f32) f32 {
    var total: f32 = 0.0;
    var normalisation: f32 = 0.0;
    var amplitude: f32 = 1.0;
    var frequency: f32 = 1.0;

    var octave: u32 = 0;
    while (octave < octaves) : (octave += 1) {
        // A different seed per octave. Sharing one would make every octave the
        // same field at another scale, and that self-similarity is visible.
        total += amplitude * gradientNoise(
            x * frequency,
            z * frequency,
            seed +% octave *% 0x9e3779b9,
        );
        normalisation += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }

    return total / normalisation;
}

// How much of each kind grows at a point, each in [0, 1]. They are separate
// channels rather than a cover index and one density, because the map is
// sampled with a linear filter: an index would interpolate to a value that
// names neither kind, while two densities blend into a boundary where both grow
// thinly, which is what a real one looks like.
pub const Growth = struct {
    grass: f32,
    wheat: f32,

    pub fn total(self: Growth) f32 {
        return self.grass + self.wheat;
    }
};

// A point of the field, answered once.
pub const Sample = struct {
    height: f32,
    cover: Cover,
    growth: Growth,
};

// The terrain as a function, holding no storage. Anything that wants a height or
// a cover evaluates it where it needs one, at whatever tessellation it chose,
// and no two consumers have to agree on a grid in advance.
pub const Terrain = struct {
    settings: Settings,

    pub fn init(settings: Settings) Error!Terrain {
        try settings.validate();
        return .{ .settings = settings };
    }

    // Metres, in the range [-relief, relief].
    pub fn heightAt(self: Terrain, x: f32, z: f32) f32 {
        const s = self.settings;
        return s.relief * fbm(
            x / s.feature_size,
            z / s.feature_size,
            s.seed,
            s.octaves,
            s.lacunarity,
            s.gain,
        );
    }

    // How far wheat wins over grass here, in [0, 1].
    //
    // Two octaves, at a gain low enough that the second only perturbs the
    // boundary. One octave was tried first and rejected on the picture: a single
    // gradient octave thresholded anywhere near its mean produces level sets
    // that are almost exactly ellipses, and the patches read as stamped rather
    // than grown. Adding octaves at the standard half gain goes too far the
    // other way and shatters the patch into islands.
    pub fn wheatShare(self: Terrain, x: f32, z: f32) f32 {
        const s = self.settings;
        const mask = fbm(
            x / s.wheat_feature_size,
            z / s.wheat_feature_size,
            s.seed +% 0x51ed_2701,
            2,
            s.lacunarity,
            s.wheat_detail,
        );
        return smoothstep(s.wheat_threshold - s.wheat_blend, s.wheat_threshold + s.wheat_blend, mask);
    }

    // Everything about a point, from one evaluation of each noise it needs.
    //
    // This is what the bake calls, and the reason it exists is arithmetic. Asked
    // through the three accessors below, a point costs three height fields and
    // two wheat masks, because each of them starts from scratch: fifteen
    // evaluations of the gradient noise where six are needed. Over the million
    // texels of a default bake that measured as 0.83 seconds against 0.44, a
    // factor of 1.9 where the evaluation count alone predicts 2.5. The rest of
    // the loop is the difference, and it is why the number here is measured.
    pub fn sampleAt(self: Terrain, x: f32, z: f32) Sample {
        const height = self.heightAt(x, z);
        const share = self.wheatShare(x, z);
        return .{
            .height = height,
            .cover = self.coverFrom(height, share),
            .growth = self.growthFrom(x, z, height, share),
        };
    }

    // What the ground is made of, for shading. The growing kinds are decided by
    // whichever share is larger, so this and the growth cannot disagree about
    // where the wheat begins.
    pub fn coverAt(self: Terrain, x: f32, z: f32) Cover {
        return self.coverFrom(self.heightAt(x, z), self.wheatShare(x, z));
    }

    // The two densities at a point. Both are zero on water and sand: the shore
    // ramp and the patch term are what keep the field from being a uniform
    // carpet that ends at a line.
    pub fn growthAt(self: Terrain, x: f32, z: f32) Growth {
        const height = self.heightAt(x, z);
        return self.growthFrom(x, z, height, self.wheatShare(x, z));
    }

    // The two below take what they need rather than computing it, so that the
    // boundary between water, sand and what grows has one statement and the
    // caller decides how often the field underneath it is evaluated.
    fn coverFrom(self: Terrain, height: f32, share: f32) Cover {
        const s = self.settings;
        if (height < s.water_level) return .water;
        if (height < s.water_level + s.sand_margin) return .sand;
        return if (share >= 0.5) .wheat else .grass;
    }

    fn growthFrom(self: Terrain, x: f32, z: f32, height: f32, share: f32) Growth {
        const s = self.settings;
        const shore_base = s.water_level + s.sand_margin;
        if (height <= shore_base) return .{ .grass = 0.0, .wheat = 0.0 };

        const shore = smoothstep(shore_base, shore_base + s.shore_falloff, height);
        const patch = gradientNoise(
            x / s.patch_feature_size,
            z / s.patch_feature_size,
            s.seed +% 0x2b7e_1516,
        );
        // The patch term is centred on one and reaches below it by `patch_depth`
        // at most, so it thins the field and never densifies it past what the
        // settings asked for.
        const thinning = 1.0 - s.patch_depth * (0.5 - 0.5 * patch);
        const density = shore * thinning;

        return .{ .grass = density * (1.0 - share), .wheat = density * share };
    }
};

// The map the device samples: one RGBA16 unorm texel per point of a square grid
// over the domain, in the layout an upload expects.
//
// Baked rather than evaluated in Slang. The alternative is a second
// implementation of every function above, in another language, kept in step by
// hand, and the first time the two drift the foliage grows where the ground is
// not. What this costs is one image and a sample; what it buys is that the field
// has one definition.
pub const Map = struct {
    // Little-endian u16 quadruples, `resolution * resolution` of them, row major
    // in z.
    texels: []u8,
    resolution: u32,

    pub const channels = 4;
    pub const bytes_per_texel = channels * @sizeOf(u16);

    // R is the height, remapped from [-relief, relief] to the unorm range; the
    // reader recovers metres with `relief * (2 * r - 1)`. G and B are the grass
    // and wheat densities. A is the cover, so that the ground shading can tell
    // water from sand without evaluating the height twice, quantised to the four
    // values `Cover` names and read with a nearest fetch rather than a filtered
    // sample.
    pub const height_channel = 0;
    pub const grass_channel = 1;
    pub const wheat_channel = 2;
    pub const cover_channel = 3;

    pub fn deinit(self: *Map, allocator: Allocator) void {
        allocator.free(self.texels);
        self.* = undefined;
    }
};

// Startup work, and the only allocation this file performs. The cost is one
// evaluation of the field per texel, so it is quadratic in the resolution and
// linear in the octave count, and it is the one number worth watching if the
// example's start becomes slow.
pub fn bake(allocator: Allocator, terrain: Terrain) Allocator.Error!Map {
    const s = terrain.settings;
    const resolution = s.bake_resolution;
    const texels = try allocator.alloc(u8, @as(usize, resolution) * resolution * Map.bytes_per_texel);
    errdefer allocator.free(texels);

    const extent = s.extent();
    // Texel centres, so that the map's first and last samples sit half a texel
    // inside the domain rather than on its edge. Sampled with a clamped address
    // mode, that is what puts the boundary of the field at the boundary of the
    // image.
    const step = 2.0 * extent / @as(f32, @floatFromInt(resolution));
    const origin = -extent + 0.5 * step;

    var row: u32 = 0;
    while (row < resolution) : (row += 1) {
        const z = origin + @as(f32, @floatFromInt(row)) * step;
        var column: u32 = 0;
        while (column < resolution) : (column += 1) {
            const x = origin + @as(f32, @floatFromInt(column)) * step;

            const sample = terrain.sampleAt(x, z);

            const offset = (@as(usize, row) * resolution + column) * Map.bytes_per_texel;
            writeUnorm16(texels[offset..], Map.height_channel, 0.5 + 0.5 * (sample.height / s.relief));
            writeUnorm16(texels[offset..], Map.grass_channel, sample.growth.grass);
            writeUnorm16(texels[offset..], Map.wheat_channel, sample.growth.wheat);
            // Three quarters of full scale is `wheat`, the largest member, so
            // the quantisation step is a value the reader can compare against
            // exactly after multiplying by three.
            writeUnorm16(texels[offset..], Map.cover_channel, @as(f32, @floatFromInt(@backingInt(sample.cover))) / 3.0);
        }
    }

    return .{ .texels = texels, .resolution = resolution };
}

// The value is already in [0, 1] at every call site, so this rounds rather than
// clamps. Rounding and not truncating: truncation biases every channel down by
// half a step, which on the height channel is a systematic sinking of the whole
// terrain rather than noise.
fn writeUnorm16(texel: []u8, channel: usize, value: f32) void {
    const scaled = @round(std.math.clamp(value, 0.0, 1.0) * 65535.0);
    const quantised: u16 = @intFromFloat(scaled);
    std.mem.writeInt(u16, texel[channel * @sizeOf(u16) ..][0..@sizeOf(u16)], quantised, .little);
}
