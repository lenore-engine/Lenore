const std = @import("std");
const gpu = @import("lenore-gpu");

const images = @import("images.zig");

const Allocator = std.mem.Allocator;
const log = std.log.scoped(.environment);
const DecodedImage = images.DecodedImage;
const SourcePixel = images.SourcePixel;

// Reading a prefiltered environment: the two cubemaps the surfaces reflect and
// the table that carries the split-sum approximation's scale and bias.
//
// The layout is the one the Khronos tool produces, which is what the sample
// environments ship as. Nothing here bakes: an environment is an input, and a
// runtime bake would be a different subsystem with a different cost.

pub const EnvironmentError = error{
    NonContiguousDecodedImage,
};

// Half a gigabyte is past anything the Khronos set publishes and short of a
// length that would be a denial of service by itself.
pub const max_environment_bytes: usize = 512 << 20;

// The three files, and the keys the cache holds them under so the references
// taken here can be given back.
pub const environment_keys = [3][]const u8{
    "environment:lambertian",
    "environment:ggx",
    "environment:lut",
};

// The range each channel of the lookup table covers.
//
// It has to carry a scale in red and a bias in green. The set Khronos publishes
// beside the environments also contains a single-channel table whose data sits
// in blue, and reading that one returns (0, 0) everywhere: the specular
// reflection vanishes and, where the base colour is white, the energy term
// divides zero by zero. That is a black model with clean validation, which is
// why the channels are measured rather than the file name trusted.
pub const LutChannels = struct {
    // Low and high of the red channel, which is the scale.
    scale: [2]u8 = .{ 255, 0 },
    // Low and high of the green channel, which is the bias.
    bias: [2]u8 = .{ 255, 0 },

    // Whether both channels carry a gradient rather than a constant. A table
    // that varies in neither is the wrong file.
    pub fn varies(self: LutChannels) bool {
        return self.scale[1] > self.scale[0] and self.bias[1] > self.bias[0];
    }
};

// Measured over the decoded table rather than asserted, and separate from the
// read so that what makes a table wrong can be stated without a device.
pub fn lutChannels(texels: []const SourcePixel) LutChannels {
    var found: LutChannels = .{};
    for (texels) |texel| {
        found.scale[0] = @min(found.scale[0], texel.r);
        found.scale[1] = @max(found.scale[1], texel.r);
        found.bias[0] = @min(found.bias[0], texel.g);
        found.bias[1] = @max(found.bias[1], texel.g);
    }
    return found;
}

// What one cubemap cost the staging pool. The number that says whether the
// ceiling is set too low: a stall is a full wait on the graphics queue.
pub const CubeLoad = struct {
    key: []const u8,
    blocks: u32,
    stalls: u32,
};

pub const Loaded = struct {
    environment: gpu.Environment,
    keys: [3][]const u8,
    lut: LutChannels,
    cubes: [2]CubeLoad,

    // Gives back the three references this load took. The images outlive them
    // only as long as something holds one.
    pub fn release(self: *const Loaded, textures: *gpu.TextureCache) void {
        for (self.keys) |key| textures.release(key);
    }
};

// Reads a prefiltered environment and hands back what the scene set is written
// from.
//
// Each map goes through its own transfer, so the staging pool is reclaimed
// between them and one file's worth is all that is ever in flight. It is the
// pool that makes that true rather than the sizing: a single GGX level is fifty
// megabytes, more than the whole ceiling, and it travels a block at a time.
pub fn load(
    allocator: Allocator,
    io: std.Io,
    context: *const gpu.Context,
    staging: *gpu.StagingPool,
    textures: *gpu.TextureCache,
    pool: *const gpu.OneShotPool,
    directory: []const u8,
) !Loaded {
    var root = try std.Io.Dir.cwd().openDir(io, directory, .{});
    defer root.close(io);

    const lambertian_bytes = try root.readFileAlloc(
        io,
        "lambertian/diffuse.ktx2",
        allocator,
        .limited(max_environment_bytes),
    );
    defer allocator.free(lambertian_bytes);
    const ggx_bytes = try root.readFileAlloc(
        io,
        "ggx/specular.ktx2",
        allocator,
        .limited(max_environment_bytes),
    );
    defer allocator.free(ggx_bytes);
    const lut_bytes = try root.readFileAlloc(
        io,
        "lut_ggx.png",
        allocator,
        .limited(max_environment_bytes),
    );
    defer allocator.free(lut_bytes);

    // Reduced before it is uploaded rather than after, because what costs is
    // sampling it: the map is read along each fragment's own surface normal, and
    // that direction is not coherent between neighbouring pixels, so a map too
    // large for cache is paid for per fragment rather than once.
    //
    // A map already at or below the target, or one whose extent the target does
    // not divide, is uploaded as it stands. Refusing an environment over a
    // resampling that is an optimisation would be the wrong trade.
    const lambertian = try acquireLambertian(
        allocator,
        textures,
        staging,
        context,
        pool,
        lambertian_bytes,
    );
    const ggx = try acquireCube(
        textures,
        staging,
        context,
        pool,
        environment_keys[1],
        ggx_bytes,
    );

    // The lookup table is data and not colour: it tabulates a scale and a bias,
    // and reading it through an sRGB transfer function returns wrong numbers
    // that still look like a plausible gradient.
    var decoded = try DecodedImage.loadFromBytes(allocator, lut_bytes);
    defer decoded.deinit(allocator);
    if (decoded.stride != decoded.cols) return error.NonContiguousDecodedImage;

    var lut_setup: gpu.Transfer = try .begin(context, pool.handle, staging);
    const lut = try textures.acquireRgba8(
        environment_keys[2],
        .{
            .width = decoded.cols,
            .height = decoded.rows,
            .bytes = std.mem.sliceAsBytes(decoded.data),
        },
        .r8g8b8a8_unorm,
        gpu.environmentSampler,
        &lut_setup,
    );
    // The reference taken above must outlive this submission: releasing it
    // sooner destroys an image a recorded copy still names.
    try lut_setup.finish();
    lut_setup.deinit();

    return .{
        .environment = .{ .lambertian = lambertian.bound, .ggx = ggx.bound, .lut = lut },
        .keys = environment_keys,
        .lut = lutChannels(decoded.data),
        .cubes = .{ lambertian.load, ggx.load },
    };
}

// The irradiance cube, reduced where the file allows it.
//
// Falling back to the file's own bytes keeps one upload path for both cases:
// what changes is which bytes are handed over, not whether the environment
// loads.
fn acquireLambertian(
    allocator: Allocator,
    textures: *gpu.TextureCache,
    staging: *gpu.StagingPool,
    context: *const gpu.Context,
    pool: *const gpu.OneShotPool,
    bytes: []const u8,
) !AcquiredCube {
    const reduced = reducedLambertian(allocator, bytes) catch |err| {
        log.warn("irradiance map kept at its own size: {t}", .{err});
        return acquireCube(textures, staging, context, pool, environment_keys[0], bytes);
    } orelse
        return acquireCube(textures, staging, context, pool, environment_keys[0], bytes);
    defer allocator.free(reduced);

    var setup: gpu.Transfer = try .begin(context, pool.handle, staging);
    const bound = try textures.acquireCube(
        environment_keys[0],
        reduced,
        lambertian_extent,
        cube_texel_bytes,
        lambertian_format,
        gpu.environmentSampler,
        &setup,
    );
    // The reference taken above must outlive this submission: releasing it
    // sooner destroys an image a recorded copy still names.
    try setup.finish();
    const measured: CubeLoad = .{
        .key = environment_keys[0],
        .blocks = staging.blockCount(),
        .stalls = setup.flushes,
    };
    setup.deinit();
    return .{ .bound = bound, .load = measured };
}

// The file's base level averaged down, or null where this file is not one the
// reduction applies to: already small enough, more than one level, or an extent
// the target does not divide.
fn reducedLambertian(allocator: Allocator, bytes: []const u8) !?[]u8 {
    if (!gpu.isKtx2(bytes)) return null;
    const file = try gpu.parseKtx2(bytes);
    if (file.format != lambertian_format) return null;
    if (file.kind != .cube) return null;
    // One level is what Khronos ships. A chain would have to be reduced level by
    // level and the coarse ones are already small, so it is left alone.
    if (file.level_count != 1) return null;
    if (file.width != file.height) return null;
    if (file.width <= lambertian_extent) return null;
    if (file.width % lambertian_extent != 0) return null;

    const level = file.levels()[0];
    const payload = bytes[@intCast(level.byte_offset)..][0..@intCast(level.byte_length)];
    return try reduceCube(allocator, payload, file.width, lambertian_extent);
}

const lambertian_format: gpu.vk.Format = .r16g16b16a16_sfloat;

const AcquiredCube = struct {
    bound: gpu.BoundTexture,
    load: CubeLoad,
};

fn acquireCube(
    textures: *gpu.TextureCache,
    staging: *gpu.StagingPool,
    context: *const gpu.Context,
    pool: *const gpu.OneShotPool,
    key: []const u8,
    bytes: []const u8,
) !AcquiredCube {
    var setup: gpu.Transfer = try .begin(context, pool.handle, staging);
    const bound = try textures.acquireKtx2(
        key,
        bytes,
        .r16g16b16a16_sfloat,
        .cube,
        gpu.environmentSampler,
        &setup,
    );
    // The reference taken above must outlive this submission: releasing it
    // sooner destroys an image a recorded copy still names.
    try setup.finish();
    const measured: CubeLoad = .{
        .key = key,
        .blocks = staging.blockCount(),
        .stalls = setup.flushes,
    };
    setup.deinit();
    return .{ .bound = bound, .load = measured };
}

// The irradiance map is the cosine convolution of the environment, so it holds
// no detail above about two spherical-harmonic bands. Khronos ships it at 1024
// square with one level, which is 48 MB of a signal that 32 square carries, and
// the cost is not the memory: every shaded fragment samples it along its own
// surface normal, and neighbouring pixels do not agree about that direction, so
// a map too large for cache is paid for per fragment.
//
// Measured on this project rather than assumed. See `Progression.md` under
// 2026-08-10 for the numbers and the configuration they were taken at.
pub const lambertian_extent: u32 = 32;

// Four halves per texel, which is what `r16g16b16a16_sfloat` is.
const cube_channels = 4;
const cube_texel_bytes = cube_channels * @sizeOf(f16);

pub const ReduceError = error{
    // The source is not a whole multiple of the target, so no block of source
    // texels maps onto one target texel.
    ExtentNotDivisible,
    // The payload does not hold six square faces of the declared extent.
    FaceBytesMismatch,
} || Allocator.Error;

// Averages every `factor` by `factor` block of each face into one texel, where
// `factor` is the ratio of the two extents.
//
// A plain mean over the block, in single precision, with no weighting by the
// solid angle a texel subtends. That weighting varies across a cube face and an
// exact resampling would carry it; for a signal this smooth the difference did
// not survive being looked at, and the same arithmetic produced the map the
// comparison was judged against.
//
// The caller owns the returned faces. They are six squares of `target_extent`,
// contiguous and in the source's order, which is the layout an upload of a cube
// level expects.
pub fn reduceCube(
    allocator: Allocator,
    faces: []const u8,
    source_extent: u32,
    target_extent: u32,
) ReduceError![]u8 {
    if (target_extent == 0 or source_extent % target_extent != 0)
        return error.ExtentNotDivisible;
    const source_face_texels = @as(usize, source_extent) * source_extent;
    if (faces.len != gpu.ktx2CubeFaces * source_face_texels * cube_texel_bytes)
        return error.FaceBytesMismatch;

    const factor = source_extent / target_extent;
    const target_face_texels = @as(usize, target_extent) * target_extent;
    const result = try allocator.alloc(u8, gpu.ktx2CubeFaces * target_face_texels * cube_texel_bytes);
    errdefer allocator.free(result);

    const divisor: f32 = @floatFromInt(factor * factor);
    for (0..gpu.ktx2CubeFaces) |face| {
        const source_face = faces[face * source_face_texels * cube_texel_bytes ..];
        const target_face = result[face * target_face_texels * cube_texel_bytes ..];
        for (0..target_extent) |row| {
            for (0..target_extent) |column| {
                var sums: [cube_channels]f32 = @splat(0);
                for (0..factor) |block_row| {
                    const source_row = row * factor + block_row;
                    for (0..factor) |block_column| {
                        const source_column = column * factor + block_column;
                        const at = (source_row * source_extent + source_column) * cube_texel_bytes;
                        for (&sums, 0..) |*sum, channel| {
                            // Read as bytes rather than as a typed slice: the
                            // payload is a run inside a file and nothing
                            // promises it starts on a two-byte boundary.
                            const bits = std.mem.readInt(
                                u16,
                                source_face[at + channel * @sizeOf(f16) ..][0..2],
                                .little,
                            );
                            sum.* += @floatCast(@as(f16, @bitCast(bits)));
                        }
                    }
                }
                const at = (row * target_extent + column) * cube_texel_bytes;
                for (sums, 0..) |sum, channel| {
                    const mean: f16 = @floatCast(sum / divisor);
                    std.mem.writeInt(
                        u16,
                        target_face[at + channel * @sizeOf(f16) ..][0..2],
                        @bitCast(mean),
                        .little,
                    );
                }
            }
        }
    }
    return result;
}
