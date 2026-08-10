const std = @import("std");
const gpu = @import("lenore-gpu");

const images = @import("images.zig");

const Allocator = std.mem.Allocator;
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

    const lambertian = try acquireCube(
        textures,
        staging,
        context,
        pool,
        environment_keys[0],
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
