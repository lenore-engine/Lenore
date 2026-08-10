const std = @import("std");
const gpu = @import("lenore-gpu");
const ktx = @import("lenore-ktx");

const testing = std.testing;

// The writer's real contract is the reader that has to accept what it produces,
// and the two live in different modules. This is the only place both are
// present, which is why the check is here and not in either of them.
//
// A writer tested only against its own idea of the format is a writer that
// agrees with itself. Every field this exercises was written from the
// specification and read back by code that was written from the specification
// separately, so a misreading has to be made twice to pass.

fn image(allocator: std.mem.Allocator, width: u32, height: u32) ![]u8 {
    const pixels = try allocator.alloc(u8, @as(usize, width) * height * 4);
    for (0..height) |y| {
        for (0..width) |x| {
            const at = (y * width + x) * 4;
            pixels[at + 0] = @intCast((x * 255) / @max(width - 1, 1));
            pixels[at + 1] = @intCast((y * 255) / @max(height - 1, 1));
            pixels[at + 2] = if ((x / 2 + y / 2) % 2 == 0) 240 else 15;
            pixels[at + 3] = 255;
        }
    }
    return pixels;
}

fn convert(
    allocator: std.mem.Allocator,
    width: u32,
    height: u32,
    semantic: ktx.Semantic,
) ![]u8 {
    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const pixels = try image(allocator, width, height);
    defer allocator.free(pixels);
    return ktx.convert(allocator, threaded.io(), pixels, width, height, .{ .semantic = semantic });
}

test "a converted colour texture parses as the format it claims" {
    const allocator = testing.allocator;
    const file = try convert(allocator, 32, 16, .colour);
    defer allocator.free(file);

    try testing.expect(gpu.isKtx2(file));
    const parsed = try gpu.parseKtx2(file);

    try testing.expectEqual(gpu.Ktx2Kind.texture_2d, parsed.kind);
    try testing.expectEqual(@as(u32, 1), parsed.face_count);
    try testing.expectEqual(@as(u32, 32), parsed.width);
    try testing.expectEqual(@as(u32, 16), parsed.height);
    // The reader maps the header's format code back to a Vulkan format and
    // validates the data format descriptor against it, so colour arriving as
    // sRGB is the sampler being able to undo the transfer that was applied.
    //
    // Compared as the code rather than as an enumerant: the umbrella does not
    // name Vulkan, and the code is exactly what the writer wrote.
    try testing.expectEqual(
        ktx.Semantic.colour.vkFormat(),
        @as(u32, @intCast(@intFromEnum(parsed.format))),
    );
    try testing.expectEqual(@as(u32, 4), parsed.block_height);
}

test "a converted data texture parses as linear" {
    const allocator = testing.allocator;
    const file = try convert(allocator, 16, 16, .data);
    defer allocator.free(file);

    const parsed = try gpu.parseKtx2(file);
    try testing.expectEqual(
        ktx.Semantic.data.vkFormat(),
        @as(u32, @intCast(@intFromEnum(parsed.format))),
    );
}

test "the reader accepts the whole chain and agrees on every level" {
    const allocator = testing.allocator;
    const file = try convert(allocator, 32, 8, .data);
    defer allocator.free(file);

    const parsed = try gpu.parseKtx2(file);
    // The reader refuses a partial chain for a 2D texture, so reaching here at
    // all is the writer producing a complete one.
    try testing.expectEqual(ktx.levelCount(32, 8), parsed.level_count);

    const levels = parsed.levels();
    var previous_offset: u64 = std.math.maxInt(u64);
    for (levels, 0..) |level, index| {
        const extent = ktx.levelExtent(32, 8, index);
        try testing.expectEqual(extent[0], level.width);
        try testing.expectEqual(extent[1], level.height);
        // The size the reader computes from the extent and the block geometry,
        // against the size the writer computed from the same extent through a
        // different expression.
        try testing.expectEqual(ktx.encodedSize(extent[0], extent[1]), level.byte_length);
        try testing.expectEqual(level.byte_length, level.face_byte_length);
        try testing.expectEqual(@as(u64, 0), level.byte_offset % parsed.level_alignment);
        // Ascending mip index from the reader, descending offsets in the file:
        // the two orders are opposite and the reader undoes it.
        try testing.expect(level.byte_offset < previous_offset);
        previous_offset = level.byte_offset;
    }
}

// The awkward extents. A power of two agrees with almost any arithmetic; these
// are where a chain length taken from one axis, a level that rounds the wrong
// way, or a block count that forgets to round up all show.
test "extents that are not powers of two survive the round trip" {
    const allocator = testing.allocator;
    const cases = [_][2]u32{
        .{ 1, 1 },
        .{ 3, 3 },
        .{ 5, 1 },
        .{ 7, 13 },
        .{ 64, 1 },
        .{ 17, 40 },
    };

    for (cases) |case| {
        const file = try convert(allocator, case[0], case[1], .colour);
        defer allocator.free(file);

        const parsed = try gpu.parseKtx2(file);
        try testing.expectEqual(case[0], parsed.width);
        try testing.expectEqual(case[1], parsed.height);
        try testing.expectEqual(ktx.levelCount(case[0], case[1]), parsed.level_count);

        const last = parsed.levels()[parsed.level_count - 1];
        try testing.expectEqual(@as(u32, 1), last.width);
        try testing.expectEqual(@as(u32, 1), last.height);
        // One block, whatever the extent above it was.
        try testing.expectEqual(@as(u64, 16), last.byte_length);
    }
}
