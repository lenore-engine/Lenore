const std = @import("std");
const lenore = @import("lenore");
const zignal = @import("zignal");

const testing = std.testing;

const Texel = zignal.Rgba(u8);

fn texel(r: u8, g: u8, b: u8) Texel {
    return .{ .r = r, .g = g, .b = b, .a = 255 };
}

test "a table that ramps in both channels is the one to sample" {
    // Distinct values in every channel, and a blue channel that ramps the other
    // way. A scan reading the wrong channel lands on a different range rather
    // than on the same one by luck.
    const table = [_]Texel{
        texel(10, 40, 200),
        texel(60, 90, 150),
        texel(30, 70, 180),
    };

    const found = lenore.lutChannels(&table);
    try testing.expectEqual([2]u8{ 10, 60 }, found.scale);
    try testing.expectEqual([2]u8{ 40, 90 }, found.bias);
    try testing.expect(found.varies());
}

test "the single-channel table Khronos also ships is rejected" {
    // The one whose data sits in blue. Reading it gives a scale and a bias of
    // zero everywhere, the specular term vanishes, and a white base colour
    // divides zero by zero. It decodes cleanly and renders a black model, which
    // is why the channels are measured rather than the file name trusted.
    const table = [_]Texel{
        texel(0, 0, 12),
        texel(0, 0, 200),
        texel(0, 0, 96),
    };

    const found = lenore.lutChannels(&table);
    try testing.expectEqual(false, found.varies());
}

test "a table that varies in only one of the two channels is rejected" {
    const scale_only = [_]Texel{ texel(10, 50, 0), texel(90, 50, 0) };
    try testing.expectEqual(false, lenore.lutChannels(&scale_only).varies());

    const bias_only = [_]Texel{ texel(50, 10, 0), texel(50, 90, 0) };
    try testing.expectEqual(false, lenore.lutChannels(&bias_only).varies());
}

test "a table of one texel cannot vary" {
    // A range needs two values. One texel gives a low equal to its high, which
    // is a constant table however plausible its colour.
    const single = [_]Texel{texel(128, 200, 64)};
    const found = lenore.lutChannels(&single);
    try testing.expectEqual([2]u8{ 128, 128 }, found.scale);
    try testing.expectEqual(false, found.varies());
}

test "an empty table reports the range it cannot have" {
    // The scan starts from an inverted range, so nothing to measure leaves it
    // inverted rather than reading as a full one.
    const found = lenore.lutChannels(&.{});
    try testing.expectEqual([2]u8{ 255, 0 }, found.scale);
    try testing.expectEqual([2]u8{ 255, 0 }, found.bias);
    try testing.expectEqual(false, found.varies());
}

test "the environment names three files and holds each under its own key" {
    // Three distinct keys. Two that collided would make the cache hand the same
    // image back for two of the maps, which draws a plausible picture.
    try testing.expectEqual(@as(usize, 3), lenore.environment_keys.len);
    for (lenore.environment_keys, 0..) |key, index| {
        for (lenore.environment_keys[index + 1 ..]) |other| {
            try testing.expect(!std.mem.eql(u8, key, other));
        }
    }
}

// Six faces of `extent` square, every texel the same colour, in the layout a
// cube level is stored in.
fn constantCube(allocator: std.mem.Allocator, extent: u32, colour: [4]f16) ![]u8 {
    const texels = @as(usize, extent) * extent * 6;
    const bytes = try allocator.alloc(u8, texels * 8);
    var at: usize = 0;
    while (at < bytes.len) : (at += 8) {
        for (colour, 0..) |channel, index|
            std.mem.writeInt(u16, bytes[at + index * 2 ..][0..2], @bitCast(channel), .little);
    }
    return bytes;
}

fn texelAt(bytes: []const u8, index: usize, channel: usize) f16 {
    return @bitCast(std.mem.readInt(u16, bytes[index * 8 + channel * 2 ..][0..2], .little));
}

test "a constant cube reduces to the same constant" {
    // The mean of equal values is that value, so anything the filter does to the
    // indexing shows up as a texel that is not the colour it started as.
    const allocator = std.testing.allocator;
    const source = try constantCube(allocator, 8, .{ 0.25, 0.5, 1.0, 2.0 });
    defer allocator.free(source);

    const reduced = try lenore.reduceCube(allocator, source, 8, 2);
    defer allocator.free(reduced);

    try std.testing.expectEqual(@as(usize, 6 * 2 * 2 * 8), reduced.len);
    for (0..6 * 2 * 2) |index| {
        try std.testing.expectEqual(@as(f16, 0.25), texelAt(reduced, index, 0));
        try std.testing.expectEqual(@as(f16, 0.5), texelAt(reduced, index, 1));
        try std.testing.expectEqual(@as(f16, 1.0), texelAt(reduced, index, 2));
        try std.testing.expectEqual(@as(f16, 2.0), texelAt(reduced, index, 3));
    }
}

test "a block averages into one texel" {
    // Two by two down to one, with values chosen so the mean is exact in f16 and
    // the test does not rest on a tolerance. Only the first face is filled with
    // anything interesting, which also pins that faces are not read across.
    const allocator = std.testing.allocator;
    const source = try constantCube(allocator, 2, .{ 0, 0, 0, 0 });
    defer allocator.free(source);
    const values = [_]f16{ 1, 2, 3, 4 };
    for (values, 0..) |value, index|
        std.mem.writeInt(u16, source[index * 8 ..][0..2], @bitCast(value), .little);

    const reduced = try lenore.reduceCube(allocator, source, 2, 1);
    defer allocator.free(reduced);

    try std.testing.expectEqual(@as(usize, 6 * 8), reduced.len);
    try std.testing.expectEqual(@as(f16, 2.5), texelAt(reduced, 0, 0));
    // The faces after it were zero and stay zero.
    for (1..6) |face| try std.testing.expectEqual(@as(f16, 0), texelAt(reduced, face, 0));
}

test "rows and columns are not transposed" {
    // A face whose two halves differ by row. Transposing the block walk would
    // still average four values and would still be a plausible picture.
    const allocator = std.testing.allocator;
    const source = try constantCube(allocator, 4, .{ 0, 0, 0, 0 });
    defer allocator.free(source);
    for (0..4) |row| {
        for (0..4) |column| {
            const value: f16 = if (row < 2) 1.0 else 5.0;
            const at = (row * 4 + column) * 8;
            std.mem.writeInt(u16, source[at..][0..2], @bitCast(value), .little);
        }
    }

    const reduced = try lenore.reduceCube(allocator, source, 4, 2);
    defer allocator.free(reduced);

    // The top row of the result comes from the top half and the bottom from the
    // bottom, so they differ; a transpose would make all four equal to three.
    try std.testing.expectEqual(@as(f16, 1.0), texelAt(reduced, 0, 0));
    try std.testing.expectEqual(@as(f16, 1.0), texelAt(reduced, 1, 0));
    try std.testing.expectEqual(@as(f16, 5.0), texelAt(reduced, 2, 0));
    try std.testing.expectEqual(@as(f16, 5.0), texelAt(reduced, 3, 0));
}

test "an extent that does not divide is refused" {
    const allocator = std.testing.allocator;
    const source = try constantCube(allocator, 6, .{ 1, 1, 1, 1 });
    defer allocator.free(source);

    try std.testing.expectError(
        error.ExtentNotDivisible,
        lenore.reduceCube(allocator, source, 6, 4),
    );
    try std.testing.expectError(
        error.ExtentNotDivisible,
        lenore.reduceCube(allocator, source, 6, 0),
    );
}

test "a payload that is not six square faces is refused" {
    // Caught before anything is allocated or read, because the length is what
    // every index below is derived from.
    const allocator = std.testing.allocator;
    const source = try constantCube(allocator, 4, .{ 1, 1, 1, 1 });
    defer allocator.free(source);

    try std.testing.expectError(
        error.FaceBytesMismatch,
        lenore.reduceCube(allocator, source[0 .. source.len - 8], 4, 2),
    );
    try std.testing.expectError(
        error.FaceBytesMismatch,
        lenore.reduceCube(allocator, source, 8, 2),
    );
}
