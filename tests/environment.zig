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
