const std = @import("std");
const lenore = @import("lenore");

const testing = std.testing;

test "a console without a font never opens" {
    var console: lenore.Console = .{ .font = null };
    console.show("/");
    try testing.expect(!console.open);
}

test "a console opens with its prefix typed and the caret after it" {
    var console: lenore.Console = .{ .font = 0 };
    console.show("/tp");
    try testing.expect(console.open);
    try testing.expect(console.focus_pending);
    try testing.expectEqualStrings("/tp", console.line());
    try testing.expectEqual(@as(usize, 3), console.state.caret);

    console.hide();
    try testing.expect(!console.open);
    try testing.expect(!console.focus_pending);
}

test "a prefix and a reply longer than their buffers are cut, not refused" {
    var console: lenore.Console = .{ .font = 0 };
    const long: [400]u8 = @splat('a');
    console.show(&long);
    try testing.expectEqual(console.buffer.len, console.line().len);

    console.reply(true, "{s}", .{&long});
    try testing.expectEqual(console.reply_buffer.len, console.reply_len);
    try testing.expect(console.reply_failed);
}

test "overlay lines are cut at their capacity and dropped past the last" {
    var lines: lenore.DebugLines = .{};
    const long: [200]u8 = @splat('b');
    lines.add("{s}", .{&long});
    try testing.expectEqual(@as(usize, 120), lines.line(0).len);
    for (1..20) |index| lines.add("line {d}", .{index});
    try testing.expectEqual(@as(usize, 16), lines.count);
    try testing.expectEqualStrings("line 15", lines.line(15));
}

test "an overlay without a font cannot be toggled open" {
    var overlay: lenore.DebugOverlay = .{ .font = null };
    overlay.toggle();
    try testing.expect(!overlay.open);
}
