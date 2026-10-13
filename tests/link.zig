const std = @import("std");
const lenore = @import("lenore");

const testing = std.testing;

test "bytes cross in the order sent, wake the reader, and stop at the limit" {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    var pipe: lenore.Pipe = .{ .limit = 6 };
    defer pipe.deinit(testing.allocator);
    var event: std.Io.Event = .unset;
    pipe.wake = &event;

    try testing.expect(try pipe.send(io, testing.allocator, "abc"));
    try testing.expect(event.isSet());
    try testing.expect(try pipe.send(io, testing.allocator, "de"));
    try testing.expect(!try pipe.send(io, testing.allocator, "fg"));

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(testing.allocator);
    pipe.take(io, &out);
    try testing.expectEqualStrings("abcde", out.items);
    out.clearRetainingCapacity();
    pipe.take(io, &out);
    try testing.expectEqual(@as(usize, 0), out.items.len);
    try testing.expect(try pipe.send(io, testing.allocator, "fg"));
}

test "an empty send is a sent message and wakes nobody" {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    var pipe: lenore.Pipe = .{ .limit = 0 };
    defer pipe.deinit(testing.allocator);
    var event: std.Io.Event = .unset;
    pipe.wake = &event;

    try testing.expect(try pipe.send(io, testing.allocator, ""));
    try testing.expect(!event.isSet());
}
