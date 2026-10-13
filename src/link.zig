// A connection between a host and one client inside one process: a pipe of
// bytes each way. Neither side reads anything of the other but these bytes, so
// what crosses is a protocol the application defines, and a host that later
// runs in another process keeps the same messages.
//
// A pipe is written by one thread and taken by another, under a mutex of its
// own. A send wakes the reader through the event it named. Taking swaps the
// pending bytes for the reader's emptied list, so bytes are copied once, and
// once both lists have grown to what a pass carries neither side allocates.
// Both sides allocate with the same allocator, since the lists change hands.

const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Pipe = struct {
    mutex: std.Io.Mutex = .init,
    pending: std.ArrayList(u8) = .empty,
    // The reader's, set after every send. Named before either side runs.
    wake: ?*std.Io.Event = null,
    // Bytes waiting past which `send` refuses more, so a reader that stopped
    // taking does not let the writer grow without end.
    limit: usize = std.math.maxInt(usize),

    pub fn deinit(self: *Pipe, allocator: Allocator) void {
        self.pending.deinit(allocator);
        self.* = undefined;
    }

    // Whole messages only, so the reader never sees half of one. False when
    // they would take the pipe past its limit, and nothing is sent.
    pub fn send(self: *Pipe, io: std.Io, allocator: Allocator, bytes: []const u8) Allocator.Error!bool {
        if (bytes.len == 0) return true;
        self.mutex.lockUncancelable(io);
        const room = self.pending.items.len + bytes.len <= self.limit;
        if (room) self.pending.appendSlice(allocator, bytes) catch |err| {
            self.mutex.unlock(io);
            return err;
        };
        const wake = self.wake;
        self.mutex.unlock(io);
        if (room) if (wake) |event| event.set(io);
        return room;
    }

    // Everything sent since the last take, into `out`, which must be empty.
    pub fn take(self: *Pipe, io: std.Io, out: *std.ArrayList(u8)) void {
        std.debug.assert(out.items.len == 0);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.mem.swap(std.ArrayList(u8), &self.pending, out);
    }
};

// The two directions. Each pipe's limit is the application's, set from what
// its own messages are: a direction whose reader takes every pass needs one
// only against a reader that stopped.
pub const Link = struct {
    to_host: Pipe = .{},
    to_client: Pipe = .{},

    pub fn deinit(self: *Link, allocator: Allocator) void {
        self.to_host.deinit(allocator);
        self.to_client.deinit(allocator);
        self.* = undefined;
    }
};
