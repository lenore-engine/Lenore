const std = @import("std");
const lenore = @import("lenore");
const platform = @import("lenore-platform");

const testing = std.testing;

// Drives the clock by hand. Same shape as lenore-platform's own clock suite:
// std.Io.failing with the two entry points the clock uses replaced.
const FakeIo = struct {
    now_ns: i96,
    vtable: std.Io.VTable,

    fn init(now_ns: i96) FakeIo {
        return .{ .now_ns = now_ns, .vtable = std.Io.failing.vtable.* };
    }

    fn io(self: *FakeIo) std.Io {
        self.vtable.now = now;
        return .{ .userdata = self, .vtable = &self.vtable };
    }

    fn now(userdata: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp {
        const self: *FakeIo = @ptrCast(@alignCast(userdata.?));
        return .{ .nanoseconds = self.now_ns };
    }

    fn advance(self: *FakeIo, nanoseconds: i96) void {
        self.now_ns += nanoseconds;
    }
};

const ms = std.time.ns_per_ms;

test "a frame's delta is the interval, and the index counts from zero" {
    var fake: FakeIo = .init(5_000);
    var frame_clock: lenore.FrameClock = .start(.init(fake.io()));

    // Three different intervals. Equal ones would let an accumulator that
    // reused the previous frame's value pass.
    const intervals = [_]u64{ 16 * ms, 33 * ms, 8 * ms };
    var expected_elapsed: u64 = 0;
    for (intervals, 0..) |interval, index| {
        fake.advance(@intCast(interval));
        expected_elapsed += interval;

        const time = frame_clock.tick();
        try testing.expectEqual(index, time.index);
        try testing.expectEqual(interval, time.interval_ns);
        try testing.expectEqual(expected_elapsed, time.elapsed_ns);
        try testing.expectEqual(false, time.stalled());
        try testing.expectApproxEqAbs(
            @as(f32, @floatCast(lenore.seconds(interval))),
            time.delta,
            1e-9,
        );
    }
}

test "a stalled frame is clamped for the world and reported in full" {
    var fake: FakeIo = .init(0);
    var frame_clock: lenore.FrameClock = .start(.init(fake.io()));

    // Well past the bound, and not a multiple of it, so a clamp that returned
    // the interval or a fraction of it is visible.
    const stall = 2_500 * ms;
    fake.advance(stall);
    const time = frame_clock.tick();

    try testing.expect(time.stalled());
    try testing.expectEqual(@as(u64, stall), time.interval_ns);
    try testing.expectApproxEqAbs(@as(f32, 0.1), time.delta, 1e-9);
    // The clamp shortens the step, never the clock: the next frame is still
    // placed where the wall clock puts it.
    try testing.expectEqual(@as(u64, stall), time.elapsed_ns);
}

test "the exact bound is not a stall, and one nanosecond past it is" {
    var fake: FakeIo = .init(0);
    var frame_clock: lenore.FrameClock = .start(.init(fake.io()));

    fake.advance(@intCast(lenore.max_frame_delta_ns));
    const at_bound = frame_clock.tick();
    try testing.expectEqual(false, at_bound.stalled());
    try testing.expectApproxEqAbs(@as(f32, 0.1), at_bound.delta, 1e-9);

    fake.advance(@intCast(lenore.max_frame_delta_ns + 1));
    try testing.expect(frame_clock.tick().stalled());
}

// The reason `tick` subtracts nanoseconds rather than two elapsed seconds.
// Measured with numpy: at ten hours of uptime the f32 spacing is 3.906 ms, so
// differencing f32 elapsed there turns a 60 Hz frame into 16.602 ms or worse.
test "a frame late in a long run keeps its interval exactly" {
    const ten_hours: i96 = 10 * 3600 * std.time.ns_per_s;
    var fake: FakeIo = .init(0);
    var frame_clock: lenore.FrameClock = .start(.init(fake.io()));

    fake.advance(ten_hours);
    _ = frame_clock.tick();

    const interval: u64 = 16_666_667;
    fake.advance(@intCast(interval));
    const time = frame_clock.tick();

    try testing.expectEqual(interval, time.interval_ns);
    // Tighter than the 3.906 ms an f32 elapsed could resolve there, and tighter
    // than the 0.065 ms error that differencing would produce.
    try testing.expectApproxEqAbs(@as(f32, 0.016666667), time.delta, 1e-6);
}

test "the fps window closes on time and separates the worst frame from the mean" {
    var counter: lenore.FpsCounter = .init(150 * ms);

    // Nine frames of 10 ms and one of 60: the window fills at exactly 150 ms
    // with a mean of 15 and a worst of 60. Equal frames would let a counter
    // that reported the mean as the worst pass, and the 60 is not the frame
    // that closes the window, so neither can stand in for the other.
    var frame: lenore.FrameTime = .{
        .elapsed_ns = 0,
        .interval_ns = 10 * ms,
        .delta = 0.01,
        .index = 0,
    };
    for (0..8) |_| try testing.expectEqual(null, counter.record(frame));

    frame.interval_ns = 60 * ms;
    try testing.expectEqual(null, counter.record(frame));

    frame.interval_ns = 10 * ms;
    const report = counter.record(frame).?;
    try testing.expectEqual(@as(u32, 10), report.frames);
    try testing.expectEqual(@as(u64, 150 * ms), report.elapsed_ns);
    try testing.expectApproxEqAbs(@as(f32, 15), report.mean_ms, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 60), report.worst_ms, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 10.0 / 0.15), report.fps, 1e-3);
}

test "a closed window carries nothing into the next one" {
    var counter: lenore.FpsCounter = .init(10 * ms);
    const slow: lenore.FrameTime = .{
        .elapsed_ns = 0,
        .interval_ns = 40 * ms,
        .delta = 0.04,
        .index = 0,
    };
    const quick: lenore.FrameTime = .{
        .elapsed_ns = 0,
        .interval_ns = 12 * ms,
        .delta = 0.012,
        .index = 1,
    };

    const first = counter.record(slow).?;
    try testing.expectApproxEqAbs(@as(f32, 40), first.worst_ms, 1e-4);

    // The 40 ms frame must not survive into this window as either the worst or
    // a summand.
    const second = counter.record(quick).?;
    try testing.expectEqual(@as(u32, 1), second.frames);
    try testing.expectEqual(@as(u64, 12 * ms), second.elapsed_ns);
    try testing.expectApproxEqAbs(@as(f32, 12), second.worst_ms, 1e-4);
}

test "phase splits partition the total" {
    var fake: FakeIo = .init(1_000);
    var timer: lenore.PhaseTimer = .begin(.init(fake.io()));

    fake.advance(3 * ms);
    const first = timer.split();
    fake.advance(7 * ms);
    const second = timer.split();
    fake.advance(11 * ms);
    // Dropped on purpose: a phase nobody keeps is lost time, and the total
    // still spans it.
    _ = timer.split();

    try testing.expectEqual(@as(u64, 3 * ms), first);
    try testing.expectEqual(@as(u64, 7 * ms), second);
    try testing.expectEqual(@as(u64, 21 * ms), timer.total());
}

test "the total excludes a phase that is still running" {
    var fake: FakeIo = .init(0);
    var timer: lenore.PhaseTimer = .begin(.init(fake.io()));

    fake.advance(5 * ms);
    _ = timer.split();
    fake.advance(9 * ms);

    try testing.expectEqual(@as(u64, 5 * ms), timer.total());
}
