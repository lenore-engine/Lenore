const std = @import("std");
const platform = @import("lenore-platform");

// The largest step one frame may advance the world by.
//
// It exists so that a stall does not teleport what the frame animates. A load
// hitch, a swapchain recreation and a debugger break all produce one long
// interval, and advancing every clip by it moves them that far in a single
// step. Past this bound the world runs slower than the wall clock instead,
// which is a state a viewer recovers from smoothly and a jump is not.
//
// A tenth of a second is a choice rather than a derivation. It has to sit above
// any frame that is merely slow, which at ten frames per second it does by
// several times over, and below any interval a viewer reads as a hitch. Godot
// bounds the same thing by a different route, capping how many fixed steps one
// frame may run rather than the step itself: eight steps at sixty per second is
// 0.133 s, the same order (main/main.cpp:2180 for the default,
// main/main.cpp:4850 for where it shortens the frame).
//
// Authored in nanoseconds because that is the unit it is compared in. Deriving
// it from a float of seconds would land a tenth of a second at 100000001 ns.
pub const max_frame_delta_ns: u64 = std.time.ns_per_s / 10;

pub fn seconds(nanoseconds: u64) f64 {
    return @as(f64, @floatFromInt(nanoseconds)) / std.time.ns_per_s;
}

// One frame's reading of the clock.
//
// Only `FrameClock.tick` produces one, which is what makes "the delta is taken
// once per frame" a property of the type rather than of where an assignment
// happens to sit in a loop body.
//
// One float, and it is the one that advances the world. Everything else is the
// nanoseconds the clock measured, so a consumer that wants seconds asks for
// them and a consumer that wants precision keeps it.
pub const FrameTime = struct {
    // Nanoseconds since the clock started.
    elapsed_ns: u64,
    // What the wall clock measured for this frame, unclamped. Timing reports
    // read this and never `delta`: the clamp protects the animation, and a
    // report taken from the clamped value would hide the stall it exists to
    // survive.
    interval_ns: u64,
    // Seconds to advance the world by, `interval_ns` clamped to
    // `max_frame_delta_ns`. Every animator takes this and nothing else.
    delta: f32,
    // Frames ticked, from zero for the first.
    index: u64,

    // Whether the clamp was reached, which is exactly when `delta` and
    // `interval_ns` describe different amounts of time.
    pub fn stalled(self: FrameTime) bool {
        return self.interval_ns > max_frame_delta_ns;
    }
};

// The frame's clock, and the only place an interval becomes a delta.
pub const FrameClock = struct {
    clock: platform.Clock,
    origin: u64,
    previous: u64,
    index: u64,

    pub fn start(clock: platform.Clock) FrameClock {
        const now = clock.now();
        return .{ .clock = clock, .origin = now, .previous = now, .index = 0 };
    }

    // The interval is taken in nanoseconds and converted once, rather than by
    // subtracting two elapsed values in seconds.
    //
    // Measured with numpy: f32 spacing at one hour of uptime is 0.244 ms, so a
    // 16.667 ms frame differenced from f32 elapsed reads as 16.602 ms, and at
    // ten hours the spacing is 3.906 ms and a 60 Hz frame is unrecoverable. The
    // u64 interval is exact and 584 years wide, and the single conversion below
    // sees a number small enough for f32 to hold it.
    pub fn tick(self: *FrameClock) FrameTime {
        const now = self.clock.now();
        // The clock is monotonic, so this cannot wrap. Two readings inside one
        // tick of the OS clock are equal rather than decreasing, which gives a
        // zero delta: a frame that advanced nothing, which every animator
        // already handles.
        const interval = now - self.previous;
        self.previous = now;

        const index = self.index;
        self.index += 1;
        return .{
            .elapsed_ns = now - self.origin,
            .interval_ns = interval,
            .delta = @floatCast(seconds(@min(interval, max_frame_delta_ns))),
            .index = index,
        };
    }
};

// A window over frame intervals, reported when the window fills.
//
// Measured in time rather than in a count of frames: the question is how the
// last second went, and a count-based window is a different length of time on
// every machine that runs it.
pub const FpsCounter = struct {
    window_ns: u64,
    frames: u32 = 0,
    summed_ns: u64 = 0,
    worst_ns: u64 = 0,

    pub const Report = struct {
        frames: u32,
        // The window that actually elapsed, which is a little longer than the
        // one asked for: it closes on the first frame that fills it.
        elapsed_ns: u64,
        fps: f32,
        mean_ms: f32,
        // The single worst frame of the window. The mean hides the thing a
        // frame budget is kept for: one frame of 100 ms among sixty of 16 ms
        // moves the mean by 1.4 ms and is the only one anybody sees.
        worst_ms: f32,
    };

    pub fn init(window_ns: u64) FpsCounter {
        return .{ .window_ns = window_ns };
    }

    // Null until the window fills, and the window it closes on the frame after.
    pub fn record(self: *FpsCounter, time: FrameTime) ?Report {
        self.frames += 1;
        self.summed_ns += time.interval_ns;
        self.worst_ns = @max(self.worst_ns, time.interval_ns);
        if (self.summed_ns < self.window_ns) return null;

        const elapsed = self.summed_ns;
        const report: Report = .{
            .frames = self.frames,
            .elapsed_ns = elapsed,
            // A window of zero length is reachable: a caller may ask for one,
            // and a clock that has not moved fills it with frames of no
            // duration. Reported as no rate rather than divided into.
            .fps = if (elapsed > 0)
                @floatCast(@as(f64, @floatFromInt(self.frames)) / seconds(elapsed))
            else
                0,
            .mean_ms = if (self.frames > 0)
                @floatCast(seconds(elapsed) * 1000.0 / @as(f64, @floatFromInt(self.frames)))
            else
                0,
            .worst_ms = @floatCast(seconds(self.worst_ns) * 1000.0),
        };
        self.frames = 0;
        self.summed_ns = 0;
        self.worst_ns = 0;
        return report;
    }
};

// Where a load spends its time: one clock and one moving boundary.
//
// Every phase is closed by a `split`, which is what keeps an interval from being
// counted in two of them. A phase whose split is dropped is lost time rather
// than double-counted time, and `total` spans the whole sequence either way.
//
// What each phase is called belongs to whoever is loading. This is the boundary
// and nothing else.
pub const PhaseTimer = struct {
    clock: platform.Clock,
    origin: u64,
    last: u64,

    pub fn begin(clock: platform.Clock) PhaseTimer {
        const now = clock.now();
        return .{ .clock = clock, .origin = now, .last = now };
    }

    // Nanoseconds since the previous boundary, and the new boundary.
    pub fn split(self: *PhaseTimer) u64 {
        const now = self.clock.now();
        const elapsed = now - self.last;
        self.last = now;
        return elapsed;
    }

    // Up to the last boundary, not up to now: the total is the sum of the
    // splits taken, so a phase still running is not in it.
    pub fn total(self: *const PhaseTimer) u64 {
        return self.last - self.origin;
    }
};
