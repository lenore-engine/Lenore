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

// The stages of one frame, in the order the loop closes them.
//
// The order is the type: `FrameMetrics` takes each split against this sequence,
// so a stage closed twice or skipped is a mistake the frame reports rather than
// one that shows up as a phase reading low.
pub const Phase = enum {
    events,
    ui,
    wait,
    acquire,
    update,
    world,
    rings,
    record,
    submit,
    present,
};

// Where a frame's host time went.
//
// CPU only: this is what the engine spent preparing and recording, not what the
// device spent drawing. The two are unrelated when the present blocks, which
// under FIFO it usually does, and `wait_ns` is where that shows up.
//
// One field per `Phase`, in that order. Every one is closed by a split of one
// clock, so an interval cannot be counted in two of them and the ten partition
// the frame exactly. Ten readings at about 22 ns each is under a quarter of a
// microsecond against a frame of milliseconds, which is why there is no switch
// to turn it off.
pub const FramePhases = struct {
    // Polling the platform, and with it whatever the loop spent between the end
    // of the previous frame and the start of this one. A frame the loop
    // skipped, for a stale swapchain or a minimised window, has its time folded
    // in here rather than being lost: it is host time and it belongs to the
    // frame that follows it.
    events_ns: u64 = 0,
    // Registering the frame's interactive regions, draining the input queue and
    // routing it against them. Not the overlay's geometry, which is written
    // with the rest of the frame's rings and counted there.
    ui_ns: u64 = 0,
    // Blocked on the previous use of this frame's slot. Under FIFO this is the
    // presentation engine's pace rather than any work of ours, so a large value
    // here beside small ones elsewhere is a frame that is waiting, not slow.
    wait_ns: u64 = 0,
    acquire_ns: u64 = 0,
    // The driver's own frame: whatever it advances with the delta. Kept apart
    // from `world_ns` because that one is the engine's animation and plan, and
    // a single number over both answers neither question.
    update_ns: u64 = 0,
    // Animation, the draw plan and the joint ring.
    world_ns: u64 = 0,
    // Writing the frame's rings, the scene's and the overlay's.
    rings_ns: u64 = 0,
    // Planning and recording the command buffer.
    record_ns: u64 = 0,
    submit_ns: u64 = 0,
    present_ns: u64 = 0,

    // The field one phase closes. Named from the tag, so the two lists cannot
    // drift apart: a `Phase` with no field here does not compile.
    pub fn of(self: *FramePhases, phase: Phase) *u64 {
        switch (phase) {
            inline else => |tag| return &@field(self, @tagName(tag) ++ "_ns"),
        }
    }

    pub fn add(self: *FramePhases, other: FramePhases) void {
        inline for (@typeInfo(FramePhases).@"struct".fields) |field| {
            @field(self, field.name) += @field(other, field.name);
        }
    }

    // The same phases divided by the frames they were summed over, which is
    // what a report wants. A window of no frames divides nothing.
    pub fn mean(self: FramePhases, frames: u32) FramePhases {
        if (frames == 0) return .{};
        var out: FramePhases = .{};
        inline for (@typeInfo(FramePhases).@"struct".fields) |field| {
            @field(out, field.name) = @field(self, field.name) / frames;
        }
        return out;
    }

    pub fn total(self: FramePhases) u64 {
        var sum: u64 = 0;
        inline for (@typeInfo(FramePhases).@"struct".fields) |field| {
            sum += @field(self, field.name);
        }
        return sum;
    }
};

// What a run measures of its own frames: where the host time went, and how fast
// they came.
//
// One type and not two, because the two are one invariant. A window closes on
// the frame that fills it and reports the mean of the phases over exactly the
// frames it counted. Close the rate before that frame's phases have been added
// and the mean divides one count of frames by another, with every frame in it
// short of whatever the loop does after the call. Neither shows in the answer:
// the numbers stay plausible and the phases simply stop summing to the frame.
//
// It reads no clock. Splits and intervals come in as nanoseconds, which is what
// makes the whole of it exercisable from a host test: the arithmetic and the
// order are the parts that can be wrong, and neither needs a device or a real
// second to pass.
pub const FrameMetrics = struct {
    // This frame's phases, and the window they are accumulating into.
    phases: FramePhases,
    window: FramePhases,
    fps: FpsCounter,

    // The last window closed. `last_fps` is null until one has, and
    // `closed_windows` is how many there have been: a driver reporting each one
    // needs to know whether what it is looking at is new, and the report's own
    // numbers are not an identity, since two windows can measure the same
    // duration.
    last_phases: FramePhases,
    last_fps: ?FpsCounter.Report,
    closed_windows: u64,

    // How many phases of this frame have been closed. Present only where an
    // assert would run: it is written once per phase and read nowhere else, so
    // a build with the checks off carries neither the field nor the store.
    phases_closed: if (checking) usize else void,

    const checking = std.debug.runtime_safety;

    pub fn init(window_ns: u64) FrameMetrics {
        return .{
            .phases = .{},
            .window = .{},
            .fps = .init(window_ns),
            .last_phases = .{},
            .last_fps = null,
            .closed_windows = 0,
            .phases_closed = if (checking) 0 else {},
        };
    }

    // Opens a frame. A frame the loop abandons is simply never ended, and the
    // next `beginFrame` drops what it had recorded: its host time belongs to
    // the frame that follows it, which counts it in `events_ns`.
    pub fn beginFrame(self: *FrameMetrics) void {
        self.phases = .{};
        if (checking) self.phases_closed = 0;
    }

    // Closes one phase with what the timer measured for it.
    pub fn record(self: *FrameMetrics, phase: Phase, elapsed_ns: u64) void {
        if (checking) {
            std.debug.assert(@intFromEnum(phase) == self.phases_closed);
            self.phases_closed = @intFromEnum(phase) + 1;
        }
        self.phases.of(phase).* = elapsed_ns;
    }

    // Closes the frame: its phases join the window, and the window closes with
    // it when the interval fills one.
    //
    // Every phase has to have been closed first, which is what the assert says.
    // The alternative is a report whose phases are short by however much of the
    // frame came after the call.
    pub fn endFrame(self: *FrameMetrics, time: FrameTime) void {
        if (checking) std.debug.assert(self.phases_closed == @typeInfo(Phase).@"enum".fields.len);

        self.window.add(self.phases);
        if (self.fps.record(time)) |report| {
            self.last_fps = report;
            self.last_phases = self.window.mean(report.frames);
            self.window = .{};
            self.closed_windows += 1;
        }
    }
};
