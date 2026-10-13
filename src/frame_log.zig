// A line to the log for every measuring window the loop closes: the rate, where
// the host's time went, and what the device spent on its passes.
//
// A component rather than a function an application calls, so that adding it
// to `Engine.run`'s tuple is the whole of putting a frame report on a run. The
// device line appears only when the engine was started with `gpu_timing`, and
// only once a timed frame has come back.

const std = @import("std");

const engine_module = @import("engine.zig");
const Engine = engine_module.Engine;
const Hooks = engine_module.Hooks;
const Level = @import("level.zig").Level;
const FrameTime = @import("time.zig").FrameTime;

const log = std.log.scoped(.frame);

pub const FrameLog = struct {
    pub const hooks: Hooks(FrameLog) = .{ .update = update };

    // Windows already reported. A window is recognised by the count of closed
    // ones rather than by its numbers, which two windows can share.
    reported: u64 = 0,

    fn update(self: *FrameLog, engine: *Engine, _: *Level, _: FrameTime) !void {
        const metrics = &engine.metrics;
        if (metrics.closed_windows == self.reported) return;
        self.reported = metrics.closed_windows;
        const rate = metrics.last_fps orelse return;

        // Every phase, because together they partition the frame: a frame
        // that is slow and one that is waiting differ only in which of them is
        // large.
        const host = metrics.last_phases;
        log.info("{d:.0} fps, mean {d:.2} ms, worst {d:.2} ms; host ms: events {d:.2} ui {d:.2} wait {d:.2} " ++
            "acquire {d:.2} update {d:.2} world {d:.2} rings {d:.2} record {d:.2} submit {d:.2} present {d:.2}", .{
            rate.fps,                      rate.mean_ms,
            rate.worst_ms,                 milliseconds(host.events_ns),
            milliseconds(host.ui_ns),      milliseconds(host.wait_ns),
            milliseconds(host.acquire_ns), milliseconds(host.update_ns),
            milliseconds(host.world_ns),   milliseconds(host.rings_ns),
            milliseconds(host.record_ns),  milliseconds(host.submit_ns),
            milliseconds(host.present_ns),
        });

        const device = engine.last_gpu orelse return;
        log.info("  device ms: shadow {d:.2} depth {d:.2} main {d:.2} bloom {d:.2} post {d:.2} total {d:.2}", .{
            milliseconds(device.get(.shadow)),
            milliseconds(device.get(.depth)),
            milliseconds(device.get(.main)),
            milliseconds(device.get(.bloom)),
            milliseconds(device.get(.post)),
            milliseconds(device.total()),
        });
    }
};

fn milliseconds(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-6;
}
