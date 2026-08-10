const std = @import("std");
const platform = @import("lenore-platform");

// DebugAllocator keeps metadata per allocation to catch leaks and use-after-free
// and pays time and memory for it. A build with safety off takes smp_allocator,
// which std documents as the one designed for ReleaseFast.
const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

pub fn main() !void {
    const gpa = if (checking) debug_allocator.allocator() else std.heap.smp_allocator;
    defer if (checking) {
        if (debug_allocator.deinit() == .leak) std.log.err("memory leaked", .{});
    };

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const clock: platform.Clock = .init(threaded.io());

    var host: platform.Platform = try .init();
    defer host.deinit();

    var window = try host.createWindow(.{ .width = 1280, .height = 720 }, "lenore-platform");
    defer window.deinit();

    var input: platform.Input = try .init(
        gpa,
        clock,
        platform.initial_event_capacity,
        platform.max_event_capacity,
    );
    defer input.deinit();

    // Routes this window's events into the queue and submits the first
    // SurfaceMetrics, so the size is available after the first pump.
    window.captureInput(&input);

    // xdg-shell.xml, xdg_surface: mapping requires a buffer attachment after
    // configure. This graphics-free example never attaches one, so it runs
    // invisibly and exits after a bounded event-pump demonstration.
    const deadline_ns = clock.now() + 5 * std.time.ns_per_s;
    while (!window.shouldClose()) {
        const now_ns = clock.now();
        if (now_ns >= deadline_ns) break;

        const remaining_seconds = @as(f64, @floatFromInt(deadline_ns - now_ns)) / std.time.ns_per_s;
        host.waitEventsTimeout(remaining_seconds);

        const batch = input.takeBatch() catch |err| switch (err) {
            error.InputEventOverflow => {
                std.log.warn("input overflowed; batch discarded ({d} so far)", .{input.overflowCount()});
                continue;
            },
        };
        defer input.releaseBatch();

        for (batch) |event| report(event);
    }
}

fn report(event: platform.Event) void {
    switch (event.payload) {
        .key => |key| std.log.info("key {t} {t}", .{ key.physical, key.action }),
        .text => |text| std.log.info("text {s}", .{text.bytes[0..text.len]}),
        .cursor => |cursor| std.log.info(
            "cursor {d:.1} {d:.1}",
            .{ cursor.logical_position[0], cursor.logical_position[1] },
        ),
        .mouse_button => |button| std.log.info("button {t} {t}", .{ button.button, button.action }),
        .scroll => |scroll| std.log.info("scroll {d:.2}", .{scroll.line_delta[1]}),
        .focus => |focus| std.log.info("focus {}", .{focus.focused}),
        .surface_metrics => |metrics| std.log.info(
            "surface {d}x{d} scale {d:.2}",
            .{ metrics.framebuffer_extent.width, metrics.framebuffer_extent.height, metrics.scale[0] },
        ),
    }
}
