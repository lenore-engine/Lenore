const std = @import("std");
const platform = @import("lenore-platform");

const Allocator = std.mem.Allocator;
const log = std.log.scoped(.lenore);

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

pub const AppOptions = struct {
    title: [:0]const u8,
    preferred_extent: platform.Extent2D = .{ .width = 1280, .height = 720 },
    initial_event_capacity: usize = platform.initial_event_capacity,
    max_event_capacity: usize = platform.max_event_capacity,
};

pub const App = struct {
    allocator: Allocator,
    io_threaded: std.Io.Threaded,
    io: std.Io,
    clock: platform.Clock,
    platform_host: platform.Platform,
    window: platform.Window,
    input: platform.Input,

    // Initialization is in-place because native callbacks retain &self.input.
    // App must not move after captureInput installs that address.
    pub fn init(self: *App, allocator: Allocator, options: AppOptions) !void {
        self.allocator = allocator;
        self.io_threaded = .init(allocator, .{});
        errdefer self.io_threaded.deinit();
        self.io = self.io_threaded.io();
        self.clock = .init(self.io);

        self.platform_host = try .init();
        errdefer self.platform_host.deinit();
        self.window = try self.platform_host.createWindow(options.preferred_extent, options.title);
        errdefer self.window.deinit();
        self.input = try .init(
            allocator,
            self.clock,
            options.initial_event_capacity,
            options.max_event_capacity,
        );
        errdefer self.input.deinit();

        self.window.captureInput(&self.input);
    }

    pub fn deinit(self: *App) void {
        self.window.deinit();
        self.input.deinit();
        self.platform_host.deinit();
        self.io_threaded.deinit();
        self.* = undefined;
    }
};

pub fn main() !void {
    const allocator = if (checking) debug_allocator.allocator() else std.heap.smp_allocator;
    defer if (checking) {
        if (debug_allocator.deinit() == .leak) log.err("memory leaked", .{});
    };

    var app: App = undefined;
    try app.init(allocator, .{ .title = "Lenore" });
    defer app.deinit();

    while (!app.window.shouldClose()) {
        app.platform_host.waitEvents();

        const batch = app.input.takeBatch() catch |err| switch (err) {
            error.InputEventOverflow => {
                log.warn("input overflowed; batch discarded ({d} so far)", .{app.input.overflowCount()});
                continue;
            },
        };
        defer app.input.releaseBatch();
        for (batch) |event| report(event);
    }
}

fn report(event: platform.Event) void {
    switch (event.payload) {
        .key => |key| log.info("key {t} {t}", .{ key.physical, key.action }),
        .text => |text| log.info("text {s}", .{text.bytes[0..text.len]}),
        .cursor => |cursor| log.info(
            "cursor {d:.1} {d:.1}",
            .{ cursor.logical_position[0], cursor.logical_position[1] },
        ),
        .mouse_button => |button| log.info("button {t} {t}", .{ button.button, button.action }),
        .scroll => |scroll| log.info("scroll {d:.2}", .{scroll.line_delta[1]}),
        .focus => |focus| log.info("focus {}", .{focus.focused}),
        .surface_metrics => |metrics| log.info(
            "surface {d}x{d} scale {d:.2}",
            .{ metrics.framebuffer_extent.width, metrics.framebuffer_extent.height, metrics.scale[0] },
        ),
    }
}
