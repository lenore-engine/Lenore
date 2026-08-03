const std = @import("std");
const gpu = @import("lenore-gpu");
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
    present: gpu.PresentModePreference = .fifo,
};

// How many frames the host may be preparing while the device works on earlier
// ones. Two lets the host record the next frame while the device finishes the
// current one, without the latency a deeper queue adds.
const frames_in_flight = 2;

// Until anything is drawn, the window shows this. It is not a placeholder for a
// clear colour policy: whoever renders replaces the whole call.
const clear_colour = [4]f32{ 0.05, 0.06, 0.09, 1.0 };

pub const App = struct {
    allocator: Allocator,
    io_threaded: std.Io.Threaded,
    io: std.Io,
    clock: platform.Clock,
    platform_host: platform.Platform,
    window: platform.Window,
    input: platform.Input,

    context: gpu.Context,
    swapchain: gpu.Swapchain,
    frames: [frames_in_flight]gpu.Frame,
    frame_index: usize,
    // The framebuffer extent as the platform last reported it. Tracked from
    // surface events rather than queried: the window exposes no getter, because
    // a compositor tells the client its size and being told is the whole point.
    surface_extent: platform.Extent2D,
    // Set when a present or acquire reports the swapchain no longer matches the
    // surface. The recreation happens at the top of the next frame, where no
    // work is in flight against the images being replaced.
    swapchain_stale: bool,

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

        self.context = try .init(allocator, options.title, self.window.nativeHandles());
        errdefer self.context.deinit();
        log.info("device: {s}", .{self.context.deviceName()});

        self.swapchain = try .init(
            &self.context,
            allocator,
            options.preferred_extent,
            options.present,
        );
        errdefer self.swapchain.deinit();

        var created: usize = 0;
        errdefer for (self.frames[0..created]) |frame| frame.deinit(&self.context);
        for (&self.frames) |*frame| {
            frame.* = try .init(&self.context);
            created += 1;
        }

        self.frame_index = 0;
        self.surface_extent = self.swapchain.currentExtent();
        self.swapchain_stale = false;
    }

    // Vulkan specification, vkDestroyDevice: every use must have completed, and
    // a fence does not establish that. Presentation is queued after the
    // submission a fence covers and still holds the swapchain image and the
    // semaphore it waited on, so the whole device has to drain.
    pub fn deinit(self: *App) void {
        self.context.waitIdle() catch |err| {
            log.err("device did not drain during teardown: {t}", .{err});
        };
        for (self.frames) |frame| frame.deinit(&self.context);
        self.swapchain.deinit();
        self.context.deinit();

        self.window.deinit();
        self.input.deinit();
        self.platform_host.deinit();
        self.io_threaded.deinit();
        self.* = undefined;
    }

    pub fn run(self: *App) !void {
        while (!self.window.shouldClose()) {
            self.platform_host.pollEvents();
            self.drainInput();

            // A minimised window has nothing to present to, and the swapchain
            // cannot be created at a zero extent. Blocking on the event queue
            // rather than spinning is what keeps an idle window free.
            if (self.surface_extent.width == 0 or self.surface_extent.height == 0) {
                self.platform_host.waitEvents();
                continue;
            }
            try self.renderFrame();
        }
    }

    fn drainInput(self: *App) void {
        const batch = self.input.takeBatch() catch |err| switch (err) {
            error.InputEventOverflow => {
                log.warn(
                    "input overflowed; batch discarded ({d} so far)",
                    .{self.input.overflowCount()},
                );
                return;
            },
        };
        defer self.input.releaseBatch();

        for (batch) |event| {
            if (event.payload == .surface_metrics)
                self.surface_extent = event.payload.surface_metrics.framebuffer_extent;
            report(event);
        }
    }

    // One clear-and-present. Nothing is drawn: this is the frame loop and the
    // swapchain paths, exercised before there is anything to draw with.
    fn renderFrame(self: *App) !void {
        // Recreation happens here, at the top, rather than where the staleness
        // was noticed. At this point the previous frame's submission is the only
        // work outstanding, and waiting for every slot below is what makes
        // replacing the images safe.
        if (self.swapchain_stale or !self.swapchain.matchesExtent(self.surface_extent))
            try self.recreateSwapchain();

        const frame = self.frames[self.frame_index];
        try frame.waitForGpu(&self.context);

        // An acquire that fails this way has signalled nothing and consumed
        // nothing, and the fence is still signalled because only a submission
        // resets it. Returning here therefore leaves the loop able to run again.
        const acquired = self.swapchain.acquireNextImage(frame.image_acquired) catch |err| switch (err) {
            error.OutOfDateKHR => {
                self.swapchain_stale = true;
                return;
            },
            else => return err,
        };
        if (acquired.state == .suboptimal) self.swapchain_stale = true;

        const commands = try frame.beginCommands(&self.context);
        try self.swapchain.recordClear(commands, acquired.image_index, clear_colour);
        try frame.submit(&self.context, .{
            .wait = frame.image_acquired,
            // The first thing this frame does to the image is clear it.
            .wait_stage = .{ .clear_bit = true },
            .signal = try self.swapchain.renderFinishedSemaphore(acquired.image_index),
            // Signalled after everything, including the barrier that puts the
            // image into the layout presentation requires. A tighter stage would
            // signal before that transition completed.
            .signal_stage = .{ .all_commands_bit = true },
        });

        const presented = self.swapchain.present(acquired.image_index) catch |err| switch (err) {
            error.OutOfDateKHR => {
                self.swapchain_stale = true;
                return;
            },
            else => return err,
        };
        if (presented == .suboptimal) self.swapchain_stale = true;

        self.frame_index = (self.frame_index + 1) % frames_in_flight;
    }

    // Vulkan specification, vkDestroySwapchainKHR: every use of an image
    // acquired from the swapchain must have completed, which includes the
    // presentation queued after the last submission. A fence covers the
    // submission and not the presentation, so this drains the device instead.
    // Resizing is cold enough to pay for it.
    fn recreateSwapchain(self: *App) !void {
        try self.context.waitIdle();
        try self.swapchain.recreate(self.surface_extent);
        self.swapchain_stale = false;
        log.info("swapchain: {d}x{d}", .{
            self.swapchain.currentExtent().width,
            self.swapchain.currentExtent().height,
        });
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
    try app.run();
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
