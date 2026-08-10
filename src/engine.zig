const std = @import("std");
const gpu = @import("lenore-gpu");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const level_module = @import("level.zig");
// The shading every pass below is built from. Named here rather than taken as
// an option, because this is the engine's own composition and what it draws
// with is the engine's answer. An application wanting another one builds the
// passes itself.
const shaders = @import("shaders.zig");
const time_module = @import("time.zig");

const Allocator = std.mem.Allocator;
const log = std.log.scoped(.lenore);
const FpsCounter = time_module.FpsCounter;
const FrameClock = time_module.FrameClock;
const FrameTime = time_module.FrameTime;
const Level = level_module.Level;

// The device, the window and the one frame loop.
//
// There is exactly one loop in this project and applications drive it rather
// than writing their own. That is the whole reason the engine exists as
// something separate from the harness: a second application meant a second copy
// of the same twenty decisions about staleness, acquisition and ordering, and
// the copies were where they drifted.
//
// What varies between applications reaches the loop two ways. `Look` is what
// the frame is drawn with and is read every frame, so changing it changes the
// next frame. A driver is a value with any of the hooks below, called at the one
// point in the frame where each is answerable.

// How many frames the host may be preparing while the device works on earlier
// ones. Two lets the host record the next frame while the device finishes the
// current one, without the latency a deeper queue adds.
const frames_in_flight = 2;

// What runs on top of the loop. Every driver declares all four, and they are
// called at the one point in the frame where each is answerable:
//
//   onEvent(driver, engine, event) !void
//       Every input event, before the frame that reacts to it.
//   onResize(driver, engine, extent) !void
//       After the swapchain and the render targets have followed the surface.
//   onRecord(driver, engine, level, commands) !void
//       Inside the main pass, after the scene and before it closes. See the
//       call site for what may be recorded there and what may not.
//   onFrame(driver, engine, level, time) !void
//       After the frame is recorded and before it is submitted. Every counter
//       the recording moved is final here, and nothing has been queued yet.
//
// Required rather than optional, and the reason is a trap that was sprung twice
// in one day. Optional hooks were selected with `@hasDecl`, which does not see a
// declaration that is not `pub` from another file, so a driver whose methods
// were declared without it compiled, ran, and was never called: no keys, no
// resize, no probes, and a background left at whatever the look defaulted to.
// Nothing reported anything, because nothing was wrong as far as the compiler
// could tell. Requiring all of them turns that into a missing declaration,
// which is an error at the call below.
//
// A driver that wants none of them uses `NoDriver`, which says so.
pub const NoDriver = struct {
    pub fn onEvent(_: *NoDriver, _: *Engine, _: platform.Event) !void {}
    pub fn onResize(_: *NoDriver, _: *Engine, _: platform.Extent2D) !void {}
    pub fn onRecord(_: *NoDriver, _: *Engine, _: *Level, _: gpu.vk.CommandBuffer) !void {}
    pub fn onFrame(_: *NoDriver, _: *Engine, _: *Level, _: FrameTime) !void {}
};

// What every frame is drawn with. Read at the top of each frame, so a driver
// that writes it between frames changes the next one.
pub const Look = struct {
    // The block the shader reads. Whose lights these are is the application's
    // decision: an asset's own, or one diagnostic sun over every model.
    lights: []const gpu.LightUniform = &.{},
    sun_shadow: scene.SunShadowSettings = .{},
    background: gpu.Background = .clear,
    // Null runs no chain and composites none. Not a look with everything at
    // zero: that would still cost the chain.
    bloom: ?gpu.BloomSettings = null,
    post: gpu.PostSettings = .{},
};

// Where a frame's host time went.
//
// CPU only: this is what the engine spent preparing and recording, not what the
// device spent drawing. The two are unrelated when the present blocks, which
// under FIFO it usually does, and `wait_ns` is where that shows up.
//
// Every phase is closed by a split of one clock, so an interval cannot be
// counted in two of them. Seven readings a frame at about 22 ns each is under a
// fifth of a microsecond against a frame of milliseconds, which is why there is
// no switch to turn it off.
pub const FramePhases = struct {
    // Polling the platform and draining the input queue, and with them whatever
    // the loop spent between the end of the previous frame and the start of
    // this one. A frame the loop skipped, for a stale swapchain or a minimised
    // window, has its time folded in here rather than being lost: it is host
    // time and it belongs to the frame that follows it.
    events_ns: u64 = 0,
    // Blocked on the previous use of this frame's slot. Under FIFO this is the
    // presentation engine's pace rather than any work of ours, so a large value
    // here beside small ones elsewhere is a frame that is waiting, not slow.
    wait_ns: u64 = 0,
    acquire_ns: u64 = 0,
    // Animation, the draw plan and the joint ring.
    world_ns: u64 = 0,
    // Writing the frame's rings.
    update_ns: u64 = 0,
    // Planning and recording the command buffer.
    record_ns: u64 = 0,
    submit_ns: u64 = 0,
    present_ns: u64 = 0,

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

// The directional light the shadow map is fitted to, and the fit.
//
// One value rather than two, because the index carried to the device and the
// direction the fit was built from have to name the same entry. Two fields set
// separately agree by order, which is a thing that stops being true.
pub const Sun = struct {
    index: u32,
    fit: scene.SunShadowFit,
};

// What the frame knows when it decides whether to re-record the shadow map.
//
// Split from the engine because it is the whole of the policy and every input
// is a bool, so the rule can be stated against a table instead of against a
// device.
pub const ShadowBakeRequest = struct {
    // Shadows are switched on and a light was fitted. Either being false makes
    // the map unread this frame.
    shadows_on: bool,
    // What the previous frame decided, which is what makes the off-to-on edge
    // visible.
    were_on: bool,
    // The fit changed, or nothing has been baked since a level was installed.
    map_stale: bool,
    casters_move: bool,

    // Off is `reuse` and not a third state, because the renderer has none: it
    // bakes once on its own account so a lookup never reads an image nothing
    // wrote, and `reuse` never opens the pass again after that. The shader
    // returns before it samples when the strength is zero, so an unread map
    // costs a frame nothing, and the leftover is one bake at load.
    //
    // Switching shadows back on always re-bakes. While they were off nothing
    // kept the map current: casters moved and the sun may have, so it holds
    // whichever moment last wrote it.
    pub fn decide(self: ShadowBakeRequest) gpu.ShadowBake {
        if (!self.shadows_on) return .reuse;
        if (!self.were_on) return .rebake;
        if (self.map_stale) return .rebake;
        return if (self.casters_move) .rebake else .reuse;
    }
};

pub const Options = struct {
    title: [:0]const u8 = "Lenore",
    extent: platform.Extent2D = .{ .width = 1280, .height = 720 },
    present: gpu.PresentModePreference = .fifo,

    // What one frame's rings hold. The joint bound is what a level's joint plan
    // packs into, so the two are the same number and not two copies of it.
    frame_capacity: gpu.FrameCapacity = .{ .instances = 64, .joints = 256 },
    // What the morph prepass holds. The Khronos morph samples reach two
    // registrations of eight targets, and a face rig is the shape that would
    // move these.
    morph_capacity: gpu.MorphCapacity = .{ .meshes = 16, .weights = 256 },
    // The largest material count a level may carry. A capacity and not a count:
    // the descriptor table is allocated once here, so one device serves models
    // of different sizes without rebuilding a pipeline.
    material_capacity: u32 = 64,

    // The sun shadow map, square, and the number a fit is computed against.
    //
    // Provisional, and matched to the reference's own choice rather than
    // measured: what would move it is a scene whose shadow edges are too coarse
    // at this resolution beside the cost of baking it, and neither has been
    // recorded.
    shadow_map_size: u32 = 2048,

    fps_window_ns: u64 = std.time.ns_per_s,
};

pub const Engine = struct {
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
    // surface events rather than queried: a compositor tells the client its
    // size, and being told is the whole point.
    surface_extent: platform.Extent2D,
    // Set when a present or acquire reports the swapchain no longer matches the
    // surface. The recreation happens at the top of the next frame, where no
    // work is in flight against the images being replaced.
    swapchain_stale: bool,
    // Scoped to one `run`, which clears it on entry. An application that leaves
    // the loop to do something the loop cannot do, such as replace the level,
    // re-enters it afterwards, and a request that outlived the run it was made
    // in would end the next one before its first frame.
    exit_requested: bool,
    // Whether the loop blocks on the event queue instead of running free. It
    // still draws one frame per wake, because input has to keep answering while
    // paused and a frame is what answers it. Owned by the application, which is
    // the only thing that knows when there is nothing to watch.
    paused: bool,

    memory: gpu.MemoryAllocator,
    staging: gpu.StagingPool,
    setup_pool: gpu.OneShotPool,
    textures: gpu.TextureCache,
    samplers: gpu.SamplerCache,
    morph_pass: gpu.MorphPass,
    renderer: gpu.Renderer,
    materials: gpu.MaterialStorage,

    camera: scene.Camera,
    look: Look,
    sun: ?Sun,

    // Set when the map no longer describes what a lookup would read: the sun was
    // refitted, or shadows were switched back on after a spell where nothing
    // kept the map current. Cleared by the bake that answers it.
    shadow_dirty: bool,
    // What the previous frame decided, so the off-to-on edge is detectable. A
    // level or a look can change between any two frames.
    shadows_were_on: bool,

    frame_clock: FrameClock,
    fps: FpsCounter,
    // This frame's phases, and the window they are accumulating into. The
    // window is reported beside the rate and reset with it.
    phases: FramePhases,
    phase_window: FramePhases,
    last_phases: FramePhases,
    // The last window the counter closed, or null until one has. A driver reads
    // it rather than being called back, because a rate is a thing to look at
    // when convenient and not an event.
    last_fps: ?FpsCounter.Report,
    // How many windows have closed. A driver that reports each one needs to
    // know whether what it is looking at is new, and the report's own numbers
    // are not an identity: two windows can measure the same duration.
    fps_windows: u64,
    // Frames that reached `record`. Not the presented count: a present that
    // fails leaves the frame recorded, and everything the recording counted
    // happened either way.
    recorded_frames: u64,
    presented_frames: u64,

    // Initialization is in place because native callbacks retain &self.input.
    // The engine must not move after the window captures that address.
    pub fn init(self: *Engine, allocator: Allocator, options: Options) !void {
        self.allocator = allocator;
        self.io_threaded = .init(allocator, .{});
        errdefer self.io_threaded.deinit();
        self.io = self.io_threaded.io();
        self.clock = .init(self.io);

        self.platform_host = try .init();
        errdefer self.platform_host.deinit();
        self.window = try self.platform_host.createWindow(options.extent, options.title);
        errdefer self.window.deinit();

        // The compositor tells a client its size and nothing here can ask.
        // Without reading those events the swapchain keeps the extent it was
        // created with, and the camera's aspect stops matching the window.
        self.input = try .init(
            allocator,
            self.clock,
            platform.initial_event_capacity,
            platform.max_event_capacity,
        );
        errdefer self.input.deinit();
        self.window.captureInput(&self.input);

        self.context = try .init(allocator, options.title, self.window.nativeHandles());
        errdefer self.context.deinit();

        self.swapchain = try .init(&self.context, allocator, options.extent, options.present);
        errdefer self.swapchain.deinit();

        var created: usize = 0;
        errdefer for (self.frames[0..created]) |frame| frame.deinit(&self.context);
        for (&self.frames) |*frame| {
            frame.* = try .init(&self.context);
            created += 1;
        }

        self.memory = try .init(&self.context, allocator, self.io, .{
            .device_buffer_block_size = 64 << 20,
            .device_image_block_size = 64 << 20,
            .upload_buffer_block_size = 16 << 20,
            .readback_buffer_block_size = 8 << 20,
        });
        errdefer _ = self.memory.deinit();

        // One pool for the whole run: every level, the fallbacks and the
        // environments draw on it in turn, and it is reclaimed by whichever
        // transfer finishes.
        self.staging = try .init(&self.context, &self.memory, allocator, .{});
        errdefer self.staging.deinit();

        self.setup_pool = try .init(&self.context);
        errdefer self.setup_pool.deinit(&self.context);

        var cache_setup: gpu.Transfer = try .begin(&self.context, self.setup_pool.handle, &self.staging);
        self.textures = try .init(&self.context, &self.memory, allocator, &cache_setup);
        errdefer _ = self.textures.deinit();
        try cache_setup.finish();
        cache_setup.deinit();

        self.samplers = .init(&self.context);
        errdefer self.samplers.deinit(allocator);
        const post_sampler = try self.samplers.get(allocator, .{
            .address_mode_u = .clamp_to_edge,
            .address_mode_v = .clamp_to_edge,
        });

        // Built even for a run whose models have no morph targets: it owns a
        // pipeline and a descriptor pool rather than per-model state, and a pass
        // created conditionally would be a second path through the frame loop
        // that the sample corpus barely exercises.
        self.morph_pass = try .init(
            &self.context,
            &self.memory,
            allocator,
            frames_in_flight,
            options.morph_capacity,
            shaders.morph,
        );
        errdefer self.morph_pass.deinit();

        const extent = self.swapchain.currentExtent();
        self.renderer = try .init(
            &self.context,
            &self.memory,
            allocator,
            .{ .width = extent.width, .height = extent.height },
            frames_in_flight,
            options.frame_capacity,
            options.material_capacity,
            self.swapchain.surface_format.format,
            post_sampler,
            options.shadow_map_size,
            shaders.renderer,
        );
        errdefer self.renderer.deinit();

        self.materials = try .init(&self.context, &self.memory, options.material_capacity);
        errdefer self.materials.deinit();

        self.frame_index = 0;
        self.surface_extent = extent;
        self.swapchain_stale = false;
        self.exit_requested = false;
        self.paused = false;
        self.camera = .{
            .anchor = .{ .orbit = .{ .target = @splat(0), .distance = 0.01 } },
            .yaw = -std.math.pi / 4.0,
            .pitch = -0.35,
        };
        self.look = .{};
        self.sun = null;
        self.shadow_dirty = true;
        self.shadows_were_on = false;
        self.frame_clock = .start(self.clock);
        self.fps = .init(options.fps_window_ns);
        self.last_fps = null;
        self.fps_windows = 0;
        self.phases = .{};
        self.phase_window = .{};
        self.last_phases = .{};
        self.recorded_frames = 0;
        self.presented_frames = 0;
    }

    // Vulkan specification, vkDestroyDevice: every use must have completed, and
    // a fence does not establish that. Presentation is queued after the
    // submission a fence covers and still holds the swapchain image and the
    // semaphore it waited on, so the whole device has to drain.
    pub fn deinit(self: *Engine) void {
        self.context.waitIdle() catch |err| {
            log.err("device did not drain during teardown: {t}", .{err});
        };

        self.materials.deinit();
        self.renderer.deinit();
        self.morph_pass.deinit();
        self.samplers.deinit(self.allocator);
        // Both report whether anything was still held, and the answer is a
        // return value nothing else can see. A reference outstanding here is an
        // image the run never gave back, and device memory is the one
        // allocation a host leak check does not cover.
        if (self.textures.deinit() == .leak) log.err("texture references outstanding", .{});
        self.setup_pool.deinit(&self.context);
        self.staging.deinit();
        if (self.memory.deinit() == .leak) log.err("device memory leaked", .{});

        for (self.frames) |frame| frame.deinit(&self.context);
        self.swapchain.deinit();
        self.context.deinit();

        self.window.deinit();
        self.input.deinit();
        self.platform_host.deinit();
        self.io_threaded.deinit();
        self.* = undefined;
    }

    // What a level needs from the device that outlives it.
    pub fn deps(self: *Engine) level_module.Deps {
        return .{
            .io = self.io,
            .clock = self.clock,
            .context = &self.context,
            .memory = &self.memory,
            .staging = &self.staging,
            .setup_pool = &self.setup_pool,
            .textures = &self.textures,
            .morph_pass = &self.morph_pass,
        };
    }

    // Installs a level and points the camera at it.
    //
    // The sun is resolved here rather than per frame because it is a function of
    // the level's extent and the lights, and neither moves inside a frame. Set
    // `look.lights` before calling: what lights a scene is the application's,
    // and the fit is taken from the block the shader will read rather than from
    // a second copy of it.
    pub fn install(
        self: *Engine,
        level: *const Level,
        model: *const @import("lenore-gltf").importer.Model,
    ) !void {
        try level.install(model, &self.renderer, &self.materials, &self.textures);

        self.sun = null;
        self.refitSun(level.world.sphere);
        _ = self.frameCamera(level.world.sphere);
    }

    // Releases a level and returns the device to a state a different one can be
    // installed into.
    //
    // Three things outlive a level and hold something of it. The prepass keeps a
    // registration and a destination buffer per morphed mesh; the renderer keeps
    // a record per material and descriptor sets pointing at that level's images;
    // the caches hold the images themselves. All three are given back here, and
    // the drain comes first because every one of them destroys something a
    // submitted frame may still be reading.
    pub fn unload(self: *Engine, level: *Level) void {
        self.context.waitIdle() catch |err| {
            log.err("device did not drain before unloading: {t}", .{err});
        };
        self.morph_pass.reset();
        self.renderer.clearMaterials();
        level.deinit(&self.textures);
        self.sun = null;
        // Nothing is installed, so nothing can be recorded until something is.
        // `Renderer.plan` enforces that on its own account; this is the map's
        // half of the same statement.
        self.shadow_dirty = true;
    }

    // Re-fits the shadow map to the first directional light in the block.
    //
    // Cheap enough to call whenever the sun may have moved. A fit that still
    // lines up is kept: `stale` measures the drift against a one-texel budget at
    // this resolution, so a sun nudged by a fraction of a texel costs nothing
    // and the map is not re-baked for a difference no texel can hold.
    //
    // The light block itself is the caller's, and moving a light there changes
    // the shading with no fit and no bake at all. Only the map needs this.
    pub fn refitSun(self: *Engine, sphere: res.Sphere) void {
        // The first directional light, so the index carried to the device and
        // the direction the fit was built from name the same entry rather than
        // two that agree by order.
        for (self.look.lights, 0..) |light, index| {
            if (light.kind != .directional) continue;
            // Toward the sun, which is what a fit takes; a light's own
            // direction is the way it travels.
            const travel = light.direction;
            const toward: res.Vec3 = .{ -travel[0], -travel[1], -travel[2] };

            if (self.sun) |current| {
                if (current.index == index and !current.fit.stale(toward)) return;
            }

            const fit = scene.SunShadowFit.compute(
                sphere,
                toward,
                self.renderer.shadowMapSize(),
            ) catch |err| {
                // A scene with no extent, or a light with no direction. One
                // light less of a shadow is still a frame, and which one it was
                // is not recoverable from anywhere else.
                log.warn("sun shadow fit rejected: {t}", .{err});
                self.sun = null;
                return;
            };
            self.sun = .{ .index = @intCast(index), .fit = fit };
            self.shadow_dirty = true;
            return;
        }
        self.sun = null;
    }

    // Fits the camera around a sphere and returns the distance it stood off at.
    //
    // A vertical field of view narrows horizontally below aspect one, so the
    // distance grows by the inverse aspect there and the same sphere stays
    // inside both axes when a compositor gives the window a portrait extent.
    pub fn frameCamera(self: *Engine, sphere: res.Sphere) f32 {
        const distance = @max(sphere.radius * 3.0 / @min(self.aspect(), 1.0), 0.01);
        self.camera.anchor = .{ .orbit = .{
            .target = .{ sphere.centre[0], sphere.centre[1], sphere.centre[2] },
            .distance = distance,
        } };
        self.camera.projection = .{ .perspective = .{
            .near = @max(distance * 0.01, 1.0e-4),
            .far = distance * 10.0,
        } };
        return distance;
    }

    pub fn aspect(self: *const Engine) f32 {
        const extent = self.swapchain.currentExtent();
        return @as(f32, @floatFromInt(extent.width)) / @as(f32, @floatFromInt(extent.height));
    }

    // Ends the loop after the frame in progress. A window closing does the same
    // thing; this is for an application that has seen what it came for, or that
    // wants the loop back to do work between runs. Why it ended is the caller's
    // to record: the engine knows only that it was asked.
    pub fn requestExit(self: *Engine) void {
        self.exit_requested = true;
    }

    pub fn run(self: *Engine, level: *Level, driver: anytype) !void {
        self.exit_requested = false;

        // One timer for the whole run rather than one per frame. A timer built
        // at the top of a frame cannot measure what happened before it was
        // built, and what happened there is every wait the loop does outside a
        // phase: the phases would then sum to a fraction of the frame and the
        // report would say the host was idle when it was blocked.
        var phase: time_module.PhaseTimer = .begin(self.clock);

        while (!self.window.shouldClose() and !self.exit_requested) {
            // Blocking here rather than at the end of the frame is what makes a
            // pause cost nothing: the loop sleeps until the compositor has
            // something, then runs one whole iteration on it. The frame that
            // iteration draws is what keeps the camera and the toggles alive.
            if (self.paused) self.platform_host.waitEvents();
            self.platform_host.pollEvents();
            try self.drainInput(driver);
            if (self.exit_requested) break;

            // A minimised window has nothing to present to, and a swapchain
            // cannot be created at a zero extent. Blocking on the event queue
            // rather than spinning is what keeps an idle window free.
            if (self.surface_extent.width == 0 or self.surface_extent.height == 0) {
                self.platform_host.waitEvents();
                continue;
            }
            if (self.swapchain_stale or !self.swapchain.matchesExtent(self.surface_extent))
                try self.recreate(driver);

            self.phases = .{};
            self.phases.events_ns = phase.split();

            const frame = self.frames[self.frame_index];
            try frame.waitForGpu(&self.context);
            self.phases.wait_ns = phase.split();

            // An acquire that fails this way has signalled nothing and consumed
            // nothing, and the fence is still signalled because only a
            // submission resets it, so the loop can run again.
            const acquired = self.swapchain.acquireNextImage(frame.image_acquired) catch |err| switch (err) {
                error.OutOfDateKHR => {
                    self.swapchain_stale = true;
                    continue;
                },
                else => return err,
            };
            if (acquired.state == .suboptimal) self.swapchain_stale = true;
            self.phases.acquire_ns = phase.split();

            const extent = self.swapchain.currentExtent();
            const ratio = self.aspect();
            const view_projection = try self.camera.viewProjection(ratio);
            const ray_basis = try self.camera.rayBasis(ratio);
            const eye = self.camera.placement().position;

            // One reading of the clock per frame, and the only one. Everything
            // that advances takes the delta it produced.
            const time = self.frame_clock.tick();
            // The same matrix the camera block below is filled from, so what a
            // draw is culled against and what it is transformed by cannot come
            // from two different cameras.
            const state = try level.world.update(
                &self.morph_pass,
                self.frame_index,
                time,
                eye,
                view_projection,
            );
            self.phases.world_ns = phase.split();

            try self.renderer.update(self.frame_index, .{
                // The scene's clip space goes in and the framebuffer's comes
                // out. The conversion is named here rather than performed
                // inside the renderer, so the value a frame is filled with says
                // which of the two it is.
                .camera = gpu.vulkanClipCamera(.{
                    .view_projection = view_projection,
                    .position = .{ eye[0], eye[1], eye[2], 1 },
                    // The same pose and the same aspect the matrix above was
                    // built from, so the background's rays and the geometry's
                    // clip coordinates cannot describe two different cameras.
                    .ray_right = lane(ray_basis.right),
                    .ray_up = lane(ray_basis.up),
                    .ray_front = lane(ray_basis.front),
                }),
                .models = state.instances,
                .joints = state.joints,
                .lights = self.look.lights,
                // Off unless a directional light was found and fitted. The
                // block's strength is what the shader tests, so an absent sun is
                // one value rather than a branch on both sides.
                .sun_shadow = if (self.sun) |sun| .{
                    .view_projection = sun.fit.view_proj,
                    .strength = self.look.sun_shadow.clampedStrength(),
                    .normal_offset = self.look.sun_shadow.normalOffsetWorld(&sun.fit),
                    .light = sun.index,
                } else .off,
            });

            self.phases.update_ns = phase.split();

            const commands = try frame.beginCommands(&self.context);
            // Before any rendering is opened: a dispatch cannot be recorded
            // inside one, and the barrier the prepass ends with orders its
            // writes against the vertex fetch of the draws below.
            self.morph_pass.record(commands, self.frame_index);

            // Everything a frame can be refused for is refused here, before a
            // single command is recorded. What comes back is what every stage
            // below takes, so a frame that failed cannot be half drawn.
            const frame_plan = try self.renderer.plan(.{
                .frame_index = self.frame_index,
                // Every batch, and how many of them the camera can see. The main
                // pass draws that prefix and the shadow bake draws all of them:
                // a caster the camera cannot see still casts into what it can.
                .batches = state.records,
                .visible = state.visible_records,
                .background = self.look.background,
                .bloom = self.look.bloom,
                .post = self.look.post,
            });
            // The frame, stage by stage. This sequence is the engine's and not
            // the renderer's: which passes a frame is made of, and in what
            // order, is composition. Each stage carries its own precondition,
            // and they are met by this order alone.
            //
            // The bake first, because it opens a rendering of its own and the
            // map it writes is sampled by every fragment the main pass shades.
            self.renderer.recordShadowBake(commands, self.shadowBake(level), frame_plan);

            self.renderer.beginMain(commands);
            self.renderer.recordScene(commands, frame_plan);
            // Inside the main pass and after the scene, which is the only place
            // an application's own draws can go without disturbing it: binding
            // a descriptor set through a different pipeline layout invalidates
            // what is already bound, and the scene's sets have to be the last
            // word for the draws that read them.
            //
            // What a driver records here writes the HDR target and is tone
            // mapped with everything else, so it is in scene radiance and not
            // in display colour. The chain and the operator run after it.
            //
            // A driver that records nothing costs the call and no commands.
            try driver.onRecord(self, level, commands);
            self.renderer.endMain(commands);

            // Between the two passes: the chain reads what the main pass wrote
            // and the post pass reads what the chain leaves.
            self.renderer.recordBloom(commands, frame_plan);
            self.renderer.recordPost(
                commands,
                .{
                    .image = self.swapchain.images[acquired.image_index].image,
                    .view = self.swapchain.images[acquired.image_index].view,
                    .extent = .{ .width = extent.width, .height = extent.height },
                },
                frame_plan,
            );
            self.recorded_frames += 1;
            self.phases.record_ns = phase.split();

            self.phase_window.add(self.phases);
            if (self.fps.record(time)) |report| {
                self.last_fps = report;
                self.last_phases = self.phase_window.mean(report.frames);
                self.phase_window = .{};
                self.fps_windows += 1;
            }
            try driver.onFrame(self, level, time);

            try frame.submit(&self.context, .{
                .wait = frame.image_acquired,
                // The first thing the frame does to the presentable image is
                // render into it, which the post pass's barrier prepares.
                .wait_stage = .{ .color_attachment_output_bit = true },
                .signal = try self.swapchain.renderFinishedSemaphore(acquired.image_index),
                // After everything, including the transition to the presentable
                // layout. A tighter stage would signal before it completed.
                .signal_stage = .{ .all_commands_bit = true },
            });
            self.phases.submit_ns = phase.split();

            const presented = self.swapchain.present(acquired.image_index) catch |err| switch (err) {
                error.OutOfDateKHR => {
                    self.swapchain_stale = true;
                    continue;
                },
                else => return err,
            };
            if (presented == .suboptimal) self.swapchain_stale = true;

            self.phases.present_ns = phase.split();
            self.frame_index = (self.frame_index + 1) % frames_in_flight;
            self.presented_frames += 1;
        }

        try self.context.waitIdle();
    }

    fn shadowBake(self: *Engine, level: *const Level) gpu.ShadowBake {
        const request: ShadowBakeRequest = .{
            .shadows_on = self.look.sun_shadow.enabled and self.sun != null,
            .were_on = self.shadows_were_on,
            .map_stale = self.shadow_dirty,
            .casters_move = level.world.castersMove(),
        };
        self.shadows_were_on = request.shadows_on;
        if (request.decide() == .rebake) self.shadow_dirty = false;
        return request.decide();
    }

    fn drainInput(self: *Engine, driver: anytype) !void {
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
            try driver.onEvent(self, event);
        }
    }

    // Vulkan specification, vkDestroySwapchainKHR: every use of an acquired
    // image must have completed, which includes the presentation queued after
    // the last submission. A fence covers the submission and not the
    // presentation, so this drains the device instead. Resizing is cold enough
    // to pay for it.
    fn recreate(self: *Engine, driver: anytype) !void {
        try self.context.waitIdle();
        try self.swapchain.recreate(self.surface_extent);

        const extent = self.swapchain.currentExtent();
        try self.renderer.resize(.{ .width = extent.width, .height = extent.height });
        self.swapchain_stale = false;

        try driver.onResize(self, extent);
    }
};

// A direction in the four lanes a uniform block's vector field has. The last one
// is the alignment's padding and is read by nothing.
fn lane(direction: res.Vec3) [4]f32 {
    return .{ direction[0], direction[1], direction[2], 0 };
}
