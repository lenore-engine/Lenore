const std = @import("std");
const gpu = @import("lenore-gpu");
const imui = @import("lenore-imui");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const font_module = @import("font.zig");
const level_module = @import("level.zig");
// The shading every pass below is built from. Named here rather than taken as
// an option, because this is the engine's own composition and what it draws
// with is the engine's answer. An application wanting another one builds the
// passes itself.
const shaders = @import("shaders.zig");
const time_module = @import("time.zig");
const translate = @import("translate.zig");
const ui_module = @import("ui.zig");

const Allocator = std.mem.Allocator;
const log = std.log.scoped(.lenore);
const FrameClock = time_module.FrameClock;
const FrameMetrics = time_module.FrameMetrics;
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
// the frame is drawn with, taken once below the last hook that can write it. A
// driver is a value with the hooks below, called at the one point in the frame
// where each is answerable.

// How many frames the host may be preparing while the device works on earlier
// ones. Two lets the host record the next frame while the device finishes the
// current one, without the latency a deeper queue adds.
const frames_in_flight = 2;

// What runs on top of the loop. Every driver declares all six, and they are
// called at the one point in the frame where each is answerable:
//
//   onUiRegions(driver, engine, level, ui) !void
//       Every interactive area the frame has, in painter's order. Called
//       before the frame's events are routed, which is what lets them be
//       routed against geometry that is this frame's rather than last one's.
//       Nothing may be drawn here and nothing read: what a widget got out of
//       the frame is not knowable until every region exists.
//   onEvent(driver, engine, event, ui_consumed) !void
//       Every input event, with what the UI did about it. `ui_consumed` is
//       this frame's answer and not the previous frame's, so an application
//       that suppresses its own binding on it does so on the frame the click
//       lands.
//   onResize(driver, engine, extent) !void
//       After the swapchain and the render targets have followed the surface.
//   onUpdate(driver, engine, level, time) !void
//       The frame's simulation, between the clock's one reading and the camera
//       the frame is built from. Everything a driver advances with the delta
//       goes here, and everything it wants this frame drawn with: the look,
//       the lights and the camera are all read below it.
//   onCompute(driver, engine, level, commands) !void
//       Before any rendering opens. Compute work ends with its own dependency
//       from storage writes to the stage that consumes them.
//   onRecord(driver, engine, level, commands) !void
//       Inside the main pass, after the scene and before it closes. See the
//       call site for what may be recorded there and what may not.
//   onUiDraw(driver, engine, level, ui) !void
//       The overlay's geometry, built into the frame's own rings, and where a
//       widget reads what the routing decided. It carries no command buffer
//       and records nothing: what is appended here is drawn later, over the
//       finished picture, in display colour.
//
// Nothing is called between the recording and the submission, because there is
// nothing a driver could only answer there. The device has produced nothing at
// that point: the command buffer is still open, `Frame.submit` is what ends it,
// and what the recording cost the device is read a ring apart, where this
// slot's fence next signals. What the recording moves on the host is counters
// that outlive the frame, so a driver reading one at the top of the next frame
// reads the same number.
//
// What a hook writes reaches the frame through two readings and no others. The
// camera is read below `onUpdate`, into the matrices the scene is culled and
// transformed by; the look is read below `onUiDraw`, once, for every stage that
// draws. A write above its reading is in this frame and a write below it is in
// the next.
//
// Stated and not enforced, which is a decision. A write on the wrong side of a
// reading costs a frame of latency and never a broken frame, and nothing can
// tell the two cases apart: a driver deliberately setting up the next frame
// writes exactly what a driver that missed the deadline writes. The overlay's
// own phases are refused at runtime because a registration in the wrong pass
// corrupts the widget tree, which is a different kind of mistake and gets a
// different answer.
//
// The UI is two hooks rather than one because the loop does its own work
// between them. Registration, routing and drawing are three passes over the
// same widget tree in a fixed order, and the middle one is the engine's: it
// drains the platform's queue, translates it and routes it. A single hook
// would have to be handed the batch and trusted to route it, and a driver that
// simply did not would draw a UI that never responds, with nothing to report.
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
    pub fn onUiRegions(_: *NoDriver, _: *Engine, _: *Level, _: *imui.WidgetContext) !void {}
    pub fn onEvent(_: *NoDriver, _: *Engine, _: platform.Event, _: bool) !void {}
    pub fn onResize(_: *NoDriver, _: *Engine, _: platform.Extent2D) !void {}
    pub fn onUpdate(_: *NoDriver, _: *Engine, _: *Level, _: FrameTime) !void {}
    pub fn onCompute(_: *NoDriver, _: *Engine, _: *Level, _: gpu.vk.CommandBuffer) !void {}
    pub fn onRecord(_: *NoDriver, _: *Engine, _: *Level, _: gpu.vk.CommandBuffer) !void {}
    pub fn onUiDraw(_: *NoDriver, _: *Engine, _: *Level, _: *imui.WidgetContext) !void {}
};

// What every frame is drawn with.
//
// The frame copies it below the last hook that can write it and reads nothing
// but the copy afterwards. That is what gives the struct one deadline instead
// of one per field: the lights and the shadow settings are wanted where the
// rings are filled, the background, the bloom and the post operator where the
// frame is planned, and the shadow decision later still.
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

    // The three capacities below take zero for "this application draws none of
    // these", which is what an application with no asset means. The engine
    // raises each to one rather than refusing it: the per-frame rings and the
    // material storage both decline a zero capacity on their own account, and
    // an application should say what it draws instead of the smallest figure
    // that gets past them. One element per frame slot is what it costs.

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

    // What one frame's overlay may hold. The defaults are the pass's own, and
    // an application that draws more than a few panels sets its own.
    ui_capacity: gpu.UiCapacity = .{},
    // What one frame's widget tree may hold. Separate from `ui_capacity`
    // because these are the UI module's arrays: a renderer has no word for a
    // region, and a capacity it cannot use does not belong in its struct.
    ui_widgets: ui_module.Capacity = .{},
    // The faces and the coverage atlas. Built whether or not an application
    // loads a font, for the reason the morph prepass is: the alternative is a
    // second path through the frame loop. What it costs an application that
    // draws no text is the atlas twice over, once on the host and once on the
    // device, plus one staging copy of it per frame slot.
    font_capacity: font_module.Capacity = .{},

    fps_window_ns: u64 = std.time.ns_per_s,

    // Whether the frame writes timestamps around its passes. Off by default:
    // the numbers have one reader, and a frame that has none should record no
    // commands for it.
    gpu_timing: bool = false,
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
    // Declared before the subsystems that borrow it and torn down after them,
    // which is the ordering the borrow requires. Lifetimes are the engine's to
    // own, not a subsystem's: the ring this is keyed to is the one the frame
    // loop below drives, and the texture cache and the morph prepass both hand
    // their device resources to it.
    retirement: gpu.ResourceRetirement,
    textures: gpu.TextureCache,
    samplers: gpu.SamplerCache,
    morph_pass: gpu.MorphPass,
    ui_pass: gpu.UiPass,
    // The neutral image a solid fill samples. Registered once, from the texture
    // cache's own white texel, so the overlay creates no image of its own to
    // draw a rectangle.
    ui_white: res.ImageHandle,

    // The fonts, and the one image every face's coverage is packed into.
    //
    // One image for all of them, because a draw command names an image: a
    // second one is a second command, and a label in two weights would cost two
    // draws instead of one. It holds four channels of coverage per texel, which
    // is what lets one pipeline draw a solid fill, a grayscale glyph and a
    // glyph measured per colour stripe with no branch between them.
    fonts: font_module.Fonts,
    glyph_image: gpu.Image,
    // One copy of the atlas per frame slot. Sized to the whole of it so that a
    // dirty region always fits and there is no second upload path for the case
    // where it does not: loading a face dirties most of the atlas at once, and
    // splitting that across frames would be a mechanism serving one event.
    glyph_staging: gpu.PerFrame(u32),
    glyph_atlas: res.ImageHandle,

    // The overlay's host side: the widget tree's arrays, the façade over them
    // and the translation from what the window says. It names no device, and
    // the memory a frame's geometry goes into is handed to it per frame from
    // the pass above.
    //
    // It is why the engine may not move after `init`, along with the input
    // callbacks: it holds pointers into its own fields.
    ui: ui_module.Host,
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
    // Where the host time of a frame went and how fast the frames came. A
    // driver reads its last window rather than being called back, because a
    // rate is a thing to look at when convenient and not an event.
    metrics: FrameMetrics,
    // Null where the device cannot carry a timestamp on the graphics queue, or
    // where nothing asked for one. Absent rather than idle: eight timestamps a
    // frame is device work, and a frame nobody is measuring should not pay it.
    //
    // Apart from `metrics`, which reads no clock and touches no device. Where
    // the marks below go is which passes a frame is made of, and that is this
    // file's answer rather than a measurement's.
    gpu_timer: ?gpu.GpuTimer,
    // The last frame slot's decomposition, read where its fence signalled.
    // Null until a frame has been submitted and come back.
    last_gpu: ?gpu.GpuTimings,
    // Frames that reached `record`. Not the presented count: a present that
    // fails leaves the frame recorded, and everything the recording counted
    // happened either way.
    recorded_frames: u64,
    presented_frames: u64,

    // Initialization is in place because native callbacks retain &self.input.
    // The engine must not move after the window captures that address.
    pub fn init(self: *Engine, allocator: Allocator, options: Options) !void {
        // Zero means none, and none is served by the smallest ring the modules
        // below will build. Done once here rather than at each of the four use
        // sites, so an application that asks for nothing cannot get a ring of
        // one in some places and a refusal in others.
        const frame_capacity: gpu.FrameCapacity = .{
            .instances = @max(options.frame_capacity.instances, 1),
            .joints = @max(options.frame_capacity.joints, 1),
        };
        const morph_capacity: gpu.MorphCapacity = .{
            .meshes = @max(options.morph_capacity.meshes, 1),
            .weights = @max(options.morph_capacity.weights, 1),
        };
        const material_capacity = @max(options.material_capacity, 1);

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

        // One list per frame slot, so a resource released mid-frame is destroyed
        // once the slot that could still be reading it comes round again.
        self.retirement = try .init(allocator, frames_in_flight);
        errdefer self.retirement.deinit(allocator);

        var cache_setup: gpu.Transfer = try .begin(&self.context, self.setup_pool.handle, &self.staging);
        self.textures = try .init(
            &self.context,
            &self.memory,
            allocator,
            &cache_setup,
            &self.retirement,
        );
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
            morph_capacity,
            shaders.morph,
            &self.retirement,
        );
        errdefer self.morph_pass.deinit();

        const extent = self.swapchain.currentExtent();
        self.renderer = try .init(
            &self.context,
            &self.memory,
            allocator,
            .{ .width = extent.width, .height = extent.height },
            frames_in_flight,
            frame_capacity,
            material_capacity,
            self.swapchain.surface_format.format,
            post_sampler,
            options.shadow_map_size,
            shaders.renderer,
        );
        errdefer self.renderer.deinit();

        self.materials = try .init(&self.context, &self.memory, material_capacity);
        errdefer self.materials.deinit();

        // After the renderer, because the pipeline is built against the format
        // the swapchain chose, and that is the image the overlay is composited
        // onto rather than the HDR target the scene writes.
        self.ui_pass = try .init(
            &self.context,
            &self.memory,
            allocator,
            frames_in_flight,
            options.ui_capacity,
            shaders.ui,
            self.swapchain.surface_format.format,
        );
        errdefer self.ui_pass.deinit();

        // The texture cache's white texel, registered rather than created: it is
        // already uploaded, already pinned and never released, and a solid fill
        // is a draw that samples it.
        const white = try self.textures.fallback(.white, .{});
        self.ui_white = try self.ui_pass.registry.add(
            &self.context,
            allocator,
            white.view,
            white.sampler,
        );

        self.fonts = try .init(allocator, options.font_capacity);
        errdefer self.fonts.deinit(allocator);

        const atlas_size = options.font_capacity.atlas_size;
        self.glyph_image = try .init(&self.context, &self.memory, .{
            .width = atlas_size,
            .height = atlas_size,
            .format = .r8g8b8a8_unorm,
            .usage = .{ .sampled_bit = true, .transfer_dst_bit = true },
            // No swizzle. A glyph measured per colour stripe stores three
            // different numbers and a fourth for the pixel as a whole, so the
            // channels are read as they are; the atlas writes one coverage into
            // all four when there is only one, which is what the view used to
            // do with a mapping.
        });
        errdefer self.glyph_image.deinit();

        // A quarter of the atlas's bytes, because a copy names a byte offset
        // and `vkCmdCopyBufferToImage` requires a multiple of four: a ring of
        // `u32` carries that alignment in the type rather than in arithmetic
        // this file would have to keep true. Four channels to a texel makes the
        // division exact.
        self.glyph_staging = try .init(
            &self.context,
            &self.memory,
            frames_in_flight,
            @as(usize, atlas_size) * atlas_size * font_module.atlas_channels / 4,
            .{ .transfer_src_bit = true },
        );
        errdefer self.glyph_staging.deinit();

        const glyph_sampler = try self.samplers.get(allocator, .{
            // A glyph quad is snapped to whole pixels and sized in whole
            // texels, so the filter reads texel centres and linear returns what
            // nearest would. Clamping matters more: the atlas has one texel of
            // unwritten padding around every glyph, and a repeat mode would
            // wrap the far edge into it.
            .address_mode_u = .clamp_to_edge,
            .address_mode_v = .clamp_to_edge,
            .mipmap_mode = .nearest,
            .anisotropic = false,
        });
        self.glyph_atlas = try self.ui_pass.registry.add(
            &self.context,
            allocator,
            self.glyph_image.view,
            glyph_sampler,
        );

        // Into the layout the frame path expects to find it in. Nothing has
        // been copied, so what the image holds is undefined; that is sound
        // because a texel is only ever sampled through a placement, and a
        // placement exists only for a glyph whose rows have been uploaded.
        var atlas_setup: gpu.Transfer = try .begin(&self.context, self.setup_pool.handle, &self.staging);
        self.glyph_image.recordLayoutTransition(
            atlas_setup.commandBuffer(),
            .to_transfer_destination,
        );
        self.glyph_image.recordLayoutTransition(
            atlas_setup.commandBuffer(),
            .toShaderRead(.{ .fragment_shader_bit = true }),
        );
        try atlas_setup.finish();
        atlas_setup.deinit();

        // Over the first slot, which the loop replaces with the slot it is
        // about to fill before anything is appended.
        try self.ui.init(
            allocator,
            options.ui_widgets,
            self.ui_pass.storage(0),
            self.input.inputState().metrics,
            self.ui_white,
        );
        errdefer self.ui.deinit(allocator);

        // Refused rather than faked where the device says a timestamp on this
        // queue is worth nothing: a pool that cannot be written would report
        // every pass as taking no time, which reads like an answer.
        self.gpu_timer = null;
        self.last_gpu = null;
        if (options.gpu_timing) {
            const support = self.context.timestampSupport();
            if (support.available()) {
                self.gpu_timer = try .init(&self.context, support, frames_in_flight);
            } else {
                log.warn(
                    "device timings were asked for; this queue reports {d} valid timestamp bits",
                    .{support.valid_bits},
                );
            }
        }
        errdefer if (self.gpu_timer) |*timer| timer.deinit(&self.context);

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
        self.metrics = .init(options.fps_window_ns);
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

        if (self.gpu_timer) |*timer| timer.deinit(&self.context);
        self.materials.deinit();
        self.renderer.deinit();
        self.ui_pass.deinit();
        self.ui.deinit(self.allocator);
        // After the pass, whose registry held the view and the sampler these
        // two are, and after the drain at the top of this function, which is
        // what makes destroying an image the last frame sampled correct.
        self.glyph_staging.deinit();
        self.glyph_image.deinit();
        self.fonts.deinit(self.allocator);
        self.morph_pass.deinit();
        self.samplers.deinit(self.allocator);
        // Both report whether anything was still held, and the answer is a
        // return value nothing else can see. A reference outstanding here is an
        // image the run never gave back, and device memory is the one
        // allocation a host leak check does not cover.
        if (self.textures.deinit() == .leak) log.err("texture references outstanding", .{});
        // After the cache, because releasing anything it still held would have
        // gone here. The device drained at the top of this function, which is
        // what makes destroying the remainder correct: at teardown there is no
        // later frame for it to wait for.
        //
        // Nothing is reported. A run that changed levels in its last frames ends
        // with resources still queued, which is the type working rather than a
        // fault, and a warning that fires on every clean exit is one nobody
        // reads.
        self.retirement.deinit(self.allocator);
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
    // the caches hold the images themselves. All three are given back here.
    //
    // The drain is for the meshes, and only for them. Images and the prepass's
    // destinations go to the frame ring; `Mesh.deinit` still destroys up to six
    // buffers inline, and any of them may be one a submitted frame is reading.
    //
    // It stays that way by decision rather than by omission. Meshes are
    // destroyed here and on a failed load, both inside a level change, which is
    // already a moment of reading files and compressing textures; a drain there
    // is not where stutter comes from. Retiring them would mean changing
    // `OwningStorage`, whose stated contract is that it destroys what it holds,
    // to buy nothing that can be measured.
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

    // Gives a device resource back, to be destroyed once no frame can be reading
    // it. Takes ownership: the value is moved in and the caller's copy is spent.
    //
    // This is how an application frees a buffer or an image it made itself, and
    // it is correct from any hook. Destroying one directly is correct only where
    // the device happens to be idle, which is true inside `onResize` because
    // `recreate` drains first and is true nowhere else the application can see.
    // Nothing in a signature says which is which, so this exists to make the
    // question not arise.
    //
    // Infallible for the reason `TextureCache.release` is: teardown paths have
    // nowhere to report, and what happens when the queue will not take the
    // resource is stated once, at `gpu.retireOrDestroy`.
    pub fn retire(self: *Engine, resource: gpu.RetiredResource) void {
        gpu.retireOrDestroy(&self.retirement, self.allocator, resource);
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

    // What a driver lays its UI out against: logical units multiplied by this
    // are framebuffer pixels, which is the space a region is tested in.
    //
    // Identity until the window has described itself. That is the same picture
    // an unscaled display gives, so a driver that ignores the distinction is
    // wrong only on a scaled output and not before the first frame.
    pub fn uiScale(self: *const Engine) imui.ScaleFactor {
        const metrics = self.input.inputState().metrics orelse return .identity;
        return translate.uiScale(metrics) orelse .identity;
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
            // The surface as the compositor last described it. The pump folds
            // metrics into the polled state as it delivers them, so this is
            // current without the batch having been drained, and the batch is
            // not drained until the regions it will be routed against exist.
            if (self.input.inputState().metrics) |metrics|
                self.surface_extent = metrics.framebuffer_extent;

            // A minimised window has nothing to present to, and a swapchain
            // cannot be created at a zero extent. Blocking on the event queue
            // rather than spinning is what keeps an idle window free.
            if (self.surface_extent.width == 0 or self.surface_extent.height == 0) {
                self.platform_host.waitEvents();
                continue;
            }
            if (self.swapchain_stale or !self.swapchain.matchesExtent(self.surface_extent))
                try self.recreate(driver);

            self.metrics.beginFrame();
            self.metrics.record(.events, phase.split());

            const extent = self.swapchain.currentExtent();
            // The overlay's first two passes, and the reason they sit above the
            // fence rather than beside the drawing.
            //
            // Nothing here writes the frame slot. `Canvas.begin` touches no
            // vertex or index memory, registration emits no geometry, and the
            // clip array both do write is ordinary host memory rather than one
            // of the device-read rings. What `PerFrame` forbids before a slot's
            // fence has signalled is writing that slot.
            //
            // What the placement buys is that every event is routed against
            // this frame's regions before the driver hears about it. A driver
            // reading the previous frame's answer instead is wrong in exactly
            // the case that matters: a panel that has just appeared or moved
            // under the pointer takes the click and starts a camera drag with
            // it, and the drag lasts until the button comes back up.
            try self.ui.beginFrame(self.ui_pass.storage(self.frame_index), .{
                .x = 0,
                .y = 0,
                .width = @floatFromInt(extent.width),
                .height = @floatFromInt(extent.height),
            });
            {
                // A frame that fails here stops in a pass `beginFrame` refuses,
                // so without this the next `run` reports `InvalidPhase` and the
                // real failure, a widget tree too large for its capacity or a
                // scope left open, is the one nobody sees. Leaving the loop and
                // re-entering it is ordinary: it is how a level is replaced.
                errdefer self.ui.abandonFrame();
                try driver.onUiRegions(self, level, &self.ui.widgets);
                try self.ui.beginRouting();
                try self.drainInput(driver);
                try self.ui.finishRouting();
            }
            self.metrics.record(.ui, phase.split());

            // Here and not above the routing, which is the phase machine's
            // doing rather than a preference: `beginFrame` refuses a context
            // that is still registering or routing, so a frame left in either
            // would take the next `run` down with it. Leaving the loop to swap
            // a level and re-entering it is what the walker does.
            if (self.exit_requested) break;

            const frame = self.frames[self.frame_index];
            try frame.waitForGpu(&self.context);
            // The fence is what makes this safe: everything this slot submitted
            // has completed, so anything retired while it was recording can go.
            // Immediately after the wait rather than later in the frame, so a
            // texture released this frame is not held for an extra round.
            self.retirement.beginFrame(self.frame_index);
            // The fence this slot was submitted with has signalled, so its
            // queries hold results and reading them waits for nothing. Taken
            // before anything overwrites the slot, which the reset below does.
            if (self.gpu_timer) |*timer| self.last_gpu = timer.read(&self.context, self.frame_index);
            self.metrics.record(.wait, phase.split());

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
            self.metrics.record(.acquire, phase.split());

            // One reading of the clock per frame, and the only one. Everything
            // that advances takes the delta it produced.
            //
            // Below the acquire, which is the last point a frame can be
            // abandoned from. A tick taken above one and then thrown away
            // takes its interval with it: the world would advance by less than
            // the wall clock, and the rate would read high by whatever the
            // abandoned frames cost.
            const time = self.frame_clock.tick();

            // The frame's simulation, and the reason it is here rather than
            // after the recording. Everything below reads what it writes: the
            // camera the frame is built from, the lights, and the look every
            // pass is planned against. A driver advancing them after the
            // recording instead would have each frame drawn from the state
            // before the input that produced it, at every rate.
            //
            // After the wait and the acquire rather than before them. Under
            // FIFO most of a frame is spent in `wait_ns`, so simulating above
            // it would hand the recording a state that much older, which is
            // the latency this placement exists to remove.
            try driver.onUpdate(self, level, time);
            self.metrics.record(.update, phase.split());

            const ratio = self.aspect();
            const view_projection = try self.camera.viewProjection(ratio);
            const ray_basis = try self.camera.rayBasis(ratio);
            const eye = self.camera.placement().position;

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
            self.metrics.record(.world, phase.split());

            // The overlay's geometry, written straight into this frame slot's
            // rings. It has to be here and cannot be earlier: the slot is only
            // safe to write once its fence has signalled, which `waitForGpu`
            // above established, and it has to be before the recording that
            // reads it.
            //
            // What a widget reads here is this frame's result. The regions were
            // registered before the fence and the frame's events have already
            // been routed against them, so a click and what it does appear in
            // the same picture rather than one apart.
            try driver.onUiDraw(self, level, &self.ui.widgets);

            // One reading of the look per frame, below the last hook that can
            // write one. Every stage below takes this copy and none of them
            // reads the field again, which is what gives the whole struct a
            // single deadline: the lights and the shadow settings are wanted
            // here, the background, the bloom and the post operator when the
            // frame is planned, and the shadow decision later still. Read
            // separately they would land a driver's two writes in two
            // different frames, and which field went into which is not
            // something a picture shows.
            const look = self.look;

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
                .lights = look.lights,
                // Off unless a directional light was found and fitted. The
                // block's strength is what the shader tests, so an absent sun is
                // one value rather than a branch on both sides.
                .sun_shadow = if (self.sun) |sun| .{
                    .view_projection = sun.fit.view_proj,
                    .strength = look.sun_shadow.clampedStrength(),
                    .normal_offset = look.sun_shadow.normalOffsetWorld(&sun.fit),
                    .light = sun.index,
                } else .off,
            });

            self.metrics.record(.rings, phase.split());

            const commands = try frame.beginCommands(&self.context);
            // Before anything is recorded into the frame and outside any
            // rendering, which is where a query pool may be reset. Every slot
            // this frame will write is cleared here, so a pass it skips leaves
            // its pair unwritten and reads as zero rather than as whatever the
            // frame two ago left there.
            if (self.gpu_timer) |*timer| timer.reset(commands, &self.context, self.frame_index);
            // The glyphs rasterised since the last frame, if any. Here because
            // a copy cannot be recorded inside a rendering and the overlay that
            // samples them is recorded inside one, and in this frame rather
            // than through the install path every other upload takes: that path
            // submits and waits, which is right for a level and not for
            // something a frame can produce.
            try self.uploadGlyphs(commands);
            // Before any rendering is opened: a dispatch cannot be recorded
            // inside one, and each compute owner ends with the barrier that
            // hands its writes to the stages below. The morph pass is the
            // engine's; application compute follows it through the same command
            // buffer and cannot escape the frame's fence lifetime.
            self.morph_pass.record(commands, self.frame_index);
            try driver.onCompute(self, level, commands);

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
                .background = look.background,
                .bloom = look.bloom,
                .post = look.post,
            });
            // The frame, stage by stage. This sequence is the engine's and not
            // the renderer's: which passes a frame is made of, and in what
            // order, is composition. Each stage carries its own precondition,
            // and they are met by this order alone.
            //
            // The bake first, because it opens a rendering of its own and the
            // map it writes is sampled by every fragment the main pass shades.
            self.mark(commands, .shadow, .begin);
            self.renderer.recordShadowBake(commands, self.shadowBake(look, level), frame_plan);
            self.mark(commands, .shadow, .end);

            self.mark(commands, .main, .begin);
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
            self.mark(commands, .main, .end);

            // Between the two passes: the chain reads what the main pass wrote
            // and the post pass reads what the chain leaves.
            self.mark(commands, .bloom, .begin);
            self.renderer.recordBloom(commands, frame_plan);
            self.mark(commands, .bloom, .end);

            self.mark(commands, .post, .begin);
            const present_target: gpu.PostTarget = .{
                .image = self.swapchain.images[acquired.image_index].image,
                .view = self.swapchain.images[acquired.image_index].view,
                .extent = .{ .width = extent.width, .height = extent.height },
            };
            self.renderer.beginPost(commands, present_target);
            self.renderer.recordPost(commands, frame_plan);
            // Inside the same rendering and after the operator, which is the
            // only place an overlay can go: before it the target holds scene
            // radiance, and after `endPost` the image belongs to presentation.
            self.ui_pass.record(
                commands,
                self.frame_index,
                self.ui.commands(),
                present_target.extent,
            );
            self.renderer.endPost(commands, present_target);
            self.mark(commands, .post, .end);
            self.recorded_frames += 1;
            self.metrics.record(.record, phase.split());

            if (self.gpu_timer) |*timer| timer.markSubmitted(self.frame_index);
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
            self.metrics.record(.submit, phase.split());

            // Held rather than unwrapped, so that the last two splits and the
            // measurement below are taken on the path a stale surface takes as
            // well. A window closed before them would carry a mean over frames
            // whose two ends were counted as nothing.
            const presented = self.swapchain.present(acquired.image_index);
            self.metrics.record(.present, phase.split());
            self.metrics.endFrame(time);

            const present_state = presented catch |err| switch (err) {
                // The frame was submitted and this slot is in flight, so the
                // slot is deliberately not advanced: the next frame takes it
                // again and waits on the fence this submission signals.
                error.OutOfDateKHR => {
                    self.swapchain_stale = true;
                    continue;
                },
                else => return err,
            };
            if (present_state == .suboptimal) self.swapchain_stale = true;

            self.frame_index = (self.frame_index + 1) % frames_in_flight;
            self.presented_frames += 1;
        }

        try self.context.waitIdle();
    }

    // Opens a face over bytes the caller keeps, and over a file this reads.
    //
    // The two are the two kinds of font an application has. A game's is in its
    // binary or its archive and already lives for the length of the program, so
    // it is borrowed and nothing is copied; a system font is a file on the host
    // that has to be read and held resident, because FreeType parses it in
    // place. Neither uploads: what a face writes is host coverage, and the
    // frame that follows carries it to the device.
    //
    // The bytes handed to `loadFont` must outlive the engine.
    // `rendering` is how the display wants glyphs made, which is the host's
    // answer and not the font's. A caller with no answer passes `.{}` and gets
    // the reference target's; a caller that asked the host passes
    // `Rendering.fromHost`. `loadSystemFont` does the asking itself.
    pub fn loadFont(
        self: *Engine,
        bytes: []const u8,
        face_index: u32,
        pixel_size: u32,
        rendering: font_module.Rendering,
    ) font_module.Error!font_module.FontId {
        return self.fonts.add(self.allocator, bytes, face_index, pixel_size, rendering);
    }

    pub fn loadFontFile(
        self: *Engine,
        io: std.Io,
        directory: std.Io.Dir,
        path: []const u8,
        face_index: u32,
        pixel_size: u32,
        rendering: font_module.Rendering,
    ) !font_module.FontId {
        return self.fonts.addFile(self.allocator, io, directory, path, face_index, pixel_size, rendering);
    }

    // Opens the font the host prefers for a role, or reports that it has none.
    //
    // A role and not a name: `sans-serif` is a question the machine answers out
    // of its own configuration, and it is the only sense in which "the system
    // font" means anything once the program leaves the machine it was built on.
    // What comes back is a file the engine reads and holds, which is the second
    // of the two ownership cases `loadFont` and `loadFontFile` divide.
    //
    // Null is the host having no answer to give: no fontconfig, or a
    // configuration that matched nothing. An application that meets it draws no
    // text, which is a decision it has to be able to take anyway, since a font
    // is a thing a host may simply not have.
    pub fn loadSystemFont(
        self: *Engine,
        io: std.Io,
        request: platform.SystemFontRequest,
        pixel_size: u32,
    ) !?font_module.FontId {
        var path: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const found = try platform.findSystemFont(&path, request) orelse return null;
        // The same answer carries how the host wants text drawn, so asking for
        // the file and asking for the mode is one question. A caller that opens
        // a font of its own is the one that has to ask separately.
        // Absolute, so the directory the path is resolved against is not read.
        return try self.loadFontFile(
            io,
            .cwd(),
            found.path,
            found.index,
            pixel_size,
            .fromHost(found.rendering),
        );
    }

    // The image a run of glyphs is drawn from. `ui_white` is its counterpart
    // for a solid fill, and a draw naming either is one command.
    pub fn glyphAtlas(self: *const Engine) res.ImageHandle {
        return self.glyph_atlas;
    }

    // Shapes and places a run against the atlas, rasterising anything new.
    //
    // On the engine rather than reached through `fonts` because the atlas it
    // writes into is uploaded by the frame loop, and a caller shaping through
    // another route would leave coverage the device never receives.
    pub fn shapeText(
        self: *Engine,
        id: font_module.FontId,
        source: []const u8,
        glyphs: []res.ShapedGlyph,
        placements: []res.GlyphPlacement,
    ) font_module.Error!res.GlyphRun {
        return self.fonts.shape(self.allocator, id, source, glyphs, placements);
    }

    pub fn fontMetrics(self: *const Engine, id: font_module.FontId) font_module.Error!res.FontMetrics {
        return self.fonts.metrics(id);
    }

    // A caption, shaped once and measurable before it is drawn.
    //
    // This is what a layout node is sized from: `advance` is the width it needs
    // and `Label.height` the height, both available before there is a rectangle
    // to draw into. Then the same value is handed to `Context.label`, so the
    // line that was measured is the line that is drawn.
    //
    // Shaping once is the point. The four parts of a label come from four
    // places — the run from the shaper, the metrics from the face, the width
    // from the run and the atlas from the engine — and a caller assembling them
    // by hand either shapes twice, once to measure and once to draw, or carries
    // the pieces between the two passes itself. It is also where the parts can
    // silently disagree: metrics of one face over a run shaped with another
    // draws a line at the wrong height with nothing to say so.
    //
    // **The label borrows `glyphs` and `placements`.** The run points into
    // them, so both must outlive the drawing and not only this call. A caller
    // that measures in one pass and draws in another therefore owns arrays that
    // live across the frame, which is the same rule every other borrowed slice
    // in this engine follows.
    pub fn shapeLabel(
        self: *Engine,
        id: font_module.FontId,
        source: []const u8,
        glyphs: []res.ShapedGlyph,
        placements: []res.GlyphPlacement,
    ) font_module.Error!imui.Label {
        const shaped = try self.shapeText(id, source, glyphs, placements);
        return .{
            .run = shaped,
            .metrics = try self.fontMetrics(id),
            .advance = font_module.advance(shaped.glyphs),
            .atlas = self.glyphAtlas(),
        };
    }

    // Carries whatever the atlas has gained to the image, as whole rows.
    //
    // Rows and not the dirty rectangle's columns: the coverage is stored row by
    // row at the atlas's full width, so a run of rows is one contiguous copy
    // while a rectangle is one copy per row of it. What that costs is the
    // texels either side of a new glyph, on the frames a new glyph appears.
    //
    // Two barriers around it, and the first is what the image being sampled
    // every frame makes necessary: the copy has to wait for the fragment reads
    // of frames still in flight, and the sampling below has to wait for the
    // copy. Nothing is recorded at all on a frame that rasterised nothing,
    // which after a face is loaded is every frame a Latin interface draws.
    fn uploadGlyphs(self: *Engine, commands: gpu.vk.CommandBuffer) !void {
        const box = self.fonts.takeDirty() orelse return;

        // In bytes, which is four to a texel. The row is the unit of the copy
        // and the atlas is the unit of the ring, so both are that many times
        // what the texel count says.
        const row_bytes = @as(usize, self.fonts.atlasSize()) * font_module.atlas_channels;
        const first = @as(usize, box.y) * row_bytes;
        const length = @as(usize, box.height) * row_bytes;
        const slot = std.mem.sliceAsBytes(self.glyph_staging.slice(self.frame_index));
        @memcpy(slot[0..length], self.fonts.coverage()[first..][0..length]);

        self.glyph_image.recordLayoutTransition(
            commands,
            .fromShaderRead(.{ .fragment_shader_bit = true }),
        );
        try self.glyph_image.recordCopyFrom(
            self.glyph_staging.storageBuffer(),
            commands,
            .{
                .buffer_offset = self.glyph_staging.dynamicOffset(self.frame_index),
                .mip_level = 0,
                .first_row = box.y,
                .row_count = box.height,
            },
        );
        self.glyph_image.recordLayoutTransition(
            commands,
            .toShaderRead(.{ .fragment_shader_bit = true }),
        );
    }

    // One pass boundary, or nothing when no timer exists. Written from here
    // rather than from inside the renderer because which passes a frame is made
    // of is composition, and this loop is where that sequence is stated.
    fn mark(self: *const Engine, commands: gpu.vk.CommandBuffer, pass: gpu.GpuPass, edge: gpu.GpuTimestampEdge) void {
        const timer = if (self.gpu_timer) |*value| value else return;
        timer.write(commands, &self.context, self.frame_index, pass, edge);
    }

    // Takes the frame's look rather than reading the field, so the decision and
    // the block the shader tests come from the same reading.
    fn shadowBake(self: *Engine, look: Look, level: *const Level) gpu.ShadowBake {
        const request: ShadowBakeRequest = .{
            .shadows_on = look.sun_shadow.enabled and self.sun != null,
            .were_on = self.shadows_were_on,
            .map_stale = self.shadow_dirty,
            .casters_move = level.world.castersMove(),
        };
        self.shadows_were_on = request.shadows_on;
        if (request.decide() == .rebake) self.shadow_dirty = false;
        return request.decide();
    }

    // The frame's events, to the UI first and to the driver with what the UI
    // did about it. Called inside the routing pass, which is what makes the
    // second half of that sentence true.
    fn drainInput(self: *Engine, driver: anytype) !void {
        const batch = self.input.takeBatch() catch |err| switch (err) {
            error.InputEventOverflow => {
                log.warn(
                    "input overflowed; batch discarded ({d} so far)",
                    .{self.input.overflowCount()},
                );
                // The discarded batch may have held the release that ends a
                // gesture in progress, and a widget waiting for one it will
                // never get stays held for as long as the pointer is not
                // pressed again. Abandoning the gesture is what the UI's
                // `cancel` is for, and it does nothing when there is none.
                try self.ui.cancel();
                return;
            },
        };
        defer self.input.releaseBatch();

        for (batch) |event| {
            try driver.onEvent(self, event, try self.ui.route(event));
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
