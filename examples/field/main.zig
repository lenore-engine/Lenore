const std = @import("std");
const gpu = @import("lenore-gpu");
const imui = @import("lenore-imui");
const lenore = @import("lenore");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const grass_module = @import("grass.zig");
const ground = @import("ground.zig");
const terrain = @import("terrain.zig");

const log = std.log.scoped(.field);

// A plain, walked at eye height, with the frame's own cost on screen beside it.
//
// This is the stand the picture work is measured on, and every decision in it
// serves being measured twice. The scene is generated from a seed and takes no
// asset, so two runs draw the same field. The sun is a slider and not a clock,
// so a time of day can be returned to exactly. And the panel reports what each
// pass cost on the device next to the control that switched it on, because a
// toggle that only answers "prettier?" is not an instrument.
//
// The foliage is not geometry this file hands to a level. It is planted in a
// compute dispatch and drawn indirect, from `grass.zig`, so no blade is ever
// host memory and the level holds the ground alone.
//
// Usage: run-field
//
//   left drag on the scene     look
//   W A S D                    walk, shift to run
//   space                      jump
//   exposure                   the whole picture, before the operator
//   sun                        elevation, from noon down past the horizon
//   density, rings             how much grass is planted, and how far out
//   bloom, shadows             on and off, with their cost beside them
//   escape                     quit

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

// Metres. A person's eye over the ground under them, and the speeds a person
// crosses a field at.
const eye_height: f32 = 1.68;
const walk_speed: f32 = 4.2;
const run_multiplier: f32 = 3.0;

// Metres per second squared, downward, and the speed a jump leaves the ground
// at. Earth's own value rather than a tuned one: a field seen from a body that
// falls at the rate a body falls at is the thing being judged, and a lighter
// gravity would make every hop read as slow motion.
//
// The jump speed follows from it: apex is v^2 / 2g, so 4.4 gives just under a
// metre. Enough to see the field from a little above without becoming a way to
// travel.
const gravity: f32 = 9.81;
const jump_speed: f32 = 4.4;

// How far inside the domain the camera is kept. The ground ends at the extent
// and the view would otherwise reach past its edge, which is a hole in the world
// and not a horizon.
const wall_margin: f32 = 2.0;

// Just short of vertical. The camera basis stays orthonormal at the poles, so
// this is not a singularity guard: it is that a walking view which can look at
// its own feet and then past them has rolled over, and nothing in the picture
// says so.
const max_pitch: f32 = std.math.pi / 2.0 - 0.05;

// The sun, in radians. Elevation is the slider; the azimuth is fixed, because
// two angles make a position that cannot be returned to by remembering one
// number, and this example's whole purpose is returning to a moment.
const sun_azimuth: f32 = 2.44;
const sun_colour = res.Vec3{ 1.0, 0.96, 0.90 };
// Lux, near enough. The operator and the exposure are what put it on screen, and
// the number that matters is that it does not change while the elevation does:
// a sun that dims as it sets hides whether the fall in the picture came from the
// atmosphere or from the slider.
const sun_intensity: f32 = 24.0;

// Wide, because nothing here has settled the scene's absolute scale yet. The
// scattering integral puts the zenith around a tenth of a unit at the sun
// intensity above, while the sun's own disc is its irradiance divided by a solid
// angle of a hundred-thousandth of a steradian, so the two ends of the picture
// are five orders apart and the operator has to be aimed by hand until an
// automatic exposure aims it.
const exposure_range: imui.SliderRange = .{ .min = 0.05, .max = 8.0 };
// From overhead to well under the horizon. Past zero the sun is below the
// ground: what should happen there is the atmosphere's answer, and the range
// reaches it so that the absence of one is visible.
const elevation_range: imui.SliderRange = .{ .min = -0.25, .max = 1.5 };

// How much of the field the planter is allowed to plant, and how many rings it
// dispatches at all.
//
// Both are here because the cost of the foliage is not in any row the panel can
// report: the planting dispatch is recorded before the main pass opens, and the
// engine's timer brackets four passes that do not include it. Until it does,
// these two are the instrument. Rings at zero dispatches nothing and draws
// nothing, which is the direct measurement of what the whole pass costs.
const density_range: imui.SliderRange = .{ .min = 0.0, .max = 1.0 };
const rings_range: imui.SliderRange = .{ .min = 0.0, .max = 4.0 };

// Logical units: what a person sees at the size they see it, whatever the
// output's scale.
const panel_width: f32 = 268;
const panel_inset: f32 = 16;
const row_height: f32 = 24;
const row_gap: f32 = 6;
const panel_padding: f32 = 12;
const label_gap: f32 = 8;
const label_points: f32 = 13;

fn hex(comptime value: *const [6:0]u8) res.PremultipliedColor {
    return imui.SrgbColor.fromHex(value).premultiplied();
}

const palette = struct {
    const panel = hex("161c26");
    const track = hex("323d52");
    const control = hex("293347");
    const hovered = hex("3f4d6b");
    const held = hex("6379a8");
    const disabled = hex("212633");
    const border = hex("7d90bb");
    const focus = hex("e8b84b");
    const mark = hex("e6ebf4");
    const reading = hex("9fb4d8");
};

const control_style: imui.ButtonStyle = .{
    .normal_fill = palette.control,
    .hovered_fill = palette.hovered,
    .held_fill = palette.held,
    .disabled_fill = palette.disabled,
    .border = palette.border,
    .focused_border = palette.focus,
    .border_width = 1,
    .radius = 4,
};

const checkbox_style: imui.CheckboxStyle = .{
    .box = control_style,
    .mark = palette.mark,
    .mark_inset = 5,
    .mark_radius = 2,
};

const slider_style: imui.SliderStyle = .{
    .track = palette.track,
    .disabled_track = palette.disabled,
    .knob = control_style,
    .track_thickness = 6,
    .knob_width = 14,
};

const label_style: imui.LabelStyle = .{
    .normal_text = palette.mark,
    .disabled_text = palette.track,
    .vertical = .middle,
};

const reading_style: imui.LabelStyle = .{
    .normal_text = palette.reading,
    .disabled_text = palette.track,
    .vertical = .middle,
};

// The rows, in the order they are laid out. The four that respond come first and
// the readings follow, so that the split between what is registered as a widget
// and what is only drawn is the split between two contiguous ranges rather than
// a property to check per row.
const Row = enum(u32) {
    exposure = 2,
    sun,
    density,
    rings,
    bloom,
    shadows,
    frame,
    shadow_ms,
    depth_ms,
    main_ms,
    bloom_ms,
    post_ms,
    blades,

    const first = @backingInt(Row.exposure);
    const last_interactive = @backingInt(Row.shadows);
    const count = @backingInt(Row.blades) - first + 1;

    fn index(self: Row) usize {
        return @backingInt(self);
    }
};

const panel_node: u32 = 1;
const node_count = Row.first + Row.count;
const panel_height: f32 = panel_padding * 2 +
    @as(f32, @floatFromInt(Row.count)) * row_height +
    @as(f32, @floatFromInt(Row.count - 1)) * row_gap;

const layout_nodes = build: {
    var nodes: [node_count]imui.LayoutNode = undefined;
    nodes[0] = .{
        .child_start = 0,
        .child_count = 1,
        .arrangement = .overlay,
        .padding = .{
            .left = panel_inset,
            .top = panel_inset,
            .right = panel_inset,
            .bottom = panel_inset,
        },
        // Under `overlay` the main axis is horizontal, so this is the top right
        // corner: right along the main axis, top across it.
        .main_alignment = .end,
        .cross_alignment = .start,
    };
    nodes[panel_node] = .{
        .child_start = 1,
        .child_count = Row.count,
        .arrangement = .column,
        .width = .{ .fixed = panel_width },
        .height = .{ .fixed = panel_height },
        .padding = .{
            .left = panel_padding,
            .top = panel_padding,
            .right = panel_padding,
            .bottom = panel_padding,
        },
        .gap = row_gap,
        .cross_alignment = .stretch,
    };
    for (Row.first..node_count) |row| nodes[row] = .{ .height = .{ .fixed = row_height } };
    break :build nodes;
};

const layout_children = build: {
    var children: [node_count - 1]imui.NodeIndex = undefined;
    children[0] = panel_node;
    for (Row.first..node_count, 1..) |row, slot| children[slot] = @intCast(row);
    break :build children;
};

// A checkbox is square and sits at the left of its row; its word takes the rest.
fn boxOf(row: imui.LogicalRect) imui.LogicalRect {
    return .{ .x = row.x, .y = row.y, .width = row.height, .height = row.height };
}

fn captionOf(row: imui.LogicalRect) imui.LogicalRect {
    const left = row.height + label_gap;
    return .{
        .x = row.x + left,
        .y = row.y,
        .width = @max(row.width - left, 0),
        .height = row.height,
    };
}

// Which movement keys are down. Held state and not per-event motion: a key
// repeat arrives at the system's repeat rate, so walking driven by events would
// travel at whatever that is set to.
const Held = struct {
    forward: bool = false,
    back: bool = false,
    left: bool = false,
    right: bool = false,
    fast: bool = false,
    jump: bool = false,

    fn set(self: *Held, key: platform.PhysicalKey, down: bool) bool {
        switch (key) {
            .w => self.forward = down,
            .s => self.back = down,
            .a => self.left = down,
            .d => self.right = down,
            .shift_left, .shift_right => self.fast = down,
            .space => self.jump = down,
            else => return false,
        }
        return true;
    }
};

// Mouse look and walking. This owns the interpretation of input; where the
// camera is stays in `scene.Camera`, the way the shared orbit controller keeps
// its angles there.
const Walk = struct {
    looking: bool = false,
    last_cursor: [2]f32 = .{ 0, 0 },
    held: Held = .{},

    // Metres per second, upward. Carried between frames because it is the whole
    // of what makes a fall a fall: a camera placed on the ground every frame
    // has no state and cannot leave it.
    vertical_velocity: f32 = 0.0,
    // Whether the feet are on the ground. Read to decide a jump, and written by
    // the landing test rather than inferred from the height, so that a frame in
    // which the ground rose to meet a rising camera is not a landing.
    grounded: bool = false,

    const sensitivity: f32 = 0.0022;

    fn beginLook(self: *Walk, position: [2]f32) void {
        self.looking = true;
        self.last_cursor = position;
    }

    fn lookTo(self: *Walk, camera: *scene.Camera, position: [2]f32) void {
        if (!self.looking) return;
        camera.yaw += (position[0] - self.last_cursor[0]) * sensitivity;
        camera.pitch = std.math.clamp(
            camera.pitch - (position[1] - self.last_cursor[1]) * sensitivity,
            -max_pitch,
            max_pitch,
        );
        self.last_cursor = position;
    }

    // Focus loss can arrive without a release, and a key that went down before
    // it would otherwise stay down forever. The fall is not cleared: losing the
    // window does not catch a body in mid air.
    fn cancel(self: *Walk) void {
        self.looking = false;
        self.held = .{};
    }

    // Puts the feet on the ground and stops any fall. Used once before the first
    // frame, where letting `advance` do it would mean falling from wherever the
    // camera happened to be constructed.
    fn stand(self: *Walk, camera: *scene.Camera, field: terrain.Terrain) void {
        const eye = eyeOf(camera.*);
        camera.anchor = .{ .eye = .{
            eye[0],
            field.heightAt(eye[0], eye[2]) + eye_height,
            eye[2],
        } };
        self.vertical_velocity = 0.0;
        self.grounded = true;
    }

    fn advance(self: *Walk, camera: *scene.Camera, field: terrain.Terrain, delta: f32) void {
        // Horizontal, from the azimuth alone. Taking the camera's own front
        // would make looking up climb, and a walk that leaves the ground when
        // the view tilts is a flight with extra steps.
        const forward = res.Vec3{ @cos(camera.yaw), 0.0, @sin(camera.yaw) };
        const right = res.Vec3{ -@sin(camera.yaw), 0.0, @cos(camera.yaw) };

        var direction = res.Vec3{ 0.0, 0.0, 0.0 };
        if (self.held.forward) direction += forward;
        if (self.held.back) direction -= forward;
        if (self.held.right) direction += right;
        if (self.held.left) direction -= right;

        const length = @sqrt(direction[0] * direction[0] + direction[2] * direction[2]);
        var eye = eyeOf(camera.*);
        if (length > 0.0) {
            // Normalised, so that two keys together do not travel by the
            // diagonal's extra factor.
            const speed = walk_speed * (if (self.held.fast) run_multiplier else 1.0) * delta;
            const scale: res.Vec3 = @splat(speed / length);
            eye += direction * scale;
        }

        const limit = field.settings.extent() - wall_margin;
        eye[0] = std.math.clamp(eye[0], -limit, limit);
        eye[2] = std.math.clamp(eye[2], -limit, limit);

        // A jump is taken only from the ground. Tested against the flag rather
        // than against the height, so that holding the key while falling does
        // not launch again the instant the feet touch.
        if (self.grounded and self.held.jump) {
            self.vertical_velocity = jump_speed;
            self.grounded = false;
        }

        // Semi-implicit Euler: the velocity is advanced first and the position
        // by the new velocity. It costs the same as the explicit form and does
        // not gain energy over a long fall, which the explicit one does.
        self.vertical_velocity -= gravity * delta;

        const ground_height = field.heightAt(eye[0], eye[2]);
        var feet = eye[1] - eye_height + self.vertical_velocity * delta;
        if (feet <= ground_height) {
            // Landed. Walking downhill lands every frame, which is right: the
            // ground is what stops the fall, whether it fell a metre or a
            // millimetre.
            feet = ground_height;
            self.vertical_velocity = 0.0;
            self.grounded = true;
        } else {
            self.grounded = false;
        }
        eye[1] = feet + eye_height;

        camera.anchor = .{ .eye = eye };
    }
};

// The camera's position, whichever way it is anchored. A walk always writes an
// eye anchor, so the orbit case is only the first frame of a camera that has not
// been placed yet.
fn eyeOf(camera: scene.Camera) res.Vec3 {
    return switch (camera.anchor) {
        .eye => |position| position,
        .orbit => |orbit| orbit.target,
    };
}

const Driver = struct {
    pub const hooks: lenore.Hooks(Driver) = .{
        .ui_regions = onUiRegions,
        .event = onEvent,
        .update = onUpdate,
        .compute = onCompute,
        .record = onRecord,
        .ui_draw = onUiDraw,
    };

    field: terrain.Terrain,
    walk: Walk = .{},
    // Borrowed. It owns device resources and is torn down by `main` after the
    // loop, where the device has drained.
    grass: *grass_module.Grass,
    // Seconds since the run began, which is what the wind is a function of. Not
    // the frame's own delta: a wave built from accumulated deltas drifts, and
    // this example exists to be returned to.
    elapsed: f32 = 0.0,

    // What the panel holds. Every widget is a call, so what the interface
    // remembers is exactly the values its controls stand for.
    exposure: f32 = 1.0,
    sun_elevation: f32 = 0.16,
    density: f32 = 1.0,
    // Held as a float because that is what a slider writes. It reaches the pass
    // rounded, and the reading beside it shows the rounded value so that the
    // number on screen is the number dispatched.
    rings: f32 = 4.0,
    bloom: bool = true,
    shadows: bool = true,

    // The light block the engine's `Look` points at. It lives here rather than
    // in `main` because the elevation slider rewrites it every frame, and a
    // slice into a caller's stack that a driver writes through is an arrangement
    // that holds only while nobody moves either.
    lights: [gpu.max_lights]gpu.LightUniform = undefined,

    font: ?lenore.FontId,

    // This frame's rectangles, solved while registering and read again while
    // drawing. Both passes have to place a widget identically, and solving twice
    // is two chances to place it differently.
    rects: [node_count]imui.LogicalRect = undefined,
    scale: imui.ScaleFactor = .identity,
    panel_rect: res.Rect = .{ .x = 0, .y = 0, .width = 0, .height = 0 },

    parents: [node_count]imui.NodeIndex = undefined,
    order: [node_count]imui.NodeIndex = undefined,
    measured: [node_count]imui.LogicalSize = undefined,
    scratch: [node_count]imui.LogicalRect = undefined,

    pub fn onUiRegions(
        self: *Driver,
        engine: *lenore.Engine,
        _: *lenore.Level,
        ui: *imui.WidgetContext,
    ) !void {
        const extent = engine.swapchain.currentExtent();
        self.scale = engine.uiScale();
        try imui.solveLayout(
            .{ .nodes = &layout_nodes, .children = &layout_children },
            .{
                .x = 0,
                .y = 0,
                .width = @as(f32, @floatFromInt(extent.width)) / self.scale.x,
                .height = @as(f32, @floatFromInt(extent.height)) / self.scale.y,
            },
            .{
                .parents = &self.parents,
                .order = &self.order,
                .measured = &self.measured,
                .rects = &self.scratch,
            },
            &self.rects,
        );
        self.panel_rect = try self.rects[panel_node].toFramebufferFilled(self.scale);

        // The panel's own background, registered first and so under everything:
        // `hitTest` answers with the last region containing the point. Without
        // it, a drag that misses a control by a pixel would reach the scene and
        // swing the view.
        try ui.register(.{ .string = "panel" }, self.panel_rect, .{});

        try ui.pushId(.{ .string = "controls" });
        defer ui.popId();
        try ui.pushClip(self.panel_rect);
        defer ui.popClip();

        // Only the interactive prefix. The readings below are drawn and never
        // registered: a label is not a target, and a click that lands on a
        // number should reach the panel behind it and stop there.
        for (Row.first..Row.last_interactive + 1) |row| {
            const rect = if (row == Row.bloom.index() or row == Row.shadows.index())
                boxOf(self.rects[row])
            else
                self.rects[row];
            try ui.register(
                .{ .integer = @intCast(row) },
                try rect.toFramebufferFilled(self.scale),
                .{ .focusable = true },
            );
        }
    }

    pub fn onEvent(
        self: *Driver,
        engine: *lenore.Engine,
        event: platform.Event,
        ui_consumed: bool,
    ) !void {
        // What the UI took, it took, and this is this frame's answer rather than
        // the previous frame's. A look that began on the scene keeps going: once
        // it holds the pointer the UI has no region under it and the motion
        // arrives here unconsumed.
        if (ui_consumed) {
            if (event.payload == .mouse_button and event.payload.mouse_button.action == .press)
                self.walk.looking = false;
            return;
        }

        switch (event.payload) {
            .cursor => |cursor| self.walk.lookTo(&engine.camera, cursor.logical_position),
            .mouse_button => |button| if (button.button == .left) switch (button.action) {
                .press => self.walk.beginLook(button.logical_position),
                .release => {
                    self.walk.lookTo(&engine.camera, button.logical_position);
                    self.walk.looking = false;
                },
                .repeat => {},
            },
            .focus => |focus| if (!focus.focused) self.walk.cancel(),
            .key => |key| switch (key.action) {
                .press => {
                    if (key.physical == .escape) return engine.requestExit();
                    _ = self.walk.held.set(key.physical, true);
                },
                .release => _ = self.walk.held.set(key.physical, false),
                // A repeat says the key is still down, which it already is.
                .repeat => {},
            },
            else => {},
        }
    }

    pub fn onCompute(
        self: *Driver,
        engine: *lenore.Engine,
        _: *lenore.Level,
        commands: gpu.vk.CommandBuffer,
    ) !void {
        // The count the frame in this slot produced last time round. Its fence
        // has signalled by the time a driver is called, so the copy has landed
        // and reading it waits for nothing.
        self.grass.readCount(engine.frame_index);
        self.grass.plant(engine, commands, self.elapsed);
    }

    pub fn onRecord(
        self: *Driver,
        engine: *lenore.Engine,
        _: *lenore.Level,
        commands: gpu.vk.CommandBuffer,
    ) !void {
        const direction = self.sunDirection();
        try self.grass.draw(engine, commands, self.elapsed, direction, .{
            sun_colour[0] * sun_intensity,
            sun_colour[1] * sun_intensity,
            sun_colour[2] * sun_intensity,
        });
    }

    pub fn onUpdate(
        self: *Driver,
        engine: *lenore.Engine,
        _: *lenore.Level,
        time: lenore.FrameTime,
    ) !void {
        self.elapsed += time.delta;
        self.walk.advance(&engine.camera, self.field, time.delta);
        self.grass.settings.density = self.density;
        self.grass.settings.rings = @intFromFloat(@round(self.rings));
        try self.applyLook(engine);
    }

    // The whole of the look, rebuilt every frame from what the panel holds.
    //
    // Rebuilt rather than written when a control changes: there are five values
    // and one place they take effect, so the alternative is five edges to notice
    // and a state that is correct only if none was missed.
    // Down from the sky toward the ground, which is the direction light travels.
    // Elevation is measured from the horizon upward, so a positive elevation
    // gives a negative Y.
    //
    // One statement, because three things read it: the light block the ground is
    // shaded from, the background that draws the disc, and the foliage. Two of
    // them get it through the block and the third is handed it directly, so a
    // second copy here would be the one that drifts.
    fn sunDirection(self: *const Driver) [3]f32 {
        return .{
            @cos(sun_azimuth) * @cos(self.sun_elevation),
            -@sin(self.sun_elevation),
            @sin(sun_azimuth) * @cos(self.sun_elevation),
        };
    }

    fn applyLook(self: *Driver, engine: *lenore.Engine) !void {
        const heading = self.sunDirection();
        const direction = res.Vec3{ heading[0], heading[1], heading[2] };
        self.lights[0] = lenore.packLight(try .directional(sun_colour, sun_intensity, direction));

        engine.look = .{
            .lights = self.lights[0..1],
            // Without this the background pass is never recorded and every
            // pixel no surface covered stays the clear colour, whatever shader
            // was handed to the engine. The name is the engine's word for
            // "draw the background" rather than a statement about what the
            // background is.
            .background = .environment,
            .sun_shadow = .{
                .enabled = self.shadows,
                .strength = 1.0,
                .normal_offset_texels = 1.0,
            },
            .bloom = if (self.bloom) .{} else null,
            .post = .{ .exposure = self.exposure },
        };
    }

    pub fn onUiDraw(
        self: *Driver,
        engine: *lenore.Engine,
        _: *lenore.Level,
        ui: *imui.WidgetContext,
    ) !void {
        const canvas = try ui.drawList();
        try canvas.addQuad(self.panel_rect, .{}, palette.panel, engine.ui_white);

        // The same scope and clip the registering pass pushed, in the same
        // order. That is what makes a widget here the one registered there:
        // nothing checks it, and a pass that walked differently would ask for an
        // identity this frame never registered.
        try ui.pushId(.{ .string = "controls" });
        defer ui.popId();
        try ui.pushClip(self.panel_rect);
        defer ui.popClip();

        var text: [48]u8 = undefined;

        _ = try ui.slider(
            .{ .integer = @intCast(Row.exposure.index()) },
            try self.rects[Row.exposure.index()].toFramebufferFilled(self.scale),
            &self.exposure,
            exposure_range,
            slider_style,
            true,
        );
        try self.caption(engine, ui, self.rects[Row.exposure.index()], "exposure", .start);
        try self.caption(
            engine,
            ui,
            self.rects[Row.exposure.index()],
            try std.fmt.bufPrint(&text, "{d:.2}", .{self.exposure}),
            .end,
        );

        _ = try ui.slider(
            .{ .integer = @intCast(Row.sun.index()) },
            try self.rects[Row.sun.index()].toFramebufferFilled(self.scale),
            &self.sun_elevation,
            elevation_range,
            slider_style,
            true,
        );
        try self.caption(engine, ui, self.rects[Row.sun.index()], "sun", .start);
        try self.caption(
            engine,
            ui,
            self.rects[Row.sun.index()],
            // Degrees, because that is the unit the picture is discussed in: a
            // sun at four degrees is a sunset and one at 0.07 radians is a
            // number.
            try std.fmt.bufPrint(&text, "{d:.1} deg", .{
                self.sun_elevation * 180.0 / std.math.pi,
            }),
            .end,
        );

        _ = try ui.slider(
            .{ .integer = @intCast(Row.density.index()) },
            try self.rects[Row.density.index()].toFramebufferFilled(self.scale),
            &self.density,
            density_range,
            slider_style,
            true,
        );
        try self.caption(engine, ui, self.rects[Row.density.index()], "density", .start);
        try self.caption(
            engine,
            ui,
            self.rects[Row.density.index()],
            try std.fmt.bufPrint(&text, "{d:.2}", .{self.density}),
            .end,
        );

        _ = try ui.slider(
            .{ .integer = @intCast(Row.rings.index()) },
            try self.rects[Row.rings.index()].toFramebufferFilled(self.scale),
            &self.rings,
            rings_range,
            slider_style,
            true,
        );
        try self.caption(engine, ui, self.rects[Row.rings.index()], "rings", .start);
        try self.caption(
            engine,
            ui,
            self.rects[Row.rings.index()],
            try std.fmt.bufPrint(&text, "{d}", .{self.grass.settings.rings}),
            .end,
        );

        _ = try ui.checkbox(
            .{ .integer = @intCast(Row.bloom.index()) },
            try boxOf(self.rects[Row.bloom.index()]).toFramebufferFilled(self.scale),
            &self.bloom,
            checkbox_style,
            true,
        );
        try self.caption(engine, ui, captionOf(self.rects[Row.bloom.index()]), "bloom", .start);

        _ = try ui.checkbox(
            .{ .integer = @intCast(Row.shadows.index()) },
            try boxOf(self.rects[Row.shadows.index()]).toFramebufferFilled(self.scale),
            &self.shadows,
            checkbox_style,
            true,
        );
        try self.caption(engine, ui, captionOf(self.rects[Row.shadows.index()]), "shadows", .start);

        try self.readings(engine, ui, &text);
    }

    // The cost of the frame, beside the controls that decide it.
    //
    // The host figure is the mean and the worst of the last closed window rather
    // than this frame: a per-frame number changes faster than it can be read,
    // and the worst is the one a budget is kept for. The device figures are the
    // last completed frame's, which is the most recent that exists at all, since
    // a timestamp is only readable once its fence has signalled.
    fn readings(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        text: []u8,
    ) !void {
        const frame_row = self.rects[Row.frame.index()];
        try self.caption(engine, ui, frame_row, "frame", .start);
        if (engine.metrics.last_fps) |report| {
            try self.reading(engine, ui, frame_row, try std.fmt.bufPrint(text, "{d:.2} / {d:.2} ms", .{
                report.mean_ms,
                report.worst_ms,
            }));
        } else {
            try self.reading(engine, ui, frame_row, "measuring");
        }

        const passes = [_]struct { row: Row, pass: gpu.GpuPass, name: []const u8 }{
            .{ .row = .shadow_ms, .pass = .shadow, .name = "shadow" },
            .{ .row = .depth_ms, .pass = .depth, .name = "depth" },
            .{ .row = .main_ms, .pass = .main, .name = "main" },
            .{ .row = .bloom_ms, .pass = .bloom, .name = "bloom" },
            .{ .row = .post_ms, .pass = .post, .name = "post" },
        };
        for (passes) |entry| {
            const row = self.rects[entry.row.index()];
            try self.caption(engine, ui, row, entry.name, .start);
            if (engine.last_gpu) |timings| {
                try self.reading(engine, ui, row, try std.fmt.bufPrint(text, "{d:.3} ms", .{
                    @as(f64, @floatFromInt(timings.get(entry.pass))) * 1e-6,
                }));
            } else {
                // The device declined to carry timestamps on this queue, or
                // nothing has come back yet. Saying so beats a column of zeros,
                // which reads as a pass that costs nothing.
                try self.reading(engine, ui, row, "no timer");
            }
        }

        // What the device planted, which the host has no other way of knowing:
        // the count was written by a dispatch, read by an indirect draw, and
        // copied back only so this line can exist.
        const blades_row = self.rects[Row.blades.index()];
        try self.caption(engine, ui, blades_row, "blades", .start);
        if (self.grass.planted) |planted| {
            const capacity = self.grass.settings.capacity;
            // Over capacity is reported rather than hidden. The planter counts
            // past the buffer and the draw takes what fits, so a field that says
            // more than it holds is drawing less grass than it decided on.
            try self.reading(engine, ui, blades_row, if (planted > capacity)
                try std.fmt.bufPrint(text, "{d} > {d}", .{ planted, capacity })
            else
                try std.fmt.bufPrint(text, "{d}", .{planted}));
        } else {
            try self.reading(engine, ui, blades_row, "measuring");
        }
    }

    fn caption(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        row: imui.LogicalRect,
        source: []const u8,
        horizontal: imui.LabelStyle.Horizontal,
    ) !void {
        var style = label_style;
        style.horizontal = horizontal;
        try self.write(engine, ui, row, source, style);
    }

    fn reading(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        row: imui.LogicalRect,
        source: []const u8,
    ) !void {
        var style = reading_style;
        style.horizontal = .end;
        try self.write(engine, ui, row, source, style);
    }

    // One run of text, shaped and drawn where it is named.
    //
    // The scratch is this call's, on the stack, and nothing outlives it. Shaping
    // allocates nothing as long as the glyphs are ones the face was loaded with,
    // which for these words and numbers they are.
    fn write(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        row: imui.LogicalRect,
        source: []const u8,
        style: imui.LabelStyle,
    ) !void {
        const font = self.font orelse return;

        var glyphs: [64]res.ShapedGlyph = undefined;
        var placements: [64 * res.SubpixelBuckets.most.count()]res.GlyphPlacement = undefined;
        const line = try engine.shapeLabel(font, source, &glyphs, &placements);
        try ui.label(try row.toFramebufferFilled(self.scale), style, line, true);
    }
};

// The example's own background, in place of the engine's environment cube.
//
// `@embedFile` yields bytes and SPIR-V is words, so the array is copied into an
// aligned constant: an embedded file carries no alignment of its own and the
// reinterpretation has to be valid rather than merely likely. The engine does
// the same for its own shading, for the same reason.
fn skyShader() gpu.SkyShader {
    const bytes = @embedFile("sky").*;
    const aligned: [bytes.len]u8 align(@alignOf(u32)) = bytes;
    const count = aligned.len / @sizeOf(u32);
    // Vulkan specification, VkShaderModuleCreateInfo: codeSize is a multiple of
    // four. Dividing without this would drop a partial word and hand the driver
    // a module shorter than the file.
    comptime std.debug.assert(count * @sizeOf(u32) == aligned.len);
    const words = @as([*]const u32, @ptrCast(&aligned))[0..count];
    return .{
        .spirv = words,
        .vertex_entry = "vertexMain",
        .fragment_entry = "fragmentMain",
    };
}

// The font this host prefers for an interface, or nothing. A host with no
// fontconfig and a host whose configuration matches no font both answer that
// way, and the panel is still usable: every control changes the picture.
fn loadFont(engine: *lenore.Engine, io: std.Io) ?lenore.FontId {
    return engine.loadSystemFont(io, .{}, engine.fontPixels(label_points)) catch |err| {
        log.warn("the host's font did not open: {t}", .{err});
        return null;
    } orelse {
        log.warn("the host offers no font, so the panel draws no captions", .{});
        return null;
    };
}

pub fn main(process: std.process.Init.Minimal) !void {
    _ = process;
    const gpa = if (checking) debug_allocator.allocator() else std.heap.smp_allocator;
    defer if (checking) {
        if (debug_allocator.deinit() == .leak) log.err("host memory leaked", .{});
    };

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const field: terrain.Terrain = try .init(.{});

    // The model is built, uploaded, and then dropped. Nothing downstream reads
    // it after `install`: the level owns the device resources and the world owns
    // the mesh records, so keeping the host copy alive would hold nine megabytes
    // of interchange vertices for the run.
    var model = try ground.build(gpa, field, .{});
    defer model.deinit(gpa);

    var engine: lenore.Engine = undefined;
    try engine.init(gpa, .{
        .title = "Lenore field",
        .material_capacity = @intCast(model.materials.len),
        // Asked for, because the panel reports per-pass device time and a
        // reading of zero would otherwise be indistinguishable from a pass that
        // is free. A device that cannot carry one says so in the panel.
        .gpu_timing = true,
        // Scattered air rather than the environment cube. The sun it draws is
        // read from the same light block the ground is lit by, so the two cannot
        // be on separate scales.
        .background_shader = skyShader(),
    });
    defer engine.deinit();
    log.info("device: {s}", .{engine.context.deviceName()});

    var level = try lenore.Level.init(gpa, engine.deps(), &model, .{
        // Two meshes, no skinning. The engine's default holds sixty-four
        // instances, and asking for what this scene draws is what makes a later
        // growth in the scene a change to this line rather than a silent fit.
        .capacity = .{ .instances = 4, .joints = 1 },
    });
    defer level.deinit(&engine.textures);
    try engine.install(&level, &model);

    // `install` framed the camera on the level, which is what an asset viewer
    // wants and what this is not. It sets the near plane to a hundredth of the
    // standoff distance, and the standoff is three times the scene's radius, so
    // a world reaching nine hundred metres puts the near plane at thirty-eight.
    // Everything within thirty-eight metres of a camera standing on the ground
    // is then clipped, which is most of the lower half of the view.
    //
    // Near is a property of the eye and not of the scene: a person cannot get
    // closer to the ground than the ground under them. Far is the corner of the
    // skirt, which is the horizon radius times root two, plus room for the
    // camera to stand away from the centre.
    //
    // Half a metre and not the smallest value that avoids clipping, because the
    // depth buffer is a float cleared to one, which is the unreversed mapping:
    // its precision is spent near the eye and the far field gets what is left.
    // The resolution at a distance d runs about d^2 / (near * 2^24), so at nine
    // hundred metres a near plane of five centimetres resolves a metre, and the
    // water sits nine tenths of one below the ground. Half a metre resolves ten
    // centimetres there and clips nothing: the ground is 1.68 m away when the
    // camera looks straight down at its own feet.
    engine.camera.projection = .{ .perspective = .{ .near = 0.5, .far = 1400.0 } };

    // After `install`, because it is built against the main pass's attachment
    // formats and those follow the renderer. Torn down before the engine, and
    // the engine drains the device on the way out.
    var grass = try grass_module.Grass.init(&engine, gpa, field, .{});
    defer grass.deinit();
    log.info("grass: {d} candidates a frame, capacity {d}", .{
        grass.candidates(),
        grass.settings.capacity,
    });

    var driver: Driver = .{ .field = field, .grass = &grass, .font = loadFont(&engine, io) };
    // Placed before the first frame so that the ground is under the camera on
    // frame zero rather than a fall onto it from wherever the camera was
    // constructed.
    driver.walk.stand(&engine.camera, field);
    try driver.applyLook(&engine);

    try engine.run(&level, .{&driver});
}
