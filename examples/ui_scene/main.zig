const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const imui = @import("lenore-imui");
const lenore = @import("lenore");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const OrbitControl = @import("example-orbit").OrbitControl;

const log = std.log.scoped(.ui_scene);

// A panel over a scene: the exit criterion of the UI stage, and the first thing
// that drives the widget module outside a test.
//
// What it is for is the four properties that cannot be answered on the host,
// each of which has a test and none of which has ever met a real cursor: that a
// click lands on the widget under the pointer and not its neighbour, that the
// order things are drawn in and the order they are hit in agree, that a drag
// keeps its capture when the pointer leaves the widget, and that a region is
// not hit outside the clip it was registered under. Every control below exists
// to make one of them visible, which is also why each one changes the picture:
// a widget whose effect cannot be seen cannot be checked by looking.
//
// The captions are the fifth thing, added when text became a widget. All three
// horizontal alignments are used at once — the row's name at the start, the
// exposure reading at the end, the button's caption centred in it — so a pair
// of them exchanged is visible rather than plausible. The font is the host's
// preferred one, asked for by role, and a host that offers none draws the panel
// without captions.
//
// Usage: run-ui_scene -- <model.glb>
//
//   left drag on the scene     orbit
//   left drag on the panel     the widget under the pointer, and the scene
//                              stays still
//   exposure slider            brightens the whole picture
//   bloom checkbox             the glow around what is brighter than white
//   frame button               puts the camera back where it started

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

// The one light, so that a model with no lights of its own is still lit and the
// exposure slider has something to expose. Direction and intensity are the
// harness's, for the reasons recorded there.
const key_light_direction = res.Vec3{ 0.6, -0.75, 0.28 };
const key_light_intensity = std.math.pi;

// Logical units throughout: what a person sees at the size they see it,
// whatever the output's scale. The conversion to pixels happens once, where a
// rectangle is registered.
const panel_width: f32 = 220;
const panel_inset: f32 = 16;
const row_height: f32 = 26;
const row_gap: f32 = 10;
const panel_padding: f32 = 12;

// Cut so that the last row is half outside it: the top of the third row plus
// half a row, and no bottom padding.
//
// Deliberate, and the panel's own reason for having a fixed height rather than
// an intrinsic one: the clip a region carries is the thing with no host test
// that a cursor can answer. The bottom button's upper half responds and its
// lower half must not, and both halves are one rectangle, so a clip that is not
// applied and a clip applied to the wrong rectangle look different.
const panel_height: f32 = panel_padding + 2 * (row_height + row_gap) + row_height * 0.5;

const exposure_range: imui.SliderRange = .{ .min = 0.25, .max = 4 };

// Between a checkbox and the word that names it, in logical units like every
// other measurement of the panel.
const label_gap: f32 = 8;

// The caption size in the same logical units the panel is laid out in. It
// reaches pixels once, where the font is opened.
const label_points: f32 = 13;

// What the three rows say. The exposure row also carries its value, which is
// the only text here that changes between frames.
const exposure_caption = "exposure";
const bloom_caption = "bloom";
const frame_caption = "frame";

fn hex(comptime value: *const [6:0]u8) res.PremultipliedColor {
    return imui.SrgbColor.fromHex(value).premultiplied();
}

// Enough contrast between the states that a photograph of the window answers
// which one a widget is in.
const palette = struct {
    const panel = hex("1d2433");
    const track = hex("39445c");
    const control = hex("2f3b52");
    const hovered = hex("46567a");
    const held = hex("6e86bb");
    const disabled = hex("262c38");
    const border = hex("8fa6d8");
    const focus = hex("f2c14e");
    const mark = hex("e8ecf5");
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

const label_style: imui.LabelStyle = .{
    .normal_text = palette.mark,
    .disabled_text = palette.track,
    .vertical = .middle,
};

const slider_style: imui.SliderStyle = .{
    .track = palette.track,
    .disabled_track = palette.disabled,
    .knob = control_style,
    .track_thickness = 6,
    .knob_width = 14,
};

// The layout tree, which is fixed: what varies between frames is the rectangle
// it is solved against, not its shape.
//
// Node zero is the root and fills the window. The panel is its one child, held
// to the top right corner by the root's alignment rather than by a computed
// position.
const Row = enum(u32) { exposure = 2, bloom, frame };
const node_count = 5;

const layout_nodes = [node_count]imui.LayoutNode{
    .{
        .child_start = 0,
        .child_count = 1,
        .arrangement = .overlay,
        .padding = .{ .left = panel_inset, .top = panel_inset, .right = panel_inset, .bottom = panel_inset },
        // Under `overlay` the main axis is the horizontal one, so this is the
        // top right corner: right along the main axis, top across it.
        .main_alignment = .end,
        .cross_alignment = .start,
    },
    .{
        .child_start = 1,
        .child_count = 3,
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
    },
    .{ .height = .{ .fixed = row_height } },
    .{ .height = .{ .fixed = row_height } },
    .{ .height = .{ .fixed = row_height } },
};

const layout_children = [_]imui.NodeIndex{ 1, @intFromEnum(Row.exposure), @intFromEnum(Row.bloom), @intFromEnum(Row.frame) };

const panel_node: u32 = 1;

// A checkbox is square and sits at the left of its row.
fn boxOf(row: imui.LogicalRect) imui.LogicalRect {
    return .{ .x = row.x, .y = row.y, .width = row.height, .height = row.height };
}

// The rest of that row, which is where its caption goes. It is drawn and not
// registered: a label is not a target, and a click on the word next to a
// checkbox reaching the box would be a second widget wearing one identity.
fn captionOf(row: imui.LogicalRect) imui.LogicalRect {
    const left = row.height + label_gap;
    return .{
        .x = row.x + left,
        .y = row.y,
        .width = @max(row.width - left, 0),
        .height = row.height,
    };
}

const Driver = struct {
    orbit: OrbitControl = .{},

    // What the panel holds, and the only state a frame carries: every widget is
    // a call, so what a UI remembers is exactly the values its controls stand
    // for.
    exposure: f32 = 1,
    bloom: bool = false,

    // Where the camera goes back to, taken once from the model's bounds.
    framing: res.Sphere,

    // Null when the host offered no font. The panel then draws exactly what it
    // drew before there were captions, which is the whole of what this example
    // was written to demonstrate.
    font: ?lenore.FontId,

    // This frame's rectangles, solved in the registering pass and read again in
    // the drawing pass. Both passes have to place a widget identically, and
    // solving twice is two chances to place it differently.
    rects: [node_count]imui.LogicalRect = undefined,
    scale: imui.ScaleFactor = .identity,
    panel_rect: res.Rect = .{ .x = 0, .y = 0, .width = 0, .height = 0 },

    // Scratch for one solve. It says nothing between frames and is a field only
    // because the frame loop allocates nothing.
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

        // The panel's own background, registered before the controls and so
        // under them: `hitTest` answers with the last region that contains the
        // point. Without it a click that misses a control by a pixel would
        // reach the scene through the panel and swing the camera.
        try ui.register(.{ .string = "panel" }, self.panel_rect, .{});

        try ui.pushId(.{ .string = "controls" });
        defer ui.popId();
        // Every control is registered under the panel's clip, including the one
        // that hangs past its edge. The drawing pass pushes the same clip, so
        // what is hit and what is drawn are cut by one rectangle.
        try ui.pushClip(self.panel_rect);
        defer ui.popClip();

        try ui.register(
            .{ .integer = @intFromEnum(Row.exposure) },
            try self.rects[@intFromEnum(Row.exposure)].toFramebufferFilled(self.scale),
            .{ .focusable = true },
        );
        try ui.register(
            .{ .integer = @intFromEnum(Row.bloom) },
            try boxOf(self.rects[@intFromEnum(Row.bloom)]).toFramebufferFilled(self.scale),
            .{ .focusable = true },
        );
        try ui.register(
            .{ .integer = @intFromEnum(Row.frame) },
            try self.rects[@intFromEnum(Row.frame)].toFramebufferFilled(self.scale),
            .{ .focusable = true },
        );
    }

    pub fn onEvent(
        self: *Driver,
        engine: *lenore.Engine,
        event: platform.Event,
        ui_consumed: bool,
    ) !void {
        // What the UI took, it took. This is this frame's answer and not the
        // previous frame's, so a press that lands on a panel which appeared
        // this frame does not also start an orbit that lasts until the button
        // comes back up.
        //
        // A drag that began on the scene keeps going: once the orbit holds the
        // pointer the UI has no region under it, and the motion arrives here
        // unconsumed.
        if (ui_consumed) {
            if (event.payload == .mouse_button and event.payload.mouse_button.action == .press)
                self.orbit.cancel();
            return;
        }

        switch (event.payload) {
            .cursor => |cursor| self.orbit.dragTo(
                &engine.camera,
                cursor.logical_position,
                event.timestamp_ns,
            ),
            .mouse_button => |button| if (button.button == .left) switch (button.action) {
                .press => self.orbit.begin(button.logical_position, event.timestamp_ns),
                .release => self.orbit.end(
                    &engine.camera,
                    button.logical_position,
                    event.timestamp_ns,
                ),
                .repeat => {},
            },
            .focus => |focus| if (!focus.focused) self.orbit.cancel(),
            .key => |key| if (key.action == .press and key.physical == .escape)
                engine.requestExit(),
            else => {},
        }
    }

    pub fn onResize(_: *Driver, _: *lenore.Engine, _: platform.Extent2D) !void {}
    pub fn onCompute(_: *Driver, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}
    pub fn onRecord(_: *Driver, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}

    pub fn onUiDraw(
        self: *Driver,
        engine: *lenore.Engine,
        _: *lenore.Level,
        ui: *imui.WidgetContext,
    ) !void {
        const canvas = try ui.drawList();
        try canvas.addQuad(self.panel_rect, .{}, palette.panel, engine.ui_white);

        // The same scope and the same clip the registering pass pushed, in the
        // same order. That is what makes a widget here the widget that was
        // registered there: nothing checks it, and a pass that walked
        // differently asks for an identity this frame never registered.
        try ui.pushId(.{ .string = "controls" });
        defer ui.popId();
        try ui.pushClip(self.panel_rect);
        defer ui.popClip();

        if (try ui.slider(
            .{ .integer = @intFromEnum(Row.exposure) },
            try self.rects[@intFromEnum(Row.exposure)].toFramebufferFilled(self.scale),
            &self.exposure,
            exposure_range,
            slider_style,
            true,
        )) engine.look.post.exposure = self.exposure;

        // Over the track, which is where a value belongs when the control it
        // belongs to is the whole row: the word at one end, the number the drag
        // produced at the other.
        const exposure_row = self.rects[@intFromEnum(Row.exposure)];
        try self.caption(engine, ui, exposure_row, exposure_caption, .start);
        var reading: [16]u8 = undefined;
        try self.caption(engine, ui, exposure_row, try std.fmt.bufPrint(
            &reading,
            "{d:.2}",
            .{self.exposure},
        ), .end);

        if (try ui.checkbox(
            .{ .integer = @intFromEnum(Row.bloom) },
            try boxOf(self.rects[@intFromEnum(Row.bloom)]).toFramebufferFilled(self.scale),
            &self.bloom,
            checkbox_style,
            true,
        )) engine.look.bloom = if (self.bloom) .{} else null;

        try self.caption(
            engine,
            ui,
            captionOf(self.rects[@intFromEnum(Row.bloom)]),
            bloom_caption,
            .start,
        );

        // Half of this one is outside the panel, and only the half inside
        // responds. Its lower edge is where a clip that is not being applied
        // announces itself.
        if (try ui.button(
            .{ .integer = @intFromEnum(Row.frame) },
            try self.rects[@intFromEnum(Row.frame)].toFramebufferFilled(self.scale),
            control_style,
            true,
        )) {
            self.orbit.cancel();
            _ = engine.frameCamera(self.framing);
        }

        // Centred in the button, and half of it is below the panel's edge like
        // the button itself: the caption is cut by the same clip, which is what
        // says the label went through the canvas the widgets did.
        try self.caption(
            engine,
            ui,
            self.rects[@intFromEnum(Row.frame)],
            frame_caption,
            .center,
        );
    }

    // One caption, shaped and placed in the row it names.
    //
    // The scratch is the frame's, on the stack: a run is shaped into it and
    // drawn from it, and nothing outlives the call. Shaping allocates nothing
    // as long as the glyphs are ones the face was loaded with, which for these
    // three words and a number they are.
    //
    // The three horizontal alignments are all used across the panel, which is
    // deliberate: swap two of them and the picture says so.
    fn caption(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        row: imui.LogicalRect,
        source: []const u8,
        horizontal: imui.LabelStyle.Horizontal,
    ) !void {
        const font = self.font orelse return;

        // One placement per glyph per subpixel bucket, reserved at the most any
        // face wants: how many a face has is settled when it is opened, and an
        // array is sized before that.
        var glyphs: [64]res.ShapedGlyph = undefined;
        var placements: [64 * res.SubpixelBuckets.most.count()]res.GlyphPlacement = undefined;
        // Shaped once, and the value carries its own width and height. The
        // arrays it borrows are this call's, which is enough here because the
        // caption is measured and drawn in one place; a caller that sized a
        // layout node from it would have to keep them for the frame.
        const line = try engine.shapeLabel(font, source, &glyphs, &placements);

        var style = label_style;
        style.horizontal = horizontal;
        try ui.label(try row.toFramebufferFilled(self.scale), style, line, true);
    }

    pub fn onUpdate(self: *Driver, engine: *lenore.Engine, _: *lenore.Level, time: lenore.FrameTime) !void {
        self.orbit.advance(&engine.camera, time.delta);
    }
};

// The font this host prefers for an interface, or nothing.
//
// The size is `label_points` against the scale the engine knows now, which is
// the output's once the window has reported its metrics and one before that. A
// face is a face at one size, so this is fixed for the run either way.
//
// Nothing is not a failure. A host with no fontconfig and a host whose
// configuration matches no font both answer that way, and the panel is legible
// without captions: every control changes the picture, which is what this
// example is read by.
fn loadFont(engine: *lenore.Engine, io: std.Io) ?lenore.FontId {
    const scale = engine.uiScale();
    const pixels: u32 = @intFromFloat(@round(label_points * scale.y));

    return engine.loadSystemFont(io, .{}, pixels) catch |err| {
        log.warn("the host's font did not open: {t}", .{err});
        return null;
    } orelse {
        log.warn("the host offers no font, so the panel draws no captions", .{});
        return null;
    };
}

pub fn main(process: std.process.Init.Minimal) !void {
    const gpa = if (checking) debug_allocator.allocator() else std.heap.smp_allocator;
    defer if (checking) {
        if (debug_allocator.deinit() == .leak) log.err("host memory leaked", .{});
    };

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var argument_iterator: std.process.Args.Iterator = try .initAllocator(process.args, gpa);
    defer argument_iterator.deinit();
    _ = argument_iterator.next();
    const model_path = argument_iterator.next() orelse {
        log.err("usage: run-ui_scene -- <model.glb>", .{});
        return error.BadArguments;
    };

    // The loader confines a document's references to a root, so it takes a
    // directory and a name inside it rather than a path.
    const model_directory = std.fs.path.dirname(model_path) orelse ".";
    const model_name = std.fs.path.basename(model_path);
    var root = try std.Io.Dir.cwd().openDir(io, model_directory, .{});
    defer root.close(io);

    var loaded = try gltf.loader.open(gpa, io, root, model_name);
    defer loaded.deinit(gpa);
    var model = try gltf.importer.build(gpa, &loaded.document, loaded.directory);
    defer model.deinit(gpa);
    try gltf.importer.readExternalImages(gpa, io, root, &model);

    var engine: lenore.Engine = undefined;
    try engine.init(gpa, .{
        .title = "Lenore UI",
        .material_capacity = @intCast(@max(model.materials.len, 1)),
    });
    defer engine.deinit();
    log.info("device: {s}", .{engine.context.deviceName()});

    var light_block: [gpu.max_lights]gpu.LightUniform = undefined;
    light_block[0] = lenore.packLight(try .directional(
        .{ 1, 1, 1 },
        key_light_intensity,
        key_light_direction,
    ));
    engine.look = .{
        .lights = light_block[0..1],
        .sun_shadow = .{ .enabled = true, .strength = 1, .normal_offset_texels = 1 },
    };

    // The engine's own default, named here because the level is built against
    // the same number the frame's rings hold.
    var level = try lenore.Level.init(gpa, engine.deps(), &model, .{
        .capacity = .{ .instances = 64, .joints = 256 },
    });
    defer level.deinit(&engine.textures);
    try engine.install(&level, &model);

    var driver: Driver = .{ .framing = level.world.sphere, .font = loadFont(&engine, io) };
    try engine.run(&level, &driver);
}
