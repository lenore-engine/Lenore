const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const imui = @import("lenore-imui");
const lenore = @import("lenore");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");

const shell = @import("shell.zig");
const theme = @import("theme.zig");

const log = std.log.scoped(.editor);

// The editor: an application over the engine, in the engine's own repository.
//
// It lives here rather than in `examples/` because it is not a demonstration of
// anything, and here rather than in a repository of its own because an editor
// is a tool of the engine. What is not the engine's is the game, which consumes
// the public surface as a stranger would and belongs outside.
//
// **Nothing in this file may reach inside a module.** The editor is the first
// application written against the whole composition, so what it cannot say
// through the public surface is a gap in that surface rather than a reason to
// look past it.
//
// The frame is a top bar, a viewport, a dock of two panels and a status bar,
// with a splitter between the dock and the viewport and another between the two
// panels. Its shape is a layout literal in `shell.zig`, solved once per frame;
// this file registers regions in the rectangles that come out and draws in
// them. Nothing here computes a panel's edge.
//
// **The viewport is not a region.** Everything else is, so a click on the dock
// or on a bar is the interface's, and what lands on the viewport reaches the
// driver because no widget wanted it. That is also what tells the wheel apart:
// over the list it scrolls, over the viewport nothing scrolls and the camera
// moves.
//
// What the reference interface has and this does not, stated so that it is
// absence rather than oversight:
//
// **A scene outliner over nodes.** `gltf.importer.Model` carries meshes,
// materials, images, lights, skins and animation templates, and no node list at
// all. The dock lists materials because a material is the only thing in a
// loaded document that carries a name a person can read. Nodes need their names
// and their hierarchy carried through the importer first.
//
// **Menus.** A menu draws over everything registered before it and takes input
// in front of it, which is a second painter layer rather than another widget.
// The bar therefore has no menu titles: a title that opened nothing would be a
// promise the interface does not keep.
//
// **Icons.** There is no icon path, so the toolbar the reference draws down the
// left of the viewport is not here.

const usage =
    \\usage: run-editor [-- <model.glb>]
    \\
    \\  drag             orbit the camera, over the viewport
    \\  drag a splitter  move the dock's edge, or the split inside it
    \\  wheel            dolly over the viewport, scroll over the list
    \\  tab              move the keyboard between the field and the list
    \\  up, down         move the selection, with the list focused
    \\  escape           drop the keyboard focus, or quit when nothing holds it
;

// The caption size, in logical units. Fourteen is what a comparison of four
// raster sizes put first on the reference display, which is a measured
// preference for one face on one display rather than a rule, so it stays in the
// application.
const label_points: f32 = 14;

// One line at a time is shaped, and the longest name a document is likely to
// carry is well inside this. A name past it leaves its row without a caption,
// which is visible rather than silent.
const max_glyphs = 128;
const max_placements = max_glyphs * res.SubpixelBuckets.most.count();

// How much text the filter holds. A filter is a word or two, and a field that
// stops accepting at the end of a phrase is telling the truth about itself.
const filter_capacity = 64;

// What the field says when it is empty, so that an empty box is a box with a
// purpose rather than a box.
const filter_placeholder = "Filter materials";

// Where the version comes from is `build.zig.zon`, which nothing here can read
// at comptime. Written out until there is somewhere to read it from.
const version = "LENORE ENGINE 0.0.1";

// The viewport's own light, over the viewer's shoulder and a little to the
// side, so that a face turned towards the camera is lit and the two faces away
// from it fall off at different rates. Straight down the view axis flattens
// everything it shows.
const key_light_direction = res.Vec3{ 0.6, -0.75, 0.28 };
// Pi, which is what turns a unit radiance into the irradiance a Lambertian
// surface receives from a full hemisphere.
const key_light_intensity = std.math.pi;
const key_light_colour = res.Vec3{ 1, 1, 1 };

// What the frame's rings and the material storage are sized for.
//
// The editor derives them from the document, which it can because it loads one
// before the device exists. The engine cannot: `FrameSet` is built inside
// `Renderer.init`, lives in set 0 and is addressed by dynamic offsets, so
// re-sizing it once a level is known means rewriting those descriptors.
//
// **The floor is what a document opened later has to fit**, and it is the one
// number here that is a guess rather than a measurement. Opening a document
// larger than the rings were built for cannot work until the engine can resize
// them, so the editor says so at the point of opening rather than failing with a
// capacity error from three layers down.
//
// The ceilings are measured: 10 240 draws and 10 240 materials are what
// `NodePerformanceTest` asks for, and 1 062 joints are what `BrainStem` asks
// for. A document past them is refused by name.
const capacity_floor: Capacities = .{
    .instances = 1024,
    .joints = 512,
    .materials = 64,
    .morph_meshes = 16,
    .morph_weights = 256,
};
const capacity_ceiling: Capacities = .{
    .instances = 10240,
    .joints = 2048,
    .materials = 10240,
    .morph_meshes = 1024,
    .morph_weights = 16384,
};

const Capacities = struct {
    instances: usize,
    joints: usize,
    materials: u32,
    morph_meshes: u32,
    morph_weights: usize,

    // What this document needs, or an error naming what it exceeded.
    //
    // Counted rather than estimated: a draw per mesh, a joint run per skinned
    // mesh, and one morph destination per mesh that carries targets. Those are
    // the same three numbers the level and the prepass validate against, so a
    // document that passes here is one they accept.
    fn forDocument(model: *const gltf.importer.Model) error{DocumentTooLarge}!Capacities {
        var joints: usize = 0;
        var morph_meshes: u32 = 0;
        var morph_weights: usize = 0;
        for (model.meshes) |*mesh| {
            if (mesh.skin) |skin| joints += skin.joint_count;
            if (mesh.morph) |deltas| {
                morph_meshes += 1;
                morph_weights += deltas.target_count;
            }
        }

        const wanted: Capacities = .{
            .instances = model.meshes.len,
            .joints = joints,
            .materials = @intCast(model.materials.len),
            .morph_meshes = morph_meshes,
            .morph_weights = morph_weights,
        };
        if (wanted.instances > capacity_ceiling.instances or
            wanted.joints > capacity_ceiling.joints or
            wanted.materials > capacity_ceiling.materials or
            wanted.morph_meshes > capacity_ceiling.morph_meshes or
            wanted.morph_weights > capacity_ceiling.morph_weights)
        {
            log.err(
                "document needs {d} draws, {d} joints, {d} materials, {d} morphed meshes and {d} weights; the ceiling is {d}, {d}, {d}, {d}, {d}",
                .{
                    wanted.instances,              wanted.joints,
                    wanted.materials,              wanted.morph_meshes,
                    wanted.morph_weights,          capacity_ceiling.instances,
                    capacity_ceiling.joints,       capacity_ceiling.materials,
                    capacity_ceiling.morph_meshes, capacity_ceiling.morph_weights,
                },
            );
            return error.DocumentTooLarge;
        }

        return .{
            .instances = @max(wanted.instances, capacity_floor.instances),
            .joints = @max(wanted.joints, capacity_floor.joints),
            .materials = @max(wanted.materials, capacity_floor.materials),
            .morph_meshes = @max(wanted.morph_meshes, capacity_floor.morph_meshes),
            .morph_weights = @max(wanted.morph_weights, capacity_floor.morph_weights),
        };
    }
};

// Dark, but not the black a window shows before anything is drawn: an empty
// viewport and a viewport that failed to render should not look the same.
const viewport_clear: [4]f32 = .{ 0.035, 0.036, 0.042, 1 };

// The modes the reference bar offers. Only one of them has anything behind it,
// and the other three are drawn disabled rather than left out: what an editor
// will be is part of what its bar says, and a disabled control says it without
// pretending to work.
const Mode = enum { two_d, three_d, script, game };

// How wide one tab is, in logical units. Wide enough for the longest caption a
// strip carries at the body size, which is "FileSystem".
const tab_width: f32 = 84;

// The document strip when nothing is open, which is one tab saying so rather
// than an empty strip: an editor with no scene has a place for one.
const empty_document_tabs = [_]Tab{.{ .caption = "[empty]" }};

const filter_key: imui.IdKey = .{ .string = "scene.filter" };
const list_key: imui.IdKey = .{ .string = "scene.list" };
const rows_key: imui.IdKey = .{ .string = "scene.rows" };
const modes_key: imui.IdKey = .{ .string = "menu.modes" };
const tools_key: imui.IdKey = .{ .string = "viewport.tools" };

// The tab strips, one scope each, so a tab's identity is its index inside its
// own strip and two strips cannot name the same widget.
const scene_tabs_key: imui.IdKey = .{ .string = "tabs.scene" };
const files_tabs_key: imui.IdKey = .{ .string = "tabs.files" };
const output_tabs_key: imui.IdKey = .{ .string = "tabs.output" };
const inspector_tabs_key: imui.IdKey = .{ .string = "tabs.inspector" };
const document_tabs_key: imui.IdKey = .{ .string = "tabs.document" };

// The surfaces that take a click so that it does not fall through to the
// camera. The viewport is deliberately not among them.
const opaque_panels = [_]struct { name: shell.Name, key: imui.IdKey }{
    .{ .name = .menu_bar, .key = .{ .string = "panel.menu" } },
    .{ .name = .status_bar, .key = .{ .string = "panel.status" } },
    .{ .name = .left_dock, .key = .{ .string = "panel.left" } },
    .{ .name = .right_dock, .key = .{ .string = "panel.right" } },
    .{ .name = .output, .key = .{ .string = "panel.output" } },
    .{ .name = .document_tabs, .key = .{ .string = "panel.document-tabs" } },
    .{ .name = .viewport_toolbar, .key = .{ .string = "panel.viewport-tools" } },
};

// The four edges the shell can drag, paired with the node each one sits in.
const splitters = [_]struct {
    name: shell.Name,
    key: imui.IdKey,
    drag: *const fn (*shell.Shell, shell.Frame, imui.Interaction) void,
}{
    .{ .name = .left_edge, .key = shell.left_edge_key, .drag = shell.Shell.dragLeftEdge },
    .{ .name = .right_edge, .key = shell.right_edge_key, .drag = shell.Shell.dragRightEdge },
    .{ .name = .left_split, .key = shell.left_split_key, .drag = shell.Shell.dragLeftSplit },
    .{ .name = .centre_split, .key = shell.centre_split_key, .drag = shell.Shell.dragCentreSplit },
};

// One strip's worth of tabs. `ready` is false for a tab that names something
// the editor will show and does not yet: drawn disabled rather than left out,
// because what a panel will hold is part of what the panel says, and a disabled
// tab says it without pretending to work.
const Tab = struct { caption: []const u8, ready: bool = true };

// In `Mode`'s own order, because the strip's index is the enum's value: a tab
// chosen by index becomes a mode without a table between them.
const modes_as_tabs = [_]Tab{
    .{ .caption = "2D", .ready = false },
    .{ .caption = "3D" },
    .{ .caption = "Script", .ready = false },
    .{ .caption = "Game", .ready = false },
};

const scene_tabs = [_]Tab{ .{ .caption = "Scene" }, .{ .caption = "Import", .ready = false } };
const files_tabs = [_]Tab{
    .{ .caption = "FileSystem" },
    .{ .caption = "History", .ready = false },
};
const output_tabs = [_]Tab{
    .{ .caption = "Output" },
    .{ .caption = "Debugger", .ready = false },
    .{ .caption = "Profiler", .ready = false },
};
const inspector_tabs = [_]Tab{
    .{ .caption = "Inspector" },
    .{ .caption = "Signals", .ready = false },
};

// The viewport's tools, which are what a click in it would do. None of them
// does anything yet, so all of them are disabled: the row exists because the
// row is where they go, and an empty row would be a void the reader has to
// guess about.
const viewport_tools = [_]Tab{
    .{ .caption = "Select", .ready = false },
    .{ .caption = "Move", .ready = false },
    .{ .caption = "Rotate", .ready = false },
    .{ .caption = "Scale", .ready = false },
};

// Where the mode strip begins, so that it is centred on the bar rather than run
// from its left edge like every other strip.
fn modeStripLeft(frame: shell.Frame) f32 {
    const bar = frame.rect(.menu_bar);
    const total = tab_width * frame.scale.x * modes_as_tabs.len;
    return bar.x + (bar.width - total) * 0.5;
}

fn rect(x: f32, y: f32, width: f32, height: f32) res.Rect {
    return .{ .x = x, .y = y, .width = width, .height = height };
}

// A rectangle inset on the horizontal axis alone, for a bar whose height is
// already one line. Only this axis, because insetting a row of twenty-two by a
// gutter of eight leaves six, and a caption centred in six is a caption drawn
// outside its own bar. Where a panel wants padding on every side, the layout
// carries it and nothing here insets anything.
fn insetX(area: res.Rect, by: f32) res.Rect {
    const taken = @min(by, area.width * 0.5);
    return .{
        .x = area.x + taken,
        .y = area.y,
        .width = @max(area.width - 2 * taken, 0),
        .height = area.height,
    };
}

// `area` split into a left part of `width` and the rest.
fn splitLeft(area: res.Rect, width: f32) struct { res.Rect, res.Rect } {
    const taken = @min(width, area.width);
    return .{
        .{ .x = area.x, .y = area.y, .width = taken, .height = area.height },
        .{
            .x = area.x + taken,
            .y = area.y,
            .width = area.width - taken,
            .height = area.height,
        },
    };
}

// One line's worth off the top of `area`, and what is left under it.
fn splitTop(area: res.Rect, height: f32) struct { res.Rect, res.Rect } {
    const taken = @min(height, area.height);
    return .{
        .{ .x = area.x, .y = area.y, .width = area.width, .height = taken },
        .{
            .x = area.x,
            .y = area.y + taken,
            .width = area.width,
            .height = area.height - taken,
        },
    };
}

const Driver = struct {
    pub const hooks: lenore.Hooks(Driver) = .{
        .ui_regions = onUiRegions,
        .event = onEvent,
        .ui_draw = onUiDraw,
    };

    white: res.ImageHandle,
    // Null when the host had no font. Every panel still draws and still routes,
    // which keeps a missing font a line in the log rather than a refusal to
    // start.
    font: ?lenore.FontId,

    shell_state: shell.Shell = .{},
    // Solved in the registering pass and read again in the drawing one, so both
    // passes place a widget in the same rectangle. A resize routed between them
    // moves the swapchain and not this, which is the answer that keeps the two
    // agreeing.
    frame: shell.Frame = .{},
    mode: Mode = .three_d,

    // The file the document came from, for the panel that lists it, and the tab
    // that names it. Both borrow the path `main` holds for the whole run.
    document_name: ?[]const u8 = null,
    document_tab: ?[1]Tab = null,

    // The document's materials, borrowed from the model `main` holds for the
    // whole run.
    materials: []const res.MaterialInfo,

    // Which materials the filter admits, in document order. Sized to the
    // document at startup, so a frame allocates nothing and no document is too
    // long to list.
    matched: []u32,
    matched_count: usize = 0,

    filter_bytes: [filter_capacity]u8 = undefined,
    filter: imui.TextFieldState = .{},

    // Read by both passes and written at the end of the drawing one, so the
    // rows a frame registered are the rows it drew. A wheel therefore takes
    // effect on the next frame, which is the step every other piece of
    // per-frame state takes.
    list_scroll: f32 = 0,
    selected: ?u32 = null,

    // The camera, which is the editor's policy and not the engine's.
    framing: res.Sphere,
    orbiting: bool = false,
    last_pointer: ?[2]f32 = null,

    const orbit_sensitivity: f32 = 0.006;
    // Short of straight down, so the basis is never asked what it does at the
    // pole.
    const pitch_limit: f32 = std.math.pi / 2.0 - 0.01;
    // One turn of the wheel as a proportion of the distance to the pivot. A
    // multiplicative step is what makes the approach feel the same far away and
    // close in.
    const dolly_rate: f32 = 0.12;

    pub fn onEvent(
        self: *Driver,
        engine: *lenore.Engine,
        event: platform.Event,
        taken: bool,
    ) !void {
        switch (event.payload) {
            .key => |key| switch (key.physical) {
                // Escape drops the keyboard first and quits second. The UI
                // reports whether it took the keystroke, so a field losing
                // focus and the editor closing are told apart by the UI rather
                // than by a flag kept here.
                .escape => if (key.action == .press and !taken) engine.requestExit(),
                else => {},
            },
            .mouse_button => |button| {
                if (button.button != .left) return;
                switch (button.action) {
                    .press => {
                        // Only where the UI wanted nothing, which is the
                        // viewport: everything else registers a region.
                        if (taken) return;
                        self.orbiting = true;
                        self.last_pointer = button.logical_position;
                    },
                    .release => {
                        self.orbiting = false;
                        self.last_pointer = null;
                    },
                    .repeat => {},
                }
            },
            .cursor => |cursor| {
                const previous = self.last_pointer orelse {
                    if (self.orbiting) self.last_pointer = cursor.logical_position;
                    return;
                };
                self.last_pointer = cursor.logical_position;
                if (!self.orbiting) return;
                orbit(
                    engine,
                    cursor.logical_position[0] - previous[0],
                    cursor.logical_position[1] - previous[1],
                );
            },
            .scroll => |wheel| {
                if (taken) return;
                self.dolly(engine, wheel.line_delta[1]);
            },
            .focus => |focus| if (!focus.focused) {
                // A drag that ends outside the window ends with no release, and
                // a camera left orbiting would swing on the next motion after
                // the window came back.
                self.orbiting = false;
                self.last_pointer = null;
            },
            else => {},
        }
    }

    fn orbit(engine: *lenore.Engine, dx: f32, dy: f32) void {
        engine.camera.yaw += dx * orbit_sensitivity;
        engine.camera.pitch = std.math.clamp(
            engine.camera.pitch - dy * orbit_sensitivity,
            -pitch_limit,
            pitch_limit,
        );
    }

    fn dolly(self: *Driver, engine: *lenore.Engine, lines: f32) void {
        const anchor = switch (engine.camera.anchor) {
            .orbit => |*orbit_anchor| orbit_anchor,
            // A camera detached from a pivot has no distance to change, and
            // nothing here detaches one.
            .eye => return,
        };
        anchor.distance = std.math.clamp(
            anchor.distance * @exp(-lines * dolly_rate),
            @max(self.framing.radius * 0.05, 1.0e-3),
            self.framing.radius * 100,
        );
    }

    // The registering pass, which runs before the frame's events are routed.
    //
    // Everything it registers is a function of state as it stood when the frame
    // began, and the drawing pass walks the same state. That is what makes the
    // two passes agree: the filter's edits, the wheel and the splitters all
    // land at the end of the drawing pass and are seen by the frame after this
    // one.
    pub fn onUiRegions(
        self: *Driver,
        engine: *lenore.Engine,
        _: *lenore.Level,
        ui: *imui.WidgetContext,
    ) !void {
        const extent = engine.swapchain.currentExtent();
        try self.frame.solve(
            .{ @floatFromInt(extent.width), @floatFromInt(extent.height) },
            engine.uiScale(),
            self.shell_state,
        );

        for (opaque_panels) |panel| try ui.register(panel.key, self.frame.rect(panel.name), .{});
        for (splitters) |bar| try ui.register(bar.key, self.frame.rect(bar.name), .{});

        try self.registerTabs(ui, modes_key, self.frame.rect(.menu_bar), &modes_as_tabs, modeStripLeft(self.frame));
        try self.registerTabs(ui, tools_key, self.frame.rect(.viewport_toolbar), &viewport_tools, null);
        try self.registerTabs(ui, document_tabs_key, self.frame.rect(.document_tabs), self.documentTabs(), null);
        try self.registerTabs(ui, scene_tabs_key, self.frame.rect(.scene_tabs), &scene_tabs, null);
        try self.registerTabs(ui, files_tabs_key, self.frame.rect(.files_tabs), &files_tabs, null);
        try self.registerTabs(ui, output_tabs_key, self.frame.rect(.output_tabs), &output_tabs, null);
        try self.registerTabs(ui, inspector_tabs_key, self.frame.rect(.inspector_tabs), &inspector_tabs, null);

        const list_area = self.frame.rect(.scene_content);
        try ui.register(filter_key, self.frame.rect(.scene_filter), .{ .focusable = true });
        try ui.register(list_key, list_area, .{ .scrollable = true, .focusable = true });

        self.refreshMatches();

        const first, const count = self.rowWindow(list_area);
        try ui.pushClip(list_area);
        try ui.pushId(rows_key);
        for (self.matched[first..][0..count], first..) |material, row| {
            try ui.register(.{ .integer = material }, self.rowRect(list_area, row), .{});
        }
        ui.popId();
        ui.popClip();
    }

    // One strip's regions, in its own scope so a tab is named by its index.
    //
    // The rectangles are computed once here and once again where the strip is
    // drawn, both through `tabRect`, because the two passes have to place a tab
    // in the same place and one arithmetic used twice is what makes that true.
    fn registerTabs(
        self: *const Driver,
        ui: *imui.WidgetContext,
        scope: imui.IdKey,
        strip: res.Rect,
        tabs: []const Tab,
        left: ?f32,
    ) !void {
        try ui.pushId(scope);
        defer ui.popId();
        for (tabs, 0..) |tab, index| {
            try ui.register(
                .{ .integer = index },
                self.tabRect(strip, tabs, index, left),
                .{ .enabled = tab.ready, .focusable = tab.ready },
            );
        }
    }

    // Every material whose name contains the filter, or every material when the
    // filter is empty.
    //
    // A plain substring over bytes, folded for ASCII only. A case rule for the
    // rest of Unicode is a table this file does not carry, and a filter that
    // quietly used a wrong one would be worse than one that says it matches
    // bytes.
    fn refreshMatches(self: *Driver) void {
        const needle = self.filter_bytes[0..self.filter.len];
        self.matched_count = 0;
        for (self.materials, 0..) |material, index| {
            if (needle.len != 0 and !containsFold(material.name, needle)) continue;
            self.matched[self.matched_count] = @intCast(index);
            self.matched_count += 1;
        }
    }

    pub fn onUiDraw(
        self: *Driver,
        engine: *lenore.Engine,
        _: *lenore.Level,
        ui: *imui.WidgetContext,
    ) !void {
        var scratch: Scratch = .{ .font = self.font, .white = self.white };
        const canvas = try ui.drawList();

        // The surfaces first, so everything below is drawn over its own panel
        // rather than over whatever the scene left there.
        for ([_]shell.Name{ .menu_bar, .status_bar, .left_dock, .right_dock, .output }) |name| {
            try canvas.addQuad(self.frame.rect(name), .{}, theme.panel.premultiplied(), self.white);
        }
        for ([_]shell.Name{ .document_tabs, .viewport_toolbar }) |name| {
            try canvas.addQuad(self.frame.rect(name), .{}, theme.raised.premultiplied(), self.white);
        }

        try self.drawMenuBar(engine, ui, &scratch);
        try self.drawCentre(engine, ui, &scratch);
        try self.drawScene(engine, ui, &scratch);
        try self.drawFiles(engine, ui, &scratch);
        try self.drawOutput(engine, ui, &scratch);
        try self.drawInspector(engine, ui, &scratch);
        try self.drawStatusBar(engine, ui, &scratch);

        for (splitters) |bar| {
            _ = try ui.splitter(bar.key, self.frame.rect(bar.name), theme.splitter());
        }

        // The drags and the wheel are applied last, so everything above drew
        // against the geometry its regions were registered under.
        for (splitters) |bar| bar.drag(&self.shell_state, self.frame, try ui.interaction(bar.key));

        const list_area = self.frame.rect(.scene_content);
        const content = @as(f32, @floatFromInt(self.matched_count)) * self.frame.rowHeight();
        _ = try ui.scroll(list_key, content, list_area.height, &self.list_scroll);

        const list = try ui.interaction(list_key);
        if (list.adjust != 0) self.moveSelection(list.adjust);
        if (list.focused or list.adjust != 0) self.revealSelection(list_area, content);
    }

    fn drawMenuBar(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        scratch: *Scratch,
    ) !void {
        const bar = self.frame.rect(.menu_bar);
        try scratch.line(
            engine,
            ui,
            insetX(bar, shell.metrics.gutter * self.frame.scale.x),
            theme.label(),
            "LENORE",
            true,
        );

        const chosen = try self.drawTabs(
            engine,
            ui,
            scratch,
            modes_key,
            bar,
            &modes_as_tabs,
            @backingInt(self.mode),
            modeStripLeft(self.frame),
        );
        self.mode = @fromBackingInt(@intCast(chosen));
    }

    fn drawCentre(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        scratch: *Scratch,
    ) !void {
        _ = try self.drawTabs(
            engine,
            ui,
            scratch,
            document_tabs_key,
            self.frame.rect(.document_tabs),
            self.documentTabs(),
            0,
            null,
        );
        _ = try self.drawTabs(
            engine,
            ui,
            scratch,
            tools_key,
            self.frame.rect(.viewport_toolbar),
            &viewport_tools,
            viewport_tools.len,
            null,
        );

        // The projection, over the picture at its corner, which is where an
        // editor puts what the view is rather than in a panel about it.
        const viewport = self.frame.rect(.viewport);
        const chip = rect(
            viewport.x + shell.metrics.gutter * self.frame.scale.x,
            viewport.y + shell.metrics.gutter * self.frame.scale.y,
            110 * self.frame.scale.x,
            shell.metrics.row * self.frame.scale.y,
        );
        const canvas = try ui.drawList();
        try canvas.addQuad(chip, .{}, theme.raised.premultiplied(), self.white);
        try scratch.line(
            engine,
            ui,
            insetX(chip, shell.metrics.gutter * self.frame.scale.x),
            theme.labelDim(),
            "Perspective",
            true,
        );
    }

    fn drawScene(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        scratch: *Scratch,
    ) !void {
        try scratch.tabBackground(ui, self.frame, self.frame.rect(.scene_tabs));
        _ = try self.drawTabs(
            engine,
            ui,
            scratch,
            scene_tabs_key,
            self.frame.rect(.scene_tabs),
            &scene_tabs,
            0,
            null,
        );

        const filter_area = self.frame.rect(.scene_filter);
        if (self.font) |font| {
            const style = theme.field(self.frame.scale);
            // Twice, and the two are not the same run. The first is the text as
            // it was drawn last, which is what a click is resolved against; the
            // second is what the keystroke left behind, which is what is about
            // to be drawn. Shaping once would put the caret a frame behind the
            // typing.
            const before = try engine.shapeLabel(
                font,
                self.filter_bytes[0..self.filter.len],
                &scratch.glyphs,
                &scratch.placements,
            );
            _ = try ui.editText(
                filter_key,
                filter_area,
                style,
                &self.filter_bytes,
                &self.filter,
                before,
                true,
            );
            const after = try engine.shapeLabel(
                font,
                self.filter_bytes[0..self.filter.len],
                &scratch.glyphs,
                &scratch.placements,
            );
            try ui.textField(filter_key, filter_area, style, &self.filter, after, true);

            // Over the field rather than inside it, because a placeholder is
            // not the field's content: shaped into the run instead, it would be
            // what the caret is measured against and a click on an empty field
            // would land inside a word nobody typed.
            if (self.filter.len == 0) {
                try scratch.line(
                    engine,
                    ui,
                    insetX(filter_area, style.padding),
                    theme.labelFaint(),
                    filter_placeholder,
                    true,
                );
            }
        }

        const canvas = try ui.drawList();
        const list_area = self.frame.rect(.scene_content);
        try canvas.addQuad(list_area, .{}, theme.sunken.premultiplied(), self.white);

        const first, const count = self.rowWindow(list_area);
        try ui.pushClip(list_area);
        defer ui.popClip();
        try ui.pushId(rows_key);
        defer ui.popId();

        for (self.matched[first..][0..count], first..) |material, row| {
            const key: imui.IdKey = .{ .integer = material };
            const area = self.rowRect(list_area, row);
            const state = try ui.interaction(key);
            if (state.pressed) self.selected = material;

            if (self.selected == material) {
                try canvas.addQuad(area, .{}, theme.row_selected.premultiplied(), self.white);
            } else if (state.hovered) {
                try canvas.addQuad(area, .{}, theme.row_hovered.premultiplied(), self.white);
            }
            try scratch.line(
                engine,
                ui,
                insetX(area, shell.metrics.gutter * self.frame.scale.x),
                if (self.selected == material) theme.label() else theme.labelDim(),
                self.materials[material].name,
                true,
            );
        }
    }

    fn drawFiles(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        scratch: *Scratch,
    ) !void {
        try scratch.tabBackground(ui, self.frame, self.frame.rect(.files_tabs));
        _ = try self.drawTabs(
            engine,
            ui,
            scratch,
            files_tabs_key,
            self.frame.rect(.files_tabs),
            &files_tabs,
            0,
            null,
        );

        // One entry, which is the document that was opened. A directory tree
        // needs the editor to be able to open a second document, which is the
        // thing the file panel exists for and the thing it cannot do yet.
        const row, _ = splitTop(self.frame.rect(.files_content), self.frame.rowHeight());
        try scratch.line(
            engine,
            ui,
            row,
            theme.labelDim(),
            self.document_name orelse "no document",
            true,
        );
    }

    fn drawOutput(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        scratch: *Scratch,
    ) !void {
        try scratch.tabBackground(ui, self.frame, self.frame.rect(.output_tabs));
        _ = try self.drawTabs(
            engine,
            ui,
            scratch,
            output_tabs_key,
            self.frame.rect(.output_tabs),
            &output_tabs,
            0,
            null,
        );

        const row, _ = splitTop(self.frame.rect(.output_content), self.frame.rowHeight());
        try scratch.line(engine, ui, row, theme.labelFaint(), "The engine's log is not routed here yet.", true);
    }

    // What is known about the selected material.
    //
    // Read-only, and stated as such rather than drawn in fields that do not
    // accept a value. A field that looked editable and was not would be worse
    // than a line of text, and the editable one is a widget that does not exist
    // yet: a number is dragged as well as typed, which a text field alone does
    // not do.
    fn drawInspector(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        scratch: *Scratch,
    ) !void {
        try scratch.tabBackground(ui, self.frame, self.frame.rect(.inspector_tabs));
        _ = try self.drawTabs(
            engine,
            ui,
            scratch,
            inspector_tabs_key,
            self.frame.rect(.inspector_tabs),
            &inspector_tabs,
            0,
            null,
        );

        var area = self.frame.rect(.inspector_content);
        const row_height = self.frame.rowHeight();

        const index = self.selected orelse {
            const empty, _ = splitTop(area, row_height);
            try scratch.line(engine, ui, empty, theme.labelFaint(), "Nothing selected", true);
            return;
        };
        const material = self.materials[index];

        const name_row, area = splitTop(area, row_height);
        try scratch.line(engine, ui, name_row, theme.label(), material.name, true);
        _, area = splitTop(area, shell.metrics.gutter * self.frame.scale.y);

        var buffer: [96]u8 = undefined;
        const factors = material.factors;
        const lines = [_][]const u8{
            std.fmt.bufPrint(buffer[0..32], "metallic  {d:.3}", .{factors.metallic}) catch "metallic  ?",
            std.fmt.bufPrint(buffer[32..64], "roughness {d:.3}", .{factors.roughness}) catch "roughness ?",
            std.fmt.bufPrint(buffer[64..96], "alpha     {d:.3}", .{factors.base_colour[3]}) catch "alpha     ?",
        };
        for (lines) |line| {
            const row, area = splitTop(area, row_height);
            try scratch.line(engine, ui, row, theme.labelDim(), line, true);
        }
    }

    fn drawStatusBar(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        scratch: *Scratch,
    ) !void {
        const area = insetX(self.frame.rect(.status_bar), shell.metrics.gutter * self.frame.scale.x);
        var buffer: [96]u8 = undefined;

        // Null until the first window of frames has closed, which is the first
        // second of the run. A dash rather than a zero: no measurement and a
        // measurement of nothing are different things.
        const line = if (engine.metrics.last_fps) |report|
            std.fmt.bufPrint(&buffer, "{d:.0} fps   {d:.1} ms   worst {d:.1} ms   {d} materials", .{
                report.fps,
                report.mean_ms,
                report.worst_ms,
                self.materials.len,
            }) catch "—"
        else
            std.fmt.bufPrint(&buffer, "—   {d} materials", .{self.materials.len}) catch "—";

        try scratch.line(engine, ui, area, theme.labelDim(), line, true);

        var right = theme.labelFaint();
        right.horizontal = .end;
        try scratch.line(engine, ui, area, right, version, true);
    }

    // One strip of tabs, and which of them is chosen after this frame.
    //
    // The chosen one wears the accent and the rest are flat, which is what makes
    // a strip read as one control with a state rather than as several buttons.
    fn drawTabs(
        self: *Driver,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        scratch: *Scratch,
        scope: imui.IdKey,
        strip: res.Rect,
        tabs: []const Tab,
        active: usize,
        left: ?f32,
    ) !usize {
        var chosen = active;
        try ui.pushId(scope);
        defer ui.popId();

        for (tabs, 0..) |tab, index| {
            const area = self.tabRect(strip, tabs, index, left);
            const style = if (index == active)
                theme.buttonActive(self.frame.scale)
            else
                theme.buttonFlat(self.frame.scale);
            if (try ui.button(.{ .integer = index }, area, style, tab.ready)) chosen = index;

            var caption = if (tab.ready) theme.label() else theme.labelFaint();
            caption.horizontal = .center;
            try scratch.line(engine, ui, area, caption, tab.caption, tab.ready);
        }
        return chosen;
    }

    // Where one tab of a strip sits.
    //
    // Each is as wide as a nominal caption needs rather than as wide as its own
    // shaped text: measured from the glyphs, a tab would move under the pointer
    // between one font and the next, and a tab that is not where it was
    // registered is a tab that answers for its neighbour.
    //
    // `left` places the strip explicitly, for the one strip that is centred on
    // its bar rather than run from its left edge.
    fn tabRect(
        self: *const Driver,
        strip: res.Rect,
        tabs: []const Tab,
        index: usize,
        left: ?f32,
    ) res.Rect {
        const scale = self.frame.scale;
        const width = tab_width * scale.x;
        const inset = shell.metrics.gutter * scale.x;
        const origin = left orelse (strip.x + inset);
        const height = @max(strip.height - 2 * scale.y, 0);
        _ = tabs;
        return rect(
            origin + width * @as(f32, @floatFromInt(index)),
            strip.y + (strip.height - height) * 0.5,
            width,
            height,
        );
    }

    fn documentTabs(self: *const Driver) []const Tab {
        return if (self.document_tab) |*tab| tab[0..1] else &empty_document_tabs;
    }

    // Moves the selection by `adjust` rows, where up is one row earlier.
    //
    // The arrows arrive as a signed sum where up is positive, because that is
    // what a value reads them as. A list runs the other way: the row after this
    // one is further down the screen.
    //
    // Selection is by material and not by row, so that a filter narrowing the
    // list does not move it onto whatever now sits at that position.
    // Which rows are inside the list at this offset, as a first and a length.
    //
    // Only these are registered and only these are drawn. A document with ten
    // thousand materials would otherwise register ten thousand regions for a
    // list showing forty, and the region capacity is a frame's ceiling rather
    // than a document's.
    //
    // One row past the bottom, because a list scrolled by half a row shows part
    // of one more.
    fn rowWindow(self: *const Driver, list_area: res.Rect) struct { usize, usize } {
        const height = self.frame.rowHeight();
        if (height <= 0 or self.matched_count == 0) return .{ 0, 0 };
        const first: usize = @intFromFloat(@max(@floor(self.list_scroll / height), 0));
        if (first >= self.matched_count) return .{ self.matched_count, 0 };
        const shown: usize = @intFromFloat(@ceil(list_area.height / height) + 1);
        return .{ first, @min(shown, self.matched_count - first) };
    }

    fn rowRect(self: *const Driver, list_area: res.Rect, row: usize) res.Rect {
        const height = self.frame.rowHeight();
        const top = list_area.y + @as(f32, @floatFromInt(row)) * height - self.list_scroll;
        return rect(list_area.x, top, list_area.width, height);
    }

    fn moveSelection(self: *Driver, adjust: i32) void {
        if (self.matched_count == 0) return;
        const last: i64 = @intCast(self.matched_count - 1);
        const current: i64 = if (self.selectedRow()) |row| @intCast(row) else 0;
        const moved = std.math.clamp(current - adjust, 0, last);
        self.selected = self.matched[@intCast(moved)];
    }

    fn selectedRow(self: *const Driver) ?usize {
        const material = self.selected orelse return null;
        for (self.matched[0..self.matched_count], 0..) |candidate, row| {
            if (candidate == material) return row;
        }
        return null;
    }

    // Scrolls just far enough to put the selected row inside the list.
    //
    // The same shape as the field's own scroll and for the same reason: a caret
    // and a selected row are both what the user is looking at, and a list that
    // jumped to centre one would move rows the user was reading.
    fn revealSelection(self: *Driver, list_area: res.Rect, content: f32) void {
        const row = self.selectedRow() orelse return;
        const top = @as(f32, @floatFromInt(row)) * self.frame.rowHeight();
        const bottom = top + self.frame.rowHeight();

        var offset = self.list_scroll;
        if (top < offset) offset = top;
        if (bottom > offset + list_area.height) offset = bottom - list_area.height;
        self.list_scroll = std.math.clamp(offset, 0, @max(content - list_area.height, 0));
    }
};

// The arrays a caption is shaped into, and the two things drawing one needs.
//
// One per frame on the stack, reused by every line: a run is shaped into it and
// drawn out of it before the next line is shaped, so nothing here outlives the
// call that used it.
const Scratch = struct {
    font: ?lenore.FontId,
    white: res.ImageHandle,
    glyphs: [max_glyphs]res.ShapedGlyph = undefined,
    placements: [max_placements]res.GlyphPlacement = undefined,

    // One line of text in a rectangle, or nothing when there is no font and
    // nothing when the line is longer than the scratch holds.
    //
    // A line that does not fit leaves its rectangle empty rather than the frame
    // without a picture, which is a report that costs nothing to read.
    fn line(
        self: *Scratch,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        area: res.Rect,
        style: imui.LabelStyle,
        content: []const u8,
        enabled: bool,
    ) !void {
        const font = self.font orelse return;
        const shaped = engine.shapeLabel(font, content, &self.glyphs, &self.placements) catch return;
        return ui.label(area, style, shaped, enabled);
    }

    // A strip of tabs, as a surface with a rule under it.
    //
    // The rule is a whole pixel: a hairline at a fractional height is drawn at
    // partial coverage on both sides instead of being a line.
    fn tabBackground(
        self: *Scratch,
        ui: *imui.WidgetContext,
        frame: shell.Frame,
        area: res.Rect,
    ) !void {
        const canvas = try ui.drawList();
        try canvas.addQuad(area, .{}, theme.raised.premultiplied(), self.white);
        const rule = @max(@round(frame.scale.y), 1);
        try canvas.addQuad(
            rect(area.x, area.y + area.height - rule, area.width, rule),
            .{},
            theme.border.premultiplied(),
            self.white,
        );
    }

    // A panel's title, over the rule that separates it from the panel's body.
    fn header(
        self: *Scratch,
        engine: *lenore.Engine,
        ui: *imui.WidgetContext,
        frame: shell.Frame,
        area: res.Rect,
        title: []const u8,
    ) !void {
        const canvas = try ui.drawList();
        try canvas.addQuad(area, .{}, theme.raised.premultiplied(), self.white);
        // The rule is a whole pixel: a hairline at a fractional height is drawn
        // at partial coverage on both sides instead of being a line.
        const rule = @max(@round(frame.scale.y), 1);
        try canvas.addQuad(
            rect(area.x, area.y + area.height - rule, area.width, rule),
            .{},
            theme.border.premultiplied(),
            self.white,
        );
        return self.line(
            engine,
            ui,
            insetX(area, shell.metrics.gutter * frame.scale.x),
            theme.label(),
            title,
            true,
        );
    }
};

// Whether `haystack` contains `needle`, comparing ASCII letters without regard
// to case and every other byte exactly.
fn containsFold(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    var start: usize = 0;
    while (start + needle.len <= haystack.len) : (start += 1) {
        if (equalsFold(haystack[start..][0..needle.len], needle)) return true;
    }
    return false;
}

fn equalsFold(a: []const u8, b: []const u8) bool {
    for (a, b) |left, right| {
        if (std.ascii.toLower(left) != std.ascii.toLower(right)) return false;
    }
    return true;
}

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

pub fn main(process: std.process.Init.Minimal) !void {
    const gpa = if (checking) debug_allocator.allocator() else std.heap.smp_allocator;
    defer if (checking) {
        if (debug_allocator.deinit() == .leak) log.err("host memory leaked", .{});
    };

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var arguments: std.process.Args.Iterator = try .initAllocator(process.args, gpa);
    defer arguments.deinit();
    _ = arguments.next();

    // A document is optional. An editor that refused to start without one
    // would have no state in which to open a second, and opening is a thing
    // the interface has to be able to do from inside itself.
    const model_path = arguments.next();

    var loaded: ?gltf.loader.Loaded = null;
    defer if (loaded) |*value| value.deinit(gpa);
    var model: ?gltf.importer.Model = null;
    defer if (model) |*value| value.deinit(gpa);

    if (model_path) |path| {
        // The loader confines a document's references to a root, so it takes a
        // directory and a name inside it rather than a path.
        var root = try std.Io.Dir.cwd().openDir(io, std.fs.path.dirname(path) orelse ".", .{});
        defer root.close(io);

        loaded = try gltf.loader.open(gpa, io, root, std.fs.path.basename(path));
        model = try gltf.importer.build(gpa, &loaded.?.document, loaded.?.directory);
        try gltf.importer.readExternalImages(gpa, io, root, &model.?);
    }

    const materials: []const res.MaterialInfo =
        if (model) |*document| document.materials else &.{};

    // Sized before the device, because the rings cannot be resized after it.
    const capacities: Capacities =
        if (model) |*document| try .forDocument(document) else capacity_floor;

    var engine: lenore.Engine = undefined;
    try engine.init(gpa, .{
        .title = "Lenore editor",
        .material_capacity = capacities.materials,
        .frame_capacity = .{
            .instances = capacities.instances,
            .joints = capacities.joints,
        },
        .morph_capacity = .{
            .meshes = capacities.morph_meshes,
            .weights = capacities.morph_weights,
        },
        // Without it the panels, the rows and the routing all stand with no
        // captions. An editor that refused to open because fontconfig answered
        // nothing would be harder to diagnose than one that opens silent.
        .interface_font = .{ .points = label_points },
    });
    defer engine.deinit();
    log.info("device: {s}", .{engine.context.deviceName()});

    engine.renderer.clear_colour = viewport_clear;

    // One light, because a document that brought none would otherwise be a
    // silhouette and an editor that cannot see what it opened is not an
    // editor. It is the editor's own and not the document's: a viewport light
    // is there to make geometry readable, which is why it comes from over the
    // viewer's shoulder rather than from wherever the asset put one.
    //
    // A document with its own lights gets them as well, through the level.
    var light_block: [gpu.max_lights]gpu.LightUniform = undefined;
    light_block[0] = lenore.packLight(try .directional(
        key_light_colour,
        key_light_intensity,
        key_light_direction,
    ));
    engine.look = .{
        .lights = light_block[0..1],
        .sun_shadow = .{ .enabled = true, .strength = 1, .normal_offset_texels = 1 },
    };

    // The same numbers the rings were built with, because a level that planned
    // for more than a frame can hold would be refused on the frame rather than
    // at load.
    var level = if (model) |*document|
        try lenore.Level.init(gpa, engine.deps(), document, .{
            .capacity = .{
                .instances = capacities.instances,
                .joints = capacities.joints,
            },
        })
    else
        lenore.Level.empty(gpa);
    defer level.deinit(&engine.textures);

    // With nothing installed the world has no bounds to frame, so the camera
    // is put around a unit sphere at the origin: that is where a document
    // opened later will be, and it is a view rather than a camera at nothing.
    const framing: res.Sphere = if (model) |*document| blk: {
        try engine.install(&level, document);
        break :blk level.world.sphere;
    } else blk: {
        const unit: res.Sphere = .{ .centre = .{ 0, 0, 0 }, .radius = 1 };
        _ = engine.frameCamera(unit);
        break :blk unit;
    };

    // Sized to the document rather than to a guess, so that a filter matching
    // everything still writes inside it and the frame allocates nothing.
    const matched = try gpa.alloc(u32, materials.len);
    defer gpa.free(matched);

    const document_name: ?[]const u8 =
        if (model_path) |path| std.fs.path.basename(path) else null;

    var driver: Driver = .{
        .white = engine.ui_white,
        .font = engine.interface_font,
        .materials = materials,
        .matched = matched,
        .framing = framing,
        .document_name = document_name,
        .document_tab = if (document_name) |name| .{.{ .caption = name }} else null,
    };
    log.info("{s}", .{usage});

    try engine.run(&level, .{&driver});
}
