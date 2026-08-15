const std = @import("std");
const gpu = @import("lenore-gpu");
const imui = @import("lenore-imui");
const lenore = @import("lenore");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const text = @import("lenore-text");

const log = std.log.scoped(.ui_panel);

// The overlay contract, on the device.
//
// This is an instrument rather than a demonstration. What it draws is chosen so
// that each thing the host cannot check announces itself by being visibly wrong,
// and nothing else is here: no asset, no camera, no shading of its own.
//
// Four things are being read off the screen, and each has its own figure.
//
// **Orientation.** The corner markers are four different colours and one of them
// is doubled in size. A flip about either axis exchanges a pair, and the sizes
// say which axis it was. Nothing on the host reaches this: the vertex stage
// converts pixels to clip space assuming the viewport maps clip Y downward, and
// no test compiles a shader.
//
// **Premultiplied compositing.** The wash over the middle is drawn at half
// coverage over the opaque plate, and the plate is drawn over the cleared
// background. If the blend state multiplied by alpha a second time the wash
// would come out at a quarter and read as much darker than the swatch beside
// it, which is drawn at the same colour and full coverage over nothing.
//
// **The scissor.** The striped block is drawn as one rectangle far wider than
// the clip it is pushed under, so what appears is the clip's width exactly. An
// off-by-one in the rounding shows as a bright or dark seam at its right edge
// against the plate behind it.
//
// **That the overlay is not tone mapped.** The white bar is at 1.0 and the
// clear colour behind it is 1.0 as well. The clear goes through the operator
// and the bar does not, so they must not match: the bar is white and the
// background is the grey the operator made of it. If they match, the overlay
// is being drawn before the operator rather than after.
//
// **The glyph atlas**, which is three answers in one figure and needs a font on
// the host. A rule is drawn at the baseline the text is placed on, so glyphs
// must sit on it and the descenders of `g` and `y` must cross it; a wrong sign
// on the vertical flip puts the whole line below the rule. The second line
// holds characters outside the range rasterised when the face was loaded, which
// is the only thing here that exercises the upload inside the frame: if that
// copy or its barriers are wrong, that line is corrupt or missing while the
// first one is perfect.
//
// And the text is white on a dark plate, which is where the second blend source
// shows. Coverage is measured per colour stripe, so a glyph edge carries a
// tint; a pipeline that scaled the destination by one alpha instead would draw
// the same letters with grey edges, and the two are told apart by looking at a
// stem edge rather than by reading a number.

const usage =
    \\usage: ui_panel
    \\
    \\  space         hold to clear the overlay, to see the picture under it
    \\  escape        quit
    \\
    \\Draws its text with the font this host prefers. Without one the other four
    \\figures still stand.
;

// A system font, which is the case the engine itself is for: an editor draws
// with what the host has, where a game ships its own. Which font that is is the
// host's answer and not this file's, so nothing here names one.
const font_pixels = 20;

// The longer of the two lines is well inside this, and the placements are one
// per glyph per subpixel bucket. How many buckets a face has is settled when it
// is opened and an array is sized before that, so the most any face wants is
// what a destination reserves.
const max_glyphs = 128;
const max_placements = max_glyphs * res.SubpixelBuckets.most.count();

// The first line is printable ASCII and was rasterised when the face loaded, so
// it costs the frame nothing. The second is not, and is what makes a frame
// rasterise and upload.
const ascii_line = "Handgloves jgy 0123 on the baseline";
const beyond_line = "Ж é ± ¶ ← outside the prewarmed range";

// The clear colour, in the linear units the main pass writes. One, so that the
// tone operator has something to do to it and the white bar above has something
// to differ from.
const background: [4]f32 = .{ 1, 1, 1, 1 };

const marker: f32 = 48;
const doubled: f32 = 96;

const palette = struct {
    const top_left = imui.SrgbColor.fromHex("e63946");
    const top_right = imui.SrgbColor.fromHex("2a9d8f");
    const bottom_right = imui.SrgbColor.fromHex("e9c46a");
    const bottom_left = imui.SrgbColor.fromHex("6d597a");
    const plate = imui.SrgbColor.fromHex("22223b");
    const wash = imui.SrgbColor.fromHex("f4a261");
    const stripe = imui.SrgbColor.fromHex("457b9d");
};

fn rect(x: f32, y: f32, width: f32, height: f32) res.Rect {
    return .{ .x = x, .y = y, .width = width, .height = height };
}

const Driver = struct {
    white: res.ImageHandle,
    // Null when the host had no font to open. Every other figure is unaffected,
    // which is why a missing font is a line in the log and not a refusal.
    font: ?lenore.FontId,
    hidden: bool = false,
    reported_windows: u64 = 0,

    pub fn onEvent(self: *Driver, engine: *lenore.Engine, event: platform.Event, _: bool) !void {
        switch (event.payload) {
            .key => |key| switch (key.physical) {
                .escape => if (key.action == .press) engine.requestExit(),
                .space => switch (key.action) {
                    .press => self.hidden = true,
                    .release => self.hidden = false,
                    .repeat => {},
                },
                else => {},
            },
            else => {},
        }
    }

    pub fn onResize(_: *Driver, _: *lenore.Engine, _: platform.Extent2D) !void {}
    pub fn onCompute(_: *Driver, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}
    pub fn onRecord(_: *Driver, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}

    pub fn onUiRegions(_: *Driver, _: *lenore.Engine, _: *lenore.Level, _: *imui.WidgetContext) !void {}

    pub fn onUiDraw(
        self: *Driver,
        engine: *lenore.Engine,
        _: *lenore.Level,
        ui: *imui.WidgetContext,
    ) !void {
        if (self.hidden) return;

        // Every figure below is a fixed rectangle in framebuffer pixels, so
        // this reaches past the widgets to the canvas itself. What the example
        // measures is the pass and the composition, and a widget between the
        // two would be one more thing that could explain a wrong picture.
        const canvas = try ui.drawList();

        const extent = engine.swapchain.currentExtent();
        const width: f32 = @floatFromInt(extent.width);
        const height: f32 = @floatFromInt(extent.height);

        // The corners. Top left is the doubled one, so a vertical flip puts the
        // large square at the bottom and a horizontal flip puts it on the right.
        try canvas.addQuad(rect(0, 0, doubled, doubled), .{}, palette.top_left.premultiplied(), self.white);
        try canvas.addQuad(
            rect(width - marker, 0, marker, marker),
            .{},
            palette.top_right.premultiplied(),
            self.white,
        );
        try canvas.addQuad(
            rect(width - marker, height - marker, marker, marker),
            .{},
            palette.bottom_right.premultiplied(),
            self.white,
        );
        try canvas.addQuad(
            rect(0, height - marker, marker, marker),
            .{},
            palette.bottom_left.premultiplied(),
            self.white,
        );

        // The plate, and the wash over it at half coverage.
        const plate = rect(width * 0.25, height * 0.3, width * 0.5, height * 0.4);
        try canvas.addQuad(plate, .{}, palette.plate.premultiplied(), self.white);
        try canvas.addQuad(
            rect(plate.x + 24, plate.y + 24, plate.width - 48, plate.height * 0.4),
            .{},
            palette.wash.withAlpha(0.5).premultiplied(),
            self.white,
        );
        // The same colour at full coverage, over the background rather than the
        // plate, as the thing to compare the wash against.
        try canvas.addQuad(
            rect(plate.x, plate.y - 40, 120, 32),
            .{},
            palette.wash.premultiplied(),
            self.white,
        );

        // The clipped block: one rectangle three times the width of the clip it
        // is drawn under, so what survives is the clip.
        const window = rect(plate.x + 24, plate.y + plate.height * 0.6, 160, 64);
        try canvas.pushClip(window);
        try canvas.addQuad(
            rect(window.x, window.y, window.width * 3, window.height),
            .{},
            palette.stripe.premultiplied(),
            self.white,
        );
        canvas.popClip();

        // The white bar, against the cleared background it must not match.
        try canvas.addQuad(
            rect(width * 0.25, height * 0.75, width * 0.5, 24),
            .{},
            (imui.SrgbColor{ .r = 1, .g = 1, .b = 1 }).premultiplied(),
            self.white,
        );

        try self.drawText(engine, canvas, plate);
    }

    // The two lines, and the rule the first one sits on.
    //
    // The scratch arrays are the frame's, on the stack: a run is shaped into
    // them and drawn from them, and nothing outlives the call. A line longer
    // than they hold is refused by the shaper rather than truncated, which for
    // an instrument is the right answer, since a line missing its end looks
    // like a line.
    fn drawText(
        self: *Driver,
        engine: *lenore.Engine,
        canvas: *imui.Canvas,
        plate: res.Rect,
    ) !void {
        const font = self.font orelse return;

        var glyphs: [max_glyphs]res.ShapedGlyph = undefined;
        var placements: [max_placements]res.GlyphPlacement = undefined;

        const metrics = try engine.fontMetrics(font);
        const white = (imui.SrgbColor{ .r = 1, .g = 1, .b = 1 }).premultiplied();
        // Two lines above the plate's bottom edge, so the second one's
        // descenders are over the plate and not over whatever is behind it.
        const baseline = plate.y + plate.height - 2 * metrics.lineHeight();
        const left = plate.x + 24;

        const first = try engine.shapeText(font, ascii_line, &glyphs, &placements);
        // Drawn before the glyphs so that a rule and a letter overlapping is
        // the letter's pixel, and one texel tall so that "on the line" is not a
        // matter of opinion.
        try canvas.addQuad(
            rect(left, baseline, text.advance(first.glyphs), 1),
            .{},
            palette.stripe.premultiplied(),
            self.white,
        );
        try imui.addGlyphs(canvas, first, .{ .x = left, .y = baseline }, white, engine.glyphAtlas());

        const second = try engine.shapeText(font, beyond_line, &glyphs, &placements);
        try imui.addGlyphs(
            canvas,
            second,
            .{ .x = left, .y = baseline + metrics.lineHeight() },
            white,
            engine.glyphAtlas(),
        );
    }

    pub fn onUpdate(
        self: *Driver,
        engine: *lenore.Engine,
        _: *lenore.Level,
        _: lenore.FrameTime,
    ) !void {
        if (engine.metrics.closed_windows == self.reported_windows) return;
        self.reported_windows = engine.metrics.closed_windows;

        const report = engine.metrics.last_fps orelse return;
        log.info("{d:.1} fps, overlay {s}", .{
            report.fps,
            if (self.hidden) "hidden" else "drawn",
        });
    }
};

// The host's preferred sans-serif, or nothing. Reported rather than returned as
// an error: the four figures that need no font are the ones this example was
// written for, and a host with no font at all should still be able to read them.
//
// Asked in two steps rather than through `Engine.loadSystemFont`, because which
// file the host chose belongs in an instrument's log. A line of empty boxes is
// a font without those characters in it, and nothing on the screen says which
// font that was.
fn loadFont(engine: *lenore.Engine, io: std.Io) ?lenore.FontId {
    var path: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const found = (platform.findSystemFont(&path, .{}) catch |err| {
        log.warn("the host's font could not be asked for: {t}", .{err});
        return null;
    }) orelse {
        log.warn("the host offers no font, so the text figure is not drawn", .{});
        return null;
    };

    // Absolute, so the directory it is opened against is not read. The same
    // answer carries how the host wants text drawn, which is what the face is
    // opened with: this example asks the host for both or neither.
    const rendering: lenore.FontRendering = .fromHost(found.rendering);
    const id = engine.loadFontFile(io, .cwd(), found.path, found.index, font_pixels, rendering) catch |err| {
        log.warn("{s} did not load: {t}", .{ found.path, err });
        return null;
    };
    log.info("drawing with {s}, face {d}, {t} hinting and {t} coverage", .{
        found.path,
        found.index,
        rendering.hinting,
        rendering.antialias,
    });
    return id;
}

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

pub fn main() !void {
    const gpa = if (checking) debug_allocator.allocator() else std.heap.smp_allocator;
    defer if (checking) {
        _ = debug_allocator.deinit();
    };

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var engine: lenore.Engine = undefined;
    try engine.init(gpa, .{
        .title = "Lenore: overlay",
        .extent = .{ .width = 1280, .height = 720 },
        // Nothing is drawn from an asset, and zero is how that is said.
        .frame_capacity = .{ .instances = 0, .joints = 0 },
        .morph_capacity = .{ .meshes = 0, .weights = 0 },
        .material_capacity = 0,
    });
    defer engine.deinit();

    engine.renderer.clear_colour = background;

    var level: lenore.Level = .empty(gpa);
    defer level.deinit(&engine.textures);

    var driver: Driver = .{ .white = engine.ui_white, .font = loadFont(&engine, io) };
    log.info("{s}", .{usage});

    try engine.run(&level, &driver);
}
