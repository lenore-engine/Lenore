// A debug overlay: lines of text over the game, each on a plate as wide as its
// text, in the engine's interface font.
//
// It owns no reading. The application writes the frame's lines and hands them
// in. The overlay registers no region, so a click goes through it to the game.

const std = @import("std");
const imui = @import("lenore-imui");
const res = @import("lenore-resources");

const engine_module = @import("engine.zig");
const font_module = @import("font.zig");

const Engine = engine_module.Engine;

// Logical points, turned into pixels for the output the window is on.
const margin: f32 = 8;
const padding: f32 = 3;

const plate = imui.SrgbColor.fromHex("101418").withAlpha(0.6);
const text_style: imui.LabelStyle = .{
    .normal_text = imui.SrgbColor.fromHex("e6e9ec").premultiplied(),
    .disabled_text = imui.SrgbColor.fromHex("8a96a1").premultiplied(),
    .vertical = .middle,
};

pub const max_lines = 16;
pub const line_capacity = 120;
const max_placements = line_capacity * res.SubpixelBuckets.most.count();

pub const Side = enum { left, right };

// The frame's lines, written into fixed buffers so that a frame's readout
// allocates nothing.
pub const Lines = struct {
    buffers: [max_lines][line_capacity]u8 = undefined,
    lengths: [max_lines]usize = @splat(0),
    count: usize = 0,

    // A line that does not fit is cut at the buffer; past the last line it is
    // dropped.
    pub fn add(self: *Lines, comptime format: []const u8, values: anytype) void {
        if (self.count == max_lines) return;
        const buffer = &self.buffers[self.count];
        const written = std.fmt.bufPrint(buffer, format, values) catch buffer[0..];
        self.lengths[self.count] = written.len;
        self.count += 1;
    }

    pub fn line(self: *const Lines, index: usize) []const u8 {
        return self.buffers[index][0..self.lengths[index]];
    }
};

pub const DebugOverlay = struct {
    font: ?font_module.FontId,
    open: bool = false,

    pub fn init(engine: *const Engine) DebugOverlay {
        return .{ .font = engine.interface_font };
    }

    // Does nothing without a font, so an overlay that cannot draw is never
    // reported as open.
    pub fn toggle(self: *DebugOverlay) void {
        if (self.font == null) return;
        self.open = !self.open;
    }

    // From the application's `ui_draw` hook. In the swapchain's extent, which
    // is what the frame's root clip is.
    pub fn draw(self: *const DebugOverlay, engine: *Engine, ui: *imui.WidgetContext, lines: *const Lines, side: Side) !void {
        if (!self.open) return;
        const font = self.font orelse return;
        const metrics = try engine.fontMetrics(font);
        const scale = engine.uiScale().y;
        const height = metrics.lineHeight() + padding * scale;
        const inner = padding * scale;
        const extent: f32 = @floatFromInt(engine.swapchain.currentExtent().width);
        const canvas = try ui.drawList();
        var glyphs: [line_capacity]res.ShapedGlyph = undefined;
        var placements: [max_placements]res.GlyphPlacement = undefined;
        for (0..lines.count) |index| {
            const shaped = try engine.shapeLabel(font, lines.line(index), &glyphs, &placements);
            const width = shaped.advance + 2 * inner;
            const y = margin * scale + @as(f32, @floatFromInt(index)) * height;
            const x = switch (side) {
                .left => margin * scale,
                .right => extent - margin * scale - width,
            };
            try canvas.addQuad(.{ .x = x, .y = y, .width = width, .height = height }, .{}, plate.premultiplied(), engine.ui_white);
            try ui.label(.{ .x = x + inner, .y = y, .width = shaped.advance, .height = height }, text_style, shaped, true);
        }
    }
};
