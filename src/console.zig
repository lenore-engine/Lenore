// A debug console: one line of input along the bottom of the window and the
// reply to the last command above it.
//
// It knows nothing of what a command does. The application opens it, hands it
// a reply, and reads the line when enter is pressed; what the line means is the
// application's. Which key opens it is the application's too, as is whether
// the game's own bindings stand down while it is open.
//
// Drawn with the overlay's widgets in framebuffer pixels, in the engine's
// interface font. An engine with no interface font has no console: the field
// cannot be shaped, so it cannot be typed into, and `show` does nothing.

const std = @import("std");
const imui = @import("lenore-imui");
const res = @import("lenore-resources");

const engine_module = @import("engine.zig");
const font_module = @import("font.zig");

const Engine = engine_module.Engine;

// Logical points, turned into pixels for the output the window is on.
const margin: f32 = 12;
const padding: f32 = 6;

const key: imui.IdKey = .{ .string = "lenore.console" };

const palette = struct {
    const plate = imui.SrgbColor.fromHex("101418").withAlpha(0.82);
    const field = imui.SrgbColor.fromHex("1c2229");
    const border = imui.SrgbColor.fromHex("3a4550");
    const text = imui.SrgbColor.fromHex("e6e9ec");
    const dim = imui.SrgbColor.fromHex("8a96a1");
    const failure = imui.SrgbColor.fromHex("ff8a80");
    const selection = imui.SrgbColor.fromHex("35506b");
};

// Glyph scratch a line may shape into, on the stack of each call.
const max_glyphs = 256;
const max_placements = max_glyphs * res.SubpixelBuckets.most.count();

pub const Console = struct {
    font: ?font_module.FontId,
    open: bool = false,
    // Set when the console opens, so the next registering pass hands the
    // field the keyboard.
    focus_pending: bool = false,
    // Whether this frame's registering pass saw the console open. A console
    // opened by a key is opened while the frame's events are routed, after the
    // regions were registered, and a widget drawn without its region is one
    // the UI refuses. It is drawn from the next frame.
    registered: bool = false,

    buffer: [256]u8 = undefined,
    state: imui.TextFieldState = .{},

    reply_buffer: [256]u8 = undefined,
    reply_len: usize = 0,
    reply_failed: bool = false,

    pub fn init(engine: *const Engine) Console {
        return .{ .font = engine.interface_font };
    }

    // Opens with `prefix` already typed, the caret after it. A prefix longer
    // than the line is cut at the line.
    pub fn show(self: *Console, prefix: []const u8) void {
        if (self.font == null) return;
        const length = @min(prefix.len, self.buffer.len);
        @memcpy(self.buffer[0..length], prefix[0..length]);
        self.state = .{ .len = length, .caret = length, .anchor = length };
        self.open = true;
        self.focus_pending = true;
    }

    pub fn hide(self: *Console) void {
        self.open = false;
        self.focus_pending = false;
    }

    pub fn line(self: *const Console) []const u8 {
        return self.buffer[0..self.state.len];
    }

    // The reply shown above the field, cut at what the buffer holds.
    pub fn reply(self: *Console, failed: bool, comptime format: []const u8, arguments: anytype) void {
        const written = std.fmt.bufPrint(&self.reply_buffer, format, arguments) catch self.reply_buffer[0..];
        self.reply_len = written.len;
        self.reply_failed = failed;
    }

    // From the application's `ui_regions` hook, at the root scope.
    pub fn regions(self: *Console, engine: *const Engine, ui: *imui.WidgetContext) !void {
        self.registered = self.open;
        if (!self.open) return;
        const layout = try self.layoutFor(engine);
        try ui.register(key, layout.field, .{ .focusable = true });
        if (self.focus_pending) {
            try ui.focus(key);
            self.focus_pending = false;
        }
    }

    // From the application's `ui_draw` hook, at the root scope.
    pub fn draw(self: *Console, engine: *Engine, ui: *imui.WidgetContext) !void {
        if (!self.open or !self.registered) return;
        const font = self.font orelse return;
        const layout = try self.layoutFor(engine);

        const canvas = try ui.drawList();
        try canvas.addQuad(layout.plate, .{}, palette.plate.premultiplied(), engine.ui_white);

        var glyphs: [max_glyphs]res.ShapedGlyph = undefined;
        var placements: [max_placements]res.GlyphPlacement = undefined;

        if (self.reply_len > 0) {
            const shaped = try engine.shapeLabel(font, self.reply_buffer[0..self.reply_len], &glyphs, &placements);
            const colour = if (self.reply_failed) palette.failure else palette.dim;
            try ui.label(layout.answer, .{
                .normal_text = colour.premultiplied(),
                .disabled_text = colour.premultiplied(),
            }, shaped, true);
        }

        const style: imui.TextFieldStyle = .{
            .box = .{
                .normal_fill = palette.field.premultiplied(),
                .hovered_fill = palette.field.premultiplied(),
                .held_fill = palette.field.premultiplied(),
                .disabled_fill = palette.field.premultiplied(),
                .border = palette.border.premultiplied(),
                .border_width = 1,
            },
            .normal_text = palette.text.premultiplied(),
            .disabled_text = palette.dim.premultiplied(),
            .selection = palette.selection.premultiplied(),
            .caret = palette.text.premultiplied(),
            .caret_width = 2,
            .padding = padding,
        };

        // Shaped before the edit, since a click is resolved against what was on
        // the screen, and again after it for what is drawn.
        const before = try engine.shapeLabel(font, self.line(), &glyphs, &placements);
        _ = try ui.editText(key, layout.field, style, &self.buffer, &self.state, before, true);
        const after = try engine.shapeLabel(font, self.line(), &glyphs, &placements);
        try ui.textField(key, layout.field, style, &self.state, after, true);
    }

    const Layout = struct { plate: res.Rect, answer: res.Rect, field: res.Rect };

    // Along the bottom of the window, the reply's line above the field's. In
    // the swapchain's extent, which is what the frame's root clip is.
    fn layoutFor(self: *const Console, engine: *const Engine) !Layout {
        const font = self.font orelse return error.NoFont;
        const metrics = try engine.fontMetrics(font);
        const scale = engine.uiScale().y;
        const extent = engine.swapchain.currentExtent();
        const width: f32 = @floatFromInt(extent.width);
        const height: f32 = @floatFromInt(extent.height);

        const gap = margin * scale;
        const line_height = metrics.lineHeight();
        const field_height = line_height + 2 * padding;
        const field: res.Rect = .{
            .x = gap,
            .y = height - gap - field_height,
            .width = @max(width - 2 * gap, 1),
            .height = field_height,
        };
        const answer: res.Rect = .{
            .x = gap + padding,
            .y = field.y - line_height - padding,
            .width = field.width - 2 * padding,
            .height = line_height,
        };
        return .{
            .plate = .{
                .x = 0,
                .y = answer.y - padding,
                .width = width,
                .height = height - (answer.y - padding),
            },
            .answer = answer,
            .field = field,
        };
    }
};
