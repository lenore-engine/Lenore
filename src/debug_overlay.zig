// A debug overlay: lines of text over the game, each on a plate as wide as its
// text, in the engine's interface font.
//
// It owns no reading. The application writes the frame's lines and hands them
// in, and records each frame's time into a `FrameGraph` for the graph under
// them. The overlay registers no region, so a click goes through it to the game.

const std = @import("std");
const imui = @import("lenore-imui");
const res = @import("lenore-resources");

const engine_module = @import("engine.zig");
const font_module = @import("font.zig");
const time = @import("time.zig");

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

// The last `graph_frames` frame intervals, the time `FrameTime.delta` is taken
// from, so the graph shows what the world advanced by and not only how often
// a frame came. A counter of 120 a second says nothing of how evenly they
// came: frames shown on every refresh with intervals alternating short and
// long move the world unevenly between even pictures, and one long frame in a
// second barely moves the average.
pub const graph_frames = 240;

pub const FrameGraph = struct {
    intervals_ns: [graph_frames]u32 = @splat(0),
    // Where the next interval goes; the oldest is there once `filled`.
    next: usize = 0,
    filled: usize = 0,

    // Every frame, open or not, so the graph is full when it is opened.
    pub fn record(self: *FrameGraph, frame: time.FrameTime) void {
        self.intervals_ns[self.next] = @intCast(@min(frame.interval_ns, std.math.maxInt(u32)));
        self.next = (self.next + 1) % graph_frames;
        self.filled = @min(self.filled + 1, graph_frames);
    }

    // Oldest first; `index` below `filled`.
    pub fn interval(self: *const FrameGraph, index: usize) u32 {
        const oldest = if (self.filled == graph_frames) self.next else 0;
        return self.intervals_ns[(oldest + index) % graph_frames];
    }

    pub const Summary = struct {
        median_ns: u32,
        mean_ns: f64,
        // The standard deviation: what an even run of frames at any rate has
        // none of.
        spread_ns: f64,
        worst_ns: u32,
        // Past `long_ratio` of the median, which at a steady rate is a refresh
        // or more missed.
        long: usize,
    };

    pub const long_ratio = 1.5;

    pub fn summary(self: *const FrameGraph) ?Summary {
        if (self.filled == 0) return null;
        var sorted: [graph_frames]u32 = undefined;
        const values = sorted[0..self.filled];
        @memcpy(values, self.intervals_ns[0..self.filled]);
        std.mem.sort(u32, values, {}, std.sort.asc(u32));
        const median = values[values.len / 2];

        var sum: f64 = 0;
        for (values) |value| sum += @floatFromInt(value);
        const mean = sum / @as(f64, @floatFromInt(values.len));
        var squares: f64 = 0;
        var long: usize = 0;
        for (values) |value| {
            const offset = @as(f64, @floatFromInt(value)) - mean;
            squares += offset * offset;
            if (@as(f64, @floatFromInt(value)) > long_ratio * @as(f64, @floatFromInt(median))) long += 1;
        }
        return .{
            .median_ns = median,
            .mean_ns = mean,
            .spread_ns = @sqrt(squares / @as(f64, @floatFromInt(values.len))),
            .worst_ns = values[values.len - 1],
            .long = long,
        };
    }
};

// The graph's size in logical points: one bar per frame, and the height that
// a frame of `graph_full_ns` fills, which is two refreshes at 60 Hz.
const bar_width: f32 = 2;
const graph_height: f32 = 90;
const graph_full_ns: f32 = 33.3e6;

const graph_plate = imui.SrgbColor.fromHex("101418").withAlpha(0.45);
const steady = imui.SrgbColor.fromHex("5cc85c");
const uneven = imui.SrgbColor.fromHex("e0c040");
const missed = imui.SrgbColor.fromHex("e04838");
const guide = imui.SrgbColor.fromHex("e6e9ec").withAlpha(0.5);

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

    // The frame graph at the bottom left, newest frame on the right, with a
    // line of its summary over it. Bars are coloured against the window's own
    // median rather than a refresh rate the engine does not know: steady
    // within a quarter of it, uneven to twice it, a missed refresh past that.
    // The guides are the median and twice it.
    pub fn drawGraph(self: *const DebugOverlay, engine: *Engine, ui: *imui.WidgetContext, graph: *const FrameGraph) !void {
        if (!self.open) return;
        const font = self.font orelse return;
        const totals = graph.summary() orelse return;
        const scale = engine.uiScale().y;
        const canvas = try ui.drawList();
        const surface = engine.swapchain.currentExtent();
        const bottom = @as(f32, @floatFromInt(surface.height)) - margin * scale;
        const left = margin * scale;
        const tall = graph_height * scale;
        const bar = bar_width * scale;
        const per_ns = tall / graph_full_ns;

        try canvas.addQuad(.{ .x = left, .y = bottom - tall, .width = bar * graph_frames, .height = tall }, .{}, graph_plate.premultiplied(), engine.ui_white);
        const median: f32 = @floatFromInt(totals.median_ns);
        for (0..graph.filled) |index| {
            const value: f32 = @floatFromInt(graph.interval(index));
            const colour = if (value <= 1.25 * median) steady else if (value <= 2 * median) uneven else missed;
            const length = @min(value * per_ns, tall);
            const x = left + @as(f32, @floatFromInt(graph_frames - graph.filled + index)) * bar;
            try canvas.addQuad(.{ .x = x, .y = bottom - length, .width = bar, .height = length }, .{}, colour.premultiplied(), engine.ui_white);
        }
        for ([_]f32{ 1, 2 }) |multiple| {
            const length = @min(multiple * median * per_ns, tall);
            try canvas.addQuad(.{ .x = left, .y = bottom - length, .width = bar * graph_frames, .height = @max(scale, 1) }, .{}, guide.premultiplied(), engine.ui_white);
        }

        var text: [line_capacity]u8 = undefined;
        const written = std.fmt.bufPrint(&text, "frame ms: median {d:.2} mean {d:.2} spread {d:.2} worst {d:.2}, {d} over {d:.1}x", .{
            median * 1e-6,
            totals.mean_ns * 1e-6,
            totals.spread_ns * 1e-6,
            @as(f32, @floatFromInt(totals.worst_ns)) * 1e-6,
            totals.long,
            FrameGraph.long_ratio,
        }) catch text[0..];
        const metrics = try engine.fontMetrics(font);
        const height = metrics.lineHeight() + padding * scale;
        const inner = padding * scale;
        var glyphs: [line_capacity]res.ShapedGlyph = undefined;
        var placements: [max_placements]res.GlyphPlacement = undefined;
        const shaped = try engine.shapeLabel(font, written, &glyphs, &placements);
        const y = bottom - tall - height;
        try canvas.addQuad(.{ .x = left, .y = y, .width = shaped.advance + 2 * inner, .height = height }, .{}, plate.premultiplied(), engine.ui_white);
        try ui.label(.{ .x = left + inner, .y = y, .width = shaped.advance, .height = height }, text_style, shaped, true);
    }
};
