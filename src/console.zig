// A debug console: one line of input along the bottom of the window and the
// reply to the last command above it.
//
// The console is the entity and ships no commands. What `/tp` or noclip mean
// differs between a first-person game, a platformer and a top-down one, so
// every command is the application's: it hands `Commands` a table of them,
// and the console splits the line, finds the command by name and runs it.
// `help` is the table's own, since listing what is registered belongs to the
// registry and to no game. Which key opens the console is the application's,
// as is whether the game's own bindings stand down while it is open.
//
// Drawn with the overlay's widgets in framebuffer pixels, in the engine's
// interface font. An engine with no interface font has no console: the field
// cannot be shaped, so it cannot be typed into, and `show` does nothing.

const std = @import("std");
const imui = @import("lenore-imui");
const platform = @import("lenore-platform");
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

// Lines kept for recall with the arrow keys, oldest dropped first.
const history_capacity = 16;
const line_capacity = 256;

// Glyph scratch a line may shape into, on the stack of each call.
const max_glyphs = 256;
const max_placements = max_glyphs * res.SubpixelBuckets.most.count();

pub const Console = struct {
    font: ?font_module.FontId,
    open: bool = false,
    // Whether this frame's registering pass saw the console open. A console
    // opened by a key is opened while the frame's events are routed, after the
    // regions were registered, and a widget drawn without its region is one
    // the UI refuses. It is drawn from the next frame.
    registered: bool = false,

    buffer: [line_capacity]u8 = undefined,
    state: imui.TextFieldState = .{},

    history: [history_capacity][line_capacity]u8 = undefined,
    history_lens: [history_capacity]u16 = @splat(0),
    // Lines submitted so far, of which the last `history_capacity` are kept.
    history_count: usize = 0,
    // How far back the arrow keys have gone, 0 being the line being typed.
    recalled: usize = 0,

    // Tab completion: the line as it was before the first Tab, whose
    // candidates the following presses walk, and which of them is written.
    // Any edit, recall, `show` or `hide` ends the walk.
    cycle_base: [line_capacity]u8 = undefined,
    cycle_len: usize = 0,
    cycle_selected: ?usize = null,

    reply_buffer: [256]u8 = undefined,
    reply_len: usize = 0,
    reply_failed: bool = false,

    // What a command takes, read by the line's check before it runs, by the
    // usage reply, by completion and by the grey hint. Each form is one way
    // to write the arguments: words separated by spaces, each word a choice of
    // alternatives separated by `|`, each alternative a literal or a
    // `<placeholder>` standing for any word. `/time` is
    // `set day|noon|night|midnight|<ticks>|<HH:MM>` and `add <ticks>`. A line
    // runs only when it fits one form word for word, so a command is never
    // handed too few or too many words.
    pub const Syntax = struct {
        name: []const u8,
        forms: []const []const u8 = &.{""},
    };

    pub const ArgumentError = error{ MissingArgument, MalformedArgument };

    // The words after the command's name. What they mean is the command's:
    // `~` in a coordinate, a unit, a clock time.
    pub const Words = struct {
        iterator: std.mem.TokenIterator(u8, .scalar),

        pub fn init(text: []const u8) Words {
            return .{ .iterator = std.mem.tokenizeScalar(u8, text, ' ') };
        }

        pub fn next(self: *Words) ArgumentError![]const u8 {
            return self.iterator.next() orelse error.MissingArgument;
        }

        pub fn float(self: *Words, comptime T: type) ArgumentError!T {
            return parseDecimal(T, try self.next()) orelse error.MalformedArgument;
        }

        pub fn int(self: *Words, comptime T: type) ArgumentError!T {
            const text = try self.next();
            for (text, 0..) |byte, index| switch (byte) {
                '0'...'9' => {},
                '+', '-' => if (index != 0) return error.MalformedArgument,
                else => return error.MalformedArgument,
            };
            return std.fmt.parseInt(T, text, 10) catch error.MalformedArgument;
        }
    };

    // Digits with one optional leading sign and at most one point. Narrower
    // than `std.fmt.parseFloat`, which also takes an exponent, `inf`, `nan`
    // and underscores between digits, none of which a typed command means.
    pub fn parseDecimal(comptime T: type, text: []const u8) ?T {
        var digits: usize = 0;
        var points: usize = 0;
        for (text, 0..) |byte, index| switch (byte) {
            '0'...'9' => digits += 1,
            '.' => points += 1,
            '+', '-' => if (index != 0) return null,
            else => return null,
        };
        if (digits == 0 or points > 1) return null;
        const value = std.fmt.parseFloat(T, text) catch return null;
        return if (std.math.isFinite(value)) value else null;
    }

    fn placeholder(word: []const u8) bool {
        return word.len >= 2 and word[0] == '<' and word[word.len - 1] == '>';
    }

    fn accepts(slot: []const u8, word: []const u8) bool {
        var alternatives = std.mem.splitScalar(u8, slot, '|');
        while (alternatives.next()) |alternative| {
            if (placeholder(alternative) or std.mem.eql(u8, alternative, word)) return true;
        }
        return false;
    }

    fn fits(form: []const u8, arguments: []const u8) bool {
        var slots = std.mem.tokenizeScalar(u8, form, ' ');
        var words = std.mem.tokenizeScalar(u8, arguments, ' ');
        while (slots.next()) |slot| {
            if (!accepts(slot, words.next() orelse return false)) return false;
        }
        return words.next() == null;
    }

    // `run` is called only with a line that fits one of the syntax's forms.
    // An `ArgumentError` it returns is answered with the usage, any other
    // error with its name.
    pub fn Command(comptime Context: type) type {
        return struct {
            syntax: Syntax,
            run: *const fn (Context, *Words, *Console) anyerror!void,
        };
    }

    // The application's commands, with `help` in front of them. `help`'s
    // argument is every registered name as one slot of literals, so `help
    // <name>` completes like any other argument.
    pub fn Commands(comptime Context: type, comptime commands: []const Command(Context)) type {
        return struct {
            pub const syntax: []const Syntax = blk: {
                var names: []const u8 = "help";
                for (commands, 0..) |command, index| {
                    const name = command.syntax.name;
                    if (name.len == 0 or std.mem.indexOfAny(u8, name, " |") != null)
                        @compileError("command names must be non-empty and contain neither spaces nor |");
                    if (std.mem.eql(u8, name, "help")) @compileError("help is reserved");
                    for (commands[0..index]) |previous| {
                        if (std.mem.eql(u8, name, previous.syntax.name)) @compileError("duplicate command name");
                    }
                    names = names ++ "|" ++ name;
                }
                var table: [commands.len + 1]Syntax = undefined;
                table[0] = .{ .name = "help", .forms = &.{ "", names } };
                for (commands, 1..) |command, index| table[index] = command.syntax;
                const result = table;
                break :blk &result;
            };

            // A key while the console is open. Enter and escape act on a press
            // only; Tab and the arrows on its repeats too.
            pub fn press(context: Context, console: *Console, event: platform.KeyEvent) void {
                if (event.action == .release) return;
                switch (event.physical) {
                    .enter, .numpad_enter => if (event.action == .press) submit(context, console),
                    .escape => if (event.action == .press) console.hide(),
                    .tab => console.complete(syntax, event.modifiers.shift),
                    else => console.press(event.physical),
                }
            }

            // Runs the typed line and leaves the console open on an empty one,
            // ready for the next. An empty line closes it.
            pub fn submit(context: Context, console: *Console) void {
                const text = std.mem.trim(u8, console.line(), " \t\r\n");
                if (text.len == 0 or std.mem.eql(u8, text, "/")) return console.hide();
                console.remember(text);
                execute(context, console, text);
                console.show("");
            }

            // Runs one line as if typed, its leading `/` optional, and answers
            // through the console's reply.
            pub fn execute(context: Context, console: *Console, text: []const u8) void {
                const trimmed = std.mem.trim(u8, text, " \t\r\n");
                const body = if (std.mem.startsWith(u8, trimmed, "/")) trimmed[1..] else trimmed;
                var words = Words.init(body);
                const name = words.next() catch return;
                const arguments = words.iterator.rest();
                if (std.mem.eql(u8, name, "help")) {
                    if (!fits(syntax[0].forms[0], arguments) and !fits(syntax[0].forms[1], arguments))
                        return console.usage(syntax[0], true);
                    if (words.iterator.next()) |wanted| {
                        for (syntax) |entry| {
                            if (std.mem.eql(u8, entry.name, wanted)) return console.usage(entry, false);
                        }
                    }
                    console.reply(false, "help", .{});
                    for (syntax[1..]) |entry| console.appendReply(" {s}", .{entry.name});
                    return;
                }
                inline for (commands) |command| {
                    if (std.mem.eql(u8, name, command.syntax.name)) {
                        var valid = false;
                        for (command.syntax.forms) |form| {
                            if (fits(form, arguments)) {
                                valid = true;
                                break;
                            }
                        }
                        if (!valid) return console.usage(command.syntax, true);
                        console.reply(false, "", .{});
                        command.run(context, &words, console) catch |err| switch (err) {
                            error.MissingArgument, error.MalformedArgument => console.usage(command.syntax, true),
                            else => console.reply(true, "{s}: {s}", .{ name, @errorName(err) }),
                        };
                        return;
                    }
                }
                console.reply(true, "unknown command {s}; help lists them", .{name});
            }
        };
    }

    fn appendReply(self: *Console, comptime format: []const u8, arguments: anytype) void {
        const rest = self.reply_buffer[self.reply_len..];
        const written = std.fmt.bufPrint(rest, format, arguments) catch rest;
        self.reply_len += written.len;
    }

    fn usage(self: *Console, syntax: Syntax, failed: bool) void {
        self.reply(failed, "usage: ", .{});
        for (syntax.forms, 0..) |form, index| {
            if (index != 0) self.appendReply(" or ", .{});
            self.appendReply("/{s}", .{syntax.name});
            if (form.len != 0) self.appendReply(" {s}", .{form});
        }
    }

    // What may be written where the line's last word is, in table and form
    // order without repeats, and the grey rest of the line after it.
    pub const Suggestions = struct {
        // Slices into the syntax table.
        candidates: [12][]const u8 = undefined,
        count: usize = 0,
        // Where the last word begins, which a completion replaces from.
        start: usize = 0,
        // Copied, since it joins pieces of a form that are not adjacent in it.
        hint: [line_capacity]u8 = undefined,
        hint_len: usize = 0,

        fn add(self: *Suggestions, candidate: []const u8) void {
            for (self.candidates[0..self.count]) |existing| {
                if (std.mem.eql(u8, existing, candidate)) return;
            }
            if (self.count == self.candidates.len) return;
            self.candidates[self.count] = candidate;
            self.count += 1;
        }

        fn appendHint(self: *Suggestions, text: []const u8) void {
            const length = @min(text.len, self.hint.len - self.hint_len);
            @memcpy(self.hint[self.hint_len..][0..length], text[0..length]);
            self.hint_len += length;
        }
    };

    pub fn suggest(syntax: []const Syntax, text: []const u8) Suggestions {
        var result: Suggestions = .{};
        const beginning: usize = if (std.mem.startsWith(u8, text, "/")) 1 else 0;
        result.start = if (std.mem.lastIndexOfScalar(u8, text, ' ')) |space| space + 1 else beginning;
        const prefix = text[result.start..];
        if (result.start == beginning) {
            var first = true;
            for (syntax) |entry| {
                if (!std.mem.startsWith(u8, entry.name, prefix)) continue;
                result.add(entry.name);
                if (first) {
                    result.appendHint(entry.name[prefix.len..]);
                    if (entry.name.len == prefix.len and entry.forms.len > 0 and entry.forms[0].len > 0) {
                        result.appendHint(" ");
                        result.appendHint(entry.forms[0]);
                    }
                    first = false;
                }
            }
            return result;
        }
        var complete_words = std.mem.tokenizeScalar(u8, text[beginning..result.start], ' ');
        const name = complete_words.next() orelse return result;
        for (syntax) |entry| {
            if (!std.mem.eql(u8, name, entry.name)) continue;
            var hinted = false;
            for (entry.forms) |form| {
                var words = complete_words;
                var slots = std.mem.tokenizeScalar(u8, form, ' ');
                var viable = true;
                while (words.next()) |word| {
                    if (!accepts(slots.next() orelse {
                        viable = false;
                        break;
                    }, word)) {
                        viable = false;
                        break;
                    }
                }
                if (!viable) continue;
                const slot = slots.next() orelse continue;
                var alternatives = std.mem.splitScalar(u8, slot, '|');
                var accepts_prefix = false;
                var literal: ?[]const u8 = null;
                while (alternatives.next()) |alternative| {
                    if (placeholder(alternative)) {
                        accepts_prefix = true;
                    } else if (std.mem.startsWith(u8, alternative, prefix)) {
                        result.add(alternative);
                        accepts_prefix = true;
                        if (literal == null) literal = alternative;
                    }
                }
                if (!hinted and accepts_prefix) {
                    if (prefix.len == 0) {
                        result.appendHint(form[slots.index - slot.len ..]);
                    } else {
                        if (literal) |value| result.appendHint(value[prefix.len..]);
                        const following = slots.rest();
                        if (following.len != 0) {
                            result.appendHint(" ");
                            result.appendHint(following);
                        }
                    }
                    hinted = true;
                }
            }
            return result;
        }
        return result;
    }

    // Tab: the last word becomes the next candidate for it, Shift+Tab the
    // previous one, wrapping. No space follows, so the next press still
    // replaces the same word.
    pub fn complete(self: *Console, syntax: []const Syntax, backwards: bool) void {
        if (self.cycle_selected == null) {
            self.cycle_len = self.state.len;
            @memcpy(self.cycle_base[0..self.cycle_len], self.line());
        }
        const suggestions = suggest(syntax, self.cycle_base[0..self.cycle_len]);
        if (suggestions.count == 0) return;
        const selected = if (self.cycle_selected) |previous|
            (previous + (if (backwards) suggestions.count - 1 else @as(usize, 1))) % suggestions.count
        else if (backwards) suggestions.count - 1 else 0;
        const candidate = suggestions.candidates[selected];
        if (suggestions.start + candidate.len > self.buffer.len) return;
        @memcpy(self.buffer[0..suggestions.start], self.cycle_base[0..suggestions.start]);
        @memcpy(self.buffer[suggestions.start..][0..candidate.len], candidate);
        const length = suggestions.start + candidate.len;
        self.state = .{ .len = length, .caret = length, .anchor = length };
        self.cycle_selected = selected;
    }

    // The list over the reply: of the line before Tab while Tab is walking it.
    fn listed(self: *const Console, syntax: []const Syntax) Suggestions {
        return suggest(syntax, if (self.cycle_selected != null) self.cycle_base[0..self.cycle_len] else self.line());
    }

    // A single candidate already typed in full lists nothing.
    fn suggestionRows(self: *const Console, choices: Suggestions) usize {
        if (choices.count == 1 and self.cycle_selected == null and
            std.mem.eql(u8, choices.candidates[0], self.line()[choices.start..])) return 0;
        return choices.count;
    }

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
        self.cycle_selected = null;
        self.recalled = 0;
    }

    pub fn hide(self: *Console) void {
        self.open = false;
        self.cycle_selected = null;
        self.recalled = 0;
    }

    // A key pressed while the console is open, after the UI has seen it.
    // Escape closes the console and the arrows walk the history; enter is
    // `Commands.submit`, since running a line needs the application.
    pub fn press(self: *Console, pressed: platform.PhysicalKey) void {
        switch (pressed) {
            .escape => self.hide(),
            .arrow_up => self.recall(self.recalled + 1),
            .arrow_down => if (self.recalled > 0) self.recall(self.recalled - 1),
            else => {},
        }
    }

    // The line `back` submissions ago, or an empty line at 0. Past the oldest
    // kept line nothing changes.
    fn recall(self: *Console, back: usize) void {
        self.cycle_selected = null;
        if (back > @min(self.history_count, history_capacity)) return;
        const kept = if (back == 0) "" else blk: {
            const slot = (self.history_count - back) % history_capacity;
            break :blk self.history[slot][0..self.history_lens[slot]];
        };
        self.show(kept);
        self.recalled = back;
    }

    fn remember(self: *Console, text: []const u8) void {
        const slot = self.history_count % history_capacity;
        const length = @min(text.len, line_capacity);
        @memcpy(self.history[slot][0..length], text[0..length]);
        self.history_lens[slot] = @intCast(length);
        self.history_count += 1;
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
    pub fn regions(self: *Console, engine: *const Engine, ui: *imui.WidgetContext, syntax: []const Syntax) !void {
        self.registered = self.open;
        if (!self.open) return;
        const layout = try self.layoutFor(engine, self.suggestionRows(self.listed(syntax)));
        try ui.register(key, layout.field, .{ .focusable = true });
        // Every frame and not only on opening: the keyboard is the console's
        // while it is open, and the UI's own Tab would otherwise walk the
        // focus away from it.
        try ui.focus(key);
    }

    // From the application's `ui_draw` hook, at the root scope.
    pub fn draw(self: *Console, engine: *Engine, ui: *imui.WidgetContext, syntax: []const Syntax) !void {
        if (!self.open or !self.registered) return;
        const font = self.font orelse return;
        const choices = self.listed(syntax);
        const rows = self.suggestionRows(choices);
        const layout = try self.layoutFor(engine, rows);

        const canvas = try ui.drawList();
        try canvas.addQuad(layout.plate, .{}, palette.plate.premultiplied(), engine.ui_white);

        var glyphs: [max_glyphs]res.ShapedGlyph = undefined;
        var placements: [max_placements]res.GlyphPlacement = undefined;

        for (choices.candidates[0..rows], 0..) |candidate, index| {
            var row = layout.answer;
            row.y -= row.height * @as(f32, @floatFromInt(rows - index));
            const selected = self.cycle_selected == index;
            if (selected) try canvas.addQuad(row, .{}, palette.selection.premultiplied(), engine.ui_white);
            const shaped = try engine.shapeLabel(font, candidate, &glyphs, &placements);
            const colour = (if (selected) palette.text else palette.dim).premultiplied();
            try ui.label(row, .{ .normal_text = colour, .disabled_text = colour }, shaped, true);
        }

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
        if (try ui.editText(key, layout.field, style, &self.buffer, &self.state, before, true)) {
            self.cycle_selected = null;
            self.recalled = 0;
        }
        const after = try engine.shapeLabel(font, self.line(), &glyphs, &placements);
        try ui.textField(key, layout.field, style, &self.state, after, true);
        const hint = suggest(syntax, self.line());
        if (self.state.caret == self.state.len and hint.hint_len > 0) {
            const inner = imui.textFieldLine(layout.field, style);
            var rect = inner;
            rect.x += after.advance - self.state.scroll;
            const shaped = try engine.shapeLabel(font, hint.hint[0..hint.hint_len], &glyphs, &placements);
            try ui.pushClip(inner);
            defer ui.popClip();
            try ui.label(rect, .{ .normal_text = palette.dim.premultiplied(), .disabled_text = palette.dim.premultiplied() }, shaped, true);
        }
    }

    const Layout = struct { plate: res.Rect, answer: res.Rect, field: res.Rect };

    // Along the bottom of the window, the reply's line above the field's and
    // `rows` candidates above that. In
    // the swapchain's extent, which is what the frame's root clip is.
    fn layoutFor(self: *const Console, engine: *const Engine, rows: usize) !Layout {
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
        const top = answer.y - line_height * @as(f32, @floatFromInt(rows));
        return .{
            .plate = .{
                .x = 0,
                .y = top - padding,
                .width = width,
                .height = height - (top - padding),
            },
            .answer = answer,
            .field = field,
        };
    }
};
