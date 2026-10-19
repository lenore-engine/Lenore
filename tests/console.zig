const std = @import("std");
const lenore = @import("lenore");

const testing = std.testing;

test "a console without a font never opens" {
    var console: lenore.Console = .{ .font = null };
    console.show("/");
    try testing.expect(!console.open);
}

test "a console opens with its prefix typed and the caret after it" {
    var console: lenore.Console = .{ .font = 0 };
    console.show("/tp");
    try testing.expect(console.open);
    try testing.expectEqualStrings("/tp", console.line());
    try testing.expectEqual(@as(usize, 3), console.state.caret);

    console.hide();
    try testing.expect(!console.open);
}

test "a prefix and a reply longer than their buffers are cut, not refused" {
    var console: lenore.Console = .{ .font = 0 };
    const long: [400]u8 = @splat('a');
    console.show(&long);
    try testing.expectEqual(console.buffer.len, console.line().len);

    console.reply(true, "{s}", .{&long});
    try testing.expectEqual(console.reply_buffer.len, console.reply_len);
    try testing.expect(console.reply_failed);
}

test "overlay lines are cut at their capacity and dropped past the last" {
    var lines: lenore.DebugLines = .{};
    const long: [200]u8 = @splat('b');
    lines.add("{s}", .{&long});
    try testing.expectEqual(@as(usize, 120), lines.line(0).len);
    for (1..20) |index| lines.add("line {d}", .{index});
    try testing.expectEqual(@as(usize, 16), lines.count);
    try testing.expectEqualStrings("line 15", lines.line(15));
}

test "an overlay without a font cannot be toggled open" {
    var overlay: lenore.DebugOverlay = .{ .font = null };
    overlay.toggle();
    try testing.expect(!overlay.open);
}

const Console = lenore.Console;
const Context = struct {
    calls: usize = 0,
    value: f32 = 0,

    fn time(self: *Context, words: *Console.Words, _: *Console) !void {
        _ = try words.next();
        self.value = try words.float(f32);
        self.calls += 1;
    }

    fn ping(self: *Context, _: *Console.Words, _: *Console) !void {
        self.calls += 1;
    }

    fn fail(_: *Context, words: *Console.Words, _: *Console) !void {
        const word = try words.next();
        if (std.mem.eql(u8, word, "missing")) return error.MissingArgument;
        if (std.mem.eql(u8, word, "malformed")) return error.MalformedArgument;
        return error.Unavailable;
    }
};
const Registry = Console.Commands(*Context, &.{
    .{ .syntax = .{ .name = "time", .forms = &.{ "set day|noon|night|midnight|<ticks>|<HH:MM>", "add <ticks>" } }, .run = Context.time },
    .{ .syntax = .{ .name = "ping" }, .run = Context.ping },
    .{ .syntax = .{ .name = "pick", .forms = &.{"day|dawn|night <value>"} }, .run = Context.ping },
    .{ .syntax = .{ .name = "fail", .forms = &.{"<reason>"} }, .run = Context.fail },
});

fn expectSuggestion(text: []const u8, candidates: []const []const u8, hint: []const u8, start: usize) !void {
    const found = Console.suggest(Registry.syntax, text);
    try testing.expectEqual(candidates.len, found.count);
    for (candidates, 0..) |candidate, index| try testing.expectEqualStrings(candidate, found.candidates[index]);
    try testing.expectEqualStrings(hint, found.hint[0..found.hint_len]);
    try testing.expectEqual(start, found.start);
}

fn expectReply(console: *const Console, failed: bool, reply: []const u8) !void {
    try testing.expectEqual(failed, console.reply_failed);
    try testing.expectEqualStrings(reply, console.reply_buffer[0..console.reply_len]);
}

test "suggestions complete names, literals, placeholders and help through the same table" {
    try expectSuggestion("/pi", &.{ "ping", "pick" }, "ng", 1);
    try expectSuggestion("time", &.{"time"}, " set day|noon|night|midnight|<ticks>|<HH:MM>", 0);
    try expectSuggestion("/time s", &.{"set"}, "et day|noon|night|midnight|<ticks>|<HH:MM>", 6);
    try expectSuggestion("/time set n", &.{ "noon", "night" }, "oon", 10);
    try expectSuggestion("time add ", &.{}, "<ticks>", 9);
    try expectSuggestion("time add 12", &.{}, "", 9);
    try expectSuggestion("pick d", &.{ "day", "dawn" }, "ay <value>", 5);
    try expectSuggestion("help ti", &.{"time"}, "me", 5);
    try expectSuggestion("wat ", &.{}, "", 4);
    try expectSuggestion("time bad ", &.{}, "", 9);
    try expectSuggestion("time add 12 ", &.{}, "", 12);
    try expectSuggestion("time   set  ", &.{ "day", "noon", "night", "midnight" }, "day|noon|night|midnight|<ticks>|<HH:MM>", 12);
}

test "suggestions deduplicate in form order and stop at capacity" {
    const syntax: []const Console.Syntax = &.{.{ .name = "x", .forms = &.{ "a|b <x>", "b|c <y>", "d|e|f|g|h|i|j|k|l|m|n" } }};
    const found = Console.suggest(syntax, "x ");
    try testing.expectEqual(@as(usize, 12), found.count);
    for (found.candidates[0..found.count], "abcdefghijkl") |word, byte| {
        try testing.expectEqual(@as(usize, 1), word.len);
        try testing.expectEqual(byte, word[0]);
    }
}

test "execute checks complete forms and reports usage, help and command failures" {
    var console: Console = .{ .font = 0 };
    var context: Context = .{};
    Registry.execute(&context, &console, " /time  add  -2.5 ");
    try testing.expectEqual(@as(usize, 1), context.calls);
    try testing.expectEqual(@as(f32, -2.5), context.value);
    try expectReply(&console, false, "");
    for ([_][]const u8{ "time", "time add", "time wrong 1", "time add 1 extra", "time set inf" }) |text| {
        Registry.execute(&context, &console, text);
        try expectReply(&console, true, "usage: /time set day|noon|night|midnight|<ticks>|<HH:MM> or /time add <ticks>");
    }
    try testing.expectEqual(@as(usize, 1), context.calls);
    Registry.execute(&context, &console, "ping");
    try testing.expectEqual(@as(usize, 2), context.calls);
    Registry.execute(&context, &console, "ping extra");
    try expectReply(&console, true, "usage: /ping");
    Registry.execute(&context, &console, "help");
    try expectReply(&console, false, "help time ping pick fail");
    Registry.execute(&context, &console, "help ping");
    try expectReply(&console, false, "usage: /ping");
    Registry.execute(&context, &console, "help help");
    try expectReply(&console, false, "usage: /help or /help help|time|ping|pick|fail");
    Registry.execute(&context, &console, "help wat");
    try testing.expect(console.reply_failed);
    Registry.execute(&context, &console, "/wat");
    try expectReply(&console, true, "unknown command wat; help lists them");
    for ([_][]const u8{ "fail missing", "fail malformed" }) |text| {
        Registry.execute(&context, &console, text);
        try expectReply(&console, true, "usage: /fail <reason>");
    }
    Registry.execute(&context, &console, "fail other");
    try expectReply(&console, true, "fail: Unavailable");
    const Empty = Console.Commands(void, &.{});
    Empty.execute({}, &console, "help");
    try expectReply(&console, false, "help");
}

test "plain decimal parsing refuses extended syntax and overflow" {
    for ([_][]const u8{ "1e3", "inf", "nan", "1-2", ".", "", "+", "1_0", "1.2.3", "--1", "1" ++ @as([50]u8, @splat('0')) }) |text| {
        try testing.expectEqual(@as(?f32, null), Console.parseDecimal(f32, text));
        var words = Console.Words.init(text);
        try testing.expectError(if (text.len == 0) error.MissingArgument else error.MalformedArgument, words.float(f32));
    }
    try testing.expectEqual(@as(?f64, -0.5), Console.parseDecimal(f64, "-.5"));
    try testing.expectEqual(@as(?f32, 1), Console.parseDecimal(f32, "+1."));
    var words = Console.Words.init("12 -2 .5 nope 1_0");
    try testing.expectEqual(@as(u8, 12), try words.int(u8));
    try testing.expectEqual(@as(i32, -2), try words.int(i32));
    try testing.expectEqual(@as(f32, 0.5), try words.float(f32));
    try testing.expectError(error.MalformedArgument, words.int(i32));
    try testing.expectError(error.MalformedArgument, words.int(i32));
    try testing.expectError(error.MissingArgument, words.next());
}

test "submission keeps a bounded history and recall returns to an empty line" {
    var console: Console = .{ .font = 0 };
    var context: Context = .{};
    for (0..20) |index| {
        var buffer: [32]u8 = undefined;
        console.show(try std.fmt.bufPrint(&buffer, " unknown{d} ", .{index}));
        Registry.submit(&context, &console);
        try testing.expect(console.open);
        try testing.expectEqualStrings("", console.line());
    }
    for (0..18) |_| console.press(.arrow_up);
    try testing.expectEqualStrings("unknown4", console.line());
    for (0..16) |_| console.press(.arrow_down);
    try testing.expectEqualStrings("", console.line());
    console.press(.arrow_up);
    try testing.expectEqualStrings("unknown19", console.line());
    console.show("ping");
    try testing.expectEqual(@as(usize, 0), console.recalled);
    Registry.submit(&context, &console);
    try testing.expectEqual(@as(usize, 1), context.calls);
    const count = console.history_count;
    console.show(" / ");
    Registry.submit(&context, &console);
    try testing.expect(!console.open);
    try testing.expectEqual(count, console.history_count);
}

test "Tab cycles the saved prefix forwards and backwards and show resets it" {
    var console: Console = .{ .font = 0 };
    var context: Context = .{};
    console.show("/pi");
    Registry.press(&context, &console, .{ .physical = .tab, .action = .release });
    try testing.expectEqualStrings("/pi", console.line());
    Registry.press(&context, &console, .{ .physical = .tab, .action = .press });
    try testing.expectEqualStrings("/ping", console.line());
    Registry.press(&context, &console, .{ .physical = .tab, .action = .repeat });
    try testing.expectEqualStrings("/pick", console.line());
    console.complete(Registry.syntax, false);
    try testing.expectEqualStrings("/ping", console.line());
    Registry.press(&context, &console, .{ .physical = .tab, .action = .press, .modifiers = .{ .shift = true } });
    try testing.expectEqualStrings("/pick", console.line());
    try testing.expectEqual(console.state.len, console.state.caret);
    console.show("time set n");
    console.complete(Registry.syntax, true);
    try testing.expectEqualStrings("time set night", console.line());
    console.complete(Registry.syntax, true);
    try testing.expectEqualStrings("time set noon", console.line());
    console.press(.arrow_up);
    try testing.expectEqual(@as(?usize, null), console.cycle_selected);
    Registry.press(&context, &console, .{ .physical = .escape, .action = .repeat });
    try testing.expect(console.open);
    console.show("ping");
    Registry.press(&context, &console, .{ .physical = .enter, .action = .repeat });
    try testing.expectEqual(@as(usize, 0), context.calls);
    Registry.press(&context, &console, .{ .physical = .numpad_enter, .action = .press });
    try testing.expectEqual(@as(usize, 1), context.calls);
    Registry.press(&context, &console, .{ .physical = .arrow_up, .action = .repeat });
    try testing.expectEqualStrings("ping", console.line());
    Registry.press(&context, &console, .{ .physical = .escape, .action = .press });
    try testing.expect(!console.open);
}
