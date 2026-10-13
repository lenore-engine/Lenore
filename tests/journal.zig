const std = @import("std");
const lenore = @import("lenore");

const testing = std.testing;

const Journal = lenore.Journal;

test "lines come back as they were recorded" {
    var journal: Journal = .{};
    defer journal.deinit(testing.allocator);
    try journal.record(testing.allocator, 0, "1 2 3 /fill ~ ~ ~ ~1 ~1 ~1 sand");
    try journal.record(testing.allocator, 7, "-1 0 4 /world flat");

    const entries = journal.entries.items;
    try testing.expectEqual(@as(usize, 2), entries.len);
    try testing.expectEqualStrings("1 2 3 /fill ~ ~ ~ ~1 ~1 ~1 sand", journal.lineOf(entries[0]));
    try testing.expectEqualStrings("-1 0 4 /world flat", journal.lineOf(entries[1]));
    try testing.expectEqual(@as(u64, 7), entries[1].tick);
}

test "a journal written and parsed is the same journal" {
    var journal: Journal = .{};
    defer journal.deinit(testing.allocator);
    // Two spaces inside the command survive, since the line is kept as written.
    try journal.record(testing.allocator, 0, "1 2 3 /fill ~ ~ ~  ~1 ~1 ~1 sand");
    try journal.record(testing.allocator, 40, "-5 0 4 /world cave");
    const run: lenore.JournalRun = .{ .ticks = 97, .fingerprint = 0x00ab_cdef_0123_4567 };

    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    try journal.write(&out.writer, "test journal", "flat", run);

    var parsed = try Journal.parse(testing.allocator, "test journal", out.written());
    defer parsed.journal.deinit(testing.allocator);
    try testing.expectEqual(run, parsed.run);
    try testing.expectEqualStrings("flat", parsed.start);
    try testing.expectEqual(journal.entries.items.len, parsed.journal.entries.items.len);
    for (journal.entries.items, parsed.journal.entries.items) |a, b| {
        try testing.expectEqual(a.tick, b.tick);
        try testing.expectEqualStrings(journal.lineOf(a), parsed.journal.lineOf(b));
    }
}

test "the text on disk is the documented format" {
    var journal: Journal = .{};
    defer journal.deinit(testing.allocator);
    try journal.record(testing.allocator, 3, "0 0 0 /tp ~ ~1 ~");

    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    try journal.write(&out.writer, "test journal", "flat", .{ .ticks = 5, .fingerprint = 0xff });
    try testing.expectEqualStrings(
        "test journal\nstart flat\ncommand 3 0 0 0 /tp ~ ~1 ~\nend 5 00000000000000ff\n",
        out.written(),
    );
}

test "a field holding a line break is refused" {
    var journal: Journal = .{};
    defer journal.deinit(testing.allocator);
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    const run: lenore.JournalRun = .{ .ticks = 0, .fingerprint = 0 };

    try testing.expectError(error.LineBreak, journal.write(&out.writer, "a\nb", "flat", run));
    try testing.expectError(error.LineBreak, journal.write(&out.writer, "test journal", "fl\nat", run));
    try journal.record(testing.allocator, 0, "/a\n/b");
    try testing.expectError(error.LineBreak, journal.write(&out.writer, "test journal", "flat", run));
}

test "a journal cut short or mangled is refused" {
    const cases = [_]struct { []const u8, anyerror }{
        .{ "", error.NotAJournal },
        .{ "other journal\nstart flat\n", error.NotAJournal },
        .{ "test journal\nstart flat\ncommand 0 /tp ~ ~ ~\n", error.MissingEnd },
        .{ "test journal\nstart flat\ncommand 0\n", error.MalformedLine },
        .{ "test journal\nbegin flat\n", error.MalformedLine },
        .{ "test journal\nstart flat\nend 5\n", error.MalformedLine },
        .{ "test journal\nstart flat\nend 5 ff extra\n", error.MalformedLine },
    };
    for (cases) |case| try testing.expectError(case[1], Journal.parse(testing.allocator, "test journal", case[0]));
}
