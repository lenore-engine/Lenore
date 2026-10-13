// What a run was given, in order, so that it can be run again.
//
// A deterministic simulation is a function of how it started and of the
// commands it was given, each placed on the tick it ran before. The ticks are
// the simulation's own count and not the clock's, so a replay runs the same
// steps whether the live run was fast, slow or dropped ticks to keep up. The
// end records the tick count and a fingerprint of the final state, which a
// replay is checked against.
//
// What a command means, how a start is spelt and how the state is
// fingerprinted are the application's. A command is one line of text: an
// application whose commands depend on context the simulation does not hold
// (where the camera was, say) writes that context into the line.
//
// On disk it is text, a line each:
//
//   <magic>                       the application's name for the format
//   start <start>
//   command <tick> <line>         once per command, in order
//   end <ticks> <fingerprint in hex>
//
// No engine types and no device; tested on the host.

const std = @import("std");

const Allocator = std.mem.Allocator;

// How a run ended.
pub const Run = struct {
    ticks: u64,
    fingerprint: u64,
};

pub const ParseError = error{
    NotAJournal,
    MalformedLine,
    MissingEnd,
} || std.fmt.ParseIntError || Allocator.Error;

pub const WriteError = std.Io.Writer.Error || error{
    // A line break inside a field would read back as two lines, so it is
    // refused rather than written.
    LineBreak,
};

pub const Entry = struct {
    // Ticks the simulation had taken when the command ran. It ran before the
    // next.
    tick: u64,
    // The line, as a range of the journal's text.
    start: usize,
    len: usize,
};

pub const Journal = struct {
    entries: std.ArrayList(Entry) = .empty,
    text: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Journal, allocator: Allocator) void {
        self.entries.deinit(allocator);
        self.text.deinit(allocator);
        self.* = undefined;
    }

    // On failure the journal is as it was.
    pub fn record(self: *Journal, allocator: Allocator, tick: u64, line: []const u8) Allocator.Error!void {
        try self.entries.ensureUnusedCapacity(allocator, 1);
        const start = self.text.items.len;
        try self.text.appendSlice(allocator, line);
        self.entries.appendAssumeCapacity(.{ .tick = tick, .start = start, .len = line.len });
    }

    pub fn lineOf(self: *const Journal, entry: Entry) []const u8 {
        return self.text.items[entry.start..][0..entry.len];
    }

    pub fn write(
        self: *const Journal,
        writer: *std.Io.Writer,
        magic: []const u8,
        start: []const u8,
        run: Run,
    ) WriteError!void {
        if (hasLineBreak(magic) or hasLineBreak(start)) return error.LineBreak;
        try writer.print("{s}\nstart {s}\n", .{ magic, start });
        for (self.entries.items) |entry| {
            const line = self.lineOf(entry);
            if (hasLineBreak(line)) return error.LineBreak;
            try writer.print("command {d} {s}\n", .{ entry.tick, line });
        }
        try writer.print("end {d} {x:0>16}\n", .{ run.ticks, run.fingerprint });
    }

    pub const Parsed = struct {
        journal: Journal,
        // Borrowed from the text that was parsed, and valid as long as it is.
        start: []const u8,
        run: Run,
    };

    pub fn parse(allocator: Allocator, magic: []const u8, text: []const u8) ParseError!Parsed {
        var lines = std.mem.splitScalar(u8, text, '\n');
        if (!std.mem.eql(u8, lines.next() orelse "", magic)) return error.NotAJournal;

        const start_line = lines.next() orelse return error.MalformedLine;
        const start_prefix = "start ";
        if (!std.mem.startsWith(u8, start_line, start_prefix)) return error.MalformedLine;
        const start = start_line[start_prefix.len..];

        var journal: Journal = .{};
        errdefer journal.deinit(allocator);
        while (lines.next()) |line| {
            if (line.len == 0) continue;
            var words = std.mem.tokenizeScalar(u8, line, ' ');
            const kind = words.next() orelse return error.MalformedLine;
            if (std.mem.eql(u8, kind, "end")) {
                const ticks = try std.fmt.parseInt(u64, words.next() orelse return error.MalformedLine, 10);
                const fingerprint = try std.fmt.parseInt(u64, words.next() orelse return error.MalformedLine, 16);
                if (words.next() != null) return error.MalformedLine;
                return .{
                    .journal = journal,
                    .start = start,
                    .run = .{ .ticks = ticks, .fingerprint = fingerprint },
                };
            }
            if (!std.mem.eql(u8, kind, "command")) return error.MalformedLine;
            const tick = try std.fmt.parseInt(u64, words.next() orelse return error.MalformedLine, 10);
            // The command is the rest of the line as written, spaces and all.
            const rest = words.rest();
            if (rest.len == 0) return error.MalformedLine;
            try journal.record(allocator, tick, line[line.len - rest.len ..]);
        }
        return error.MissingEnd;
    }
};

fn hasLineBreak(bytes: []const u8) bool {
    return std.mem.findScalar(u8, bytes, '\n') != null;
}
