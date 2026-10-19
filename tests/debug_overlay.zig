const std = @import("std");
const lenore = @import("lenore");

const testing = std.testing;

fn frame(interval_ns: u64) lenore.FrameTime {
    return .{ .elapsed_ns = 0, .interval_ns = interval_ns, .delta = 0, .index = 0 };
}

test "the frame graph keeps the last frames, oldest first" {
    var graph: lenore.FrameGraph = .{};
    try testing.expectEqual(@as(?lenore.FrameGraph.Summary, null), graph.summary());
    for (0..lenore.debug_graph_frames + 3) |index| graph.record(frame(index + 1));
    try testing.expectEqual(@as(usize, lenore.debug_graph_frames), graph.filled);
    // The first three were overwritten.
    try testing.expectEqual(@as(u32, 4), graph.interval(0));
    try testing.expectEqual(@as(u32, lenore.debug_graph_frames + 3), graph.interval(graph.filled - 1));
}

// The case the graph is for: a steady rate with every fourth frame a refresh
// late. The median stays the refresh, the late frames are counted, and the
// spread says the run is uneven where the mean alone would not.
test "the frame graph's summary separates late frames from the rate" {
    var graph: lenore.FrameGraph = .{};
    for (0..40) |index| graph.record(frame(if (index % 4 == 3) 16_666_666 else 8_333_333));
    const summary = graph.summary().?;
    try testing.expectEqual(@as(u32, 8_333_333), summary.median_ns);
    try testing.expectEqual(@as(u32, 16_666_666), summary.worst_ns);
    try testing.expectEqual(@as(usize, 10), summary.long);
    try testing.expectApproxEqRel(@as(f64, 10_416_666.25), summary.mean_ns, 1e-6);
    // Two values a quarter and three quarters of the time: sqrt(p (1 - p)) times their gap.
    try testing.expectApproxEqRel(@sqrt(0.25 * 0.75) * 8_333_333.0, summary.spread_ns, 1e-6);
}

test "an interval past what the graph holds is held at its largest" {
    var graph: lenore.FrameGraph = .{};
    graph.record(frame(std.math.maxInt(u64)));
    try testing.expectEqual(@as(u32, std.math.maxInt(u32)), graph.interval(0));
}
