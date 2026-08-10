const std = @import("std");
const gpu = @import("lenore-gpu");
const lenore = @import("lenore");
const platform = @import("lenore-platform");

const testing = std.testing;

// The loop is generic over its driver, so nothing compiles its body until a
// driver is named. `_ = &Engine.run` does not: a reference to a generic function
// is not an instantiation, and a body only a reach line names would stay
// unanalysed however green the build.
//
// `@TypeOf` on a call is what instantiates without running. The return type is
// an inferred error set, so the compiler has to walk the body to know it, and a
// mistake anywhere inside stops the build here. Proven by breaking a statement
// in the loop and watching this fail.

// `pub` is the whole point of this fixture. Written without it, these are
// invisible to the engine's file, and before the hooks were made required that
// compiled into a loop that called nothing.
const Full = struct {
    pub fn onEvent(_: *Full, _: *lenore.Engine, _: platform.Event) !void {}
    pub fn onResize(_: *Full, _: *lenore.Engine, _: platform.Extent2D) !void {}
    pub fn onCompute(_: *Full, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}
    pub fn onRecord(_: *Full, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}
    pub fn onFrame(_: *Full, _: *lenore.Engine, _: *lenore.Level, _: lenore.FrameTime) !void {}
};

test "the loop compiles for a driver that declares every hook" {
    var driver: Full = .{};
    const engine: *lenore.Engine = undefined;
    const level: *lenore.Level = undefined;
    _ = @TypeOf(engine.run(level, &driver));
}

test "the loop compiles for the driver that does nothing" {
    var driver: lenore.NoDriver = .{};
    const engine: *lenore.Engine = undefined;
    const level: *lenore.Level = undefined;
    _ = @TypeOf(engine.run(level, &driver));
}

fn bake(request: lenore.ShadowBakeRequest) gpu.ShadowBake {
    return request.decide();
}

test "shadows switched off never re-record the map" {
    // Whatever else is true. The renderer has already baked once on its own
    // account, the shader returns before sampling, and nothing reads what the
    // map holds.
    try testing.expectEqual(gpu.ShadowBake.reuse, bake(.{
        .shadows_on = false,
        .were_on = true,
        .map_stale = true,
        .casters_move = true,
    }));
    try testing.expectEqual(gpu.ShadowBake.reuse, bake(.{
        .shadows_on = false,
        .were_on = false,
        .map_stale = false,
        .casters_move = false,
    }));
}

test "switching shadows on re-records whatever the map last held" {
    // The one case a still scene must still bake: while shadows were off the
    // casters moved and the sun may have, and nothing updated the map.
    try testing.expectEqual(gpu.ShadowBake.rebake, bake(.{
        .shadows_on = true,
        .were_on = false,
        .map_stale = false,
        .casters_move = false,
    }));
}

test "a still scene under a still sun bakes once and reuses it" {
    try testing.expectEqual(gpu.ShadowBake.reuse, bake(.{
        .shadows_on = true,
        .were_on = true,
        .map_stale = false,
        .casters_move = false,
    }));
}

test "a moved sun re-records even though nothing else changed" {
    try testing.expectEqual(gpu.ShadowBake.rebake, bake(.{
        .shadows_on = true,
        .were_on = true,
        .map_stale = true,
        .casters_move = false,
    }));
}

test "animated casters re-record under a sun that has not moved" {
    try testing.expectEqual(gpu.ShadowBake.rebake, bake(.{
        .shadows_on = true,
        .were_on = true,
        .map_stale = false,
        .casters_move = true,
    }));
}

test "the phase window averages over the frames it summed" {
    var window: lenore.FramePhases = .{};
    // Three frames of different shapes, so a mean taken over the wrong count or
    // a field summed into its neighbour is visible.
    for ([_]lenore.FramePhases{
        .{ .wait_ns = 100, .world_ns = 20, .record_ns = 6 },
        .{ .wait_ns = 200, .world_ns = 40, .record_ns = 9 },
        .{ .wait_ns = 300, .world_ns = 60, .record_ns = 15 },
    }) |frame| window.add(frame);

    const mean = window.mean(3);
    try testing.expectEqual(@as(u64, 200), mean.wait_ns);
    try testing.expectEqual(@as(u64, 40), mean.world_ns);
    try testing.expectEqual(@as(u64, 10), mean.record_ns);
    // A phase nothing spent time in stays zero rather than picking up a
    // neighbour's total.
    try testing.expectEqual(@as(u64, 0), mean.acquire_ns);
    try testing.expectEqual(@as(u64, 250), mean.total());
}

test "a window of no frames divides nothing" {
    const window: lenore.FramePhases = .{ .wait_ns = 7 };
    try testing.expectEqual(@as(u64, 0), window.mean(0).wait_ns);
}
