const std = @import("std");
const gpu = @import("lenore-gpu");
const imui = @import("lenore-imui");
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
    pub fn onEvent(_: *Full, _: *lenore.Engine, _: platform.Event, _: bool) !void {}
    pub fn onResize(_: *Full, _: *lenore.Engine, _: platform.Extent2D) !void {}
    pub fn onUpdate(_: *Full, _: *lenore.Engine, _: *lenore.Level, _: lenore.FrameTime) !void {}
    pub fn onDepth(_: *Full, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}
    pub fn onCompute(_: *Full, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}
    pub fn onRecord(_: *Full, _: *lenore.Engine, _: *lenore.Level, _: gpu.vk.CommandBuffer) !void {}
    pub fn onUiRegions(_: *Full, _: *lenore.Engine, _: *lenore.Level, _: *imui.WidgetContext) !void {}
    pub fn onUiDraw(_: *Full, _: *lenore.Engine, _: *lenore.Level, _: *imui.WidgetContext) !void {}
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

test "the device phase is independently destructible" {
    var engine: lenore.Engine = undefined;
    try engine.initDevice(testing.allocator, .{
        .title = "Lenore device initialization test",
        .extent = .{ .width = 64, .height = 64 },
        .frame_capacity = .{ .instances = 0, .joints = 0 },
        .morph_capacity = .{ .meshes = 0, .weights = 0 },
        .material_capacity = 0,
    });
    defer engine.deinit();
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

// The render scale's arithmetic, which decides how many pixels every pass
// before the presenting one is charged for.

test "a scale of one leaves the surface untouched" {
    const surface: platform.Extent2D = .{ .width = 2560, .height = 1600 };
    try testing.expectEqual(surface, lenore.scaledExtent(surface, 1.0));
}

test "two thirds of a side is four ninths of the pixels" {
    const scaled = lenore.scaledExtent(.{ .width = 2560, .height = 1600 }, 2.0 / 3.0);
    try testing.expectEqual(@as(u32, 1707), scaled.width);
    try testing.expectEqual(@as(u32, 1067), scaled.height);

    // The ratio survives the rounding to within a thousandth, which is what
    // lets the camera's projection be taken from either extent.
    const surface_ratio = 2560.0 / 1600.0;
    const target_ratio = @as(f64, @floatFromInt(scaled.width)) /
        @as(f64, @floatFromInt(scaled.height));
    try testing.expect(@abs(target_ratio - surface_ratio) < 1.0e-3);
}

test "a side is rounded rather than truncated" {
    // 3 * 0.5 is 1.5, which floors to one and rounds to two.
    try testing.expectEqual(@as(u32, 2), lenore.scaledExtent(.{ .width = 3, .height = 3 }, 0.5).width);
}

test "no side scales away to nothing" {
    const scaled = lenore.scaledExtent(.{ .width = 1, .height = 1 }, 0.01);
    try testing.expectEqual(@as(u32, 1), scaled.width);
    try testing.expectEqual(@as(u32, 1), scaled.height);
}
