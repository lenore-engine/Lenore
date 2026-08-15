const std = @import("std");
const imui = @import("lenore-imui");
const lenore = @import("lenore");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");

const testing = std.testing;

// The engine's UI composition, without a device.
//
// What is under test here is not the widget module, which has its own suite,
// but the three things the engine adds to it: that a window event reaches the
// routing in the UI's own vocabulary, that what a widget took is reported back
// to the driver, and that a frame writes the slot it was pointed at rather than
// the one before it.

const image: res.ImageHandle = @enumFromInt(1);
const window: platform.SurfaceMetrics = .{
    .logical_size = .{ 400, 300 },
    .framebuffer_extent = .{ .width = 400, .height = 300 },
    .scale = .{ 1, 1 },
    .generation = 1,
};
const root: res.Rect = .{ .x = 0, .y = 0, .width = 400, .height = 300 };
const box: res.Rect = .{ .x = 100, .y = 100, .width = 80, .height = 40 };

// Two slots, the way a frame ring has two. Nothing about the host says which
// memory it writes; the loop hands it one per frame.
const Slots = struct {
    vertices: [2][64]res.Vertex2D = undefined,
    indices: [2][128]res.DrawIndex = undefined,
    commands: [16]res.DrawCommand = undefined,
    clips: [8]res.Rect = undefined,

    fn storage(self: *Slots, slot: usize) res.DrawListStorage {
        return .{
            .vertices = &self.vertices[slot],
            .indices = &self.indices[slot],
            .commands = &self.commands,
            .clips = &self.clips,
        };
    }
};

fn event(payload: platform.Payload) platform.Event {
    return .{ .sequence = 0, .timestamp_ns = 0, .payload = payload };
}

fn pointer(x: f32, y: f32, action: platform.KeyAction) platform.Event {
    return event(.{ .mouse_button = .{
        .button = .left,
        .action = action,
        .logical_position = .{ x, y },
        .metrics_generation = window.generation,
    } });
}

test "a window event reaches the widget under it, and one the UI cannot name does not" {
    var slots: Slots = .{};
    var host: lenore.UiHost = undefined;
    try host.init(testing.allocator, .{}, slots.storage(0), window, image);
    defer host.deinit(testing.allocator);

    try host.beginFrame(slots.storage(0), root);
    try host.widgets.register(.{ .string = "ok" }, box, .{});
    try host.beginRouting();

    // Off the widget and with nothing held: no widget wanted it, which is what
    // lets a camera act on the same press.
    try testing.expect(!try host.route(pointer(10, 10, .press)));
    // A key the UI declares nothing for. False rather than an error: most of
    // what a window says is not for the UI.
    try testing.expect(!try host.route(event(.{ .key = .{ .physical = .q, .action = .press } })));

    // On the widget, so the UI takes it. The position crosses through the
    // surface metrics rather than being passed through: at this scale the two
    // spaces agree, and the conversion has its own tests.
    try testing.expect(try host.route(pointer(120, 110, .press)));
    // Taken as well, and not because of where it is: a release of the button
    // being held ends the gesture wherever the pointer has got to, which is
    // what stops a drag from ending in whatever is behind the widget.
    try testing.expect(try host.route(pointer(10, 10, .release)));

    try host.finishRouting();
}

test "the frame writes the slot it was handed" {
    var slots: Slots = .{};
    var host: lenore.UiHost = undefined;
    try host.init(testing.allocator, .{}, slots.storage(0), window, image);
    defer host.deinit(testing.allocator);

    // A recognisable value in both slots, so that a write reaching the wrong
    // one is visible as the other's mark surviving.
    for (&slots.vertices) |*slot| for (slot) |*vertex| {
        vertex.* = .{ .position = .{ -1, -1 }, .uv = .{ 0, 0 }, .colour = .transparent };
    };

    try host.beginFrame(slots.storage(1), root);
    try host.beginRouting();
    try host.finishRouting();
    const canvas = try host.widgets.drawList();
    try canvas.addQuad(box, .{}, .white, image);

    try testing.expectEqual(@as(usize, 1), host.commands().len);
    // The geometry is in the slot the frame named and the other is untouched,
    // which is the whole of what the loop relies on when it hands over the
    // storage of the slot whose fence has signalled.
    try testing.expectEqual(@as(f32, box.x), slots.vertices[1][0].position[0]);
    try testing.expectEqual(@as(f32, -1), slots.vertices[0][0].position[0]);
}

test "a frame the caller failed in does not refuse the next one" {
    var slots: Slots = .{};
    var host: lenore.UiHost = undefined;
    try host.init(testing.allocator, .{}, slots.storage(0), window, image);
    defer host.deinit(testing.allocator);

    // A capacity of one region, reached by the second: the failure a driver
    // meets when its UI outgrows what the engine was built with.
    var small: lenore.UiHost = undefined;
    try small.init(testing.allocator, .{ .regions = 1 }, slots.storage(0), window, image);
    defer small.deinit(testing.allocator);

    try small.beginFrame(slots.storage(0), root);
    try small.widgets.register(.{ .string = "first" }, box, .{});
    try testing.expectError(
        error.RegionCapacityExceeded,
        small.widgets.register(.{ .string = "second" }, box, .{}),
    );
    small.abandonFrame();

    // The next frame runs whole. Without the abandon it would report
    // `InvalidPhase`, which names the recovery rather than the capacity and
    // sends whoever reads it looking in the wrong place.
    try small.beginFrame(slots.storage(0), root);
    try small.widgets.register(.{ .string = "first" }, box, .{});
    try small.beginRouting();
    try testing.expect(try small.route(pointer(120, 110, .press)));
    try small.finishRouting();
}
