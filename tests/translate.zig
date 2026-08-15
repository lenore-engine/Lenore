const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const imui = @import("lenore-imui");
const lenore = @import("lenore");
const platform = @import("lenore-platform");
const scene = @import("lenore-scene");
const zm = @import("zmath");

const testing = std.testing;

// Never dereferenced. A resident mesh is a device object, and everything the
// translation does with one is carry the pointer through, so identity is the
// whole of what these stand for.
var mesh_a: gpu.Mesh = undefined;
var mesh_b: gpu.Mesh = undefined;

// `lenore-gltf` does not export the document types from its root, so the type of
// this field cannot be named from outside. The importer produces these values in
// the engine's own use and never takes one, so only a fixture needs to say it.
const DocumentLight = @FieldType(gltf.importer.Light, "source");
const LightKind = @FieldType(DocumentLight, "kind");

fn documentLight(kind: LightKind, intensity: f32, world: zm.Mat) gltf.importer.Light {
    return .{
        .source = .{
            .name = null,
            .color = .{ 1, 1, 1 },
            .intensity = intensity,
            .kind = kind,
            .range = null,
            .spot = if (kind == .spot)
                .{ .inner_angle = 0.2, .outer_angle = 0.5 }
            else
                null,
        },
        .world = world,
    };
}

test "a batch keeps its mesh, material, run and face policy" {
    // Three different cull modes, three different materials, two runs of
    // different length. Equal values anywhere here would let a translation that
    // copied the wrong field pass.
    const batches = [_]lenore.RecordPlan.Batch{
        .{ .mesh = &mesh_a, .material = 7, .face_culling = .none, .front_face = .counter_clockwise, .first_instance = 0, .instance_count = 2 },
        .{ .mesh = &mesh_b, .material = 3, .face_culling = .back, .front_face = .counter_clockwise, .first_instance = 2, .instance_count = 1 },
        .{ .mesh = &mesh_a, .material = 5, .face_culling = .front, .front_face = .clockwise, .first_instance = 3, .instance_count = 4 },
    };
    const ordered = [_]u32{ 0, 1, 2, 3, 4, 5, 6 };
    const sources = [_]?gpu.MeshVertexSource{null} ** 7;
    var destination: [3]gpu.RecordBatch = undefined;

    const records = try lenore.recordBatches(&batches, &ordered, &sources, &destination);
    try testing.expectEqual(@as(usize, 3), records.len);

    try testing.expectEqual(&mesh_a, records[0].mesh);
    try testing.expectEqual(@as(u32, 7), records[0].material_index);
    try testing.expectEqual(@as(u32, 0), records[0].first_instance);
    try testing.expectEqual(@as(u32, 2), records[0].instance_count);
    try testing.expect(!records[0].cull_mode.front_bit and !records[0].cull_mode.back_bit);
    // The winding is its own state and not a second reading of the cull mode:
    // the third batch below culls the front and is wound the other way, and the
    // first culls nothing and is wound the usual way.
    try testing.expectEqual(gpu.vk.FrontFace.counter_clockwise, records[0].front_face);

    try testing.expectEqual(&mesh_b, records[1].mesh);
    try testing.expectEqual(@as(u32, 3), records[1].material_index);
    try testing.expectEqual(@as(u32, 2), records[1].first_instance);
    try testing.expectEqual(@as(u32, 1), records[1].instance_count);
    try testing.expect(records[1].cull_mode.back_bit and !records[1].cull_mode.front_bit);

    try testing.expectEqual(@as(u32, 5), records[2].material_index);
    try testing.expectEqual(@as(u32, 3), records[2].first_instance);
    try testing.expectEqual(@as(u32, 4), records[2].instance_count);
    try testing.expect(records[2].cull_mode.front_bit and !records[2].cull_mode.back_bit);
    try testing.expectEqual(gpu.vk.FrontFace.clockwise, records[2].front_face);
}

test "the vertex source is the batch's first draw, through the order" {
    // The order is not the identity, so indexing `sources` by `first_instance`
    // instead of by `ordered[first_instance]` reads a different slot. The two
    // are told apart by giving every mesh a distinguishable offset.
    const batches = [_]lenore.RecordPlan.Batch{
        .{ .mesh = &mesh_a, .material = 0, .face_culling = .none, .front_face = .counter_clockwise, .first_instance = 0, .instance_count = 1 },
        .{ .mesh = &mesh_b, .material = 1, .face_culling = .none, .front_face = .counter_clockwise, .first_instance = 1, .instance_count = 1 },
        .{ .mesh = &mesh_a, .material = 2, .face_culling = .none, .front_face = .counter_clockwise, .first_instance = 2, .instance_count = 1 },
    };
    const ordered = [_]u32{ 2, 0, 1 };
    const sources = [_]?gpu.MeshVertexSource{
        .{ .handle = .null_handle, .offset = 100 },
        null,
        .{ .handle = .null_handle, .offset = 300 },
    };
    var destination: [3]gpu.RecordBatch = undefined;

    const records = try lenore.recordBatches(&batches, &ordered, &sources, &destination);
    try testing.expectEqual(@as(u64, 300), records[0].vertex_source.?.offset);
    try testing.expectEqual(@as(u64, 100), records[1].vertex_source.?.offset);
    try testing.expectEqual(@as(?gpu.MeshVertexSource, null), records[2].vertex_source);
}

test "a destination shorter than the batch list is refused" {
    const batches = [_]lenore.RecordPlan.Batch{
        .{ .mesh = &mesh_a, .material = 0, .face_culling = .none, .front_face = .counter_clockwise, .first_instance = 0, .instance_count = 1 },
        .{ .mesh = &mesh_b, .material = 1, .face_culling = .none, .front_face = .counter_clockwise, .first_instance = 1, .instance_count = 1 },
    };
    const ordered = [_]u32{ 0, 1 };
    const sources = [_]?gpu.MeshVertexSource{ null, null };
    var destination: [1]gpu.RecordBatch = undefined;

    try testing.expectError(
        error.BatchDestinationTooSmall,
        lenore.recordBatches(&batches, &ordered, &sources, &destination),
    );
}

test "a batch reaching past the order or the sources is refused" {
    const sources = [_]?gpu.MeshVertexSource{ null, null };
    var destination: [1]gpu.RecordBatch = undefined;

    const past_order = [_]lenore.RecordPlan.Batch{
        .{ .mesh = &mesh_a, .material = 0, .face_culling = .none, .front_face = .counter_clockwise, .first_instance = 4, .instance_count = 1 },
    };
    const short_order = [_]u32{ 0, 1 };
    try testing.expectError(
        error.BatchDrawOutOfRange,
        lenore.recordBatches(&past_order, &short_order, &sources, &destination),
    );

    // In range for the order, and the draw it names is past the sources.
    const in_order = [_]lenore.RecordPlan.Batch{
        .{ .mesh = &mesh_a, .material = 0, .face_culling = .none, .front_face = .counter_clockwise, .first_instance = 0, .instance_count = 1 },
    };
    const wide_draw = [_]u32{9};
    try testing.expectError(
        error.BatchDrawOutOfRange,
        lenore.recordBatches(&in_order, &wide_draw, &sources, &destination),
    );

    // Exactly at each bound, which is the only place the comparison itself is
    // observable: a guard written with `>` admits these and reads one past the
    // end of the caller's slice.
    const at_order_end = [_]lenore.RecordPlan.Batch{
        .{ .mesh = &mesh_a, .material = 0, .face_culling = .none, .front_face = .counter_clockwise, .first_instance = 2, .instance_count = 1 },
    };
    try testing.expectError(
        error.BatchDrawOutOfRange,
        lenore.recordBatches(&at_order_end, &short_order, &sources, &destination),
    );

    const at_sources_end = [_]u32{2};
    try testing.expectError(
        error.BatchDrawOutOfRange,
        lenore.recordBatches(&in_order, &at_sources_end, &sources, &destination),
    );
}

test "an empty batch list translates to an empty record list" {
    const ordered = [_]u32{0};
    const sources = [_]?gpu.MeshVertexSource{null};
    var destination: [1]gpu.RecordBatch = undefined;

    const records = try lenore.recordBatches(&.{}, &ordered, &sources, &destination);
    try testing.expectEqual(@as(usize, 0), records.len);
}

test "the no-pose marker becomes zero and every other base passes through" {
    try testing.expectEqual(@as(u32, 0), lenore.jointBase(scene.no_joint_base));
    try testing.expectEqual(@as(u32, 0), lenore.jointBase(0));
    // Not zero and not the marker, so a function returning either constant
    // fails here.
    try testing.expectEqual(@as(u32, 37), lenore.jointBase(37));
}

test "vertex sources are refused before the pass is read" {
    // The guard returns before `pass` is dereferenced, which is what makes an
    // undefined one safe here and is the whole property under test.
    var pass: gpu.MorphPass = undefined;
    const registrations = [_]?u32{ null, null };
    var destination: [3]?gpu.MeshVertexSource = undefined;

    try testing.expectError(
        error.VertexSourceCountMismatch,
        lenore.fillVertexSources(&pass, &registrations, 0, &destination),
    );
}

test "a directional light is oriented by its node and nothing else" {
    const half_turn = zm.rotationY(std.math.pi);
    const light = try lenore.sceneLight(documentLight(.directional, 3, half_turn));

    // Untransformed it points down -Z; a half turn about Y sends it to +Z.
    const direction = light.kind.directional;
    try testing.expectApproxEqAbs(@as(f32, 0), direction[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), direction[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1), direction[2], 1e-6);
    try testing.expectEqual(@as(f32, 3), light.intensity);
}

test "a spot light is placed by its node and keeps its cone ordered" {
    const world = zm.translation(2, -5, 11);
    const light = try lenore.sceneLight(documentLight(.spot, 1, world));

    const cone = light.kind.spot;
    try testing.expectApproxEqAbs(@as(f32, 2), cone.position[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -5), cone.position[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 11), cone.position[2], 1e-5);
    // Cosine falls as the angle opens, so the inner cosine is the larger.
    try testing.expect(cone.cos_inner > cone.cos_outer);
    // A light with no authored range gets one derived rather than infinity.
    try testing.expect(cone.range > 0 and std.math.isFinite(cone.range));
}

test "each light kind packs into its own lanes and zeroes the rest" {
    const directional = lenore.packLight(try .directional(.{ 0.25, 0.5, 0.75 }, 2, .{ 0, 0, -1 }));
    try testing.expectEqual(gpu.LightUniform.Kind.directional, directional.kind);
    try testing.expectEqual(@as(f32, 0.25), directional.colour[0]);
    try testing.expectEqual(@as(f32, 0.75), directional.colour[2]);
    try testing.expectEqual(@as(f32, 2), directional.intensity);
    try testing.expectEqual(@as(f32, -1), directional.direction[2]);
    // A directional light has no position and no range, and the lanes it does
    // not use are zero rather than carrying another kind's value.
    try testing.expectEqual([3]f32{ 0, 0, 0 }, directional.position);
    try testing.expectEqual(@as(f32, 0), directional.range);

    const point = lenore.packLight(try .point(.{ 1, 1, 1 }, 4, .{ 3, 6, 9 }, 12));
    try testing.expectEqual(gpu.LightUniform.Kind.point, point.kind);
    try testing.expectEqual([3]f32{ 3, 6, 9 }, point.position);
    try testing.expectEqual(@as(f32, 12), point.range);
    try testing.expectEqual(@as(f32, 0), point.cos_outer);

    const spot = lenore.packLight(try .spot(.{ 1, 1, 1 }, 5, .{
        .position = .{ -1, -2, -3 },
        .direction = .{ 1, 0, 0 },
        .range = 8,
        .inner_angle = 0.2,
        .outer_angle = 0.5,
    }));
    try testing.expectEqual(gpu.LightUniform.Kind.spot, spot.kind);
    try testing.expectEqual([3]f32{ -1, -2, -3 }, spot.position);
    try testing.expectEqual(@as(f32, 1), spot.direction[0]);
    // Not the same number as the point light's above, so a range read from the
    // wrong branch is visible.
    try testing.expectEqual(@as(f32, 8), spot.range);
    try testing.expect(spot.cos_inner > spot.cos_outer);
}

test "a rejected light is dropped and the rest of the document still draws" {
    // The rejection is reported, and this is the one test that provokes it. The
    // runner resets this per test, so the next one still sees warnings.
    std.testing.log_level = .err;

    const identity = zm.identity();
    const document = [_]gltf.importer.Light{
        documentLight(.directional, 1, identity),
        // Negative intensity: the scene module refuses it rather than shading
        // with it.
        documentLight(.point, -1, identity),
        documentLight(.spot, 2, identity),
    };

    var block: [gpu.max_lights]gpu.LightUniform = undefined;
    const live = lenore.packDocumentLights(&document, &block);

    try testing.expectEqual(@as(usize, 2), live.len);
    // The survivors close the gap rather than leaving a hole where the
    // rejected light was.
    try testing.expectEqual(gpu.LightUniform.Kind.directional, live[0].kind);
    try testing.expectEqual(gpu.LightUniform.Kind.spot, live[1].kind);
    try testing.expectEqual(@as(f32, 2), live[1].intensity);
}

test "a document with more lights than the block holds fills it and stops" {
    const identity = zm.identity();
    var document: [gpu.max_lights + 3]gltf.importer.Light = undefined;
    for (&document, 0..) |*entry, index| {
        // A distinct intensity per light, so the prefix that survives is
        // identifiable as the first ones rather than any sixteen.
        entry.* = documentLight(.directional, @floatFromInt(index + 1), identity);
    }

    var block: [gpu.max_lights]gpu.LightUniform = undefined;
    const live = lenore.packDocumentLights(&document, &block);

    try testing.expectEqual(@as(usize, gpu.max_lights), live.len);
    try testing.expectEqual(@as(f32, 1), live[0].intensity);
    try testing.expectEqual(@as(f32, gpu.max_lights), live[gpu.max_lights - 1].intensity);
}

fn uiEvent(payload: platform.Payload) platform.Event {
    return .{ .sequence = 0, .timestamp_ns = 0, .payload = payload };
}

fn uiMetrics(generation: u32) platform.Payload {
    return .{ .surface_metrics = .{
        .logical_size = .{ 1000, 700 },
        .framebuffer_extent = .{ .width = 1501, .height = 1051 },
        .scale = .{ 1.5, 1.5 },
        .generation = generation,
    } };
}

fn uiTranslator(generation: u32) lenore.UiEvents {
    var state: platform.InputState = .{};
    state.apply(uiMetrics(generation));
    return .init(state.metrics);
}

test "a pointer position crosses in framebuffer pixels" {
    var events = uiTranslator(1);

    const moved = events.translate(uiEvent(.{ .cursor = .{
        .logical_position = .{ 500, 350 },
        .metrics_generation = 1,
    } }));
    try testing.expectEqual(
        imui.Event{ .pointer_move = .{ .x = 750.5, .y = 525.5 } },
        moved.?,
    );

    const pressed = events.translate(uiEvent(.{ .mouse_button = .{
        .button = .right,
        .action = .press,
        .modifiers = .{ .shift = true },
        .logical_position = .{ 0, 700 },
        .metrics_generation = 1,
    } }));
    try testing.expectEqual(imui.Event{ .pointer_button = .{
        .position = .{ .x = 0, .y = 1051 },
        .button = .secondary,
        .action = .press,
        .shift = true,
    } }, pressed.?);
}

test "the surface configuration is folded where it arrives in the batch" {
    var events = uiTranslator(1);
    const moved = uiEvent(.{ .cursor = .{
        .logical_position = .{ 500, 350 },
        .metrics_generation = 2,
    } });

    // The window has resized and the event says so, but the metrics that
    // describe the new surface are still ahead of it in the batch.
    try testing.expectEqual(null, events.translate(moved));

    try testing.expectEqual(null, events.translate(uiEvent(uiMetrics(2))));
    try testing.expect(events.translate(moved) != null);
}

test "a transition that cannot be placed cancels the gesture and a motion does not" {
    var events = uiTranslator(4);

    // Both carry a generation the translator has not seen. The motion is
    // dropped, because the pointer keeps the last position it was known at;
    // the transition becomes a cancel, because the release it might be is the
    // one thing that must not go missing.
    try testing.expectEqual(null, events.translate(uiEvent(.{ .cursor = .{
        .logical_position = .{ 500, 350 },
        .metrics_generation = 9,
    } })));
    try testing.expectEqual(imui.Event.cancel, events.translate(uiEvent(.{ .mouse_button = .{
        .button = .left,
        .action = .release,
        .logical_position = .{ 500, 350 },
        .metrics_generation = 9,
    } })).?);
}

test "what the UI has no word for does not cross" {
    var events = uiTranslator(1);

    for ([_]platform.MouseButton{ .back, .forward, .other }) |button| {
        try testing.expectEqual(null, events.translate(uiEvent(.{ .mouse_button = .{
            .button = button,
            .action = .press,
            .logical_position = .{ 500, 350 },
            .metrics_generation = 1,
        } })));
    }
    // A held mouse button does not repeat: the action type is shared with the
    // keyboard, where it does.
    try testing.expectEqual(null, events.translate(uiEvent(.{ .mouse_button = .{
        .button = .left,
        .action = .repeat,
        .logical_position = .{ 500, 350 },
        .metrics_generation = 1,
    } })));

    try testing.expectEqual(null, events.translate(uiEvent(.{ .key = .{
        .physical = .q,
        .action = .press,
    } })));
    try testing.expectEqual(null, events.translate(uiEvent(.{ .scroll = .{
        .line_delta = .{ 0, 1 },
    } })));
    try testing.expectEqual(null, events.translate(uiEvent(.{ .text = .{
        .transaction = 1,
        .kind = .commit,
        .begin = true,
        .end = true,
        .len = 1,
        .bytes = [_]u8{'a'} ++ [_]u8{0} ** 15,
    } })));
}

test "the keys a widget acts on keep their action and their shift" {
    var events = uiTranslator(1);

    try testing.expectEqual(imui.Event{ .key = .{
        .key = .enter,
        .action = .repeat,
        .shift = false,
    } }, events.translate(uiEvent(.{ .key = .{
        .physical = .numpad_enter,
        .action = .repeat,
    } })).?);

    try testing.expectEqual(imui.Event{ .key = .{
        .key = .tab,
        .action = .press,
        .shift = true,
    } }, events.translate(uiEvent(.{ .key = .{
        .physical = .tab,
        .action = .press,
        .modifiers = .{ .shift = true },
    } })).?);

    try testing.expectEqual(
        imui.Event{ .focus = false },
        events.translate(uiEvent(.{ .focus = .{ .focused = false } })).?,
    );
}
