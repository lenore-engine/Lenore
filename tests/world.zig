const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const lenore = @import("lenore");
const res = @import("lenore-resources");
const zm = @import("zmath");

const testing = std.testing;

// Never dereferenced: the world carries resident meshes through to the record
// batches and does nothing else with them.
var mesh_a: gpu.Mesh = undefined;

const capacity: gpu.FrameCapacity = .{ .instances = 8, .joints = 16 };

fn vertexAt(position: res.Vec3) res.Vertex3D {
    return .{
        .position = position,
        .normal = .{ 0, 1, 0 },
        .uv = .{ 0, 0 },
        .tangent = .{ 1, 0, 0, 1 },
    };
}

// One mesh occupying a unit cube centred on `centre`, so its bounds and its
// sort key are both predictable.
fn meshAt(vertices: []res.Vertex3D, material_index: u32) gltf.importer.Mesh {
    return .{
        .vertices = vertices,
        .indices = &.{},
        .streams = .{},
        .morph = null,
        .material = material_index,
        .anchor = null,
        .skin = null,
        .morph_template = null,
    };
}

// Wide and deep enough to hold every fixture below, so a test about ordering is
// not also a test about culling. Left-handed because the fixtures sit along
// positive z and that is the direction this projection looks.
fn seesEverything() zm.Mat {
    return zm.orthographicLh(1.0e6, 1.0e6, 0.1, 1.0e6);
}

// The same view cut short of the `Ordered` fixture's farthest mesh, which sits
// between z 19 and 21.
fn seesNearHalf() zm.Mat {
    return zm.orthographicLh(4, 4, 0.1, 12);
}

fn material(mode: res.MaterialInfo.Rendering.AlphaMode) res.MaterialInfo {
    return .{
        .name = "",
        .textures = .{},
        .factors = .{},
        .rendering = .{ .alpha_mode = mode },
    };
}

test "an unanchored draw is placed by the model matrix alone" {
    const placement = zm.translation(3, 0, 0);
    const placed = lenore.instanceMatrix(null, null, placement);
    try testing.expectEqual(placement, placed);
}

test "an anchor applies before the placement, not after" {
    // Two transforms that do not commute, so a composition written the other
    // way round lands somewhere else rather than merely looking different.
    var animator: res.NodeAnimator = undefined;
    var transforms = [_]zm.Mat{zm.translation(10, 0, 0)};
    animator.world_transforms = &transforms;

    const placement = zm.rotationY(std.math.pi / 2.0);
    const placed = lenore.instanceMatrix(0, &animator, placement);

    // Row vectors: the anchor's translation is carried through the rotation.
    // Anchor then rotate sends (10, 0, 0) to about (0, 0, -10); rotate then
    // anchor would leave it at (10, 0, 0).
    const origin = lenore.placePoint(.{ 0, 0, 0 }, placed);
    try testing.expectApproxEqAbs(@as(f32, 0), origin[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -10), origin[2], 1e-5);
}

test "a point carried by a matrix picks up its translation" {
    const moved = lenore.placePoint(.{ 1, 2, 3 }, zm.translation(10, 20, 30));
    try testing.expectApproxEqAbs(@as(f32, 11), moved[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 22), moved[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 33), moved[2], 1e-5);
}

test "the world unions every mesh's bounds and frames a sphere on them" {
    const allocator = testing.allocator;

    var near = [_]res.Vertex3D{ vertexAt(.{ -1, -1, -1 }), vertexAt(.{ 1, 1, 1 }) };
    var far = [_]res.Vertex3D{ vertexAt(.{ 3, 3, 3 }), vertexAt(.{ 5, 5, 5 }) };
    var meshes = [_]gltf.importer.Mesh{ meshAt(&near, 0), meshAt(&far, 0) };
    var materials = [_]res.MaterialInfo{material(.@"opaque")};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = null,
        .morph_templates = &.{},
    };
    const resident = [_]*const gpu.Mesh{ &mesh_a, &mesh_a };
    const registrations = [_]?u32{ null, null };

    var world = try lenore.World.init(allocator, &model, &resident, &registrations, .{
        .capacity = capacity,
    });
    defer world.deinit();

    try testing.expectEqual(res.Vec3{ -1, -1, -1 }, world.bounds.min);
    try testing.expectEqual(res.Vec3{ 5, 5, 5 }, world.bounds.max);
    // The sphere is the box's own: centred between the corners, with the
    // half-diagonal for a radius. Asserted rather than merely tested for being
    // positive, because the camera is framed on it and the shadow map is fitted
    // to it, and any radius at all passes that weaker test.
    try testing.expectEqual(res.Vec3{ 2, 2, 2 }, world.sphere.centre);
    try testing.expectApproxEqAbs(@sqrt(27.0), world.sphere.radius, 1e-4);
    // Nothing is skinned, so no joint capacity is taken.
    try testing.expectEqual(@as(u32, 0), world.joint_total);
    try testing.expectEqual(false, world.castersMove());
}

test "an empty mesh does not read past its vertices" {
    const allocator = testing.allocator;

    // The old hand-written bounds fold read `vertices[0]` unconditionally. A
    // primitive with no positions is degenerate rather than fatal.
    var empty: [0]res.Vertex3D = .{};
    var meshes = [_]gltf.importer.Mesh{meshAt(&empty, 0)};
    var materials = [_]res.MaterialInfo{material(.@"opaque")};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = null,
        .morph_templates = &.{},
    };
    const resident = [_]*const gpu.Mesh{&mesh_a};
    const registrations = [_]?u32{null};

    var world = try lenore.World.init(allocator, &model, &resident, &registrations, .{
        .capacity = capacity,
    });
    defer world.deinit();

    try testing.expectEqual(res.Vec3{ 0, 0, 0 }, world.bounds.min);
    try testing.expectEqual(res.Vec3{ 0, 0, 0 }, world.bounds.max);
}

// The fixture the ordering tests share: three draws whose recording order is
// deliberately not their mesh order.
const Ordered = struct {
    world: lenore.World,
    meshes: [3]gltf.importer.Mesh,
    materials: [3]res.MaterialInfo,
    vertices: [3][2]res.Vertex3D,
    model: gltf.importer.Model,

    fn init(
        self: *Ordered,
        allocator: std.mem.Allocator,
        placement: zm.Mat,
        registrations: []const ?u32,
    ) !void {
        // Mesh 0 blends and sits near the eye, mesh 1 is solid, mesh 2 blends
        // and sits far. The recording order is therefore 1, 2, 0: solid first,
        // then the blended run farthest first. Nothing about that is the mesh
        // order, which is what makes a ring filled by mesh index visible.
        self.vertices = .{
            .{ vertexAt(.{ 0, 0, 4 }), vertexAt(.{ 0, 0, 6 }) },
            .{ vertexAt(.{ 0, 0, 19 }), vertexAt(.{ 0, 0, 21 }) },
            .{ vertexAt(.{ 0, 0, 9 }), vertexAt(.{ 0, 0, 11 }) },
        };
        self.materials = .{ material(.blend), material(.@"opaque"), material(.blend) };
        self.meshes = .{
            meshAt(&self.vertices[0], 0),
            meshAt(&self.vertices[1], 1),
            meshAt(&self.vertices[2], 2),
        };
        self.model = .{
            .meshes = &self.meshes,
            .materials = &self.materials,
            .images = &.{},
            .lights = &.{},
            .skins = &.{},
            .node_animation = null,
            .morph_templates = &.{},
        };

        const resident = [_]*const gpu.Mesh{ &mesh_a, &mesh_a, &mesh_a };
        self.world = try lenore.World.init(
            allocator,
            &self.model,
            &resident,
            registrations,
            .{ .capacity = capacity, .placement = placement },
        );
    }
};

test "the blended run follows the solid one and descends away from the eye" {
    var fixture: Ordered = undefined;
    try fixture.init(testing.allocator, zm.identity(), &.{ null, null, null });
    defer fixture.world.deinit();

    try fixture.world.plan.rebuild(fixture.world.meshes, null, zm.identity(), .{ 0, 0, 0 }, seesEverything());

    const ordered = fixture.world.plan.ordered;
    try testing.expectEqual(@as(usize, 3), ordered.len);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 0 }, ordered);
}

test "the instance ring is packed in draw order and not in mesh order" {
    var fixture: Ordered = undefined;
    try fixture.init(testing.allocator, zm.identity(), &.{ null, null, null });
    defer fixture.world.deinit();

    try fixture.world.plan.rebuild(fixture.world.meshes, null, zm.identity(), .{ 0, 0, 0 }, seesEverything());

    // Each mesh has its own material, so the material index is what says which
    // draw a slot of the ring belongs to. Packed by mesh index these would read
    // 0, 1, 2.
    const instances = fixture.world.plan.instances;
    try testing.expectEqual(@as(u32, 1), instances[0].material_index);
    try testing.expectEqual(@as(u32, 2), instances[1].material_index);
    try testing.expectEqual(@as(u32, 0), instances[2].material_index);

    // And the batches name the same materials in the same order, so the record
    // a draw reads and the pipeline state it is drawn with agree.
    const records = fixture.world.plan.records;
    try testing.expectEqual(@as(usize, 3), records.len);
    try testing.expectEqual(@as(u32, 1), records[0].material_index);
    try testing.expectEqual(@as(u32, 2), records[1].material_index);
    try testing.expectEqual(@as(u32, 0), records[2].material_index);
}

test "moving the eye past the model reverses the blended run" {
    var fixture: Ordered = undefined;
    try fixture.init(testing.allocator, zm.identity(), &.{ null, null, null });
    defer fixture.world.deinit();

    // From beyond the far draw, the near one is now the farthest. A plan built
    // once and never rebuilt keeps the previous answer.
    try fixture.world.plan.rebuild(fixture.world.meshes, null, zm.identity(), .{ 0, 0, 30 }, seesEverything());
    try testing.expectEqualSlices(u32, &.{ 1, 0, 2 }, fixture.world.plan.ordered);
}

test "the sort key is measured where the draw ends up, not where it was authored" {
    var fixture: Ordered = undefined;
    // The placement carries the whole model past the eye and reverses which
    // blended draw is farthest. Sorting on the authored centre gives 1, 2, 0
    // here; sorting on the placed one gives 1, 0, 2. With the identity
    // placement the two are the same slice and neither can be told apart.
    try fixture.init(testing.allocator, zm.translation(0, 0, -100), &.{ null, null, null });
    defer fixture.world.deinit();

    try fixture.world.plan.rebuild(
        fixture.world.meshes,
        null,
        fixture.world.placement,
        .{ 0, 0, 0 },
        seesEverything(),
    );
    try testing.expectEqualSlices(u32, &.{ 1, 0, 2 }, fixture.world.plan.ordered);
}

test "an instance record carries its draw's matrix" {
    var fixture: Ordered = undefined;
    const placement = zm.translation(0, 0, -100);
    try fixture.init(testing.allocator, placement, &.{ null, null, null });
    defer fixture.world.deinit();

    try fixture.world.plan.rebuild(fixture.world.meshes, null, placement, .{ 0, 0, 0 }, seesEverything());

    // Nothing is anchored, so every draw's matrix is the placement itself. A
    // record filled with the identity, or with a matrix belonging to another
    // draw, is only visible once the placement is not the identity.
    for (fixture.world.plan.instances) |instance| {
        try testing.expectEqual(placement, instance.model);
    }
}

test "a draw is sorted on its centre, not on the near face of its box" {
    const allocator = testing.allocator;

    // Two blended draws whose boxes order one way by centre and the other way
    // by their nearest corner. A wide box in front of a small one is exactly
    // the case the two disagree on, and equal-sized boxes never do.
    var wide = [_]res.Vertex3D{ vertexAt(.{ 0, 0, 0 }), vertexAt(.{ 0, 0, 10 }) };
    var narrow = [_]res.Vertex3D{ vertexAt(.{ 0, 0, 3 }), vertexAt(.{ 0, 0, 4 }) };
    var meshes = [_]gltf.importer.Mesh{ meshAt(&wide, 0), meshAt(&narrow, 1) };
    var materials = [_]res.MaterialInfo{ material(.blend), material(.blend) };
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = null,
        .morph_templates = &.{},
    };
    const resident = [_]*const gpu.Mesh{ &mesh_a, &mesh_a };
    const registrations = [_]?u32{ null, null };

    var world = try lenore.World.init(allocator, &model, &resident, &registrations, .{
        .capacity = capacity,
    });
    defer world.deinit();

    try world.plan.rebuild(world.meshes, null, zm.identity(), .{ 0, 0, 0 }, seesEverything());
    // By centre the wide box sits at 5 and the narrow one at 3.5, so the wide
    // one is farther and is drawn first. By nearest corner it would be 0
    // against 3, and the order would be the other way round.
    try testing.expectEqualSlices(u32, &.{ 0, 1 }, world.plan.ordered);
}

test "a mesh naming a material the model does not have is refused" {
    var vertices = [_]res.Vertex3D{ vertexAt(.{ 0, 0, 0 }), vertexAt(.{ 1, 1, 1 }) };
    var meshes = [_]gltf.importer.Mesh{meshAt(&vertices, 3)};
    var materials = [_]res.MaterialInfo{material(.@"opaque")};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = null,
        .morph_templates = &.{},
    };
    const resident = [_]*const gpu.Mesh{&mesh_a};
    const registrations = [_]?u32{null};

    try testing.expectError(error.MaterialIndexOutOfRange, lenore.World.init(
        testing.allocator,
        &model,
        &resident,
        &registrations,
        .{ .capacity = capacity },
    ));

    // Exactly one past the last material, which is the only index that tells
    // `>=` apart from `>`. The one above is far enough out that both refuse it.
    meshes[0].material = 1;
    try testing.expectError(error.MaterialIndexOutOfRange, lenore.World.init(
        testing.allocator,
        &model,
        &resident,
        &registrations,
        .{ .capacity = capacity },
    ));
}

test "an anchored mesh with no animator to resolve it is refused" {
    var vertices = [_]res.Vertex3D{ vertexAt(.{ 0, 0, 0 }), vertexAt(.{ 1, 1, 1 }) };
    var meshes = [_]gltf.importer.Mesh{meshAt(&vertices, 0)};
    meshes[0].anchor = 0;
    var materials = [_]res.MaterialInfo{material(.@"opaque")};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = null,
        .morph_templates = &.{},
    };
    const resident = [_]*const gpu.Mesh{&mesh_a};
    const registrations = [_]?u32{null};

    try testing.expectError(error.AnchoredMeshWithoutAnimator, lenore.World.init(
        testing.allocator,
        &model,
        &resident,
        &registrations,
        .{ .capacity = capacity },
    ));
}

test "more draws than the frame's instance ring holds is refused" {
    var vertices = [_]res.Vertex3D{ vertexAt(.{ 0, 0, 0 }), vertexAt(.{ 1, 1, 1 }) };
    var meshes: [3]gltf.importer.Mesh = @splat(meshAt(&vertices, 0));
    var materials = [_]res.MaterialInfo{material(.@"opaque")};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = null,
        .morph_templates = &.{},
    };
    const resident = [_]*const gpu.Mesh{ &mesh_a, &mesh_a, &mesh_a };
    const registrations = [_]?u32{ null, null, null };

    try testing.expectError(error.InstanceCapacityExceeded, lenore.World.init(
        testing.allocator,
        &model,
        &resident,
        &registrations,
        // Two slots for three draws. One less than the count, not zero, so a
        // check written as `> 0` would not catch it either.
        .{ .capacity = .{ .instances = 2, .joints = 16 } },
    ));
}

test "a resident mesh list of the wrong length is refused" {
    var vertices = [_]res.Vertex3D{ vertexAt(.{ 0, 0, 0 }), vertexAt(.{ 1, 1, 1 }) };
    var meshes = [_]gltf.importer.Mesh{ meshAt(&vertices, 0), meshAt(&vertices, 0) };
    var materials = [_]res.MaterialInfo{material(.@"opaque")};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = null,
        .morph_templates = &.{},
    };
    const short = [_]*const gpu.Mesh{&mesh_a};
    const registrations = [_]?u32{ null, null };

    try testing.expectError(error.MeshCountMismatch, lenore.World.init(
        testing.allocator,
        &model,
        &short,
        &registrations,
        .{ .capacity = capacity },
    ));
}

// A skinned draw and a morphed template, which the fixtures above have neither
// of. Without one the whole skin path of `World.init` is uncovered rather than
// weakly covered: the joint bound, the empty skin, and the teardown counters
// that only matter when init fails part way.
const Skinned = struct {
    skeleton: res.SkeletonTemplate,
    skins: [1]gltf.importer.Skin,
    vertices: [2]res.Vertex3D,
    meshes: [1]gltf.importer.Mesh,
    materials: [1]res.MaterialInfo,
    morph_templates: [1]res.MorphTemplate,
    model: gltf.importer.Model,

    const Shape = struct {
        // The joint index every vertex carries. Two joints exist, so 2 is the
        // first that is out of range.
        joint: u16 = 0,
        // A skeleton with no joints at all, which is a skin that cannot pose
        // anything.
        jointless: bool = false,
    };

    fn build(self: *Skinned, allocator: std.mem.Allocator, shape: Shape) !void {
        const joint_slot = [_]u16{ 0, 1 };
        const inverse_bind = [_]zm.Mat{ zm.identity(), zm.identity() };
        self.skeleton = try .init(allocator, .{
            .slot_parent = &.{ res.no_parent, 0 },
            .slot_prefix = &.{ zm.identity(), zm.identity() },
            .bind_translations = &.{ zm.f32x4(0, 0, 0, 1), zm.f32x4(0, 1, 0, 1) },
            .bind_rotations = &.{ zm.f32x4(0, 0, 0, 1), zm.f32x4(0, 0, 0, 1) },
            .bind_scales = &.{ zm.f32x4s(1), zm.f32x4s(1) },
            .inverse_bind = if (shape.jointless) &.{} else &inverse_bind,
            .joint_slot = if (shape.jointless) &.{} else &joint_slot,
        });
        errdefer self.skeleton.deinit(allocator);

        self.skins = .{.{ .skeleton = self.skeleton, .clips = &.{}, .index = 4 }};
        self.vertices = .{ vertexAt(.{ 0, 0, 0 }), vertexAt(.{ 1, 1, 1 }) };
        for (&self.vertices) |*vertex| {
            vertex.joints = @splat(shape.joint);
            vertex.weights = .{ 1, 0, 0, 0 };
        }

        self.meshes = .{meshAt(&self.vertices, 0)};
        self.meshes[0].streams.skinned = true;
        // The document's skin index, which is deliberately not the position of
        // the skin in the imported slice.
        self.meshes[0].skin = 4;
        self.materials = .{material(.@"opaque")};

        const defaults = try allocator.alloc(f32, 2);
        defaults[0] = 0;
        defaults[1] = 0;
        self.morph_templates = .{try .init(allocator, .{ .defaults = defaults, .clips = &.{} })};

        self.model = .{
            .meshes = &self.meshes,
            .materials = &self.materials,
            .images = &.{},
            .lights = &.{},
            .skins = &self.skins,
            .node_animation = null,
            .morph_templates = &self.morph_templates,
        };
    }

    fn deinit(self: *Skinned, allocator: std.mem.Allocator) void {
        self.morph_templates[0].deinit(allocator);
        self.skeleton.deinit(allocator);
    }

    fn open(self: *Skinned, allocator: std.mem.Allocator) !lenore.World {
        const resident = [_]*const gpu.Mesh{&mesh_a};
        const registrations = [_]?u32{null};
        return lenore.World.init(allocator, &self.model, &resident, &registrations, .{
            .capacity = capacity,
        });
    }
};

test "a skinned draw takes joint slots and is found by its document index" {
    const allocator = testing.allocator;
    var fixture: Skinned = undefined;
    try fixture.build(allocator, .{});
    defer fixture.deinit(allocator);

    var world = try fixture.open(allocator);
    defer world.deinit();

    try testing.expectEqual(@as(u32, 2), world.joint_total);
    try testing.expectEqual(@as(?u32, 0), world.skin_of_mesh[0]);
    try testing.expectEqual(@as(u32, 4), world.skins[0].source_index);
    // A morph template exists, so something in this world can move.
    try testing.expectEqual(true, world.castersMove());
}

test "a joint index one past the skin's last is refused" {
    const allocator = testing.allocator;

    // Exactly at the bound. The shader has no check of its own, so this is the
    // only thing between an asset and a read past the joint array.
    var over: Skinned = undefined;
    try over.build(allocator, .{ .joint = 2 });
    defer over.deinit(allocator);
    try testing.expectError(error.JointIndexOutOfRange, over.open(allocator));

    // And the last valid one is accepted, so the check is not simply refusing
    // everything.
    var last: Skinned = undefined;
    try last.build(allocator, .{ .joint = 1 });
    defer last.deinit(allocator);
    var world = try last.open(allocator);
    world.deinit();
}

test "a skin with no joints is refused" {
    const allocator = testing.allocator;
    var fixture: Skinned = undefined;
    try fixture.build(allocator, .{ .jointless = true });
    defer fixture.deinit(allocator);

    try testing.expectError(error.EmptySkin, fixture.open(allocator));
}

test "a skinned mesh naming no skin is refused" {
    const allocator = testing.allocator;
    var fixture: Skinned = undefined;
    try fixture.build(allocator, .{});
    defer fixture.deinit(allocator);
    fixture.meshes[0].skin = null;

    try testing.expectError(error.SkinnedMeshWithoutSkin, fixture.open(allocator));
}

test "a skinned mesh naming a skin the document does not have is refused" {
    const allocator = testing.allocator;
    var fixture: Skinned = undefined;
    try fixture.build(allocator, .{});
    defer fixture.deinit(allocator);
    fixture.meshes[0].skin = 9;

    try testing.expectError(error.SkinnedMeshWithoutSkin, fixture.open(allocator));
}

test "a failure at any point during init releases everything built before it" {
    const allocator = testing.allocator;
    var fixture: Skinned = undefined;
    try fixture.build(allocator, .{});
    defer fixture.deinit(allocator);

    // Every allocation in turn. What this catches is teardown reading the wrong
    // count: a pose built and not released leaks, and the testing allocator is
    // what reports it. The bound is above the number of allocations init makes,
    // so the last rounds all succeed.
    var fail_at: usize = 0;
    while (fail_at < 32) : (fail_at += 1) {
        var failing: std.testing.FailingAllocator = .init(allocator, .{ .fail_index = fail_at });
        if (fixture.open(failing.allocator())) |built| {
            var world = built;
            world.deinit();
        } else |err| {
            try testing.expectEqual(error.OutOfMemory, err);
        }
    }
}

// A skeleton of `joints` independent slots, one joint each.
fn skeletonWith(allocator: std.mem.Allocator, joints: usize) !res.SkeletonTemplate {
    const parents = try allocator.alloc(res.Slot, joints);
    defer allocator.free(parents);
    const prefixes = try allocator.alloc(zm.Mat, joints);
    defer allocator.free(prefixes);
    const translations = try allocator.alloc(zm.Vec, joints);
    defer allocator.free(translations);
    const rotations = try allocator.alloc(zm.Quat, joints);
    defer allocator.free(rotations);
    const scales = try allocator.alloc(zm.Vec, joints);
    defer allocator.free(scales);
    const inverse_bind = try allocator.alloc(zm.Mat, joints);
    defer allocator.free(inverse_bind);
    const joint_slot = try allocator.alloc(u16, joints);
    defer allocator.free(joint_slot);

    for (0..joints) |index| {
        parents[index] = res.no_parent;
        prefixes[index] = zm.identity();
        translations[index] = zm.f32x4(0, 0, 0, 1);
        rotations[index] = zm.f32x4(0, 0, 0, 1);
        scales[index] = zm.f32x4s(1);
        inverse_bind[index] = zm.identity();
        joint_slot[index] = @intCast(index);
    }
    return res.SkeletonTemplate.init(allocator, .{
        .slot_parent = parents,
        .slot_prefix = prefixes,
        .bind_translations = translations,
        .bind_rotations = rotations,
        .bind_scales = scales,
        .inverse_bind = inverse_bind,
        .joint_slot = joint_slot,
    });
}

test "each skin's joints land in the run the offsets gave it" {
    const allocator = testing.allocator;

    // Two skins of different sizes, so the second one's run does not start at
    // zero. With a single skin every offset is zero and the packing cannot be
    // told from writing everything to the front.
    var small = try skeletonWith(allocator, 2);
    defer small.deinit(allocator);
    var large = try skeletonWith(allocator, 3);
    defer large.deinit(allocator);

    var skins = [_]gltf.importer.Skin{
        .{ .skeleton = small, .clips = &.{}, .index = 10 },
        .{ .skeleton = large, .clips = &.{}, .index = 20 },
    };
    var vertices = [_]res.Vertex3D{ vertexAt(.{ 0, 0, 0 }), vertexAt(.{ 1, 1, 1 }) };
    var meshes = [_]gltf.importer.Mesh{ meshAt(&vertices, 0), meshAt(&vertices, 0) };
    for (&meshes, [_]u32{ 10, 20 }) |*mesh, source| {
        mesh.streams.skinned = true;
        mesh.skin = source;
    }
    var materials = [_]res.MaterialInfo{material(.@"opaque")};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &skins,
        .node_animation = null,
        .morph_templates = &.{},
    };
    const resident = [_]*const gpu.Mesh{ &mesh_a, &mesh_a };
    const registrations = [_]?u32{ null, null };

    var world = try lenore.World.init(allocator, &model, &resident, &registrations, .{
        .capacity = capacity,
    });
    defer world.deinit();

    try testing.expectEqual(@as(u32, 5), world.joint_total);
    try testing.expectEqual(@as(u32, 0), world.plan.joint_bases[0]);
    try testing.expectEqual(@as(u32, 2), world.plan.joint_bases[1]);

    // A value per joint that says which skin and which slot it came from, so a
    // run written to the wrong offset is identifiable rather than merely
    // different.
    for (world.skins, 0..) |*skin, skin_index| {
        for (skin.animator.pose.joint_transforms, 0..) |*joint, slot| {
            joint.* = zm.translation(@floatFromInt(skin_index + 1), @floatFromInt(slot), 0);
        }
    }
    // Poison, so a slot nothing writes is not mistaken for a slot written
    // correctly.
    for (world.joint_storage) |*joint| joint.* = zm.translation(-1, -1, -1);

    world.packJoints();

    for (0..2) |slot| {
        try testing.expectEqual(
            zm.translation(1, @floatFromInt(slot), 0),
            world.joint_storage[slot],
        );
    }
    for (0..3) |slot| {
        try testing.expectEqual(
            zm.translation(2, @floatFromInt(slot), 0),
            world.joint_storage[2 + slot],
        );
    }
}

// A rigid hierarchy of one slot, offset from the origin, with no clips. It is
// enough to resolve an anchor, which is what geometry needs to reach the world.
fn nodeTemplateAt(allocator: std.mem.Allocator, offset: f32) !res.NodeTemplate {
    const parent = try allocator.alloc(res.Slot, 1);
    parent[0] = res.no_parent;
    const prefix = try allocator.alloc(zm.Mat, 1);
    prefix[0] = zm.identity();
    const translations = try allocator.alloc(zm.Vec, 1);
    translations[0] = zm.f32x4(offset, 0, 0, 1);
    const rotations = try allocator.alloc(zm.Quat, 1);
    rotations[0] = zm.f32x4(0, 0, 0, 1);
    const scales = try allocator.alloc(zm.Vec, 1);
    scales[0] = zm.f32x4s(1);
    const locals = try allocator.alloc(zm.Mat, 1);
    locals[0] = zm.translation(offset, 0, 0);

    return res.NodeTemplate.init(allocator, .{
        .parent = parent,
        .prefix = prefix,
        .bind_translations = translations,
        .bind_rotations = rotations,
        .bind_scales = scales,
        .bind_local = locals,
        .clips = &.{},
        .animated_slots = &.{},
    });
}

test "an anchored draw is measured where its slot puts it" {
    const allocator = testing.allocator;

    var template = try nodeTemplateAt(allocator, 100);
    defer template.deinit(allocator);

    // The vertices are baked around the origin in the anchor's space. Unioning
    // them directly would report a box at the origin; through the anchor they
    // are a hundred units away, which is where the asset puts them.
    var vertices = [_]res.Vertex3D{ vertexAt(.{ -1, 0, 0 }), vertexAt(.{ 1, 0, 0 }) };
    var meshes = [_]gltf.importer.Mesh{meshAt(&vertices, 0)};
    meshes[0].anchor = 0;
    var materials = [_]res.MaterialInfo{material(.@"opaque")};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = &template,
        .morph_templates = &.{},
    };
    const resident = [_]*const gpu.Mesh{&mesh_a};
    const registrations = [_]?u32{null};

    var world = try lenore.World.init(allocator, &model, &resident, &registrations, .{
        .capacity = capacity,
    });
    defer world.deinit();

    try testing.expectApproxEqAbs(@as(f32, 99), world.bounds.min[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 101), world.bounds.max[0], 1e-4);
    // A rigid animator exists, so the shadow map cannot be baked once and
    // reused even though this template has no clip to play.
    try testing.expectEqual(true, world.castersMove());
}

test "a skin that has a clip is playing one when the world opens" {
    const allocator = testing.allocator;

    var skeleton = try skeletonWith(allocator, 2);
    defer skeleton.deinit(allocator);

    const keys = try allocator.alloc(res.Keyframe(zm.Vec), 2);
    keys[0] = .{ .time = 0.0, .value = zm.f32x4(0, 0, 0, 0) };
    keys[1] = .{ .time = 2.0, .value = zm.f32x4(10, 0, 0, 0) };
    const channels = try allocator.alloc(res.AnimationChannel, 1);
    channels[0] = .{ .track = .{ .translation = .{ .keys = keys } }, .target_slot = 0 };
    var clips = [_]res.Animation{try .init(allocator, channels, "slide")};
    defer clips[0].deinit(allocator);

    var skins = [_]gltf.importer.Skin{.{ .skeleton = skeleton, .clips = &clips, .index = 0 }};
    var vertices = [_]res.Vertex3D{ vertexAt(.{ 0, 0, 0 }), vertexAt(.{ 1, 1, 1 }) };
    var meshes = [_]gltf.importer.Mesh{meshAt(&vertices, 0)};
    meshes[0].streams.skinned = true;
    meshes[0].skin = 0;
    var materials = [_]res.MaterialInfo{material(.@"opaque")};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &skins,
        .node_animation = null,
        .morph_templates = &.{},
    };
    const resident = [_]*const gpu.Mesh{&mesh_a};
    const registrations = [_]?u32{null};

    var world = try lenore.World.init(allocator, &model, &resident, &registrations, .{
        .capacity = capacity,
    });
    defer world.deinit();

    // A skin left unstarted holds its bind pose forever, which looks exactly
    // like a document with no animation in it.
    try testing.expectEqual(@as(?u16, 0), world.skins[0].animator.active_clip);
    // Nothing else in this model can move, so the skin is the only thing that
    // can make this true.
    try testing.expectEqual(true, world.castersMove());
    try testing.expectEqual(@as(usize, 1), world.clipCount());
    try testing.expectEqual(@as(?u16, 0), world.activeClip());
}

test "a world opened with no clip holds the bind pose" {
    const allocator = testing.allocator;

    var template = try nodeTemplateAt(allocator, 0);
    defer template.deinit(allocator);

    var vertices = [_]res.Vertex3D{ vertexAt(.{ 0, 0, 0 }), vertexAt(.{ 1, 1, 1 }) };
    var meshes = [_]gltf.importer.Mesh{meshAt(&vertices, 0)};
    var materials = [_]res.MaterialInfo{material(.@"opaque")};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = &template,
        .morph_templates = &.{},
    };
    const resident = [_]*const gpu.Mesh{&mesh_a};
    const registrations = [_]?u32{null};

    var world = try lenore.World.init(allocator, &model, &resident, &registrations, .{
        .capacity = capacity,
        .clip = null,
    });
    defer world.deinit();

    try testing.expectEqual(@as(?u16, null), world.activeClip());
}

test "switching a clip moves the skin and the rigid hierarchy together" {
    const allocator = testing.allocator;

    // Two document animations, present in both clip lists because each list
    // holds one entry per animation in the document. Switching one animator
    // and not the other is the state this replaces.
    var skeleton = try skeletonWith(allocator, 2);
    defer skeleton.deinit(allocator);

    var clips: [2]res.Animation = undefined;
    for (&clips, 0..) |*clip, index| {
        const keys = try allocator.alloc(res.Keyframe(zm.Vec), 2);
        keys[0] = .{ .time = 0.0, .value = zm.f32x4(0, 0, 0, 0) };
        keys[1] = .{ .time = 2.0, .value = zm.f32x4(@floatFromInt(index + 1), 0, 0, 0) };
        const channels = try allocator.alloc(res.AnimationChannel, 1);
        channels[0] = .{ .track = .{ .translation = .{ .keys = keys } }, .target_slot = 0 };
        clip.* = try .init(allocator, channels, "clip");
    }
    defer for (&clips) |*clip| clip.deinit(allocator);

    var skins = [_]gltf.importer.Skin{.{ .skeleton = skeleton, .clips = &clips, .index = 0 }};
    var vertices = [_]res.Vertex3D{ vertexAt(.{ 0, 0, 0 }), vertexAt(.{ 1, 1, 1 }) };
    var meshes = [_]gltf.importer.Mesh{meshAt(&vertices, 0)};
    meshes[0].streams.skinned = true;
    meshes[0].skin = 0;
    var materials = [_]res.MaterialInfo{material(.@"opaque")};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &skins,
        .node_animation = null,
        .morph_templates = &.{},
    };
    const resident = [_]*const gpu.Mesh{&mesh_a};
    const registrations = [_]?u32{null};

    var world = try lenore.World.init(allocator, &model, &resident, &registrations, .{
        .capacity = capacity,
        // Not zero, so a world that ignored the option and started the first
        // clip regardless would be visible here.
        .clip = 1,
    });
    defer world.deinit();

    try testing.expectEqual(@as(usize, 2), world.clipCount());
    try testing.expectEqual(@as(?u16, 1), world.activeClip());

    try testing.expectEqual(@as(usize, 1), try world.playClip(0));
    try testing.expectEqual(@as(?u16, 0), world.skins[0].animator.active_clip);

    // An index past what the document holds starts nothing rather than
    // failing: a caller cycling clips should not have to know the count twice.
    try testing.expectEqual(@as(usize, 0), try world.playClip(5));
}

test "a registration list of the wrong length is refused" {
    var vertices = [_]res.Vertex3D{ vertexAt(.{ 0, 0, 0 }), vertexAt(.{ 1, 1, 1 }) };
    var meshes = [_]gltf.importer.Mesh{ meshAt(&vertices, 0), meshAt(&vertices, 0) };
    var materials = [_]res.MaterialInfo{material(.@"opaque")};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        .materials = &materials,
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = null,
        .morph_templates = &.{},
    };
    const resident = [_]*const gpu.Mesh{ &mesh_a, &mesh_a };
    // The right number of meshes and the wrong number of registrations, so the
    // two length checks cannot stand in for one another.
    const registrations = [_]?u32{null};

    try testing.expectError(error.MeshCountMismatch, lenore.World.init(
        testing.allocator,
        &model,
        &resident,
        &registrations,
        .{ .capacity = capacity },
    ));
}

test "a model with no geometry is refused" {
    var model: gltf.importer.Model = .{
        .meshes = &.{},
        .materials = &.{},
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = null,
        .morph_templates = &.{},
    };

    try testing.expectError(error.NoGeometry, lenore.World.init(
        testing.allocator,
        &model,
        &.{},
        &.{},
        .{ .capacity = capacity },
    ));
}

test "a mesh outside the frustum is ordered behind the visible ones" {
    var fixture: Ordered = undefined;
    try fixture.init(testing.allocator, zm.identity(), &.{ null, null, null });
    defer fixture.world.deinit();

    // The solid mesh is the far one, so what survives is the two blended ones
    // and the visible half opens with no solid batch at all.
    try fixture.world.plan.rebuild(
        fixture.world.meshes,
        null,
        zm.identity(),
        .{ 0, 0, 0 },
        seesNearHalf(),
    );

    // Farthest of the two visible first, then the culled solid behind them.
    try testing.expectEqualSlices(u32, &.{ 2, 0, 1 }, fixture.world.plan.ordered);
    // Every mesh names its own material, so each is a batch of its own and the
    // count is two rather than the two draws it happens to equal here.
    try testing.expectEqual(@as(usize, 2), fixture.world.plan.visible_records);
    try testing.expectEqual(@as(usize, 3), fixture.world.plan.records.len);
}

test "a morphed mesh is not culled" {
    // A registration is what says the morph pass writes this mesh, and a morphed
    // mesh's bounds describe its base shape while its vertices have left it.
    var fixture: Ordered = undefined;
    try fixture.init(testing.allocator, zm.identity(), &.{ null, 1, null });
    defer fixture.world.deinit();

    try fixture.world.plan.rebuild(
        fixture.world.meshes,
        null,
        zm.identity(),
        .{ 0, 0, 0 },
        seesNearHalf(),
    );

    // The same view that culled mesh 1 in the test above leaves it visible, and
    // it is solid, so it leads the list.
    try testing.expectEqualSlices(u32, &.{ 1, 2, 0 }, fixture.world.plan.ordered);
    try testing.expectEqual(@as(usize, 3), fixture.world.plan.visible_records);
}

// One joint at the origin, which is all a skinned mesh needs to exist. The pose
// is never advanced here: what is under test is that a skinned mesh is exempt
// from culling, not where skinning puts it.
const OneJoint = struct {
    skins: [1]gltf.importer.Skin,
    meshes: [1]gltf.importer.Mesh,
    materials: [1]res.MaterialInfo,
    vertices: [2]res.Vertex3D,
    model: gltf.importer.Model,
    world: lenore.World,

    fn init(self: *OneJoint, allocator: std.mem.Allocator) !void {
        const template: res.SkeletonTemplate = try .init(allocator, .{
            .slot_parent = &.{res.no_parent},
            .slot_prefix = &.{zm.identity()},
            .bind_translations = &.{zm.f32x4(0, 0, 0, 1)},
            .bind_rotations = &.{zm.qidentity()},
            .bind_scales = &.{zm.f32x4(1, 1, 1, 0)},
            .inverse_bind = &.{zm.identity()},
            .joint_slot = &.{0},
        });
        self.skins = .{.{ .skeleton = template, .clips = &.{}, .index = 0 }};

        // Far beyond the near view, which is what makes the exemption visible.
        self.vertices = .{ vertexAt(.{ 0, 0, 40 }), vertexAt(.{ 0, 0, 42 }) };
        self.materials = .{material(.@"opaque")};
        self.meshes = .{meshAt(&self.vertices, 0)};
        self.meshes[0].streams = .{ .skinned = true };
        self.meshes[0].skin = 0;
        self.model = .{
            .meshes = &self.meshes,
            .materials = &self.materials,
            .images = &.{},
            .lights = &.{},
            .skins = &self.skins,
            .node_animation = null,
            .morph_templates = &.{},
        };

        const resident = [_]*const gpu.Mesh{&mesh_a};
        self.world = try lenore.World.init(
            allocator,
            &self.model,
            &resident,
            &.{null},
            .{ .capacity = capacity, .placement = zm.identity() },
        );
    }

    // The world owns the animator it built from the template; the template
    // itself belongs to the model, which here is this fixture.
    fn deinit(self: *OneJoint, allocator: std.mem.Allocator) void {
        self.world.deinit();
        self.skins[0].skeleton.deinit(allocator);
    }
};

test "a skinned mesh is not culled" {
    // The bounds a skinned mesh carries are its bind pose and its vertices are
    // wherever the joints put them, so a frustum test on those bounds can only
    // be wrong. A false negative here is a limb that vanishes.
    var fixture: OneJoint = undefined;
    try fixture.init(testing.allocator);
    defer fixture.deinit(testing.allocator);

    try fixture.world.plan.rebuild(
        fixture.world.meshes,
        null,
        zm.identity(),
        .{ 0, 0, 0 },
        seesNearHalf(),
    );

    try testing.expectEqual(@as(usize, 1), fixture.world.plan.visible_records);
    try testing.expect(!fixture.world.plan.cullable[0]);
}

test "the empty world is torn down by the same teardown as a built one" {
    // The allocator is the check: `deinit` frees fourteen slices and releases
    // every animator, and this hands it a value where all of them are empty. A
    // field left `undefined` by the constructor, or one teardown frees twice,
    // is a leak or a double free reported here rather than a crash in an
    // application that drew no meshes.
    var world: lenore.World = .empty(testing.allocator);
    world.deinit();
}

test "the empty world plans no draws and moves no casters" {
    var world: lenore.World = .empty(testing.allocator);
    defer world.deinit();

    // Nothing animates, so a frame drawing this asks for no shadow bake after
    // the first. That is the whole of what the loop reads out of a world it was
    // given only to satisfy its own contract.
    try testing.expect(!world.castersMove());

    // `rebuild` over no meshes rather than a branch that skips it: the record
    // path a fullscreen application takes is the same one every other frame
    // takes, and this is where that stops being an assumption.
    try world.plan.rebuild(world.meshes, null, zm.identity(), .{ 0, 0, 0 }, zm.identity());
    try testing.expectEqual(@as(usize, 0), world.plan.records.len);
    try testing.expectEqual(@as(usize, 0), world.plan.visible_records);
}
