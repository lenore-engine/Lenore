const std = @import("std");
const gltf = @import("lenore-gltf");
const lenore = @import("lenore");
const res = @import("lenore-resources");

const testing = std.testing;

// Opening a level is an upload batch and a prepass registration, so what it
// does is verified by running it against a device. What can be checked here is
// the one refusal that happens before the device is reached, which is the
// point of it happening there.
test "a model with no materials is refused before the device is reached" {
    var vertices = [_]res.Vertex3D{.{
        .position = .{ 0, 0, 0 },
        .normal = .{ 0, 1, 0 },
        .uv = .{ 0, 0 },
        .tangent = .{ 1, 0, 0, 1 },
    }};
    var meshes = [_]gltf.importer.Mesh{.{
        .vertices = &vertices,
        .indices = &.{},
        .streams = .{},
        .morph = null,
        .material = 0,
        .anchor = null,
        .skin = null,
        .morph_template = null,
    }};
    var model: gltf.importer.Model = .{
        .meshes = &meshes,
        // Geometry naming a material the document does not have. Every mesh
        // carries an index into this, so there is nothing to shade with.
        .materials = &.{},
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = null,
        .morph_templates = &.{},
    };

    // Every dependency is undefined on purpose: the refusal precedes the clock,
    // the context and the caches, and that ordering is what is under test.
    const deps: lenore.LevelDeps = undefined;
    try testing.expectError(error.NoMaterials, lenore.Level.init(
        testing.allocator,
        deps,
        &model,
        .{ .capacity = .{ .instances = 8, .joints = 16 } },
    ));
}
