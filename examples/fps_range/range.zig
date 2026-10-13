const std = @import("std");
const fps = @import("lenore-fps");
const gltf = @import("lenore-gltf");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");
const zm = @import("zmath");

const Allocator = std.mem.Allocator;

// The range: a floor, a partition, a steel plate and targets, every one of
// them a box. The box is what is drawn, what a shot crosses and what stops a
// body walking, from the one list below, so a round lands on the surface that
// can be seen and a body stops at the edge it walks into.
//
// Down range is -Z, which is where a camera at yaw -pi/2 looks.

pub const Kind = enum {
    floor,
    drywall,
    target,
    steel,

    // How the kind is drawn and what a round pays to cross it. The costs are
    // in the round's units, chosen so the ball round below gets through a
    // partition and a torso and is stopped by the plate and the floor.
    pub fn material(self: Kind) fps.Material {
        return switch (self) {
            .floor => .{ .per_metre = 400.0 },
            .drywall => .{ .per_metre = 20.0 },
            .target => .{ .per_metre = 10.0 },
            .steel => .{ .entry = 12.0, .per_metre = 400.0 },
        };
    }

    fn factors(self: Kind) res.MaterialInfo.Factors {
        return switch (self) {
            .floor => .{ .base_colour = .{ 0.30, 0.30, 0.29, 1.0 }, .metallic = 0.0, .roughness = 0.9 },
            .drywall => .{ .base_colour = .{ 0.75, 0.74, 0.70, 1.0 }, .metallic = 0.0, .roughness = 0.85 },
            .target => .{ .base_colour = .{ 0.55, 0.40, 0.22, 1.0 }, .metallic = 0.0, .roughness = 0.8 },
            .steel => .{ .base_colour = .{ 0.35, 0.36, 0.38, 1.0 }, .metallic = 1.0, .roughness = 0.45 },
        };
    }
};

// Which part of a target a box is, for the damage a hit does.
pub const Zone = enum {
    none,
    head,
    torso,

    pub fn multiplier(self: Zone) f32 {
        return switch (self) {
            .none => 0.0,
            .head => 4.0,
            .torso => 1.0,
        };
    }
};

pub const Prop = struct {
    name: []const u8,
    kind: Kind,
    zone: Zone = .none,
    // Metres: the box's centre and its whole extent on each axis.
    centre: [3]f32,
    size: [3]f32,
    // Whether a body walking into it stops. The floor is under the feet, not
    // in their way.
    solid: bool = true,

    // The unit box below, scaled and then moved: the transform a crossing is
    // taken through.
    pub fn model(self: Prop) zm.Mat {
        return zm.mul(
            zm.scaling(self.size[0], self.size[1], self.size[2]),
            zm.translation(self.centre[0], self.centre[1], self.centre[2]),
        );
    }
};

pub const unit_box: res.Aabb = .{ .min = .{ -0.5, -0.5, -0.5 }, .max = .{ 0.5, 0.5, 0.5 } };

// A target is a torso with a head on it, as two boxes, so a hit names the
// zone it struck by which box it crossed.
fn torso(comptime name: []const u8, x: f32, z: f32) Prop {
    return .{ .name = name ++ " torso", .kind = .target, .zone = .torso, .centre = .{ x, 1.30, z }, .size = .{ 0.45, 0.60, 0.25 } };
}

fn head(comptime name: []const u8, x: f32, z: f32) Prop {
    return .{ .name = name ++ " head", .kind = .target, .zone = .head, .centre = .{ x, 1.76, z }, .size = .{ 0.20, 0.25, 0.22 } };
}

pub const props = [_]Prop{
    .{ .name = "floor", .kind = .floor, .centre = .{ 0.0, -0.05, -20.0 }, .size = .{ 30.0, 0.1, 60.0 }, .solid = false },

    torso("5 m", 0.0, -5.0),
    head("5 m", 0.0, -5.0),
    torso("10 m", 1.5, -10.0),
    head("10 m", 1.5, -10.0),
    torso("25 m", -1.0, -25.0),
    head("25 m", -1.0, -25.0),

    // Behind a partition: a round that gets through the wall still reaches
    // it, with less left.
    .{ .name = "partition", .kind = .drywall, .centre = .{ -5.0, 1.25, -12.0 }, .size = .{ 4.0, 2.5, 0.1 } },
    torso("behind the partition", -5.0, -14.0),
    head("behind the partition", -5.0, -14.0),

    // Behind a plate: nothing reaches it.
    .{ .name = "plate", .kind = .steel, .centre = .{ 5.0, 1.30, -9.0 }, .size = .{ 0.6, 0.8, 0.01 } },
    torso("behind the plate", 5.0, -10.0),
    head("behind the plate", 5.0, -10.0),
};

// Where the eye starts: at the near end of the floor, facing down range.
pub const spawn: [2]f32 = .{ 0.0, 3.0 };

// Every prop as geometry, one mesh per kind so the kinds can be drawn with
// their own materials. The engine never opens a file for this: `Level.init`
// takes a model, and a generated one goes the same way a loaded one does.
pub fn build(allocator: Allocator) Allocator.Error!gltf.importer.Model {
    const kinds = @typeInfo(Kind).@"enum".field_names.len;

    const meshes = try allocator.alloc(gltf.importer.Mesh, kinds);
    var built: usize = 0;
    errdefer {
        for (meshes[0..built]) |*mesh| mesh.deinit(allocator);
        allocator.free(meshes);
    }
    for (0..kinds) |index| {
        meshes[index] = try meshOf(allocator, @fromBackingInt(@as(u2, @intCast(index))));
        built += 1;
    }

    const materials = try allocator.alloc(res.MaterialInfo, kinds);
    var named: usize = 0;
    errdefer {
        for (materials[0..named]) |*material| material.deinit(allocator);
        allocator.free(materials);
    }
    for (0..kinds) |index| {
        const kind: Kind = @fromBackingInt(@as(u2, @intCast(index)));
        materials[index] = .{
            .name = try allocator.dupe(u8, @tagName(kind)),
            .textures = .{},
            .factors = kind.factors(),
            .rendering = .{},
        };
        named += 1;
    }

    return .{
        .meshes = meshes,
        .materials = materials,
        .images = &.{},
        .lights = &.{},
        .skins = &.{},
        .node_animation = null,
        .morph_templates = &.{},
    };
}

const faces_per_box = 6;

fn meshOf(allocator: Allocator, kind: Kind) Allocator.Error!gltf.importer.Mesh {
    var count: usize = 0;
    for (props) |prop| {
        if (prop.kind == kind) count += 1;
    }
    const vertices = try allocator.alloc(res.Vertex3D, count * faces_per_box * 4);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, count * faces_per_box * 6);
    errdefer allocator.free(indices);

    var box: usize = 0;
    for (props) |prop| {
        if (prop.kind != kind) continue;
        writeBox(prop, vertices[box * faces_per_box * 4 ..][0 .. faces_per_box * 4], indices[box * faces_per_box * 6 ..][0 .. faces_per_box * 6], @intCast(box * faces_per_box * 4));
        box += 1;
    }
    return .{
        .vertices = vertices,
        .indices = indices,
        .streams = .{},
        .morph = null,
        .material = @backingInt(kind),
        .anchor = null,
        .skin = null,
        .morph_template = null,
    };
}

// Six quads, each wound counterclockwise seen from outside, which is the
// front face for a mesh under an identity transform (glTF 2.0, 3.7.4).
//
// A face along axis `a` with outward sign `s` spans the next two axes in
// cyclic order, `u` and `v`, with u x v = n for the positive face and the two
// swapped for the negative one. Corners taken (-u -v), (+u -v), (+u +v),
// (-u +v) then run counterclockwise about n.
fn writeBox(prop: Prop, vertices: []res.Vertex3D, indices: []u32, base: u32) void {
    const half: res.Vec3 = .{ prop.size[0] * 0.5, prop.size[1] * 0.5, prop.size[2] * 0.5 };
    const centre: res.Vec3 = prop.centre;
    var face: usize = 0;
    inline for (0..3) |a| {
        inline for (.{ 1.0, -1.0 }) |s| {
            var n: res.Vec3 = @splat(0.0);
            n[a] = s;
            var u: res.Vec3 = @splat(0.0);
            var v: res.Vec3 = @splat(0.0);
            const first = (a + 1) % 3;
            const second = (a + 2) % 3;
            if (s > 0.0) {
                u[first] = 1.0;
                v[second] = 1.0;
            } else {
                u[second] = 1.0;
                v[first] = 1.0;
            }

            const middle = centre + n * half;
            const du = u * half;
            const dv = v * half;
            const corners = [4]res.Vec3{ middle - du - dv, middle + du - dv, middle + du + dv, middle - du + dv };
            for (corners, 0..) |corner, k| {
                vertices[face * 4 + k] = .{
                    .position = corner,
                    .normal = n,
                    // A metre of surface to a unit of texture, so a material
                    // that gains a map later tiles it at a size that reads.
                    .uv = .{ @reduce(.Add, corner * u), @reduce(.Add, corner * v) },
                    // n x u = v, so the bitangent the shader rebuilds is +v.
                    .tangent = .{ u[0], u[1], u[2], 1.0 },
                };
            }
            const at: u32 = base + @as(u32, @intCast(face * 4));
            @memcpy(indices[face * 6 ..][0..6], &[6]u32{ at, at + 1, at + 2, at, at + 2, at + 3 });
            face += 1;
        }
    }
}

// What a body bumps into: the solid props, as their footprints on the floor,
// and the floor's own edge.
//
// Circle against rectangle, one axis at a time, the way Backrooms' collider
// works, so walking into a box at an angle slides along it.
pub const radius: f32 = 0.30;

pub fn slide(from: [2]f32, delta: [2]f32) [2]f32 {
    var at = from;
    const along_x: [2]f32 = .{ at[0] + delta[0], at[1] };
    if (!blocked(along_x)) at = along_x;
    const along_z: [2]f32 = .{ at[0], at[1] + delta[1] };
    if (!blocked(along_z)) at = along_z;
    return at;
}

pub fn blocked(at: [2]f32) bool {
    const floor = props[0];
    if (at[0] - radius < floor.centre[0] - floor.size[0] * 0.5 or
        at[0] + radius > floor.centre[0] + floor.size[0] * 0.5 or
        at[1] - radius < floor.centre[2] - floor.size[2] * 0.5 or
        at[1] + radius > floor.centre[2] + floor.size[2] * 0.5) return true;

    for (props) |prop| {
        if (!prop.solid) continue;
        const near_x = std.math.clamp(at[0], prop.centre[0] - prop.size[0] * 0.5, prop.centre[0] + prop.size[0] * 0.5);
        const near_z = std.math.clamp(at[1], prop.centre[2] - prop.size[2] * 0.5, prop.centre[2] + prop.size[2] * 0.5);
        const dx = at[0] - near_x;
        const dz = at[1] - near_z;
        if (dx * dx + dz * dz < radius * radius) return true;
    }
    return false;
}

// Every prop a shot's line crosses, tagged with the prop's index.
pub fn cross(shot: scene.Ray, into: []fps.Crossing) usize {
    var count: usize = 0;
    for (props, 0..) |prop, index| {
        if (count == into.len) break;
        const crossing = fps.crossBox(shot, prop.model(), unit_box, prop.kind.material(), @intCast(index)) orelse continue;
        into[count] = crossing;
        count += 1;
    }
    return count;
}

const testing = std.testing;

test "every face of a box faces out and is wound counterclockwise from outside" {
    var vertices: [faces_per_box * 4]res.Vertex3D = undefined;
    var indices: [faces_per_box * 6]u32 = undefined;
    const prop: Prop = .{ .name = "test", .kind = .target, .centre = .{ 1.0, 2.0, 3.0 }, .size = .{ 0.4, 0.6, 0.2 } };
    writeBox(prop, &vertices, &indices, 0);

    for (0..faces_per_box) |face| {
        const tri = indices[face * 6 ..][0..3];
        const a = vertices[tri[0]].position;
        const b = vertices[tri[1]].position;
        const c = vertices[tri[2]].position;
        const e1 = b - a;
        const e2 = c - a;
        const winding: res.Vec3 = .{
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        };
        const n = vertices[tri[0]].normal;
        // The triangle's own normal agrees with the stored one, and both
        // point away from the centre.
        try testing.expect(@reduce(.Add, winding * n) > 0.0);
        try testing.expect(@reduce(.Add, (a - res.Vec3{ 1.0, 2.0, 3.0 }) * n) > 0.0);
    }
}

test "a shot down the middle crosses the near target before the floor" {
    const shot: scene.Ray = .{ .origin = .{ 0.0, 1.3, 2.0 }, .direction = .{ 0.0, 0.0, -1.0 } };
    var crossings: [props.len]fps.Crossing = undefined;
    const count = cross(shot, &crossings);
    var hits: [props.len]fps.Hit = undefined;
    const outcome = fps.resolve(.{ .damage = 100.0, .penetration = 10.0, .range = 100.0 }, crossings[0..count], &hits);
    try testing.expect(outcome.hits >= 1);
    try testing.expectEqualStrings("5 m torso", props[hits[0].tag].name);
    try testing.expectApproxEqAbs(@as(f32, 6.875), hits[0].distance, 1e-4);
}

test "a body stops at a target and slides along it" {
    // Just short of the 5 m torso, walking into it and to the side.
    const start: [2]f32 = .{ 0.0, -4.5 };
    const moved = slide(start, .{ 0.2, -0.2 });
    try testing.expectApproxEqAbs(@as(f32, 0.2), moved[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -4.5), moved[1], 1e-6);
    try testing.expect(!blocked(.{ 0.0, 0.0 }));
    try testing.expect(blocked(.{ 0.0, 9.9 }));
}
