// The ground and the water, as geometry the engine can already draw.
//
// This produces an `importer.Model` and nothing else. The type is the engine's
// ingest vocabulary, `Level.init` takes one, and it is a plain record of owned
// slices rather than anything the glTF parser has to have produced. So a
// procedural scene reaches the device down exactly the path an asset does, with
// no branch anywhere saying which it was. That is the whole reason this file
// builds a model instead of talking to `lenore-gpu`: the alternative is a second
// upload path that only the example uses, and the first defect in it would be
// invisible to every asset the engine already loads.
//
// The foliage is not here. It never becomes a vertex on this side.

const std = @import("std");
const gltf = @import("lenore-gltf");
const res = @import("lenore-resources");

const terrain = @import("terrain.zig");

const Allocator = std.mem.Allocator;

pub const Error = error{
    SpacingNotPositive,
    GridTooLarge,
    HorizonInsideDomain,
    GrowthNotAboveOne,
};

pub const Options = struct {
    // Metres between ground vertices inside the domain. One metre against a
    // height field whose shortest period is twelve is a dozen samples per
    // feature, so the surface is smooth well before the silhouette is.
    vertex_spacing: f32 = 1.0,

    // How far the ground reaches, in metres, which is much further than the
    // domain the field is defined over.
    //
    // The two are different distances answering different questions. The domain
    // is where the field has detail and where the camera is allowed. This is
    // where the ground has to reach for there to be no gap between its edge and
    // the horizon, and that distance is set by the eye: the horizon for an eye
    // 1.68 m above a sphere of Earth's radius is 4.6 km away, and ground that
    // stops at 128 m leaves three quarters of a degree of nothing under it.
    //
    // Nine hundred rather than the full 4.6 km because the camera's far plane is
    // at a kilometre. Ground ending there sits about 0.08 degrees below the true
    // horizon, which is two pixels at 1080p and 45 degrees, and aerial
    // perspective is what closes the rest: haze reaches the sky colour before
    // the geometry runs out, which is how the reference pictures hide the same
    // edge. Until there is fog this is a seam two pixels tall.
    horizon_radius: f32 = 900.0,

    // How fast cells grow once past the domain. The noise is defined everywhere,
    // so the skirt is the same terrain and not a flat plate, but nothing out
    // there is closer than 128 m and it does not need metre-scale vertices.
    // Twenty-four rings a side reach the horizon at this ratio.
    skirt_growth: f32 = 1.25,

    // Metres per repeat of the ground's texture coordinates. Nothing samples a
    // texture with them yet. They are written because a vertex without them is a
    // vertex every later material has to regenerate, and the cost now is two
    // floats already in the layout.
    uv_scale: f32 = 4.0,

    // Linear, not sRGB: `Vertex3D.colour` is a glTF COLOR_0, and section 3.9.2
    // of the specification defines that attribute as linear values multiplying
    // the base colour factor.
    water_bed_colour: [3]f32 = .{ 0.048, 0.055, 0.038 },
    sand_colour: [3]f32 = .{ 0.44, 0.37, 0.24 },
    grass_colour: [3]f32 = .{ 0.055, 0.105, 0.030 },
    wheat_colour: [3]f32 = .{ 0.29, 0.22, 0.065 },
    // What the ground fades toward where nothing grows on it. Bare earth under a
    // thinning field, so that the patchiness in the density map is visible on
    // the ground and not only in the foliage above it.
    bare_colour: [3]f32 = .{ 0.085, 0.070, 0.045 },

    water_colour: [4]f32 = .{ 0.035, 0.090, 0.115, 0.72 },
    water_roughness: f32 = 0.06,
};

// Ground first, water second, and the order is load bearing. The draw list is
// sorted by the material's alpha mode, so the blended surface composites over an
// opaque one that is already there. It is also the order the two material
// indices below are in, and nothing but this comment keeps those two facts
// together.
pub const ground_mesh = 0;
pub const water_mesh = 1;

pub fn build(
    allocator: Allocator,
    field: terrain.Terrain,
    options: Options,
) (Error || Allocator.Error)!gltf.importer.Model {
    if (!(options.vertex_spacing > 0.0)) return error.SpacingNotPositive;
    if (!(options.skirt_growth > 1.0)) return error.GrowthNotAboveOne;

    const extent = field.settings.extent();
    if (!(options.horizon_radius > extent)) return error.HorizonInsideDomain;

    // One axis of coordinates, shared by both. The grid is a product of it with
    // itself, so the topology below is an ordinary lattice and only the spacing
    // varies: the same six indices per cell, the same winding, and no ring
    // stitching to get wrong.
    const axis = try buildAxis(allocator, extent, options);
    defer allocator.free(axis);

    const side: u32 = @intCast(axis.len);
    const vertex_count = @as(usize, side) * side;

    var meshes = try allocator.alloc(gltf.importer.Mesh, 2);
    errdefer allocator.free(meshes);
    var built: usize = 0;
    errdefer for (meshes[0..built]) |*mesh| mesh.deinit(allocator);

    // The field sampled once per vertex, kept so that the normals below can be
    // differences of neighbouring samples rather than four more evaluations of
    // the noise each. Five height fields per vertex against one, and it is also
    // the more faithful normal: it is the gradient of the surface that will be
    // drawn, not of the continuous one it was sampled from.
    const samples = try allocator.alloc(terrain.Sample, vertex_count);
    defer allocator.free(samples);

    for (axis, 0..) |z, row| {
        for (axis, 0..) |x, column| {
            samples[row * side + column] = field.sampleAt(x, z);
        }
    }

    meshes[0] = try buildGround(allocator, field, options, samples, axis);
    built = 1;
    meshes[1] = try buildWater(allocator, field, options);
    built = 2;

    var materials = try allocator.alloc(res.MaterialInfo, 2);
    errdefer allocator.free(materials);
    var named: usize = 0;
    errdefer for (materials[0..named]) |*material| material.deinit(allocator);

    materials[0] = .{
        .name = try allocator.dupe(u8, "field ground"),
        .textures = .{},
        // White, because the whole of the ground's colour is in its vertices.
        // The factor multiplies the attribute, so anything but white here would
        // tint a decision that was already made per vertex.
        .factors = .{
            .base_colour = .{ 1.0, 1.0, 1.0, 1.0 },
            .metallic = 0.0,
            .roughness = 0.92,
        },
        .rendering = .{},
    };
    named = 1;

    materials[1] = .{
        .name = try allocator.dupe(u8, "field water"),
        .textures = .{},
        .factors = .{
            .base_colour = options.water_colour,
            .metallic = 0.0,
            .roughness = options.water_roughness,
        },
        // Blended and double sided: the camera is allowed under the surface, and
        // a single quad seen from below would otherwise vanish.
        .rendering = .{ .alpha_mode = .blend, .double_sided = true },
    };
    named = 2;

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

// The coordinates one axis of the grid takes, from the far edge of the skirt
// through the domain and out the other side.
//
// Uniform inside the domain and geometrically growing outside it. The camera is
// held inside the domain by its own margin, so every vertex it can stand near is
// at the fine spacing and the coarse ones are only ever seen from at least the
// domain's half width away.
fn buildAxis(allocator: Allocator, extent: f32, options: Options) (Error || Allocator.Error)![]f32 {
    var coordinates: std.ArrayList(f32) = .empty;
    errdefer coordinates.deinit(allocator);

    // The skirt, outward from the domain edge. Built once and mirrored, so the
    // two sides cannot come out different lengths.
    var skirt: std.ArrayList(f32) = .empty;
    defer skirt.deinit(allocator);

    var cell = options.vertex_spacing * options.skirt_growth;
    var reach = extent;
    while (reach < options.horizon_radius) {
        reach = @min(reach + cell, options.horizon_radius);
        cell *= options.skirt_growth;
        try skirt.append(allocator, reach);
        if (skirt.items.len > 256) return error.GridTooLarge;
    }

    var index = skirt.items.len;
    while (index > 0) {
        index -= 1;
        try coordinates.append(allocator, -skirt.items[index]);
    }

    const inner: u32 = @intFromFloat(@ceil(2.0 * extent / options.vertex_spacing));
    // The one quantity a caller can make arbitrarily large with a single small
    // number, so it is checked rather than assumed.
    if (inner >= 4096) return error.GridTooLarge;
    const step = 2.0 * extent / @as(f32, @floatFromInt(inner));
    for (0..inner + 1) |cell_index| {
        try coordinates.append(allocator, -extent + @as(f32, @floatFromInt(cell_index)) * step);
    }

    for (skirt.items) |coordinate| try coordinates.append(allocator, coordinate);

    return coordinates.toOwnedSlice(allocator);
}

fn buildGround(
    allocator: Allocator,
    field: terrain.Terrain,
    options: Options,
    samples: []const terrain.Sample,
    axis: []const f32,
) Allocator.Error!gltf.importer.Mesh {
    const side: u32 = @intCast(axis.len);
    const vertex_count = @as(usize, side) * side;
    const cells = side - 1;

    const vertices = try allocator.alloc(res.Vertex3D, vertex_count);
    errdefer allocator.free(vertices);

    for (0..side) |row| {
        for (0..side) |column| {
            const index = row * side + column;
            const sample = samples[index];
            const x = axis[column];
            const z = axis[row];

            // Central differences where there are two neighbours and one sided
            // at the border, which is what keeps the edge of the grid from
            // taking a normal computed against a height that is not there.
            const dx = slope(samples, axis, column, row, 1, 0);
            const dz = slope(samples, axis, column, row, 0, 1);

            // The surface is a height field, so its normal is (-dh/dx, 1,
            // -dh/dz) before normalising. No cross product is needed and none is
            // taken.
            const normal = normalise(.{ -dx, 1.0, -dz });
            // Along +X, carried onto the surface. The handedness is negative
            // because the bitangent glTF reconstructs is cross(N, T) * w, and
            // that cross points along -Z while V below increases with +Z.
            const tangent = normalise(.{ 1.0, dx, 0.0 });

            vertices[index] = .{
                .position = .{ x, sample.height, z },
                .normal = .{ normal[0], normal[1], normal[2] },
                .uv = .{ x / options.uv_scale, z / options.uv_scale },
                .tangent = .{ tangent[0], tangent[1], tangent[2], -1.0 },
                .colour = groundColour(field, options, sample),
            };
        }
    }

    const indices = try allocator.alloc(u32, @as(usize, cells) * cells * 6);
    errdefer allocator.free(indices);

    var written: usize = 0;
    for (0..cells) |row| {
        for (0..cells) |column| {
            const here: u32 = @intCast(row * side + column);
            const east = here + 1;
            const north = here + side;
            const north_east = north + 1;

            // Counter-clockwise seen from above, which is what the front face
            // has to be: the winding follows the sign of the instance matrix's
            // determinant and this model is placed by the identity.
            indices[written + 0] = here;
            indices[written + 1] = north;
            indices[written + 2] = east;
            indices[written + 3] = east;
            indices[written + 4] = north;
            indices[written + 5] = north_east;
            written += 6;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .streams = .{ .colour = true },
        .morph = null,
        .material = ground_mesh,
        .anchor = null,
        .skin = null,
        .morph_template = null,
    };
}

// One quad over the whole domain at the water level. It is drawn everywhere and
// hidden by the depth test wherever the ground stands above it, which costs four
// vertices and no decision about where the shoreline runs. Trimming it to the
// water would mean deriving that contour, and the ground already carries it.
fn buildWater(
    allocator: Allocator,
    field: terrain.Terrain,
    options: Options,
) Allocator.Error!gltf.importer.Mesh {
    const level = field.settings.water_level;
    // Out to the ground's own edge and not the domain's. The surface has to
    // reach as far as the terrain it is cut into, or the sea ends before the
    // land does and the seam is worse than the one this was widened to close.
    const extent = options.horizon_radius;

    const vertices = try allocator.alloc(res.Vertex3D, 4);
    errdefer allocator.free(vertices);

    const corners = [4][2]f32{
        .{ -extent, -extent },
        .{ extent, -extent },
        .{ -extent, extent },
        .{ extent, extent },
    };
    for (corners, vertices) |corner, *vertex| {
        vertex.* = .{
            .position = .{ corner[0], level, corner[1] },
            .normal = .{ 0.0, 1.0, 0.0 },
            .uv = .{ corner[0] / options.uv_scale, corner[1] / options.uv_scale },
            .tangent = .{ 1.0, 0.0, 0.0, -1.0 },
        };
    }

    const indices = try allocator.alloc(u32, 6);
    errdefer allocator.free(indices);
    indices[0] = 0;
    indices[1] = 2;
    indices[2] = 1;
    indices[3] = 1;
    indices[4] = 2;
    indices[5] = 3;

    return .{
        .vertices = vertices,
        .indices = indices,
        // No colour stream: the water's colour is entirely its factor, and a
        // stream of four white vertices would say the same thing in a wider
        // vertex layout.
        .streams = .{},
        .morph = null,
        .material = water_mesh,
        .anchor = null,
        .skin = null,
        .morph_template = null,
    };
}

// The ground's own colour under whatever grows on it. Blended rather than
// switched, so the cover boundaries the map carries as two densities arrive here
// as gradients and not as a staircase across one vertex.
fn groundColour(
    field: terrain.Terrain,
    options: Options,
    sample: terrain.Sample,
) [4]u8 {
    const settings = field.settings;

    const base: [3]f32 = if (sample.height < settings.water_level)
        options.water_bed_colour
    else if (sample.height < settings.water_level + settings.sand_margin)
        options.sand_colour
    else base: {
        const growth = sample.growth;
        const total = growth.total();
        if (total <= 0.0) break :base options.bare_colour;
        // The two growing colours in the proportion they grow, then faded toward
        // bare earth by how much of the field is actually there.
        const share = growth.wheat / total;
        var grown: [3]f32 = undefined;
        for (&grown, options.grass_colour, options.wheat_colour) |*channel, grass, wheat| {
            channel.* = grass + (wheat - grass) * share;
        }
        for (&grown, options.bare_colour) |*channel, bare| {
            channel.* = bare + (channel.* - bare) * std.math.clamp(total, 0.0, 1.0);
        }
        break :base grown;
    };

    return .{ toUnorm8(base[0]), toUnorm8(base[1]), toUnorm8(base[2]), 255 };
}

fn toUnorm8(value: f32) u8 {
    return @intFromFloat(@round(std.math.clamp(value, 0.0, 1.0) * 255.0));
}

// The height gradient along one axis, in metres per metre. `dx` and `dz` select
// the axis and are zero or one, so the two calls differ only in which neighbour
// they reach for.
//
// The denominator is the distance between the two samples actually used, taken
// from the axis rather than assumed. That is not a refinement: the spacing grows
// by a quarter every cell out in the skirt, so a fixed step would report a slope
// wrong by that ratio and light the skirt as though it were a different shape
// from the terrain it is made of.
fn slope(
    samples: []const terrain.Sample,
    axis: []const f32,
    column: usize,
    row: usize,
    dx: usize,
    dz: usize,
) f32 {
    const side = axis.len;
    const low_column = if (dx != 0 and column > 0) column - 1 else column;
    const low_row = if (dz != 0 and row > 0) row - 1 else row;
    const high_column = if (dx != 0 and column + 1 < side) column + 1 else column;
    const high_row = if (dz != 0 and row + 1 < side) row + 1 else row;

    const low = samples[low_row * side + low_column].height;
    const high = samples[high_row * side + high_column].height;
    const span = if (dx != 0)
        axis[high_column] - axis[low_column]
    else
        axis[high_row] - axis[low_row];
    if (span == 0.0) return 0.0;
    return (high - low) / span;
}

fn normalise(v: [3]f32) [3]f32 {
    const length = @sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    // The Y component is one before normalising in both callers, so the length
    // is at least one and this cannot divide by zero.
    return .{ v[0] / length, v[1] / length, v[2] / length };
}
