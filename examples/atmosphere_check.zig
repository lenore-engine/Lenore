// The device's sky against the host's: renders the atmosphere cube for four
// heights of the sun, reads level zero back, and compares texels in chosen
// directions and on a grid over every face with `scene.skyRadiance` along
// each texel's exact direction. It proves the shader's port of the march, the
// cube's face convention and the units at once, which no host test can.
// Exits with an error past the tolerance or on a validation message.
//
//   zig build run-atmosphere_check

const std = @import("std");
const lenore = @import("lenore");
const scene = @import("lenore-scene");
const gpu = @import("lenore-gpu");

const side = 128;
const V = @Vector(3, f32);
// Half-float storage rounds by about 0.05 %, and the device marches in f32
// where the host's geometry is f64. Measured on the 780M on 2026-10-19: at
// most 0.15 %, so 3 % is a budget for a different device's arithmetic and
// still catches any real fault (a wrong face or a missing term is tens of per
// cent). Channels under 1e-5 of the sun's irradiance are compared against
// that floor rather than relative to themselves.
const tolerance: f32 = 0.03;

fn direction(index: usize) ![3]f32 {
    const x = index % side;
    const y = (index / side) % side;
    return gpu.cubeDirection(@intCast(index / (side * side)), .{
        (@as(f32, @floatFromInt(x)) + 0.5) / side,
        (@as(f32, @floatFromInt(y)) + 0.5) / side,
    });
}

fn nearest(target: [3]f32) !usize {
    var best: f32 = -std.math.inf(f32);
    var index: usize = 0;
    for (0..6 * side * side) |i| {
        const d: V = try direction(i);
        const dot = @reduce(.Add, d * @as(V, target));
        if (dot > best) {
            best = dot;
            index = i;
        }
    }
    return index;
}

fn compare(
    tables: *const scene.Tables,
    a: scene.Atmosphere,
    pixels: []const [4]f16,
    sun: [3]f32,
    top: f32,
    index: usize,
    label: []const u8,
) !f32 {
    const d = try direction(index);
    const unit = try scene.skyRadiance(tables, a, scene.planet_radius_offset, d, sun);
    var expected: [3]f32 = undefined;
    var actual: [3]f32 = undefined;
    var relative: f32 = 0;
    for (0..3) |c| {
        expected[c] = unit[c] * top;
        actual[c] = @floatCast(pixels[index][c]);
        if (!std.math.isFinite(actual[c])) return error.NonFiniteSky;
        relative = @max(relative, @abs(actual[c] - expected[c]) / @max(@abs(expected[c]), 1e-5 * top));
    }
    std.debug.print("{s} face={d} xy=({d},{d}) dir={any} GPU={any} CPU={any} relative={d:.5}\n", .{
        label,
        index / (side * side),
        index % side,
        (index / side) % side,
        d,
        actual,
        expected,
        relative,
    });
    return relative;
}

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    var engine: lenore.Engine = undefined;
    try engine.init(allocator, .{ .title = "Lenore: atmosphere check", .atmosphere = true, .gpu_timing = true });
    defer engine.deinit();
    // A grass-like ground, so the rays that end on it check the ground term
    // as well as the air.
    const a: scene.Atmosphere = .{ .ground_albedo = .{ 0.07, 0.16, 0.045 } };
    const tables = try allocator.create(scene.Tables);
    defer allocator.destroy(tables);
    try tables.build(a);
    try engine.setAtmosphere(a, tables);
    const pixels = try allocator.alloc([4]f16, 6 * side * side);
    defer allocator.free(pixels);
    const top: f32 = 10000;
    var maximum: f32 = 0;
    for ([_]f32{ 55, 10, 2, -3 }) |elevation| {
        const angle = elevation * std.math.pi / 180;
        const sun = [3]f32{ 0, @sin(angle), @cos(angle) };
        try engine.relightSky(sun, top);
        try engine.readSkyCube(pixels);
        std.debug.print("sun elevation={d} degrees; top={d}\n", .{ elevation, top });
        const targets = [_]struct { name: []const u8, d: [3]f32 }{
            .{ .name = "zenith", .d = .{ 0, 1, 0 } },
            .{ .name = "horizon toward", .d = .{ 0, 0, 1 } },
            .{ .name = "horizon away", .d = .{ 0, 0, -1 } },
            .{ .name = "45 degrees up", .d = .{ 0, 0.70710678, 0.70710678 } },
            .{ .name = "below horizon", .d = .{ 0, -0.5, 0.8660254 } },
        };
        for (targets) |target| maximum = @max(maximum, try compare(tables, a, pixels, sun, top, try nearest(target.d), target.name));
        // All faces and corners test signs that a zenith/horizon-only check
        // would miss. Read exactly the stored level-zero texels, without a
        // cube sampler or filtering hiding a face convention error.
        for (0..6) |face| for ([_]usize{ 0, 64, 127 }) |y| for ([_]usize{ 0, 64, 127 }) |x| {
            maximum = @max(maximum, try compare(tables, a, pixels, sun, top, face * side * side + y * side + x, "face grid"));
        };
    }
    std.debug.print("max relative error={d:.5}; tolerance={d:.5}\n", .{ maximum, tolerance });
    if (maximum > tolerance) return error.SkyMismatch;
    if (gpu.validationErrorCount() != 0) return error.ValidationErrors;
}
