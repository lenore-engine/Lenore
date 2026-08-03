const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const platform = b.dependency("lenore_platform", .{ .target = target, .optimize = optimize });
    const platform_import: std.Build.Module.Import = .{ .name = "lenore-platform", .module = platform.module("lenore-platform") };
    const gpu = b.dependency("lenore_gpu", .{ .target = target, .optimize = optimize });
    const gpu_import: std.Build.Module.Import = .{ .name = "lenore-gpu", .module = gpu.module("lenore-gpu") };
    const gltf = b.dependency("lenore_gltf", .{ .target = target, .optimize = optimize });
    const gltf_import: std.Build.Module.Import = .{ .name = "lenore-gltf", .module = gltf.module("lenore-gltf") };
    const resources = b.dependency("lenore_resources", .{ .target = target, .optimize = optimize });
    const resources_import: std.Build.Module.Import = .{ .name = "lenore-resources", .module = resources.module("lenore-resources") };
    const scene = b.dependency("lenore_scene", .{ .target = target, .optimize = optimize });
    const scene_import: std.Build.Module.Import = .{ .name = "lenore-scene", .module = scene.module("lenore-scene") };
    const zignal_import: std.Build.Module.Import = .{
        .name = "zignal",
        .module = b.dependency("zignal", .{ .target = target, .optimize = optimize }).module("zignal"),
    };
    // The engine composes the modules and does its own maths over what they
    // return, so it names zmath directly rather than through one of them.
    const zmath_import: std.Build.Module.Import = .{
        .name = "zmath",
        .module = b.dependency("zmath", .{}).module("root"),
    };

    const exe = b.addExecutable(.{
        .name = "lenore",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/app.zig"),
            .imports = &.{
                gltf_import,
                gpu_import,
                platform_import,
                resources_import,
                scene_import,
                zignal_import,
                zmath_import,
            },
            .target = target,
            .optimize = optimize,
        }),
    });
    // The temporary platform backend owns glfw, but Zig artifacts do not
    // propagate native library links through a module dependency.
    exe.root_module.linkLibrary(platform.artifact("glfw"));
    b.installArtifact(exe);

    const run_step = b.step("run", "Run Lenore");
    run_step.dependOn(&b.addRunArtifact(exe).step);

    // Every example lives here rather than in the module it exercises. A module
    // example would still have to be built from the umbrella to reach a window
    // or a sibling's types, and what compiles a module from its own directory
    // is its `tests/reach.zig`, not an example.
    const imports = [_]std.Build.Module.Import{
        gltf_import,
        gpu_import,
        platform_import,
        resources_import,
        scene_import,
        zignal_import,
        zmath_import,
    };
    const examples_step = b.step("examples", "Build every example");
    for (zigFilesIn(b, "examples")) |name| {
        const stem = name[0 .. name.len - ".zig".len];
        const example = b.addExecutable(.{
            .name = stem,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("examples/{s}", .{name})),
                .imports = &imports,
                .target = target,
                .optimize = optimize,
            }),
        });
        example.root_module.linkLibrary(platform.artifact("glfw"));
        examples_step.dependOn(&b.addInstallArtifact(example, .{}).step);

        const run = b.addRunArtifact(example);
        if (b.args) |args| run.addArgs(args);
        b.step(b.fmt("run-{s}", .{stem}), b.fmt("Run the {s} example", .{stem}))
            .dependOn(&run.step);
    }
}

fn zigFilesIn(b: *std.Build, dir_path: []const u8) [][]const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    const io = b.graph.io;
    var dir = b.build_root.handle.openDir(io, dir_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return names.items,
        else => std.debug.panic("cannot open {s}/: {t}", .{ dir_path, err }),
    };
    defer dir.close(io);

    var iterator = dir.iterate();
    while (iterator.next(io) catch @panic("cannot list the directory")) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".zig")) continue;
        names.append(b.allocator, b.dupe(entry.name)) catch @panic("OOM");
    }

    std.mem.sort([]const u8, names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b_: []const u8) bool {
            return std.mem.order(u8, a, b_) == .lt;
        }
    }.lessThan);
    return names.items;
}
