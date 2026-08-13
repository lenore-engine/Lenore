const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const platform = b.dependency("lenore_platform", .{ .target = target, .optimize = optimize });
    const platform_import: std.Build.Module.Import = .{ .name = "lenore-platform", .module = platform.module("lenore-platform") };
    // Forwarded rather than defaulted here: the option belongs to the module
    // that creates the pipelines, and an umbrella that swallowed it would leave
    // `-Dshader-stats` accepted and ignored.
    const shader_stats = b.option(
        bool,
        "shader-stats",
        "Create pipelines so their compiled statistics can be read back",
    ) orelse false;
    const gpu = b.dependency("lenore_gpu", .{
        .target = target,
        .optimize = optimize,
        .@"shader-stats" = shader_stats,
    });
    const gpu_import: std.Build.Module.Import = .{ .name = "lenore-gpu", .module = gpu.module("lenore-gpu") };
    // The converter is an application's tool and not the engine's: nothing in
    // `src/` imports it. What the engine gained for it is the ability to upload
    // a KTX2 image it is handed, which is format support rather than policy.
    const ktx = b.dependency("lenore_ktx", .{ .target = target, .optimize = optimize });
    const ktx_import: std.Build.Module.Import = .{ .name = "lenore-ktx", .module = ktx.module("lenore-ktx") };
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

    // Controllers are application policy, so examples share this module rather
    // than putting one in the engine or copying it into every demonstration.
    const example_orbit = b.createModule(.{
        .root_source_file = b.path("examples/common_orbit.zig"),
        .imports = &.{scene_import},
        .target = target,
        .optimize = optimize,
    });
    const example_orbit_import: std.Build.Module.Import = .{
        .name = "example-orbit",
        .module = example_orbit,
    };
    // The engine is a module and not only an executable root. Two things follow
    // from that and neither is available without it: an example can consume the
    // composition instead of reassembling it, and a test binary can be built
    // against the engine's own surface. Until there is a consumer the compiler
    // reaches, src/ is analysed only as far as one executable's `main` calls,
    // which is the weakest of the three thresholds a green build can mean.
    const engine = b.addModule("lenore", .{
        .root_source_file = b.path("src/root.zig"),
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
    });
    addShaders(b, engine, "assets/shaders");

    const engine_import: std.Build.Module.Import = .{ .name = "lenore", .module = engine };

    // There is no default executable. The engine is a module and every
    // application over it is an example, which is where the umbrella already
    // puts them; a second entry point here would be a second frame loop.

    // Every example lives here rather than in the module it exercises. A module
    // example would still have to be built from the umbrella to reach a window
    // or a sibling's types, and what compiles a module from its own directory
    // is its `tests/reach.zig`, not an example.
    const imports = [_]std.Build.Module.Import{
        engine_import,
        example_orbit_import,
        gltf_import,
        ktx_import,
        gpu_import,
        platform_import,
        resources_import,
        scene_import,
        zignal_import,
        zmath_import,
    };
    // One directory per example, `main.zig` at its root. A directory rather than
    // a bare file because an example may bring assets of its own, and the two
    // spellings would otherwise be one name meaning two things.
    const examples_step = b.step("examples", "Build every example");
    for (directoriesIn(b, "examples")) |name| {
        const module = b.createModule(.{
            .root_source_file = b.path(b.fmt("examples/{s}/main.zig", .{name})),
            .imports = &imports,
            .target = target,
            .optimize = optimize,
        });
        // Shading an example authored for itself, compiled into the example's
        // own module. Anonymous imports belong to one module, so a shader added
        // to the engine's is one this cannot embed; the engine's defaults stay
        // where they are and an example that draws with them declares no
        // directory at all.
        addShaders(b, module, b.fmt("examples/{s}/shaders", .{name}));

        const example = b.addExecutable(.{ .name = name, .root_module = module });
        example.root_module.linkLibrary(platform.artifact("glfw"));
        examples_step.dependOn(&b.addInstallArtifact(example, .{}).step);

        const run = b.addRunArtifact(example);
        if (b.args) |args| run.addArgs(args);
        b.step(b.fmt("run-{s}", .{name}), b.fmt("Run the {s} example", .{name}))
            .dependOn(&run.step);
    }

    // The umbrella's suite covers what the umbrella owns. Every module carries
    // its own, run from its own directory with no umbrella, and this does not
    // depend on those.
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = testRoot(b, "tests"),
            .imports = &.{
                engine_import,
                gltf_import,
                ktx_import,
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
    // The suite compiles the window-facing surface, which is the backend. A
    // module's library links do not reach an artifact that imports it, so this
    // is named here as well as on the engine below.
    unit_tests.root_module.linkLibrary(platform.artifact("glfw"));
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(unit_tests).step);

    // The shared example controller is not imported by the umbrella suite, and
    // addTest discovers blocks only in its root module. Give it that root so
    // frame-rate invariance and unconstrained orbit cannot stay uncompiled.
    const orbit_tests = b.addTest(.{ .root_module = example_orbit });
    test_step.dependOn(&b.addRunArtifact(orbit_tests).step);
    // addTest collects test blocks from the root module of its compilation only.
    // The suite above imports the engine rather than being it, so a `test`
    // written beside the code in src/ would never run and would stay green
    // forever. This second binary is that module.
    engine.linkLibrary(platform.artifact("glfw"));
    const module_tests = b.addTest(.{ .root_module = engine });
    test_step.dependOn(&b.addRunArtifact(module_tests).step);
}

// Vulkan 1.3 consumes SPIR-V 1.6, which is what the profile names.
const spirv_profile = "spirv_1_6";

// Every `*.slang` in the named directory becomes one SPIR-V module carrying all
// of that file's entry points, imported into `module` under the file's stem.
// Whoever embeds them names no path.
//
// A directory that does not exist contributes nothing, which is what an example
// drawing with the engine's own shading is.
//
// The shading is the engine's and not `lenore-gpu`'s. What a surface reflects,
// what the background is, which tone operator the picture is presented through
// and how the bloom chain filters are all answers to "what should this look
// like", and a module that owns the device has no business holding an opinion
// on that. `lenore-gpu` states the shape of the set it needs and creates
// modules from words it is handed; the words are authored here.
//
// One module per file rather than per stage, because a Slang file marked up
// with `[shader(...)]` attributes emits every entry point into one binary and
// keeps their names, and the pipeline selects a stage by name.
//
// `-matrix-layout-row-major` is what makes a zmath `Mat` arrive as itself.
//
// The flag names the source language's convention, not the SPIR-V decoration:
// measured, it produces `ColMajor`, and the default produces `RowMajor`. Which
// one is right follows from the decoration together with the operation, and
// only from both.
//
// Slang compiles `mul(vector, matrix)` to `OpMatrixTimesVector` either way, so
// the result is the SPIR-V matrix times a column vector: component r is the sum
// over c of column c's r-th component times v[c]. SPIR-V specification, 3.20
// Decoration: `ColMajor` means components within a column are contiguous, so
// SPIR-V's column c is the c-th four floats in memory, which is zmath's row c.
// The sum is then over rows, which is what zmath's `mul(v, m)` computes.
//
// With the default the same sum runs over zmath's columns instead, and every
// matrix reaches the shader transposed.
fn addShaders(b: *std.Build, module: *std.Build.Module, dir_path: []const u8) void {
    for (filesIn(b, dir_path, ".slang")) |name| {
        const stem = name[0 .. name.len - ".slang".len];
        const command = b.addSystemCommand(&.{
            "slangc",
            "-target",
            "spirv",
            "-profile",
            spirv_profile,
            "-matrix-layout-row-major",
            // A module with one entry point has it renamed to `main` without
            // this; several keep their names. Measured on 2026-08-08, and the
            // reflection JSON reports the source name either way, so nothing
            // reading that can see the rename. `tests/shader_reflection.zig`
            // scans the emitted words instead.
            "-fvk-use-entrypoint-name",
        });
        command.addFileArg(b.path(b.fmt("{s}/{s}", .{ dir_path, name })));
        command.addArg("-o");
        const spirv = command.addOutputFileArg(b.fmt("{s}.spv", .{stem}));

        // The compiler's own account of the layout it produced. It is imported
        // beside the words so a test can hold the hand-written mirror against
        // it, which is the only thing that keeps the two from drifting. Nothing
        // is generated from it: the Zig side stays authored.
        command.addArg("-reflection-json");
        const reflection = command.addOutputFileArg(b.fmt("{s}.json", .{stem}));

        module.addAnonymousImport(stem, .{ .root_source_file = spirv });
        module.addAnonymousImport(b.fmt("{s}_reflection", .{stem}), .{ .root_source_file = reflection });
    }
}

fn zigFilesIn(b: *std.Build, dir_path: []const u8) [][]const u8 {
    return filesIn(b, dir_path, ".zig");
}

fn directoriesIn(b: *std.Build, dir_path: []const u8) [][]const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    const io = b.graph.io;
    var dir = b.build_root.handle.openDir(io, dir_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return names.items,
        else => std.debug.panic("cannot open {s}/: {t}", .{ dir_path, err }),
    };
    defer dir.close(io);

    var iterator = dir.iterate();
    while (iterator.next(io) catch @panic("cannot list the directory")) |entry| {
        if (entry.kind != .directory) continue;
        names.append(b.allocator, b.dupe(entry.name)) catch @panic("OOM");
    }

    std.mem.sort([]const u8, names.items, {}, lessThanName);
    return names.items;
}

fn filesIn(b: *std.Build, dir_path: []const u8, extension: []const u8) [][]const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    const io = b.graph.io;
    var dir = b.build_root.handle.openDir(io, dir_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return names.items,
        else => std.debug.panic("cannot open {s}/: {t}", .{ dir_path, err }),
    };
    defer dir.close(io);

    var iterator = dir.iterate();
    while (iterator.next(io) catch @panic("cannot list the directory")) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, extension)) continue;
        names.append(b.allocator, b.dupe(entry.name)) catch @panic("OOM");
    }

    std.mem.sort([]const u8, names.items, {}, lessThanName);
    return names.items;
}

fn lessThanName(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

// Generates the test root by listing the directory, so a new test file needs no
// registration.
//
// This cannot be done at comptime: `@import` takes a string literal and there is
// no filesystem at comptime. Zig analyses lazily, so a test file nobody imports
// is silently not run, and a forgotten registration is a suite that goes green
// without it.
fn testRoot(b: *std.Build, dir_path: []const u8) std.Build.LazyPath {
    var source: std.ArrayList(u8) = .empty;
    source.appendSlice(b.allocator, "// Generated by build.zig from the test directory. Do not edit.\ntest {\n") catch @panic("OOM");
    for (zigFilesIn(b, dir_path)) |name|
        source.print(b.allocator, "    _ = @import(\"{s}/{s}\");\n", .{ dir_path, name }) catch @panic("OOM");
    source.appendSlice(b.allocator, "}\n") catch @panic("OOM");

    // The generated root sits beside a copy of the directory, so its imports
    // resolve relative to itself.
    const generated = b.addWriteFiles();
    _ = generated.addCopyDirectory(b.path(dir_path), dir_path, .{});
    return generated.add("test_root.zig", source.items);
}
