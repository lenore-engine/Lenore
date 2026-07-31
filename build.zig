const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const platform = b.dependency("lenore_platform", .{ .target = target, .optimize = optimize });
    const platform_import: std.Build.Module.Import = .{ .name = "lenore-platform", .module = platform.module("lenore-platform") };

    const exe = b.addExecutable(.{
        .name = "lenore",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/app.zig"),
            .imports = &.{platform_import},
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
}
