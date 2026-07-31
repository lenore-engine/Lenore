const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const platform = b.dependency("lenore_platform", .{ .target = target, .optimize = optimize });
    const platform_import: std.Build.Module.Import = .{ .name = "lenore-platform", .module = platform.module("lenore-platform") };

    const lenore = b.addModule("lenore", .{
        .root_source_file = b.path("src/root.zig"),
        .imports = &.{
            platform_import,
        },
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "lenore",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/app.zig"),
            .imports = &.{
                .{ .name = "lenore", .module = lenore },
                platform_import,
            },
            .target = target,
            .optimize = optimize,
        }),
    });
    // Temporary glfw link
    exe.root_module.linkLibrary(platform.artifact("glfw"));
    b.installArtifact(exe);

    const run_step = b.step("run", "Run Lenore");
    run_step.dependOn(&b.addRunArtifact(exe).step);
    // `test` step over tests/ once src.root.zig has policy in it
}
