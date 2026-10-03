const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const lenore = @import("lenore");
const platform = @import("lenore-platform");

const log = std.log.scoped(.minimal);

// The smallest application that draws a model: a window, one glTF file, one
// light, the engine's orbiting camera and one component of its own. Every
// other example has this shape and adds what it demonstrates.
//
// Usage: run-minimal -- <model.glb>
//
//   left drag    orbit
//   escape       quit

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

// What one frame may draw. One constant for the engine and the level, because
// the level plans its instances and joints against the bounds the engine sizes
// the frame's rings by.
const capacity: gpu.FrameCapacity = .{ .instances = 64, .joints = 256 };

// A component is a value whose hooks the loop calls. It names only the hooks it
// wants, and a misspelt or mistyped one is a compile error here.
const Quit = struct {
    pub const hooks: lenore.Hooks(Quit) = .{ .event = event };

    fn event(_: *Quit, engine: *lenore.Engine, input: platform.Event, ui_consumed: bool) !void {
        if (ui_consumed) return;
        switch (input.payload) {
            .key => |key| if (key.action == .press and key.physical == .escape) engine.requestExit(),
            else => {},
        }
    }
};

pub fn main(process: std.process.Init.Minimal) !void {
    const gpa = if (checking) debug_allocator.allocator() else std.heap.smp_allocator;
    defer if (checking) {
        if (debug_allocator.deinit() == .leak) log.err("host memory leaked", .{});
    };

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var arguments: std.process.Args.Iterator = try .initAllocator(process.args, gpa);
    defer arguments.deinit();
    _ = arguments.next();
    const model_path = arguments.next() orelse {
        log.err("usage: run-minimal -- <model.glb>", .{});
        return error.BadArguments;
    };

    // The loader confines a document's references to a directory, so it takes
    // the directory and a name inside it rather than a path.
    var root = try std.Io.Dir.cwd().openDir(io, std.fs.path.dirname(model_path) orelse ".", .{});
    defer root.close(io);
    var loaded = try gltf.loader.open(gpa, io, root, std.fs.path.basename(model_path));
    defer loaded.deinit(gpa);
    var model = try gltf.importer.build(gpa, &loaded.document, loaded.directory);
    defer model.deinit(gpa);
    try gltf.importer.readExternalImages(gpa, io, root, &model);

    var engine: lenore.Engine = undefined;
    try engine.init(gpa, .{
        .title = "Lenore: minimal",
        .frame_capacity = capacity,
        // A capacity: the material table is allocated once, here.
        .material_capacity = @intCast(@max(model.materials.len, 1)),
    });
    defer engine.deinit();

    // Set before `install`, which fits the sun's shadow to the light it finds
    // in this block.
    const lights = [_]gpu.LightUniform{
        lenore.packLight(try .directional(.{ 1, 1, 1 }, std.math.pi, .{ 0.6, -0.75, 0.28 })),
    };
    engine.look = .{
        .lights = &lights,
        .sun_shadow = .{ .enabled = true, .strength = 1, .normal_offset_texels = 1 },
    };

    var level = try lenore.Level.init(gpa, engine.deps(), &model, .{ .capacity = capacity });
    defer level.deinit(&engine.textures);
    try engine.install(&level, &model);

    // At each point in the frame, the components' hooks are called in tuple
    // order. `run` returns with the device drained, so the defers above may
    // destroy what they own at once.
    var orbit: lenore.OrbitCamera = .{};
    var quit: Quit = .{};
    try engine.run(&level, .{ &orbit, &quit });
}
