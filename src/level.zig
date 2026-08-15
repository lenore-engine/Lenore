const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const platform = @import("lenore-platform");
const zm = @import("zmath");

const env = @import("environment.zig");
const images = @import("images.zig");
const time_module = @import("time.zig");
const world_module = @import("world.zig");

const Allocator = std.mem.Allocator;
const PhaseTimer = time_module.PhaseTimer;
const World = world_module.World;

// One model, resident.
//
// The split against the engine is lifetime and nothing else. Everything here
// belongs to the model that was opened and dies with it; the device, the caches
// and the renderer outlive it, because a viewer that walks a corpus opens one
// device and many models. That is why the renderer is not here: it takes a
// material capacity rather than a count, so it is sized once for the largest
// model a run will meet and written into per level.
//
// What is not solved: the morph prepass has no way to give a registration back,
// so a second level registers on top of the first. Walking a corpus through one
// prepass needs that, and it belongs to `lenore-gpu`.

pub const LevelError = error{
    NoMaterials,
    TooManyMeshes,
    TooManyMaterials,
    UploadedMeshMissing,
    UploadedTextureSetMissing,
};

// Where opening a model spent itself. The phases are the engine's; what they
// are called in a report is the caller's.
pub const Timings = struct {
    upload_ns: u64 = 0,
    // Full waits on the graphics queue the upload took. The number that says
    // whether the staging ceiling is set too low for this model.
    upload_stalls: u32 = 0,
    world_ns: u64 = 0,
    environment_ns: u64 = 0,
};

// What the level needs from the device that outlives it.
pub const Deps = struct {
    io: std.Io,
    clock: platform.Clock,
    context: *const gpu.Context,
    memory: *gpu.MemoryAllocator,
    staging: *gpu.StagingPool,
    setup_pool: *const gpu.OneShotPool,
    textures: *gpu.TextureCache,
    morph_pass: *gpu.MorphPass,
};

pub const Options = struct {
    capacity: gpu.FrameCapacity,
    placement: zm.Mat = zm.identity(),
    clip: ?u16 = 0,
};

pub const Level = struct {
    allocator: Allocator,

    storage: gpu.ResourceStorage,
    uploaded: gpu.Uploaded,
    // Resolved after every insertion that can grow storage. Nothing is added
    // during the frame loop, so these stay valid for the level's life.
    meshes: []*const gpu.Mesh,

    image_loader: images.ImageLoader,
    image_records: []images.ImageRecord,
    image_report: images.Report,

    // The packed array every fragment indexes. One buffer for the whole scene
    // rather than one per material: what selects a record is the index the
    // instance carries, not which descriptor set is bound.
    packed_materials: []gpu.MaterialData,
    morph_registrations: []?u32,
    // Meshes whose shape targets did not fit the prepass. They draw at their
    // bind shape, which is a model too large for the viewer rather than a fault
    // in it, so the count is reported and not fatal.
    morph_overflow: usize,

    world: World,
    environment: ?env.Loaded,
    timings: Timings,

    // A level that holds nothing, for an application that draws without an
    // asset: a fullscreen effect, or a simulation that produces its own
    // geometry.
    //
    // The loop still needs one. It advances a world, decides a shadow bake from
    // it and hands it to every driver hook, and an application writing its own
    // loop instead would be a second copy of the twenty decisions this engine
    // exists to have only once.
    //
    // Nothing here touches the device, so it takes no `Deps` and cannot fail.
    // The frame that draws it records no batches and the whole picture comes
    // from whatever the driver records.
    pub fn empty(allocator: Allocator) Level {
        return .{
            .allocator = allocator,
            .storage = .empty,
            .uploaded = .{ .meshes = .empty, .texture_sets = .empty, .texture_keys = .empty },
            .meshes = &.{},
            .image_loader = .{ .uses = &.{}, .bound = &.{} },
            .image_records = &.{},
            .image_report = .{ .records = &.{} },
            .packed_materials = &.{},
            .morph_registrations = &.{},
            .morph_overflow = 0,
            .world = .empty(allocator),
            .environment = null,
            .timings = .{},
        };
    }

    pub fn init(
        allocator: Allocator,
        deps: Deps,
        model: *const gltf.importer.Model,
        options: Options,
    ) !Level {
        if (model.materials.len == 0) return error.NoMaterials;
        if (model.meshes.len > std.math.maxInt(u32)) return error.TooManyMeshes;
        if (model.materials.len > std.math.maxInt(u32)) return error.TooManyMaterials;

        var timer: PhaseTimer = .begin(deps.clock);

        // Every owning field starts empty, because the errdefer below is
        // `deinit` and it reads all of them.
        var self: Level = .{
            .allocator = allocator,
            .storage = .empty,
            .uploaded = .{ .meshes = .empty, .texture_sets = .empty, .texture_keys = .empty },
            .meshes = &.{},
            .image_loader = .{ .uses = &.{}, .bound = &.{} },
            .image_records = &.{},
            .image_report = .{ .records = &.{} },
            .packed_materials = &.{},
            .morph_registrations = &.{},
            .morph_overflow = 0,
            .world = undefined,
            .environment = null,
            .timings = .{},
        };
        // `world` is the one field teardown cannot be handed empty, so it is
        // tracked rather than guessed.
        var world_built = false;
        errdefer self.release(deps.textures, world_built);

        self.image_loader = try .init(allocator, model);
        self.image_records = try allocator.alloc(images.ImageRecord, model.images.len);
        @memset(self.image_records, .{});
        self.image_report = .{ .records = self.image_records };

        try self.upload(deps, model);
        self.timings.upload_ns = timer.split();

        self.packed_materials = try allocator.alloc(gpu.MaterialData, model.materials.len);
        for (model.materials, self.packed_materials) |*source, *record| record.* = .fromInfo(source);

        try self.registerMorph(deps, model);
        self.world = try .init(allocator, model, self.meshes, self.morph_registrations, .{
            .capacity = options.capacity,
            .placement = options.placement,
            .clip = options.clip,
        });
        world_built = true;
        self.timings.world_ns = timer.split();
        return self;
    }

    // Releases a level the engine never took. The cache is named because a
    // level holds references into it and gives them back here, not because
    // ownership is split: the images belong to the cache, which counts them,
    // and a level that drew no asset gives back none.
    //
    // A level that was installed goes back through `Engine.unload` instead,
    // which has three other owners to ask first. The two are not
    // interchangeable in either direction: unloading one the renderer never
    // took would clear material records belonging to whatever is installed.
    pub fn deinit(self: *Level, textures: *gpu.TextureCache) void {
        self.release(textures, true);
    }

    fn release(self: *Level, textures: *gpu.TextureCache, world_built: bool) void {
        const allocator = self.allocator;

        if (self.environment) |*loaded| loaded.release(textures);
        if (world_built) self.world.deinit();

        allocator.free(self.morph_registrations);
        allocator.free(self.packed_materials);
        allocator.free(self.image_records);
        self.image_loader.deinit(allocator);
        allocator.free(self.meshes);

        self.uploaded.deinit(allocator, &self.storage, textures);
        self.storage.deinit(allocator);
        self.* = undefined;
    }

    // Reads the prefiltered environment this level is lit and backed by.
    //
    // Outside the upload batch, and not for want of room: the batch is a
    // transaction, and an environment that fails to load should not roll the
    // scene back with it.
    pub fn openEnvironment(self: *Level, deps: Deps, directory: []const u8) !void {
        var timer: PhaseTimer = .begin(deps.clock);
        self.environment = try env.load(
            self.allocator,
            deps.io,
            deps.context,
            deps.staging,
            deps.textures,
            deps.setup_pool,
            directory,
        );
        self.timings.environment_ns = timer.split();
    }

    // Writes this level into the device objects that outlive it: one descriptor
    // set per material, the packed record buffer, and the cube the surfaces
    // reflect.
    //
    // Separate from `init` because those objects are the engine's and are sized
    // for a capacity rather than for this model, so opening a level and
    // installing one are different moments.
    pub fn install(
        self: *const Level,
        model: *const gltf.importer.Model,
        renderer: *gpu.Renderer,
        materials: *gpu.MaterialStorage,
        textures: *gpu.TextureCache,
    ) !void {
        for (self.uploaded.texture_sets.items, 0..) |handle, index| {
            const set = self.storage.textureSet(handle) orelse return error.UploadedTextureSetMissing;
            try renderer.setMaterialTextures(
                @intCast(index),
                set,
                .forMaterial(&model.materials[index]),
            );
        }

        // Nothing has been submitted for this level yet, so the cold-path
        // requirement that no frame be reading the buffer is met by there being
        // no frame.
        try materials.upload(self.packed_materials);
        renderer.setMaterialBuffer(materials);

        // Without an environment every image-based term is exactly zero rather
        // than approximately so, which is a supported state and not a degraded
        // one.
        renderer.setEnvironment(if (self.environment) |*loaded|
            loaded.environment
        else
            try gpu.Environment.neutral(textures));
    }

    // Whether the environment is something to draw behind the model. With none
    // loaded the cube behind it is the black fallback, so a background draw
    // would put black over a black clear.
    pub fn hasEnvironment(self: *const Level) bool {
        return self.environment != null;
    }

    fn upload(self: *Level, deps: Deps, model: *const gltf.importer.Model) !void {
        const allocator = self.allocator;

        var batch = try gpu.UploadBatch.begin(
            allocator,
            deps.context,
            deps.memory,
            &self.storage,
            deps.textures,
            deps.staging,
            deps.setup_pool.handle,
        );
        {
            errdefer batch.deinit();

            for (model.meshes) |*mesh| {
                _ = try batch.addMesh(u32, .{
                    .vertices = mesh.vertices,
                    .indices = mesh.indices,
                    .streams = mesh.streams,
                    // A mesh with no targets passes null and gets no morph
                    // buffer, which is what `MorphPass.register` refuses as
                    // NotMorphed.
                    .morph = if (mesh.morph) |deltas| .{
                        .positions = deltas.positions,
                        .normals = deltas.normals,
                        .target_count = deltas.target_count,
                    } else null,
                });
            }

            // Images before materials. An image is decoded once and uploaded
            // once per slot interpretation; a pass over materials would decode
            // once per naming instead.
            try self.image_loader.load(
                allocator,
                deps.io,
                deps.clock,
                &batch,
                model,
                &self.image_report,
            );

            // Nothing is decoded or uploaded here: a slot names an image the
            // pass above made resident and the sampler this use wants.
            for (model.materials) |*material| {
                _ = try batch.addTextureSet(try self.image_loader.textureSet(model, material));
            }
            // Read before finishing, which consumes the batch. Only a
            // reservation raises the count, so it is already final.
            self.timings.upload_stalls = batch.transfer.flushes;
            self.uploaded = try batch.finish();
        }

        self.meshes = try allocator.alloc(*const gpu.Mesh, model.meshes.len);
        for (self.uploaded.meshes.items, self.meshes) |handle, *resident| {
            resident.* = self.storage.mesh(handle) orelse return error.UploadedMeshMissing;
        }
    }

    fn registerMorph(self: *Level, deps: Deps, model: *const gltf.importer.Model) !void {
        self.morph_registrations = try self.allocator.alloc(?u32, model.meshes.len);
        @memset(self.morph_registrations, null);

        for (model.meshes, self.meshes, self.morph_registrations) |*mesh, resident, *registration| {
            if (mesh.morph == null) continue;
            registration.* = deps.morph_pass.register(deps.memory, resident) catch |err| switch (err) {
                error.MeshCapacityExceeded, error.WeightCapacityExceeded => {
                    self.morph_overflow += 1;
                    continue;
                },
                else => return err,
            };
        }
    }
};
