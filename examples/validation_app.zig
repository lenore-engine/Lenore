const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");
const zignal = @import("zignal");
const zm = @import("zmath");

const Allocator = std.mem.Allocator;
const log = std.log.scoped(.validation);

// The first composed frame, and the checks that say which stage is wrong when
// the picture is not what it should be.
//
// A wrong matrix, a wrong descriptor and a wrong barrier all produce the same
// black window. So every stage that can be answered on the host is answered
// here and printed beside what the device was told, and a run that draws
// nothing still says where it stopped agreeing.
//
// Usage: run-validation_app -- <model.glb> [environment-directory]
//
// The environment directory holds a prefiltered set in the shape the Khronos
// tool produces: `lambertian/diffuse.ktx2`, `ggx/specular.ktx2` and `lut_ggx.png`.
// Without one the scene is lit by its punctual lights alone, which is a
// supported state and not a degraded one: the black cubemaps the cache falls
// back to make every image-based term exactly zero.
//
// Every mesh and material is uploaded. Embedded PNG or JPEG images consumed by
// the shader are decoded to RGBA8; absent maps use the texture cache's neutral
// fallbacks.

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

const frames_in_flight = 2;

// What one frame's rings hold. The joint bound is what the scene's joint plan
// packs into, so the two are the same number and not two copies of it.
const frame_capacity: gpu.FrameCapacity = .{ .instances = 64, .joints = 256 };

// What the morph prepass holds. Both bounds are the app's: the Khronos morph
// samples reach two registrations of eight targets, and a face rig is the shape
// that would move these. Exceeding either is reported and not fatal, because a
// model too large for the viewer is not a defect in the viewer.
const morph_capacity: gpu.MorphCapacity = .{ .meshes = 16, .weights = 256 };

const SourcePixel = zignal.Rgba(u8);
const DecodedImage = zignal.Image(SourcePixel);

comptime {
    if (@sizeOf(SourcePixel) != 4 or
        @bitOffsetOf(SourcePixel, "r") != 0 or
        @bitOffsetOf(SourcePixel, "g") != 8 or
        @bitOffsetOf(SourcePixel, "b") != 16 or
        @bitOffsetOf(SourcePixel, "a") != 24)
    {
        @compileError("zignal RGBA8 no longer has byte-packed RGBA order");
    }
}

// Where a load spends its time. The app's own verification passes are a phase of
// their own because they are not loading: they walk vertex data the engine never
// reads again, and folding them into the upload phase reports a load slower than
// the engine performs.
//
// One clock and one moving boundary. Every phase is closed by a `split`, which
// is what keeps an interval from being counted in two fields; a phase whose
// result is dropped is lost time rather than double-counted time, and the total
// spans the whole sequence either way.
const Timeline = struct {
    clock: platform.Clock,
    origin: u64,
    last: u64,

    parse: u64 = 0,
    verify: u64 = 0,
    device: u64 = 0,
    upload: u64 = 0,
    renderer: u64 = 0,
    environment: u64 = 0,
    prepare: u64 = 0,

    fn begin(clock: platform.Clock) Timeline {
        const now = clock.now();
        return .{ .clock = clock, .origin = now, .last = now };
    }

    // Nanoseconds since the previous boundary, and the new boundary.
    fn split(self: *Timeline) u64 {
        const now = self.clock.now();
        const elapsed = now - self.last;
        self.last = now;
        return elapsed;
    }

    fn total(self: *const Timeline) u64 {
        return self.last - self.origin;
    }
};

fn seconds(nanoseconds: u64) f64 {
    return @as(f64, @floatFromInt(nanoseconds)) / std.time.ns_per_s;
}

// What decoding cost and how much of it was spent twice. One counter per
// document image rather than a map: the image pass walks `model.images` by
// position, so that position is the identity and a repeat is a count above one.
//
// `nanoseconds` is summed across the decode threads and `wall_nanoseconds` is
// the pass that contains them, so the two are only equal when the pool
// contributed nothing. Their ratio is the concurrency the run actually got, and
// reporting the sum alone would read as a regression.
const DecodeStats = struct {
    clock: platform.Clock,
    counts: []u32,
    nanoseconds: u64 = 0,
    wall_nanoseconds: u64 = 0,
    pixels: u64 = 0,
    redundant_pixels: u64 = 0,
    source_bytes: u64 = 0,
    peak_window_bytes: u64 = 0,

    fn calls(self: *const DecodeStats) u32 {
        var sum: u32 = 0;
        for (self.counts) |count| sum += count;
        return sum;
    }

    fn distinct(self: *const DecodeStats) u32 {
        var sum: u32 = 0;
        for (self.counts) |count| sum += @intFromBool(count > 0);
        return sum;
    }

    // One decoded image, accounted on the thread that will upload it. The
    // decode's own elapsed time comes back from the worker; nothing here is
    // touched off the recording thread.
    fn record(self: *DecodeStats, index: usize, decoded: *const DecodedImage, nanoseconds: u64) void {
        const pixels: u64 = @as(u64, decoded.cols) * decoded.rows;
        self.nanoseconds += nanoseconds;
        self.pixels += pixels;
        if (self.counts[index] > 0) self.redundant_pixels += pixels;
        self.counts[index] += 1;
    }

    fn report(self: *const DecodeStats) void {
        const megapixels = @as(f64, @floatFromInt(self.pixels)) / 1e6;
        const redundant = @as(f64, @floatFromInt(self.redundant_pixels)) / 1e6;
        const cpu = seconds(self.nanoseconds);
        const wall = seconds(self.wall_nanoseconds);
        log.info(
            "decode: {d} call(s) over {d} distinct image(s), {d:.1} Mpix from {d:.1} MB",
            .{
                self.calls(),
                self.distinct(),
                megapixels,
                @as(f64, @floatFromInt(self.source_bytes)) / (1 << 20),
            },
        );
        log.info(
            "decode: {d:.3} s of CPU across the pool in {d:.3} s of wall ({d:.2}x), {d:.1} Mpix/s delivered",
            .{ cpu, wall, if (wall > 0) cpu / wall else 0, if (wall > 0) megapixels / wall else 0 },
        );
        log.info("decode: {d:.1} Mpix of that was decoded more than once", .{redundant});
        log.info("decode: window peaked at {d:.1} MB of {d:.1} MB", .{
            @as(f64, @floatFromInt(self.peak_window_bytes)) / (1 << 20),
            @as(f64, @floatFromInt(decode_window_bytes)) / (1 << 20),
        });
    }
};

// One skin's playback. The pose borrows the template owned by the model, and
// `source_index` is the document index that a mesh carries.
const Skin = struct {
    source_index: u32,
    pose: res.SkeletonPose,
    clip: ?*const res.Animation,
};

fn skinForIndex(skins: []Skin, source_index: u32) ?*Skin {
    for (skins) |*skin| {
        if (skin.source_index == source_index) return skin;
    }
    return null;
}

// The one definition of a draw's object-to-world transform. Everything that
// needs it goes through here: the instance record the shader reads, the depth
// key the blended run is ordered by, and the bounds the camera is framed on.
//
// Rigid node animation is the only thing that moves it. Static geometry is baked
// in world space and a skinned mesh's motion is in its joint matrices, so both
// take `placement` alone.
//
// Row vector convention, so the anchor's motion applies before the placement: an
// anchored mesh's vertices are baked in its anchor's space and reach the world
// only through it.
fn instanceMatrix(anchor: ?res.Slot, animator: ?*const res.NodeAnimator, placement: zm.Mat) zm.Mat {
    const slot = anchor orelse return placement;
    // An anchor exists only where the importer produced an animation to resolve
    // it against, so the caller has already turned a missing animator into an
    // error rather than reaching here with one.
    return zm.mul(animator.?.world_transforms[slot], placement);
}

// The largest element-wise difference between two transforms. It mixes the
// rotation with the translation, so it is not a distance and is not comparable
// to anything in world units. It answers one question: whether two poses are the
// same pose.
fn matrixDrift(a: zm.Mat, b: zm.Mat) f32 {
    var worst: f32 = 0;
    inline for (0..4) |row| {
        inline for (0..4) |column| {
            worst = @max(worst, @abs(a[row][column] - b[row][column]));
        }
    }
    return worst;
}

// A point carried into world space by an instance matrix. The w lane is one, so
// the matrix's translation applies.
fn placePoint(point: res.Vec3, matrix: zm.Mat) res.Vec3 {
    const carried = zm.mul(zm.f32x4(point[0], point[1], point[2], 1.0), matrix);
    return .{ carried[0], carried[1], carried[2] };
}

const RecordPlan = scene.DrawBatches(*const gpu.Mesh, u32);

pub const BatchTranslationError = error{BatchDestinationTooSmall};

// The deliberate module-boundary copy: scene owns ordering and face policy;
// GPU owns resource pointers and Vulkan state. The destination is preallocated,
// and its capacity is checked before the first write.
fn translateRecordBatches(
    batches: []const RecordPlan.Batch,
    ordered: []const u32,
    sources: []const ?gpu.MeshVertexSource,
    destination: []gpu.RecordBatch,
) BatchTranslationError![]gpu.RecordBatch {
    if (destination.len < batches.len) return error.BatchDestinationTooSmall;

    for (batches, destination[0..batches.len]) |batch, *record| {
        record.* = .{
            .mesh = batch.mesh,
            .material_index = batch.material,
            .cull_mode = switch (batch.face_culling) {
                .none => .{},
                .back => .{ .back_bit = true },
                .front => .{ .front_bit = true },
            },
            .first_instance = batch.first_instance,
            .instance_count = batch.instance_count,
            // `first_instance` is a position within `order`, so this is the
            // first draw the batch coalesced. Any draw of the batch would do:
            // batching keys on the mesh, and this app uploads one GPU mesh per
            // imported mesh, so every draw in a batch resolves to one source.
            .vertex_source = sources[ordered[batch.first_instance]],
        };
    }
    return destination[0..batches.len];
}

// Where each mesh fetches its vertices this frame, by mesh index. A morphed mesh
// reads the slot the prepass is about to write for `frame`, and every other one
// reads its own buffer, which `RecordBatch` spells as null.
fn fillVertexSources(
    pass: *const gpu.MorphPass,
    registrations: []const ?u32,
    frame: usize,
    destination: []?gpu.MeshVertexSource,
) void {
    for (registrations, destination) |registration, *source| {
        source.* = if (registration) |index| pass.vertexSource(index, frame) else null;
    }
}

// The base an instance record carries, from what the plan assigned it. The plan
// marks an entity with no pose, and that marker is not an index: the unskinned
// pipeline reads no joint array, so it becomes zero here rather than reaching
// the device.
fn jointBase(planned: u32) u32 {
    return if (planned == scene.no_joint_base) 0 else planned;
}

// Where every draw is this frame, what order they are recorded in, and the
// instance record each one reads.
//
// Rebuilt every frame rather than once. The blended run is ordered by each
// draw's distance from the eye, and rigid node animation moves a draw between
// one frame and the next, so an order computed at load describes the bind pose
// and nothing after it.
//
// Nothing here allocates. Every buffer is owned by the caller for the length of
// the run, which is what lets this sit in the frame loop.
const DrawPlan = struct {
    // Read, in mesh order, and never written here.
    candidates: []const RecordPlan.Draw,
    // The centre of each draw in the space its instance matrix maps from. A
    // skinned mesh carries its bind placement here instead, because its instance
    // matrix is identity and the joint matrices hold its motion.
    centres: []const res.Vec3,
    joint_bases: []const u32,

    // Written in mesh order. Only `depth` of a key changes per frame; the layer
    // is the material's alpha mode and is set once.
    matrices: []zm.Mat,
    keys: []scene.DrawKey,

    // Written in draw order. `first_instance` of a batch is a position within
    // `order`, so the instance ring is packed in that order too and never by
    // mesh index.
    order: []u32,
    instances: []gpu.Instance,
    batch_storage: []RecordPlan.Batch,
    record_storage: []gpu.RecordBatch,

    // By mesh index, and rewritten by `fillVertexSources` whenever the frame
    // slot changes. Read here rather than passed in, because a rebuild reaches
    // it only through the batch translation at the end.
    vertex_sources: []const ?gpu.MeshVertexSource,

    // The last rebuild's output.
    ordered: []const u32 = &.{},
    batches: []const RecordPlan.Batch = &.{},
    records: []gpu.RecordBatch = &.{},

    fn rebuild(
        self: *DrawPlan,
        meshes: []const gltf.importer.Mesh,
        animator: ?*const res.NodeAnimator,
        placement: zm.Mat,
        eye: res.Vec3,
    ) !void {
        for (meshes, self.centres, self.matrices, self.keys) |*mesh, centre, *matrix, *key| {
            matrix.* = instanceMatrix(mesh.anchor, animator, placement);
            key.depth = scene.depthOf(eye, placePoint(centre, matrix.*));
        }

        self.ordered = try scene.orderDraws(self.keys, self.order);

        for (self.ordered, self.instances) |draw_index, *instance| {
            instance.* = .{
                .model = self.matrices[draw_index],
                .joint_base = jointBase(self.joint_bases[draw_index]),
                .material_index = meshes[draw_index].material,
            };
        }

        self.batches = try RecordPlan.build(self.candidates, self.ordered, self.batch_storage);
        self.records = try translateRecordBatches(
            self.batches,
            self.ordered,
            self.vertex_sources,
            self.record_storage,
        );
    }
};

fn frameCamera(camera: *scene.Camera, sphere: res.Sphere, aspect: f32) f32 {
    // A vertical field of view narrows horizontally below aspect one. Increase
    // distance by the inverse aspect there so the same sphere stays inside both
    // axes when a compositor gives the window a portrait extent.
    const distance = @max(sphere.radius * 3.0 / @min(aspect, 1.0), 0.01);
    camera.anchor = .{ .orbit = .{
        .target = .{ sphere.centre[0], sphere.centre[1], sphere.centre[2] },
        .distance = distance,
    } };
    camera.projection = .{ .perspective = .{
        .near = @max(distance * 0.01, 1.0e-4),
        .far = distance * 10.0,
    } };
    return distance;
}

// The weighted sum of joint matrices applied to a vertex, exactly as
// `skinnedVertexMain` computes it. Written twice on purpose: this is the host's
// account of what the shader should produce, and the two agreeing is the
// check.
fn skinnedPosition(pose: *const res.SkeletonPose, vertex: res.Vertex3D) [3]f32 {
    var skin: zm.Mat = .{ zm.f32x4s(0), zm.f32x4s(0), zm.f32x4s(0), zm.f32x4s(0) };
    inline for (0..4) |lane| {
        const weight = zm.f32x4s(vertex.weights[lane]);
        const joint = pose.joint_transforms[vertex.joints[lane]];
        inline for (0..4) |row| skin[row] += joint[row] * weight;
    }

    const position = zm.f32x4(vertex.position[0], vertex.position[1], vertex.position[2], 1);
    const skinned = zm.mul(position, skin);
    return .{ skinned[0], skinned[1], skinned[2] };
}

// Failures a check found, as distinct from errors the code returned. A check
// that fails does not stop the run: the point is to reach the frame and see
// what it does, with every disagreement already named.
var failures: u32 = 0;

fn check(passed: bool, comptime what: []const u8, args: anytype) void {
    if (passed) {
        log.info("ok    " ++ what, args);
    } else {
        failures += 1;
        log.err("FAIL  " ++ what, args);
    }
}

pub fn main(process: std.process.Init.Minimal) !void {
    const gpa = if (checking) debug_allocator.allocator() else std.heap.smp_allocator;
    defer if (checking) {
        if (debug_allocator.deinit() == .leak) log.err("host memory leaked", .{});
    };

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // Before the first read, so the parse phase is measured from a boundary that
    // no work has crossed. Input takes the same clock later.
    const clock: platform.Clock = .init(io);
    var timeline: Timeline = .begin(clock);

    var arguments: std.process.Args.Iterator = .init(process.args);
    _ = arguments.skip();
    const model_path = arguments.next() orelse {
        log.err("usage: validation_app <model.glb>", .{});
        return error.MissingModelPath;
    };

    // The loader confines every reference the document makes to a root, so it
    // takes a directory and a name inside it rather than a path. Splitting the
    // argument here is what lets the asset live anywhere while its own images
    // still cannot escape the directory it was found in.
    // Optional. Absent means no environment, which the neutral fallbacks
    // express exactly rather than approximately.
    const environment_path = arguments.next();

    const model_directory = std.fs.path.dirname(model_path) orelse ".";
    const model_name = std.fs.path.basename(model_path);
    var root = try std.Io.Dir.cwd().openDir(io, model_directory, .{});
    defer root.close(io);

    // Host side first, so a broken asset is reported before a device is touched.
    var loaded = try gltf.loader.open(gpa, io, root, model_name);
    defer loaded.deinit(gpa);
    var model = try gltf.importer.build(gpa, &loaded.document, loaded.directory);
    defer model.deinit(gpa);
    timeline.parse = timeline.split();

    const decode_counts = try gpa.alloc(u32, model.images.len);
    defer gpa.free(decode_counts);
    @memset(decode_counts, 0);
    var decode_stats: DecodeStats = .{ .clock = clock, .counts = decode_counts };

    var work: ImageWork = try .init(gpa, &model);
    defer work.deinit(gpa);

    check(model.meshes.len > 0, "model holds {d} mesh(es), {d} material(s)", .{
        model.meshes.len,
        model.materials.len,
    });
    if (model.meshes.len == 0) return error.NoGeometry;

    if (model.materials.len == 0) return error.NoMaterials;
    if (model.meshes.len > frame_capacity.instances) return error.InstanceCapacityExceeded;
    if (model.meshes.len > std.math.maxInt(u32)) return error.TooManyMeshes;
    if (model.materials.len > std.math.maxInt(u32)) return error.TooManyMaterials;

    var total_vertices: usize = 0;
    var total_indices: usize = 0;
    for (model.meshes, 0..) |*mesh, index| {
        if (mesh.material >= model.materials.len) return error.MaterialIndexOutOfRange;
        log.info("mesh {d}: {d} vertices, {d} indices, material {d}, streams {any}", .{
            index,
            mesh.vertices.len,
            mesh.indices.len,
            mesh.material,
            mesh.streams,
        });
        check(
            mesh.indices.len == 0 or mesh.indices.len % 3 == 0,
            "mesh {d} index count {d} is whole triangles",
            .{ index, mesh.indices.len },
        );
        total_vertices += mesh.vertices.len;
        total_indices += mesh.indices.len;
    }

    // Every skin owns one pose. Meshes refer to it by the document's skin index,
    // which is not necessarily the position of that skin in the imported slice.
    const skins = try gpa.alloc(Skin, model.skins.len);
    var initialized_skins: usize = 0;
    defer {
        for (skins[0..initialized_skins]) |*skin| skin.pose.deinit(gpa);
        gpa.free(skins);
    }
    for (model.skins, skins) |*source_skin, *skin| {
        skin.* = .{
            .source_index = source_skin.index,
            .pose = try res.SkeletonPose.init(gpa, &source_skin.skeleton),
            // The first clip. Which one plays is playback's choice and this app
            // has no way to make it; a skin with none holds its bind pose.
            .clip = if (source_skin.clips.len > 0) &source_skin.clips[0] else null,
        };
        initialized_skins += 1;
        skin.pose.evaluate();

        log.info("skin {d}: {d} joints over {d} slots, {d} clip(s)", .{
            source_skin.index,
            source_skin.skeleton.jointCount(),
            source_skin.skeleton.slotCount(),
            source_skin.clips.len,
        });
        if (skin.clip) |clip|
            log.info("clip: keyed from {d:.3} to {d:.3} s over {d} slots", .{
                clip.start_time,
                clip.duration,
                clip.slot_count,
            });
    }

    // Rigid node animation: one animator over the whole document, because the
    // template is one hierarchy holding every dynamic node the default scene
    // reaches. A mesh names the slot inside it that moves the mesh.
    //
    // Init propagates the bind pose, so the world transforms are readable before
    // a clip is ever played and a document with no clips still draws in place.
    var node_animator: ?res.NodeAnimator = if (model.node_animation) |*template|
        try res.NodeAnimator.init(gpa, template)
    else
        null;
    defer if (node_animator) |*animator| animator.deinit(gpa);

    if (model.node_animation) |*template| {
        log.info("rigid animation: {d} slot(s), {d} clip(s)", .{
            template.slotCount(),
            template.clips.len,
        });
        for (template.clips, 0..) |*clip, index| {
            log.info("rigid clip {d}: keyed from {d:.3} to {d:.3} s over {d} slot(s), {d} channel(s)", .{
                index,
                clip.start_time,
                clip.duration,
                clip.slot_count,
                clip.channels.len,
            });
        }
        // The first clip, for the same reason a skin takes its first: which one
        // plays is playback's choice and this app has no way to make it.
        if (template.clips.len > 0) try node_animator.?.play(0);
    }

    // An anchor is a slot inside the template, so geometry cannot carry one
    // unless the importer produced a template to resolve it against. Answered
    // here rather than at the first frame, where a null animator would read as
    // an asset that simply does not move.
    for (model.meshes) |*mesh| {
        if (mesh.anchor != null and node_animator == null) return error.AnchoredMeshWithoutAnimator;
    }

    // Whether playback moves anything a draw is attached to. Three states put
    // the same still picture on the screen and this separates them: a document
    // with nothing to drive, an animator that is never advanced, and an animator
    // that moves slots no geometry is anchored to.
    //
    // The clip is walked and the largest departure from the bind pose is kept
    // per slot. A single sample would not do: a clip loops, so at the end of its
    // span every slot is back where it started, and comparing only there reports
    // a working clip as frozen.
    // The farthest any anchored draw travels from its bind placement, reported
    // against the framed bounds once those exist.
    var anchored_travel: f32 = 0;

    if (node_animator) |*animator| {
        if (animator.active_clip) |clip_index| {
            const span = model.node_animation.?.clips[clip_index].loopSpan();
            const bind = try gpa.dupe(zm.Mat, animator.world_transforms);
            defer gpa.free(bind);
            const slot_drift = try gpa.alloc(f32, animator.world_transforms.len);
            defer gpa.free(slot_drift);
            @memset(slot_drift, 0);
            // How far a slot's origin travels from where the bind pose put it,
            // in world units. Separate from the drift above, which is a maximum
            // over every matrix element and mixes rotation with translation, so
            // it is not a distance and cannot be compared to anything.
            const slot_travel = try gpa.alloc(f32, animator.world_transforms.len);
            defer gpa.free(slot_travel);
            @memset(slot_travel, 0);

            const steps = 16;
            for (0..steps) |_| {
                animator.update(span / @as(f32, steps));
                for (
                    bind,
                    animator.world_transforms,
                    slot_drift,
                    slot_travel,
                ) |before, after, *drift, *travel| {
                    drift.* = @max(drift.*, matrixDrift(after, before));
                    // Row three is the translation in the row vector convention
                    // this project composes in.
                    const offset = after[3] - before[3];
                    travel.* = @max(travel.*, @sqrt(@reduce(.Add, offset * offset)));
                }
            }

            var worst: f32 = 0;
            for (slot_drift) |drift| worst = @max(worst, drift);
            log.info("rigid animation: the clip moves a slot by at most {d:.6} over {d:.3} s", .{
                worst,
                span,
            });
            check(span <= 0 or worst > 1e-6, "playing the clip moves at least one slot", .{});

            // An animator whose moving slots carry no geometry draws exactly the
            // same picture as no animation at all, and that is the failure this
            // slice exists to rule out. Counted over draws rather than slots,
            // because a draw is what reaches the screen.
            //
            // No draw being anchored is a legitimate state rather than a fault.
            // A skin's joints are document nodes and its clip targets them, so
            // they earn rigid slots too, while the skinned mesh that follows
            // them is never anchored: its motion is in the joint matrices. A
            // purely skinned asset therefore carries a rigid hierarchy that
            // nothing reads.
            var anchored_draws: usize = 0;
            var moved_draws: usize = 0;
            for (model.meshes) |*mesh| {
                const slot = mesh.anchor orelse continue;
                anchored_draws += 1;
                if (slot_drift[slot] > 1e-6) moved_draws += 1;
                anchored_travel = @max(anchored_travel, slot_travel[slot]);
            }
            log.info("rigid animation: {d} of {d} draw(s) anchored, {d} of those move", .{
                anchored_draws,
                model.meshes.len,
                moved_draws,
            });
            if (anchored_draws == 0) {
                log.info(
                    "rigid animation: no draw reads the {d} slot(s); the document animates joints only",
                    .{animator.world_transforms.len},
                );
            }
            check(
                span <= 0 or anchored_draws == 0 or moved_draws > 0,
                "every frame that anchors a draw moves at least one of them",
                .{},
            );

            // Back to the start, so the first frame draws the pose the checks
            // below were taken against.
            try animator.play(clip_index);
        }
    }

    for (model.meshes, 0..) |*mesh, mesh_index| {
        if (!mesh.streams.skinned) continue;

        // glTF 2.0 specification, 3.7.3.3: a skinned mesh primitive has the
        // attributes that skinning reads, and a node referencing it names the
        // skin they index. A document with the stream and no matching skin gives
        // the shader an array to index and nothing to fill it with.
        const source_skin_index = mesh.skin orelse return error.SkinnedMeshWithoutSkin;
        const skin = skinForIndex(skins, source_skin_index) orelse
            return error.SkinnedMeshWithoutSkin;
        const joint_count = skin.pose.jointCount();
        if (joint_count == 0) return error.EmptySkin;

        // The shader has no bound of its own, so answer the one that would read
        // outside the joint array once, before a device is touched.
        var highest: u32 = 0;
        for (mesh.vertices) |vertex| {
            inline for (0..4) |lane| highest = @max(highest, vertex.joints[lane]);
        }
        if (highest >= joint_count) return error.JointIndexOutOfRange;
        check(true, "mesh {d} joint indices are under the skin's {d}", .{ mesh_index, joint_count });

        // glTF 2.0 specification, 3.7.3.3: at the bind pose each joint matrix is
        // the same placement, whatever that placement is. A transpose, a wrong
        // slot or applying the inverse bind on the wrong side breaks this before
        // all three become the same torn mesh on screen.
        const bind_placement = skin.pose.joint_transforms[0];
        var worst_joint: f32 = 0;
        for (skin.pose.joint_transforms) |joint| {
            inline for (0..4) |row| {
                inline for (0..4) |column| {
                    worst_joint = @max(worst_joint, @abs(joint[row][column] - bind_placement[row][column]));
                }
            }
        }
        log.info("bind pose: joints disagree by at most {d:.6}", .{worst_joint});
        check(worst_joint < 1e-4, "every joint carries the same bind placement", .{});

        var worst_vertex_drift: f32 = 0;
        for (mesh.vertices) |vertex| {
            const skinned = skinnedPosition(&skin.pose, vertex);
            const placed = zm.mul(
                zm.f32x4(vertex.position[0], vertex.position[1], vertex.position[2], 1),
                bind_placement,
            );
            inline for (0..3) |axis|
                worst_vertex_drift = @max(worst_vertex_drift, @abs(skinned[axis] - placed[axis]));
        }
        check(worst_vertex_drift < 1e-3, "the bind pose moves every vertex by that placement alone", .{});
        log.info("bind pose: the placement moves the model's height axis to [{d:.3} {d:.3} {d:.3}]", .{
            bind_placement[2][0], bind_placement[2][1], bind_placement[2][2],
        });
    }

    // Where this app puts the model. Identity: static vertices are already in
    // world space, a skinned mesh's bind placement is in its joint matrices, and
    // an anchored mesh's is in its animator slot. It is written down rather than
    // dropped because it is what an instance matrix composes against, and the
    // composition is only correct in one order.
    const model_matrix = zm.identity();

    // Kept per mesh as well as unioned, because the blended run is ordered by
    // each draw's own distance from the eye and recomputing these bounds beside
    // the draw plan would be the same walk over every vertex a second time.
    //
    // These are centres in the space each draw's instance matrix maps from, not
    // in world space. For a static mesh the two are the same. A skinned mesh
    // folds its bind placement in here, because its instance matrix is identity.
    // An anchored mesh does not, because its instance matrix carries the anchor
    // and would otherwise apply it twice.
    const mesh_centres = try gpa.alloc(res.Vec3, model.meshes.len);
    defer gpa.free(mesh_centres);

    var world: res.Aabb = undefined;
    for (model.meshes, mesh_centres, 0..) |*mesh, *centre, index| {
        var placed = localBounds(mesh.vertices);
        if (mesh.streams.skinned) {
            const skin = skinForIndex(skins, mesh.skin.?) orelse return error.SkinnedMeshWithoutSkin;
            placed = scene.worldAabb(placed, skin.pose.joint_transforms[0]);
        }
        centre.* = scene.sphereAroundAabb(placed).centre;

        // The bind pose is what frames the camera. An anchored mesh reaches
        // world space only through its animator slot, so unioning its baked
        // vertices directly would frame on geometry sitting at the anchor's
        // origin instead of where the asset puts it.
        const bind_matrix = instanceMatrix(
            mesh.anchor,
            if (node_animator) |*animator| animator else null,
            model_matrix,
        );
        const mesh_world = scene.worldAabb(placed, bind_matrix);
        world = if (index == 0) mesh_world else scene.unionAabb(world, mesh_world);
    }
    log.info("model bounds: [{d:.3} {d:.3} {d:.3}] to [{d:.3} {d:.3} {d:.3}]", .{
        world.min[0], world.min[1], world.min[2],
        world.max[0], world.max[1], world.max[2],
    });
    check(
        world.min[0] <= world.max[0] and world.min[1] <= world.max[1] and world.min[2] <= world.max[2],
        "bounds are not inverted",
        .{},
    );

    const identity_bounds = scene.worldAabb(world, model_matrix);
    check(
        approxEqual(identity_bounds.min, world.min) and approxEqual(identity_bounds.max, world.max),
        "an identity transform leaves the bounds where they were",
        .{},
    );

    const sphere = scene.sphereAroundAabb(world);
    log.info("bounding sphere: centre [{d:.3} {d:.3} {d:.3}] radius {d:.4}", .{
        sphere.centre[0], sphere.centre[1], sphere.centre[2], sphere.radius,
    });
    check(sphere.radius > 0, "the bounding sphere has a radius", .{});

    // The camera is framed on the bind pose, so an asset whose animation carries
    // a draw further than this sphere plays partly outside the view. That is not
    // reported as a failure, because framing on the swept bounds is a choice the
    // corpus walk has to make and neither reference makes it: both frame at rest.
    // It is reported at all so that a part leaving the window is a number here
    // rather than a surprise on screen.
    if (anchored_travel > 0) {
        log.info("rigid animation: an anchored draw travels {d:.3} from a framed radius of {d:.3}", .{
            anchored_travel,
            sphere.radius,
        });
    }

    timeline.verify = timeline.split();

    // Platform and device.
    var host: platform.Platform = try .init();
    defer host.deinit();
    var window = try host.createWindow(.{ .width = 1280, .height = 720 }, "Lenore validation");
    defer window.deinit();

    // The compositor tells a client its size; nothing here can ask. Without
    // reading those events the swapchain keeps the extent it was created with
    // and the camera's aspect stops matching the window the moment it is
    // resized.
    var input: platform.Input = try .init(
        gpa,
        clock,
        platform.initial_event_capacity,
        platform.max_event_capacity,
    );
    defer input.deinit();
    window.captureInput(&input);

    var context: gpu.Context = try .init(gpa, "Lenore validation", window.nativeHandles());
    defer context.deinit();
    log.info("device: {s}", .{context.deviceName()});

    var swapchain: gpu.Swapchain = try .init(&context, gpa, .{ .width = 1280, .height = 720 }, .fifo);
    defer swapchain.deinit();

    var frames: [frames_in_flight]gpu.Frame = undefined;
    var created: usize = 0;
    defer for (frames[0..created]) |frame| frame.deinit(&context);
    for (&frames) |*frame| {
        frame.* = try .init(&context);
        created += 1;
    }

    var memory: gpu.MemoryAllocator = try .init(&context, gpa, io, .{
        .device_buffer_block_size = 64 << 20,
        .device_image_block_size = 64 << 20,
        .upload_buffer_block_size = 16 << 20,
        .readback_buffer_block_size = 8 << 20,
    });
    defer if (memory.deinit() == .leak) log.err("device memory leaked", .{});

    // One pool for the whole run: the environment, the fallbacks and the scene
    // all draw on it in turn, and it is reclaimed by whichever transfer finishes.
    // Defaults, so what a load costs here is what the module's own numbers cost.
    var staging: gpu.StagingPool = try .init(&context, &memory, gpa, .{});
    defer staging.deinit();

    var setup_pool: gpu.OneShotPool = try .init(&context);
    defer setup_pool.deinit(&context);

    var cache_setup: gpu.Transfer = try .begin(&context, setup_pool.handle, &staging);
    var textures: gpu.TextureCache = try .init(&context, &memory, gpa, &cache_setup);
    defer if (textures.deinit() == .leak) log.err("texture references outstanding", .{});
    try cache_setup.finish();
    cache_setup.deinit();
    timeline.device = timeline.split();

    var storage: gpu.ResourceStorage = .empty;
    defer storage.deinit(gpa);

    var batch = try gpu.UploadBatch.begin(
        gpa,
        &context,
        &memory,
        &storage,
        &textures,
        &staging,
        setup_pool.handle,
    );
    var uploaded: gpu.Uploaded = uploaded: {
        errdefer batch.deinit();

        for (model.meshes) |*mesh| {
            _ = try batch.addMesh(u32, .{
                .vertices = mesh.vertices,
                .indices = mesh.indices,
                .streams = mesh.streams,
                // A mesh with no targets passes null and gets no morph buffer,
                // which is what `MorphPass.register` refuses as NotMorphed.
                .morph = if (mesh.morph) |deltas| .{
                    .positions = deltas.positions,
                    .normals = deltas.normals,
                    .target_count = deltas.target_count,
                } else null,
            });
        }
        var queue: DecodeQueue = try .init(gpa, io, clock, model.images.len);
        defer queue.deinit(gpa);

        // Images first, decoded over the pool and uploaded here. Each is decoded
        // once and its pixels are freed as soon as the last slot that wants them
        // has been recorded, so the host holds at most one window of them.
        //
        // The upload stays on this thread. The transfer owns a command buffer
        // and the staging pool, and nothing about spreading the decode makes
        // either threadsafe.
        const images_started = timeline.clock.now();
        for (model.images, 0..) |*source, image_index| {
            if (work.uses[image_index].count() == 0) continue;

            const bytes = source.bytes orelse return error.ExternalImageNotLoaded;
            const size = try decodedByteSize(bytes);
            decode_stats.source_bytes += bytes.len;

            // Draining the oldest is what makes room, and it is also the upload,
            // so the window doubles as the pipeline's depth.
            while (queue.wouldExceed(size))
                try uploadDecoded(gpa, &queue, &batch, &work, &model, &decode_stats);
            queue.spawn(gpa, image_index, bytes, size);
        }
        while (queue.len > 0)
            try uploadDecoded(gpa, &queue, &batch, &work, &model, &decode_stats);
        decode_stats.wall_nanoseconds = timeline.clock.now() - images_started;
        decode_stats.peak_window_bytes = queue.peak;
        // Materials second. Nothing is decoded or uploaded here: a slot names an
        // image the pass above made resident and the sampler this use wants.
        for (model.materials) |*material| {
            _ = try batch.addTextureSet(.{
                .base_colour = try work.fill(&model, material.textures.base_colour, .base_colour),
                .metallic_roughness = try work.fill(
                    &model,
                    material.textures.metallic_roughness,
                    .metallic_roughness,
                ),
                .normal = try work.fill(&model, material.textures.normal, .normal),
                .emissive = try work.fill(&model, material.textures.emissive, .emissive),
                .occlusion = try work.fill(&model, material.textures.occlusion, .occlusion),
            });
        }
        // What the load cost the staging pool. Read before finishing, which
        // consumes the batch; only a reservation raises the count, so it is
        // already final. A stall is a full wait on the graphics queue, so this
        // is the number that says whether the ceiling is set too low.
        const stalls = batch.transfer.flushes;
        const finished = try batch.finish();
        log.info("staging: {d} block(s), {d} bytes resident, {d} stall(s)", .{
            staging.blockCount(),
            staging.residentBytes(),
            stalls,
        });
        break :uploaded finished;
    };
    defer uploaded.deinit(gpa, &storage, &textures);
    timeline.upload = timeline.split();
    decode_stats.report();
    // The property the image-first order exists for. It is not a statement about
    // this asset: any document whose materials share an image breaks it the
    // moment a pass over materials decodes instead of a pass over images.
    check(
        decode_stats.calls() == decode_stats.distinct(),
        "each of the {d} image(s) the materials name was decoded once",
        .{decode_stats.distinct()},
    );

    check(
        uploaded.meshes.items.len == model.meshes.len and
            uploaded.texture_sets.items.len == model.materials.len,
        "upload retained {d} meshes and {d} material sets",
        .{ uploaded.meshes.items.len, uploaded.texture_sets.items.len },
    );

    // Resolve pointers only after every insertion that can grow storage. No
    // resource is added during the frame loop, so these remain stable until the
    // uploaded residency is released.
    const gpu_meshes = try gpa.alloc(*const gpu.Mesh, model.meshes.len);
    defer gpa.free(gpu_meshes);
    var uploaded_vertices: usize = 0;
    var uploaded_indices: usize = 0;
    var uploaded_bounds_match = true;
    for (model.meshes, uploaded.meshes.items, gpu_meshes) |*source_mesh, handle, *gpu_mesh| {
        const resident = storage.mesh(handle) orelse return error.UploadedMeshMissing;
        gpu_mesh.* = resident;
        uploaded_vertices += resident.vertex_count;
        uploaded_indices += resident.index_count;
        const expected = localBounds(source_mesh.vertices);
        uploaded_bounds_match = uploaded_bounds_match and
            approxEqual(resident.bounds.box.min, expected.min) and
            approxEqual(resident.bounds.box.max, expected.max);
    }
    check(
        uploaded_vertices == total_vertices and uploaded_indices == total_indices,
        "uploaded {d} vertices and {d} indices",
        .{ uploaded_vertices, uploaded_indices },
    );
    check(uploaded_bounds_match, "every uploaded mesh reproduces its source box", .{});

    // The morph prepass, and one registration per morphed mesh. Built even for a
    // model with none: it owns a pipeline and a descriptor pool rather than any
    // per-model state, and a pass created conditionally would be a second code
    // path through the frame loop that the sample corpus barely exercises.
    var morph_pass: gpu.MorphPass = try .init(&context, &memory, gpa, frames_in_flight, morph_capacity);
    defer morph_pass.deinit();

    // Which registration each mesh draws through, by mesh index. Null is a mesh
    // with no shape targets, which is every mesh of most assets.
    const morph_registrations = try gpa.alloc(?u32, model.meshes.len);
    defer gpa.free(morph_registrations);
    @memset(morph_registrations, null);

    // One animator per template, and the mesh reaches its own through the
    // template index the importer resolved. Several meshes may share one: a
    // glTF mesh with two morphed primitives becomes two draws blended by one
    // node's weights.
    const morph_animators = try gpa.alloc(res.MorphAnimator, model.morph_templates.len);
    defer gpa.free(morph_animators);
    var built_animators: usize = 0;
    defer for (morph_animators[0..built_animators]) |*animator| animator.deinit(gpa);
    for (model.morph_templates, morph_animators) |*template, *animator| {
        animator.* = try res.MorphAnimator.init(gpa, template);
        built_animators += 1;
        // The first clip, for the reason a skin and the rigid animator take
        // theirs: which one plays is playback's choice and this app cannot make
        // it. A template with none holds the pose section 3.7.4 gives it.
        if (template.clips.len > 0) try animator.play(0);
    }

    var morphed_meshes: usize = 0;
    for (model.meshes, gpu_meshes, morph_registrations) |*source_mesh, gpu_mesh, *registration| {
        if (source_mesh.morph == null) continue;
        morphed_meshes += 1;
        registration.* = morph_pass.register(&memory, gpu_mesh) catch |err| switch (err) {
            error.MeshCapacityExceeded, error.WeightCapacityExceeded => {
                log.warn("morph: {t}; this mesh draws at its bind shape", .{err});
                continue;
            },
            else => return err,
        };
    }
    // Three states put the same still shape on the screen and this separates
    // them: a document whose weights nothing animates, an animator that was
    // never started, and one that is started and never advanced. The last is the
    // frame loop's and has its own probe there; the middle one is this.
    var playing_templates: usize = 0;
    var animated_templates: usize = 0;
    for (model.morph_templates, morph_animators[0..built_animators]) |*template, *animator| {
        if (template.clips.len > 0) animated_templates += 1;
        if (animator.active_clip != null) playing_templates += 1;
    }
    if (animated_templates > 0) {
        log.info("morph: {d} of {d} template(s) are playing a clip", .{
            playing_templates,
            animated_templates,
        });
        check(
            playing_templates == animated_templates,
            "every morph template with a clip is playing one",
            .{},
        );
    }

    if (morphed_meshes > 0) {
        for (model.morph_templates, 0..) |*template, index| {
            log.info("morph template {d}: {d} target(s), {d} clip(s), defaults {any}", .{
                index,
                template.targetCount(),
                template.clips.len,
                template.defaults,
            });
        }
        log.info("morph: {d} of {d} mesh(es) carry shape targets", .{
            morphed_meshes,
            model.meshes.len,
        });
        check(
            morph_pass.registrationCount() == morphed_meshes,
            "every morphed mesh reached the prepass",
            .{},
        );
    }

    // An importer that resolved a template for a mesh with no deltas, or the
    // reverse, would draw a still shape and report nothing. The two are set in
    // different walks of the document, so this is what holds them together.
    for (model.meshes, 0..) |*mesh, index| {
        check(
            (mesh.morph == null) == (mesh.morph_template == null),
            "mesh {d} carries its deltas and its weights together",
            .{index},
        );
    }

    var samplers: gpu.SamplerCache = .init(&context);
    defer samplers.deinit(gpa);
    const post_sampler = try samplers.get(gpa, .{ .address_mode_u = .clamp_to_edge, .address_mode_v = .clamp_to_edge });

    const extent = swapchain.currentExtent();
    var renderer: gpu.Renderer = try .init(
        &context,
        &memory,
        gpa,
        .{ .width = extent.width, .height = extent.height },
        frames_in_flight,
        frame_capacity,
        @intCast(model.materials.len),
        swapchain.surface_format.format,
        post_sampler,
    );
    defer renderer.deinit();
    for (uploaded.texture_sets.items, 0..) |handle, material_index| {
        const texture_set = storage.textureSet(handle) orelse return error.UploadedTextureSetMissing;
        try renderer.setMaterialTextures(
            @intCast(material_index),
            texture_set,
            model.materials[material_index].rendering.alpha_mode,
        );
    }

    // The packed array every fragment indexes, uploaded once. One buffer for the
    // whole scene rather than one per material: what selects a record is the
    // index the instance carries, not which descriptor set is bound.
    const packed_materials = try gpa.alloc(gpu.MaterialData, model.materials.len);
    defer gpa.free(packed_materials);
    for (model.materials, packed_materials) |*source, *record| record.* = .fromInfo(source);

    var materials: gpu.MaterialStorage = try .init(&context, &memory, @intCast(model.materials.len));
    defer materials.deinit();
    // Nothing has been submitted yet, so the cold-path requirement that no frame
    // be reading the buffer is met by there being no frame.
    try materials.upload(packed_materials);
    renderer.setMaterialBuffer(&materials);
    timeline.renderer = timeline.split();

    // The environment is acquired outside the upload batch, which carries meshes
    // and material texture sets. Not for want of room: the batch is a
    // transaction, and an environment that fails to load should not roll the
    // scene back with it.
    var held_environment: ?[3][]const u8 = null;
    defer if (held_environment) |keys| for (keys) |key| textures.release(key);

    const environment = if (environment_path) |path| loaded_environment: {
        const acquired = try loadEnvironment(
            gpa,
            io,
            &context,
            &staging,
            &textures,
            &setup_pool,
            path,
        );
        held_environment = acquired.keys;
        break :loaded_environment acquired.environment;
    } else neutral: {
        log.info("environment: none given, image-based lighting contributes zero", .{});
        break :neutral try gpu.Environment.neutral(&textures);
    };
    renderer.setEnvironment(environment);
    timeline.environment = timeline.split();

    // A factor-only diagnostic through the host mirror of the fragment shader.
    // Unit samples leave every factor unchanged; texture variation is visible
    // in the rendered image rather than reducible to one material value here.
    const neutral_material_samples: gpu.Shading.MaterialSamples = .{
        .base_colour = @splat(1),
        .metallic_roughness = @splat(1),
    };
    var materials_differ = false;
    const first_surface = gpu.Shading.Surface.fromMaterial(
        packed_materials[0],
        neutral_material_samples,
    );
    for (packed_materials, 0..) |record, material_index| {
        const surface = gpu.Shading.Surface.fromMaterial(record, neutral_material_samples);
        const emitted = gpu.Shading.emissive(record, @splat(1));
        materials_differ = materials_differ or
            !std.meta.eql(surface, first_surface) or
            !std.meta.eql(alphaCoverageKey(record), alphaCoverageKey(packed_materials[0]));
        log.info(
            "material {d}: base [{d:.3} {d:.3} {d:.3}], metallic {d:.3}, roughness {d:.3}, emissive factor [{d:.3} {d:.3} {d:.3}]",
            .{
                material_index,
                surface.base_colour[0],
                surface.base_colour[1],
                surface.base_colour[2],
                surface.metallic,
                surface.roughness,
                emitted[0],
                emitted[1],
                emitted[2],
            },
        );
    }

    // Whether this asset can show that the index selects anything. Materials
    // that all pack to the same factors are drawn identically however the
    // indexing behaves, so a picture from such a model proves nothing about it
    // and the run should say so rather than pass quietly.
    if (model.materials.len > 1) {
        check(
            materials_differ,
            "the {d} materials differ in their factors, so the index is observable",
            .{model.materials.len},
        );
    }

    const post_settings: gpu.PostSettings = .{};

    // The visible field carries HDR values in every channel. A copied post
    // leaves them above display range; the selected operator maps them to the
    // bounded blue field printed beside the source.
    renderer.clear_colour = .{ 1.25, 2, 3, 1 };
    const mapped_clear = try gpu.toneMap(.{
        renderer.clear_colour[0],
        renderer.clear_colour[1],
        renderer.clear_colour[2],
    }, post_settings);
    log.info(
        "diagnostic: HDR clear [{d:.2} {d:.2} {d:.2}] maps to [{d:.3} {d:.3} {d:.3}] at exposure {d:.2}",
        .{
            renderer.clear_colour[0],
            renderer.clear_colour[1],
            renderer.clear_colour[2],
            mapped_clear[0],
            mapped_clear[1],
            mapped_clear[2],
            post_settings.exposure,
        },
    );
    check(
        renderer.clear_colour[0] > 1 and
            renderer.clear_colour[1] > 1 and
            renderer.clear_colour[2] > 1 and
            mapped_clear[0] < mapped_clear[1] and
            mapped_clear[1] < mapped_clear[2] and
            mapped_clear[2] < 1,
        "the diagnostic separates copied HDR from bounded tonemapping",
        .{},
    );

    log.info("attachments: hdr {t}, depth {t}, present {t}", .{
        renderer.hdr.format,
        renderer.depth.format,
        swapchain.surface_format.format,
    });
    log.info("ring strides: camera {d} B, instances {d} B, lights {d} B", .{
        renderer.frame.camera.stride,
        renderer.frame.instances.stride,
        renderer.frame.lights.stride,
    });
    for (0..frames_in_flight) |index| {
        log.info("frame {d}: dynamic offsets {any}", .{ index, renderer.frame.dynamicOffsets(index) });
    }
    check(
        renderer.frame.dynamicOffsets(0)[0] != renderer.frame.dynamicOffsets(1)[0],
        "the two frames address different slots",
        .{},
    );

    // The camera, placed so the whole model fits. Everything below is host-side
    // and is the answer the device is about to be given.
    var camera: scene.Camera = .{
        .anchor = .{ .orbit = .{ .target = @splat(0), .distance = 0.01 } },
        .yaw = -std.math.pi / 4.0,
        .pitch = -0.35,
    };
    const aspect = @as(f32, @floatFromInt(extent.width)) / @as(f32, @floatFromInt(extent.height));
    const distance = frameCamera(&camera, sphere, aspect);
    const view_projection = try camera.viewProjection(aspect);
    const placement = camera.placement();
    log.info("camera: eye [{d:.3} {d:.3} {d:.3}] distance {d:.3} aspect {d:.4}", .{
        placement.position[0], placement.position[1], placement.position[2], distance, aspect,
    });

    // The one check a black window cannot hide behind: where a corner of the
    // model lands in clip space, computed here with the same matrix the shader
    // is handed. Every visible corner has |x| and |y| within w and z in [0, w].
    var corners_in_view: u32 = 0;
    for (aabbCorners(world)) |corner| {
        const clip = zm.mul(zm.f32x4(corner[0], corner[1], corner[2], 1), view_projection);
        if (clip[3] > 0 and
            @abs(clip[0]) <= clip[3] and
            @abs(clip[1]) <= clip[3] and
            clip[2] >= 0 and clip[2] <= clip[3]) corners_in_view += 1;
    }
    log.info("clip space: {d} of 8 bounding corners inside the view volume", .{corners_in_view});
    check(corners_in_view == 8, "the whole model is in front of the camera", .{});

    // Where those corners land after the perspective divide, as a box in
    // normalized device coordinates. A cube seen from a corner covers roughly
    // as much of one axis as the other, so a box far from square here is the
    // projection distorting it, and a square box with a distorted picture puts
    // the fault after the projection, in the viewport or the target.
    reportNdcExtent(view_projection, world, aspect, extent);

    // Which way up the picture comes out, answered in pixels rather than by
    // eye. The camera produces clip space with Y up and a framebuffer counts
    // rows downward, so with the flip unplaced the model's highest point lands
    // in the lower half of the window. Nothing in an unlit white silhouette
    // shows that; this does.
    reportUpAxis(gpu.vulkanClip(view_projection), world, extent);

    const frustum: scene.Frustum = .fromViewProj(view_projection);
    check(frustum.intersectsAabb(world), "the frustum accepts the model's bounds", .{});

    // Where this frame's instances put their joints. Planned once, because the
    // draw list does not change here; the plan is a pure function of the poses
    // being drawn, so replanning every frame would return the same answer.
    const poses = try gpa.alloc(?*const res.SkeletonPose, model.meshes.len);
    defer gpa.free(poses);
    for (model.meshes, poses) |*mesh, *pose| {
        pose.* = if (mesh.streams.skinned)
            &(skinForIndex(skins, mesh.skin.?) orelse return error.SkinnedMeshWithoutSkin).pose
        else
            null;
    }
    const joint_bases = try gpa.alloc(u32, model.meshes.len);
    defer gpa.free(joint_bases);
    const joint_total = try scene.assignJointOffsets(poses, joint_bases, @intCast(frame_capacity.joints));
    check(
        joint_total <= frame_capacity.joints,
        "the frame's {d} joint(s) fit an array of {d}",
        .{ joint_total, frame_capacity.joints },
    );

    const instances = try gpa.alloc(gpu.Instance, model.meshes.len);
    defer gpa.free(instances);
    const draw_candidates = try gpa.alloc(RecordPlan.Draw, model.meshes.len);
    defer gpa.free(draw_candidates);
    const draw_order = try gpa.alloc(u32, model.meshes.len);
    defer gpa.free(draw_order);
    const draw_keys = try gpa.alloc(scene.DrawKey, model.meshes.len);
    defer gpa.free(draw_keys);
    const draw_matrices = try gpa.alloc(zm.Mat, model.meshes.len);
    defer gpa.free(draw_matrices);
    for (model.meshes, gpu_meshes, draw_candidates, draw_keys) |
        *source_mesh,
        gpu_mesh,
        *candidate,
        *key,
    | {
        candidate.* = .{
            .mesh = gpu_mesh,
            .material = source_mesh.material,
            // Imported static geometry may combine nodes whose baked transforms
            // have opposite determinant signs, and that sign is not retained in
            // the mesh product. Skinning has the same problem per deformation.
            // No culling is the only conservative state until scene input can
            // prove one winding for the whole batch.
            .face_culling = .none,
        };

        // The layer is the material's and never changes. The depth is the
        // draw's distance from the eye and the plan rewrites it every frame.
        key.* = .{
            .layer = switch (model.materials[source_mesh.material].rendering.alpha_mode) {
                .@"opaque", .mask => .solid,
                .blend => .blended,
            },
            .depth = undefined,
        };
    }

    const scene_batch_storage = try gpa.alloc(RecordPlan.Batch, model.meshes.len);
    defer gpa.free(scene_batch_storage);
    const record_batch_storage = try gpa.alloc(gpu.RecordBatch, model.meshes.len);
    defer gpa.free(record_batch_storage);
    const vertex_sources = try gpa.alloc(?gpu.MeshVertexSource, model.meshes.len);
    defer gpa.free(vertex_sources);
    // Frame zero, so the checks below read the plan the first frame records.
    fillVertexSources(&morph_pass, morph_registrations, 0, vertex_sources);

    var draw_plan: DrawPlan = .{
        .candidates = draw_candidates,
        .centres = mesh_centres,
        .joint_bases = joint_bases,
        .matrices = draw_matrices,
        .keys = draw_keys,
        .order = draw_order,
        .instances = instances,
        .batch_storage = scene_batch_storage,
        .record_storage = record_batch_storage,
        .vertex_sources = vertex_sources,
    };

    // Once here, so the checks below read a real plan, and again every frame.
    // The camera this app builds never moves, but a rigid animator does, and a
    // blended draw hanging off an animated node changes places with its
    // neighbours while the eye stays put.
    try draw_plan.rebuild(
        model.meshes,
        if (node_animator) |*animator| animator else null,
        model_matrix,
        placement.position,
    );

    // Whether advancing the animator reaches the instance records the device
    // reads. Everything above this measures the animator's own arrays; this is
    // the only thing that follows a slot through the plan and out into what is
    // uploaded, and it is what a plan built once and never rebuilt fails.
    //
    // Two poses rather than one value, because the composition itself is written
    // in exactly one place and a check that recomputed it would agree with a
    // wrong definition as readily as a right one.
    if (node_animator) |*animator| {
        if (animator.active_clip) |clip_index| {
            const span = model.node_animation.?.clips[clip_index].loopSpan();
            const at_rest = try gpa.dupe(zm.Mat, draw_plan.matrices);
            defer gpa.free(at_rest);

            // One advance and one rebuild for the whole model, not one per
            // anchored draw: the plan is a function of the animator's pose, so
            // rebuilding it inside the walk would repeat the same answer once
            // per mesh.
            animator.update(span * 0.25);
            try draw_plan.rebuild(model.meshes, animator, model_matrix, placement.position);

            var anchored: usize = 0;
            var changed: usize = 0;
            for (model.meshes, at_rest, draw_plan.matrices) |*mesh, before, after| {
                if (mesh.anchor == null) continue;
                anchored += 1;
                if (matrixDrift(after, before) > 1e-6) changed += 1;
            }

            // Back to the pose the frame loop starts from.
            try animator.play(clip_index);
            try draw_plan.rebuild(model.meshes, animator, model_matrix, placement.position);

            if (anchored > 0) {
                log.info("rigid animation: {d} of {d} anchored draw(s) change their instance matrix", .{
                    changed,
                    anchored,
                });
                check(span <= 0 or changed > 0, "advancing the clip rewrites an instance matrix", .{});
            }
        }
    }

    const ordered_draws = draw_plan.ordered;

    var blended_draws: usize = 0;
    for (draw_keys) |key| {
        if (key.layer == .blended) blended_draws += 1;
    }
    log.info("draw order: {d} solid then {d} blended, back to front", .{
        ordered_draws.len - blended_draws,
        blended_draws,
    });

    // The host's own reading of the invariant the recorder refuses to draw
    // without. Saying it here names which side is wrong when the two disagree.
    var partitioned = true;
    var farthest_first = true;
    var seen_blended = false;
    var previous_depth: f32 = std.math.inf(f32);
    for (ordered_draws) |draw_index| {
        const key = draw_keys[draw_index];
        switch (key.layer) {
            .solid => if (seen_blended) {
                partitioned = false;
            },
            .blended => {
                if (seen_blended and key.depth > previous_depth) farthest_first = false;
                previous_depth = key.depth;
                seen_blended = true;
            },
        }
    }
    check(partitioned, "every blended draw follows every solid one", .{});
    check(farthest_first, "the blended run descends in distance from the eye", .{});

    const scene_batches = draw_plan.batches;
    const record_batches = draw_plan.records;

    var translation_matches = true;
    var has_non_zero_first_instance = false;
    // A batch names a material, and so does every instance record it draws. The
    // two are written from different walks, one over the ordered draws and one
    // over the coalesced runs, and nothing but this compares them. When they
    // disagree the picture is plausible and wrong: each draw shades with another
    // draw's material.
    var instances_match_batches = true;
    for (scene_batches, record_batches, 0..) |planned, record, index| {
        for (instances[planned.first_instance..][0..planned.instance_count]) |instance| {
            if (instance.material_index != planned.material) instances_match_batches = false;
        }
        const culling_matches = switch (planned.face_culling) {
            .none => !record.cull_mode.front_bit and !record.cull_mode.back_bit,
            .back => record.cull_mode.back_bit and !record.cull_mode.front_bit,
            .front => record.cull_mode.front_bit and !record.cull_mode.back_bit,
        };
        translation_matches = translation_matches and
            record.mesh == planned.mesh and
            record.material_index == planned.material and
            record.first_instance == planned.first_instance and
            record.instance_count == planned.instance_count and
            culling_matches;
        has_non_zero_first_instance = has_non_zero_first_instance or record.first_instance > 0;
        log.info("batch {d}: material {d}, instances [{d}..{d}), culling {t}", .{
            index,
            planned.material,
            planned.first_instance,
            planned.first_instance + planned.instance_count,
            planned.face_culling,
        });
    }
    check(translation_matches, "scene batches survive the explicit GPU translation", .{});
    check(instances_match_batches, "each batch's instance records name the batch's material", .{});
    if (record_batches.len > 1) {
        check(
            has_non_zero_first_instance,
            "a multi-batch frame records a non-zero firstInstance",
            .{},
        );
    }

    // The instance records are what the vertex stage reads the index out of, and
    // in this scene every model matrix is the same. Distinct indices over equal
    // matrices are what make the picture answer which record a draw selected:
    // with one matrix the geometry cannot show it, and the material can.
    var distinct_instance_materials = false;
    for (instances[1..]) |instance| {
        if (instance.material_index != instances[0].material_index) distinct_instance_materials = true;
    }
    if (model.materials.len > 1) {
        check(
            distinct_instance_materials,
            "the instance records carry more than one material index",
            .{},
        );
    }

    var joint_storage: [frame_capacity.joints]gpu.Joint = undefined;

    timeline.prepare = timeline.split();
    log.info(
        "load: parse {d:.3} s, verify {d:.3} s, device {d:.3} s, upload {d:.3} s, renderer {d:.3} s, environment {d:.3} s, prepare {d:.3} s, total {d:.3} s",
        .{
            seconds(timeline.parse),
            seconds(timeline.verify),
            seconds(timeline.device),
            seconds(timeline.upload),
            seconds(timeline.renderer),
            seconds(timeline.environment),
            seconds(timeline.prepare),
            seconds(timeline.total()),
        },
    );
    // Wall, not the pool's summed CPU: the question this answers is how much of
    // the phase the image pass occupies, and the summed figure exceeds the phase
    // it sits in as soon as more than one thread decodes.
    log.info("load: decoding is {d:.3} s of the {d:.3} s upload phase", .{
        seconds(decode_stats.wall_nanoseconds),
        seconds(timeline.upload),
    });

    // Where playback starts. The clock is monotonic from its own init, so this
    // is the first frame's zero rather than the process's.
    const started = clock.now();

    // The lights the frame is drawn with. The asset's own if it has any, and
    // otherwise one key light, because an unlit picture and a black one are the
    // same window and the point of this app is to tell them apart.
    var lights: [gpu.max_lights]gpu.LightUniform = undefined;
    const live = try resolveLights(model.lights, &lights);
    check(
        model.lights.len <= gpu.max_lights,
        "the model's {d} light(s) fit a block of {d}; drawing with {d}",
        .{ model.lights.len, gpu.max_lights, live.len },
    );
    for (live) |light| {
        log.info("light: {t} colour [{d:.2} {d:.2} {d:.2}] intensity {d:.3}", .{
            light.kind, light.colour[0], light.colour[1], light.colour[2], light.intensity,
        });
    }

    // Whether anything is going to be lit, answered on the host from the
    // normals the device is about to be handed. A black picture with every
    // vertex facing away is a scene, not a fault; a black picture with most of
    // them facing the light is one.
    reportLitFraction(model.meshes, live);

    // The normal as the device will read it, not as the asset stated it. It is
    // packed 10-10-10-2 on upload and expanded by the vertex input stage.
    reportPackedNormals(model.meshes);

    log.info("checks before the first frame: {d} failed", .{failures});
    log.info("validation errors before the first frame: {d}", .{gpu.validationErrorCount()});

    // The frame loop.
    var frame_index: usize = 0;
    var surface_extent = swapchain.currentExtent();
    var stale = false;
    var presented_frames: u64 = 0;
    var reported_validation: u32 = 0;
    // The rigid animator advances by a delta rather than being placed at an
    // absolute time, and this is what turns the one into the other. It keeps
    // more precision than the skins' absolute elapsed, not less: the animator
    // wraps its accumulator to the clip's span, so it never grows large enough
    // for an f32 to lose the frame's step.
    var previous_elapsed: f32 = 0;

    // Everything else about rigid animation is answered before the first frame,
    // and none of it can see the frame loop. A loop that never advances the
    // animator, and one that builds the draw plan once instead of per frame,
    // both leave every load-time check passing and put a still picture on the
    // screen. This is the only thing that separates them, so it lives here.
    //
    // Deliberately a quarter of the clip rather than a frame count: how many
    // frames that takes depends on the display, and a clip several seconds long
    // has barely started after sixty of them.
    const motion_probe: ?usize = for (model.meshes, 0..) |*mesh, index| {
        if (mesh.anchor != null) break index;
    } else null;
    const probe_span = if (node_animator) |*animator|
        if (animator.active_clip) |clip| model.node_animation.?.clips[clip].loopSpan() else 0
    else
        0;
    var probe_start: ?struct { elapsed: f32, matrix: zm.Mat } = null;
    var probe_reported = false;

    // The same question for the weights, and it needs its own probe for the same
    // reason: every morph check above runs before the first frame, so a loop
    // that never advances a weight animator, or one that writes the previous
    // frame's slot, leaves all of them green and puts a still shape on screen.
    //
    // The weights and not the vertices: what the prepass produced is on the
    // device and nothing here reads it back. This separates a blend that is not
    // being driven from one whose result never arrives, and only the first of
    // those is a host defect.
    const weight_probe: ?usize = for (morph_animators[0..built_animators], 0..) |*animator, index| {
        if (animator.active_clip != null) break index;
    } else null;
    const weight_probe_span = if (weight_probe) |index|
        model.morph_templates[index].clips[morph_animators[index].active_clip.?].loopSpan()
    else
        0;
    var weight_start: ?struct { elapsed: f32, weights: [8]f32, count: usize } = null;
    var weight_reported = false;

    while (!window.shouldClose()) {
        host.pollEvents();
        if (input.takeBatch()) |events| {
            defer input.releaseBatch();
            for (events) |event| {
                if (event.payload == .surface_metrics)
                    surface_extent = event.payload.surface_metrics.framebuffer_extent;
            }
        } else |err| switch (err) {
            error.InputEventOverflow => log.warn("input overflowed; batch discarded", .{}),
        }

        if (surface_extent.width == 0 or surface_extent.height == 0) {
            host.waitEvents();
            continue;
        }

        if (stale or !swapchain.matchesExtent(surface_extent)) {
            try context.waitIdle();
            try swapchain.recreate(surface_extent);
            const size = swapchain.currentExtent();
            try renderer.resize(.{ .width = size.width, .height = size.height });
            stale = false;
            const resized_aspect = @as(f32, @floatFromInt(size.width)) / @as(f32, @floatFromInt(size.height));
            _ = frameCamera(&camera, sphere, resized_aspect);
            const target = renderer.targetExtent();
            log.info("resized: swapchain {d}x{d}, main pass target {d}x{d}", .{
                size.width, size.height, target.width, target.height,
            });
            check(
                target.width == size.width and target.height == size.height,
                "the main pass target follows the swapchain",
                .{},
            );
            reportNdcExtent(try camera.viewProjection(resized_aspect), world, resized_aspect, size);
            reportUpAxis(gpu.vulkanClip(try camera.viewProjection(resized_aspect)), world, size);
        }

        const frame = frames[frame_index];
        try frame.waitForGpu(&context);

        const acquired = swapchain.acquireNextImage(frame.image_acquired) catch |err| switch (err) {
            error.OutOfDateKHR => {
                stale = true;
                continue;
            },
            else => return err,
        };
        if (acquired.state == .suboptimal) stale = true;

        const current = swapchain.currentExtent();
        const frame_aspect = @as(f32, @floatFromInt(current.width)) / @as(f32, @floatFromInt(current.height));
        const frame_view_projection = try camera.viewProjection(frame_aspect);
        const eye = camera.placement().position;

        // Sampling writes each clip's channels over its pose's locals and
        // leaves untargeted slots at their bind value. Evaluate each unique pose
        // once, then copy it into every draw-order run the scene plan assigned.
        const elapsed = @as(f32, @floatFromInt(clock.now() - started)) / std.time.ns_per_s;

        // Rigid node animation first: the draw plan below reads the world
        // transforms this writes, and the bounds, the depth key and the instance
        // record all come out of that one set.
        if (node_animator) |*animator| animator.update(elapsed - previous_elapsed);
        // Morph weights follow the same delta. They reach the device through the
        // prepass rather than through an instance record, so nothing between
        // here and the dispatch reads them.
        for (morph_animators[0..built_animators]) |*animator|
            animator.update(elapsed - previous_elapsed);
        previous_elapsed = elapsed;

        // Which slot the prepass writes and the draws read. It moves with the
        // frame index, so the plan below has to be rebuilt against this frame's
        // and not against the previous one's.
        fillVertexSources(&morph_pass, morph_registrations, frame_index, vertex_sources);
        for (model.meshes, morph_registrations) |*source_mesh, registration| {
            const index = registration orelse continue;
            const template = source_mesh.morph_template orelse continue;
            try morph_pass.writeWeights(frame_index, index, morph_animators[template].weights);
        }

        for (skins) |*skin| {
            if (skin.clip) |clip| {
                try clip.sample(
                    clip.cursorAt(elapsed),
                    skin.pose.local_translations,
                    skin.pose.local_rotations,
                    skin.pose.local_scales,
                );
            }
            skin.pose.evaluate();
        }
        for (poses, joint_bases) |pose, joint_base| {
            const live_pose = pose orelse continue;
            const first: usize = @intCast(joint_base);
            @memcpy(joint_storage[first..][0..live_pose.joint_transforms.len], live_pose.joint_transforms);
        }
        const joint_matrices = joint_storage[0..joint_total];

        // Where every draw is this frame. It follows the animator and the eye,
        // and it produces the batches recorded below, so it has to run after the
        // animator and before anything reads an instance record.
        try draw_plan.rebuild(
            model.meshes,
            if (node_animator) |*animator| animator else null,
            model_matrix,
            eye,
        );

        // Does the picture actually change? Read off the plan the device is
        // about to be handed, not off the animator, so that a break anywhere
        // between the two shows up here.
        // A clip with no span holds one pose by definition, so there is nothing
        // for this to separate and it would only report a still draw as broken.
        if (if (probe_span > 0) motion_probe else null) |probe| {
            if (probe_start) |start| {
                if (!probe_reported and elapsed - start.elapsed > probe_span * 0.25) {
                    const moved = matrixDrift(draw_plan.matrices[probe], start.matrix);
                    log.info("rigid animation: after {d:.3} s of frames an anchored draw moved by {d:.6}", .{
                        elapsed - start.elapsed,
                        moved,
                    });
                    check(moved > 1e-6, "the frame loop keeps the anchored draw moving", .{});
                    probe_reported = true;
                }
            } else {
                probe_start = .{ .elapsed = elapsed, .matrix = draw_plan.matrices[probe] };
            }
        }

        // Read off the animator the prepass is about to be handed, after the
        // update and before the write, which is the only window where the two
        // can be told apart.
        if (if (weight_probe_span > 0) weight_probe else null) |probe| {
            const live_weights = morph_animators[probe].weights;
            const carried = @min(live_weights.len, 8);
            if (weight_start) |start| {
                if (!weight_reported and elapsed - start.elapsed > weight_probe_span * 0.25) {
                    var moved: f32 = 0;
                    for (live_weights[0..start.count], start.weights[0..start.count]) |now, before|
                        moved = @max(moved, @abs(now - before));
                    log.info("morph: after {d:.3} s of frames a weight moved by {d:.6}", .{
                        elapsed - start.elapsed,
                        moved,
                    });
                    check(moved > 1e-6, "the frame loop keeps the blend moving", .{});
                    weight_reported = true;
                }
            } else {
                var snapshot: [8]f32 = @splat(0);
                @memcpy(snapshot[0..carried], live_weights[0..carried]);
                weight_start = .{ .elapsed = elapsed, .weights = snapshot, .count = carried };
            }
        }

        try renderer.update(frame_index, .{
            .camera = .{
                .view_projection = frame_view_projection,
                .position = .{ eye[0], eye[1], eye[2], 1 },
            },
            .models = draw_plan.instances,
            .joints = joint_matrices,
            .lights = live,
        });

        const commands = try frame.beginCommands(&context);
        // Before the main pass. `record` below opens and closes the rendering
        // itself, so this is the point the dispatches have, and the barrier the
        // prepass ends with is what orders their writes against the vertex
        // fetch of the draws that follow.
        morph_pass.record(commands, frame_index);
        try renderer.record(commands, frame_index, .{
            .image = swapchain.images[acquired.image_index].image,
            .view = swapchain.images[acquired.image_index].view,
            .extent = .{ .width = current.width, .height = current.height },
        }, draw_plan.records, post_settings);
        try frame.submit(&context, .{
            .wait = frame.image_acquired,
            // The first thing the frame does to the presentable image is render
            // into it, which the post pass's barrier prepares.
            .wait_stage = .{ .color_attachment_output_bit = true },
            .signal = try swapchain.renderFinishedSemaphore(acquired.image_index),
            // After everything, including the transition to the presentable
            // layout. A tighter stage would signal before it completed.
            .signal_stage = .{ .all_commands_bit = true },
        });

        const state = swapchain.present(acquired.image_index) catch |err| switch (err) {
            error.OutOfDateKHR => {
                stale = true;
                continue;
            },
            else => return err,
        };
        if (state == .suboptimal) stale = true;

        frame_index = (frame_index + 1) % frames_in_flight;
        presented_frames += 1;

        // Reported as it happens rather than at the end: a validation error on
        // the first frame repeats every frame afterwards, and knowing it began
        // at frame one is most of the diagnosis.
        const errors = gpu.validationErrorCount();
        if (errors != reported_validation) {
            log.err("validation errors after frame {d}: {d}", .{ presented_frames, errors });
            reported_validation = errors;
        }
        if (presented_frames == 1) log.info("first frame presented", .{});
    }

    try context.waitIdle();
    log.info("presented {d} frames", .{presented_frames});
    log.info("checks failed: {d}", .{failures});
    log.info("validation errors: {d}", .{gpu.validationErrorCount()});
    if (failures > 0 or gpu.validationErrorCount() > 0) return error.ValidationRunFailed;
}

// The key light used when a document carries none, which is most of them.
//
// The direction is fixed rather than derived from the camera, so the same asset
// shades the same way on every run and a change to the picture is a change to
// the code. It travels down and away from where the camera is placed, which puts
// the lit side toward the viewer.
//
// Intensity is pi because the BRDF's diffuse lobe divides by pi. A white
// dielectric facing this light head-on returns exactly 0.97: 0.96 of diffuse,
// which is what the Fresnel term leaves of a white base at normal incidence,
// and 0.01 of specular. So a fallback texture still comes out very near white
// and the shading is visible in how it falls off rather than in its peak.
const key_light_direction = res.Vec3{ 0.5, -0.8, -0.6 };
const key_light_intensity = std.math.pi;

// The lights a frame is drawn with, written into `out` and returned as the
// prefix of it that holds one.
//
// Every light goes through `lenore-scene` rather than being packed straight from
// the document: that is where a direction is normalized, a cone is ordered and a
// range is derived for an asset that states none, and it is the only place that
// rejects a light instead of drawing a wrong one.
fn resolveLights(
    document_lights: []const gltf.importer.Light,
    out: *[gpu.max_lights]gpu.LightUniform,
) !([]const gpu.LightUniform) {
    if (document_lights.len == 0) {
        // Through the same constructor as an asset's, so the direction is unit
        // by the same code rather than by the constant above being written
        // carefully.
        out[0] = pack(try .directional(.{ 1, 1, 1 }, key_light_intensity, key_light_direction));
        return out[0..1];
    }

    var count: usize = 0;
    for (document_lights) |entry| {
        if (count == out.len) break;
        const placed = sceneLight(entry) catch |err| {
            // A light a document got wrong is one light, not the frame. The
            // rest still draw, and the run says which one was dropped.
            log.warn("light {d} rejected: {t}", .{ count, err });
            continue;
        };
        out[count] = pack(placed);
        count += 1;
    }
    return out[0..count];
}

// One document light, placed by the transform of the node that references it.
//
// KHR_lights_punctual, "Directional" and "Spot": a light emits along the local
// -Z axis of its node. That is the direction handed to the constructor, and the
// node's world matrix turns it into where the light actually points.
fn sceneLight(entry: gltf.importer.Light) scene.LightError!scene.Light {
    const source = entry.source;
    const local_axis = res.Vec3{ 0, 0, -1 };
    const colour = res.Vec3{ source.color[0], source.color[1], source.color[2] };
    // The extension leaves `range` optional and means infinite by it. The
    // shader has no infinite case, so the distance where the falloff is spent
    // stands in for it, which is what `rangeFor` computes.
    const range = source.range orelse scene.Light.rangeFor(source.intensity);

    const unplaced: scene.Light = switch (source.kind) {
        .directional => try .directional(colour, source.intensity, local_axis),
        .point => try .point(colour, source.intensity, .{ 0, 0, 0 }, range),
        .spot => try .spot(colour, source.intensity, .{
            .position = .{ 0, 0, 0 },
            .direction = local_axis,
            .range = range,
            .inner_angle = if (source.spot) |cone| cone.inner_angle else 0,
            .outer_angle = if (source.spot) |cone| cone.outer_angle else std.math.pi / 4.0,
        }),
    };
    return unplaced.placed(entry.world);
}

// A scene light in the layout the shader reads. The two are separate types
// because `lenore-gpu` does not know `lenore-scene`, which is what keeps a
// device layout out of the module that has no device in it.
fn pack(light: scene.Light) gpu.LightUniform {
    const colour = [3]f32{ light.colour[0], light.colour[1], light.colour[2] };
    return switch (light.kind) {
        .directional => |direction| .directional(colour, light.intensity, .{
            direction[0], direction[1], direction[2],
        }),
        .point => |source| .point(colour, light.intensity, .{
            source.position[0], source.position[1], source.position[2],
        }, source.range),
        .spot => |source| .spot(colour, light.intensity, .{
            .position = .{ source.position[0], source.position[1], source.position[2] },
            .direction = .{ source.direction[0], source.direction[1], source.direction[2] },
            .range = source.range,
            .cos_inner = source.cos_inner,
            .cos_outer = source.cos_outer,
        }),
    };
}

// How much of the model faces a light, counted the way the fragment shader
// decides it: a vertex is lit when its normal has a positive dot with the
// direction toward some light.
//
// Vertex normals rather than fragments, so this is an estimate. It answers the
// only question a black window raises here, which is whether the geometry is
// pointing at the light at all.
fn reportLitFraction(meshes: []const gltf.importer.Mesh, lights: []const gpu.LightUniform) void {
    var lit: usize = 0;
    var total: usize = 0;
    for (meshes) |mesh| {
        total += mesh.vertices.len;
        for (mesh.vertices) |vertex| {
            for (lights) |light| {
                const to_light: res.Vec3 = switch (light.kind) {
                    .directional => .{ -light.direction[0], -light.direction[1], -light.direction[2] },
                    .point, .spot => .{
                        light.position[0] - vertex.position[0],
                        light.position[1] - vertex.position[1],
                        light.position[2] - vertex.position[2],
                    },
                };
                if (@reduce(.Add, vertex.normal * to_light) > 0) {
                    lit += 1;
                    break;
                }
            }
        }
    }
    log.info("lighting: {d} of {d} vertices face a light", .{ lit, total });
    check(lit > 0, "some of the model faces a light", .{});
}

// The largest distance between an authored normal and the same normal after the
// round trip through the packed vertex, over every mesh the frame uploads.
//
// Vulkan specification, Fixed-Point Data Conversion: a 10-bit snorm reads back
// as max(c / 511, -1), so a component resolves to about 1/511 and the worst a
// direction can move is a little over that.
fn reportPackedNormals(meshes: []const gltf.importer.Mesh) void {
    var worst: f32 = 0;
    var worst_mesh: usize = 0;
    var worst_at: usize = 0;
    for (meshes, 0..) |mesh, mesh_index| {
        for (mesh.vertices, 0..) |vertex, index| {
            // Through `packVertex`, which is what the upload calls. Packing the
            // components directly would test the arithmetic and not the path.
            const word = gpu.packVertex(&vertex).normal;
            const unpacked = res.Vec3{
                unpackSnorm10(word),
                unpackSnorm10(word >> 10),
                unpackSnorm10(word >> 20),
            };
            const drift = @sqrt(@reduce(.Add, (unpacked - vertex.normal) * (unpacked - vertex.normal)));
            if (drift > worst) {
                worst = drift;
                worst_mesh = mesh_index;
                worst_at = index;
            }
        }
    }

    const first = &meshes[0].vertices[0];
    const worst_source = meshes[worst_mesh].vertices[worst_at];
    log.info("normals: first source [{d:.3} {d:.3} {d:.3}] packs to 0x{X:0>8} at byte {d} of {d}", .{
        first.normal[0],
        first.normal[1],
        first.normal[2],
        gpu.packVertex(first).normal,
        @offsetOf(gpu.GpuVertex, "normal"),
        @sizeOf(gpu.GpuVertex),
    });
    log.info("normals: worst packing drift {d:.5} at mesh {d} vertex {d}, source [{d:.3} {d:.3} {d:.3}]", .{
        worst,
        worst_mesh,
        worst_at,
        worst_source.normal[0],
        worst_source.normal[1],
        worst_source.normal[2],
    });
    // Three components each within one step of 1/511, so the bound is the
    // length of that: sqrt(3)/511.
    check(worst < @sqrt(3.0) / 511.0, "the packed normals survive the round trip", .{});
}

fn unpackSnorm10(word: u32) f32 {
    const bits: u10 = @truncate(word);
    const signed: i10 = @bitCast(bits);
    return @max(@as(f32, @floatFromInt(signed)) / 511.0, -1.0);
}

fn localBounds(vertices: []const res.Vertex3D) res.Aabb {
    var min = vertices[0].position;
    var max = vertices[0].position;
    for (vertices[1..]) |vertex| {
        min = @min(min, vertex.position);
        max = @max(max, vertex.position);
    }
    return .{ .min = min, .max = max };
}

fn aabbCorners(box: res.Aabb) [8]res.Vec3 {
    var out: [8]res.Vec3 = undefined;
    for (&out, 0..) |*corner, index| {
        corner.* = .{
            if (index & 1 == 0) box.min[0] else box.max[0],
            if (index & 2 == 0) box.min[1] else box.max[1],
            if (index & 4 == 0) box.min[2] else box.max[2],
        };
    }
    return out;
}

fn approxEqual(a: res.Vec3, b: res.Vec3) bool {
    return @reduce(.And, @abs(a - b) < @as(res.Vec3, @splat(1e-5)));
}

// What makes two materials draw differently through the alpha modes, which is
// the part of a material the shaded surface above does not carry. glTF 2.0
// section 3.9.4 states that the cutoff is ignored outside MASK, so folding it
// in for the other modes would make two materials that draw alike compare as
// different.
const AlphaCoverageKey = struct {
    mode: res.MaterialInfo.Rendering.AlphaMode,
    cutoff: f32,
};

fn alphaCoverageKey(record: gpu.MaterialData) AlphaCoverageKey {
    const mode = record.alphaMode();
    return .{
        .mode = mode,
        .cutoff = if (mode == .mask) record.metallic_roughness_cutoff[2] else 0,
    };
}

// The slots this app decodes for, which is every slot a material set binds. A
// slot left out of this list and still named by a material stops the load with
// TextureNotUploaded rather than binding a fallback, so the two lists disagreeing
// is loud.
const decoded_slots = [_]gpu.MaterialSlot{
    .base_colour,
    .metallic_roughness,
    .normal,
    .emissive,
    .occlusion,
};

// What the load has to do, in the order the upload path wants it: which images
// are named at all, and in which slots.
//
// glTF's own unit is the image. Several materials name one, and its slot decides
// how it is interpreted, so an image is decoded once and uploaded once per slot
// interpretation. A pass over materials would instead decode once per naming,
// which is the same picture for more work.
const ImageWork = struct {
    // Parallel to model.images. A slot present in `uses` gets an upload, and the
    // resident image it produced lands in `bound` under the same slot.
    uses: []std.EnumSet(gpu.MaterialSlot),
    bound: []std.EnumArray(gpu.MaterialSlot, ?gpu.ResidentTexture),

    fn init(allocator: Allocator, model: *const gltf.importer.Model) !ImageWork {
        const uses = try allocator.alloc(std.EnumSet(gpu.MaterialSlot), model.images.len);
        errdefer allocator.free(uses);
        @memset(uses, .initEmpty());

        const bound = try allocator.alloc(
            std.EnumArray(gpu.MaterialSlot, ?gpu.ResidentTexture),
            model.images.len,
        );
        errdefer allocator.free(bound);
        @memset(bound, .initFill(null));

        for (model.materials) |*material| {
            inline for (decoded_slots) |slot| {
                const reference = @field(material.textures, @tagName(slot));
                if (reference.path) |key| {
                    // Before a device exists, so a document naming an image it
                    // does not carry fails while the failure is still cheap.
                    uses[try imageIndex(model, key)].insert(slot);
                }
            }
        }
        return .{ .uses = uses, .bound = bound };
    }

    fn deinit(self: *ImageWork, allocator: Allocator) void {
        allocator.free(self.bound);
        allocator.free(self.uses);
        self.* = undefined;
    }

    // What a material's slot binds: the image this work list already uploaded,
    // paired with the sampler of this reference. The sampler belongs to the
    // glTF texture rather than to the image, so two references to one image may
    // want different filtering and neither takes a second upload.
    fn fill(
        self: *const ImageWork,
        model: *const gltf.importer.Model,
        reference: anytype,
        slot: gpu.MaterialSlot,
    ) !?gpu.TextureSlot {
        const key = reference.path orelse return null;
        const index = try imageIndex(model, key);
        const held = self.bound[index].get(slot) orelse return error.TextureNotUploaded;
        return .{ .resident = .{ .texture = held, .sampler = reference.sampler } };
    }
};

fn imageIndex(model: *const gltf.importer.Model, key: []const u8) error{ImageNotFound}!usize {
    for (model.images, 0..) |*image, index| {
        if (std.mem.eql(u8, image.key, key)) return index;
    }
    return error.ImageNotFound;
}

// The three files a prefiltered environment is, and the keys they were cached
// under so the references taken here can be given back.
const AcquiredEnvironment = struct {
    environment: gpu.Environment,
    keys: [3][]const u8,
};

const environment_keys = [3][]const u8{
    "environment:lambertian",
    "environment:ggx",
    "environment:lut",
};

// Reads a prefiltered environment and hands back what the scene set is written
// from.
//
// Each map goes through its own transfer, so the staging pool is reclaimed
// between them and one file's worth is all that is ever in flight. It is the
// pool that makes that true rather than the sizing: a single GGX level is fifty
// megabytes, more than the whole ceiling, and it travels a block at a time.
fn loadEnvironment(
    allocator: Allocator,
    io: std.Io,
    context: *const gpu.Context,
    staging: *gpu.StagingPool,
    textures: *gpu.TextureCache,
    pool: *const gpu.OneShotPool,
    directory: []const u8,
) !AcquiredEnvironment {
    var root = try std.Io.Dir.cwd().openDir(io, directory, .{});
    defer root.close(io);

    const lambertian_bytes = try root.readFileAlloc(io, "lambertian/diffuse.ktx2", allocator, .limited(max_environment_bytes));
    defer allocator.free(lambertian_bytes);
    const ggx_bytes = try root.readFileAlloc(io, "ggx/specular.ktx2", allocator, .limited(max_environment_bytes));
    defer allocator.free(ggx_bytes);
    const lut_bytes = try root.readFileAlloc(io, "lut_ggx.png", allocator, .limited(max_environment_bytes));
    defer allocator.free(lut_bytes);

    const lambertian = try acquireEnvironmentCube(
        textures,
        staging,
        context,
        pool,
        environment_keys[0],
        lambertian_bytes,
    );
    const ggx = try acquireEnvironmentCube(
        textures,
        staging,
        context,
        pool,
        environment_keys[1],
        ggx_bytes,
    );

    // The lookup table is data and not colour: it tabulates a scale and a bias,
    // and reading it through an sRGB transfer function returns wrong numbers
    // that still look like a plausible gradient.
    var decoded = try DecodedImage.loadFromBytes(allocator, lut_bytes);
    defer decoded.deinit(allocator);
    if (decoded.stride != decoded.cols) return error.NonContiguousDecodedImage;

    // The table has to carry a scale in red and a bias in green. The set
    // Khronos publishes beside the environments also contains a single-channel
    // table whose data sits in blue, and reading that one returns (0, 0)
    // everywhere: the specular reflection then vanishes and, where the base
    // colour is white, the energy term divides zero by zero. That was a black
    // model with clean validation, so the channels are checked rather than the
    // file name trusted.
    var scale_range: [2]u8 = .{ 255, 0 };
    var bias_range: [2]u8 = .{ 255, 0 };
    for (decoded.data) |texel| {
        scale_range[0] = @min(scale_range[0], texel.r);
        scale_range[1] = @max(scale_range[1], texel.r);
        bias_range[0] = @min(bias_range[0], texel.g);
        bias_range[1] = @max(bias_range[1], texel.g);
    }
    check(
        scale_range[1] > scale_range[0] and bias_range[1] > bias_range[0],
        "the lookup table varies in both channels: scale {d}..{d}, bias {d}..{d}",
        .{ scale_range[0], scale_range[1], bias_range[0], bias_range[1] },
    );

    var lut_setup: gpu.Transfer = try .begin(context, pool.handle, staging);
    const lut = try textures.acquireRgba8(
        environment_keys[2],
        .{
            .width = decoded.cols,
            .height = decoded.rows,
            .bytes = std.mem.sliceAsBytes(decoded.data),
        },
        .r8g8b8a8_unorm,
        gpu.environmentSampler,
        &lut_setup,
    );
    // The reference taken above must outlive this submission: releasing it
    // sooner destroys an image a recorded copy still names.
    try lut_setup.finish();
    lut_setup.deinit();

    log.info("environment: lambertian {d}x{d} {d} level(s), ggx {d}x{d} {d} level(s), lut {d}x{d}", .{
        lambertian.width, lambertian.height, lambertian.mip_levels,
        ggx.width,        ggx.height,        ggx.mip_levels,
        lut.width,        lut.height,
    });
    check(
        ggx.mip_levels > 1,
        "the ggx chain has {d} roughness step(s)",
        .{ggx.mip_levels},
    );

    return .{
        .environment = .{ .lambertian = lambertian, .ggx = ggx, .lut = lut },
        .keys = environment_keys,
    };
}

// Half a gigabyte is past anything the Khronos set publishes and short of a
// length that would be a denial of service by itself.
const max_environment_bytes: usize = 512 << 20;

fn acquireEnvironmentCube(
    textures: *gpu.TextureCache,
    staging: *gpu.StagingPool,
    context: *const gpu.Context,
    pool: *const gpu.OneShotPool,
    key: []const u8,
    bytes: []const u8,
) !gpu.BoundTexture {
    var setup: gpu.Transfer = try .begin(context, pool.handle, staging);
    const bound = try textures.acquireKtx2(
        key,
        bytes,
        .r16g16b16a16_sfloat,
        .cube,
        gpu.environmentSampler,
        &setup,
    );
    // The reference taken above must outlive this submission: releasing it
    // sooner destroys an image a recorded copy still names.
    try setup.finish();
    log.info("environment: {s} staged through {d} block(s) and {d} stall(s)", .{
        key,
        staging.blockCount(),
        setup.flushes,
    });
    setup.deinit();
    return bound;
}

// Decodes one document image to the byte layout the upload path accepts. A
// file-backed image is an explicit error: reading it must go through the
// loader's no-symlink confinement path rather than reopening its normalized key
// less safely here.
// The decoded pixels the image pass may hold at once, over every thread.
//
// Bytes rather than a count of images: the images are not the same size, so a
// count bounds nothing.
//
// It is not simply "large enough". The window sets how many decodes run at once,
// and past a point that costs more than it buys, because the decoded pixels
// leave cache and the host and the integrated GPU share one memory bus.
// Measured on ABeautifulGame, 33 images of 2048 square, ReleaseFast on a
// sixteen-thread host at low-power with the powersave governor, as decode wall
// against summed pool CPU: 128 MiB gives 1.38 s for 9.1 s, 192 MiB gives 1.24 s
// for 11.4 s, 256 MiB gives 1.17 s for 14.5 s, and 768 MiB is worse on both at
// 1.36 s for 16.9 s. This is the knee. Buying the remaining 0.07 s costs three
// more CPU-seconds, and energy is the priority the engine states first.
//
// The run reports where the window actually peaked, so an asset that never
// reaches it says so.
const decode_window_bytes: u64 = 192 << 20;

// What one decode produces. The elapsed time is measured on the thread that did
// the work, because the recording thread cannot see when it started.
const DecodedSource = struct {
    image: DecodedImage,
    nanoseconds: u64,
};

// One work item: pure, and the reason this pass can be spread over threads at
// all. No device, no staging block, no shared mutable state. The source bytes
// are immutable for the whole pass and the pixels belong to the caller that
// awaits this.
//
// The allocator has to be threadsafe. Both of the app's are: DebugAllocator's
// `thread_safe` defaults to `!builtin.single_threaded` and SmpAllocator exists
// for this case (std/heap/debug_allocator.zig, std/heap/SmpAllocator.zig).
fn decodeSource(allocator: Allocator, clock: platform.Clock, bytes: []const u8) !DecodedSource {
    const started = clock.now();
    var decoded = DecodedImage.loadFromBytes(allocator, bytes) catch |err| switch (err) {
        error.UnsupportedImageFormat => return error.UnsupportedImageEncoding,
        else => return err,
    };
    errdefer decoded.deinit(allocator);
    const elapsed = clock.now() - started;

    if (decoded.stride != decoded.cols) return error.NonContiguousDecodedImage;
    return .{ .image = decoded, .nanoseconds = elapsed };
}

// What the decode will allocate, without decoding. Both codecs can answer from
// the header, and every decode targets `SourcePixel`, so four bytes a pixel is
// the whole cost whatever the source encoding was.
//
// This is what makes the window exact rather than an estimate: the reservation
// is charged before the work is spawned. The alternative, admitting on current
// occupancy alone, overshoots by one image per worker.
fn decodedByteSize(bytes: []const u8) !u64 {
    var reader: std.Io.Reader = .fixed(bytes);
    const pixels: u64 = switch (zignal.ImageFormat.detectFromBytes(bytes) orelse
        return error.UnsupportedImageEncoding) {
        .png => (try zignal.png.getInfo(&reader, .{})).totalPixels(),
        .jpeg => (try zignal.jpeg.getInfo(&reader, .{})).totalPixels(),
    };
    return pixels * @sizeOf(SourcePixel);
}

// Decodes running ahead of the thread that uploads them, bounded in bytes.
//
// Admission is the recording thread's job and never a worker's. `Io.async` runs
// the task inline on the calling thread once the pool is at its limit, and also
// when the future cannot be allocated or a thread cannot be spawned
// (std/Io/Threaded.zig, `async`). A worker that blocked waiting for the
// recording thread to free window space would therefore deadlock the moment it
// ran inline. Charging before the spawn is what removes that case entirely.
//
// The ring is drained in spawn order, so uploads reach the batch in image order
// exactly as a serial pass would produce them.
const DecodeQueue = struct {
    const Pending = struct {
        index: usize,
        charged: u64,
        future: Future,
    };
    const Future = std.Io.Future(@typeInfo(@TypeOf(decodeSource)).@"fn".return_type.?);

    io: std.Io,
    clock: platform.Clock,
    ring: []Pending,
    head: usize = 0,
    len: usize = 0,
    charged: u64 = 0,
    peak: u64 = 0,

    // `capacity` is the whole image count: the window is what limits how many
    // run, and a ring that cannot hold them all would be a second limit with no
    // reason behind it.
    fn init(allocator: Allocator, io: std.Io, clock: platform.Clock, capacity: usize) !DecodeQueue {
        return .{ .io = io, .clock = clock, .ring = try allocator.alloc(Pending, capacity) };
    }

    // Awaits everything still outstanding and frees what it produced. This is
    // the normal path's release as well as the failure path's: a future whose
    // task is still running owns an allocation and possibly a thread, so
    // abandoning one leaks both.
    fn deinit(self: *DecodeQueue, allocator: Allocator) void {
        while (self.len > 0) {
            var pending = self.take();
            if (pending.future.await(self.io)) |*decoded| {
                var image = decoded.image;
                image.deinit(allocator);
            } else |_| {}
        }
        allocator.free(self.ring);
        self.* = undefined;
    }

    // Whether admitting `size` would put the window over. An empty queue admits
    // anything, so an image larger than the whole window still loads; the peak
    // is then that image rather than the window, which the report shows.
    fn wouldExceed(self: *const DecodeQueue, size: u64) bool {
        return self.len > 0 and self.charged + size > decode_window_bytes;
    }

    fn spawn(self: *DecodeQueue, allocator: Allocator, index: usize, bytes: []const u8, size: u64) void {
        self.ring[(self.head + self.len) % self.ring.len] = .{
            .index = index,
            .charged = size,
            .future = self.io.async(decodeSource, .{ allocator, self.clock, bytes }),
        };
        self.len += 1;
        self.charged += size;
        self.peak = @max(self.peak, self.charged);
    }

    // The oldest entry, removed from the ring. Its charge is still held: the
    // pixels are not released until the caller has uploaded and freed them.
    fn take(self: *DecodeQueue) Pending {
        const pending = self.ring[self.head];
        self.head = (self.head + 1) % self.ring.len;
        self.len -= 1;
        return pending;
    }

    fn release(self: *DecodeQueue, pending: Pending) void {
        self.charged -= pending.charged;
    }
};

// The drain half of the queue: awaits the oldest decode, uploads it under every
// slot that names it, and gives the window its bytes back.
//
// The charge is released whether or not the decode succeeded, so a failing image
// cannot wedge the pass behind a reservation nothing will ever free.
fn uploadDecoded(
    allocator: Allocator,
    queue: *DecodeQueue,
    batch: *gpu.UploadBatch,
    work: *ImageWork,
    model: *const gltf.importer.Model,
    stats: *DecodeStats,
) !void {
    var pending = queue.take();
    defer queue.release(pending);

    var produced = try pending.future.await(queue.io);
    defer produced.image.deinit(allocator);

    const index = pending.index;
    stats.record(index, &produced.image, produced.nanoseconds);

    // The header said what the decode would allocate, and the window was charged
    // for exactly that. A disagreement means the reservation was never the size
    // it claimed to be, which is a defect in the accounting rather than in the
    // asset.
    const produced_bytes = @as(u64, produced.image.cols) *
        produced.image.rows * @sizeOf(SourcePixel);
    if (produced_bytes != pending.charged) return error.DecodeSizeMismatch;

    const uses = work.uses[index];
    const request: gpu.TextureRequest = .{
        .key = model.images[index].key,
        .source = .{ .rgba8 = .{
            .width = produced.image.cols,
            .height = produced.image.rows,
            .bytes = std.mem.sliceAsBytes(produced.image.data),
        } },
    };
    // Two slots of the same interpretation resolve to one cached image, so what
    // this costs beyond the first is a reference, not an upload.
    inline for (decoded_slots) |slot| {
        if (uses.contains(slot))
            work.bound[index].set(slot, (try batch.addTexture(slot, request)).resident());
    }
    // The alpha range, because it is what the alpha modes read and the only part
    // of a base colour a picture cannot be squinted at to confirm. An image whose
    // alpha is 255 everywhere makes MASK and BLEND indistinguishable from OPAQUE,
    // and that is a property of the asset rather than a fault in the pass.
    var alpha_low: u8 = 255;
    var alpha_high: u8 = 0;
    for (produced.image.data) |texel| {
        alpha_low = @min(alpha_low, texel.a);
        alpha_high = @max(alpha_high, texel.a);
    }
    log.info("image {d}: decoded {d}x{d} for {d} slot(s), alpha {d}..{d}", .{
        index,
        produced.image.cols,
        produced.image.rows,
        uses.count(),
        alpha_low,
        alpha_high,
    });
}

// The box the model occupies in normalized device coordinates, printed beside
// the viewport it will be stretched across.
fn reportNdcExtent(
    view_projection: zm.Mat,
    box: res.Aabb,
    aspect: f32,
    viewport: platform.Extent2D,
) void {
    var min = [2]f32{ std.math.inf(f32), std.math.inf(f32) };
    var max = [2]f32{ -std.math.inf(f32), -std.math.inf(f32) };
    for (aabbCorners(box)) |corner| {
        const clip = zm.mul(zm.f32x4(corner[0], corner[1], corner[2], 1), view_projection);
        if (!(clip[3] > 0)) continue;
        const ndc = [2]f32{ clip[0] / clip[3], clip[1] / clip[3] };
        min = .{ @min(min[0], ndc[0]), @min(min[1], ndc[1]) };
        max = .{ @max(max[0], ndc[0]), @max(max[1], ndc[1]) };
    }
    const width = max[0] - min[0];
    const height = max[1] - min[1];
    log.info(
        "ndc box: x [{d:.3} {d:.3}] y [{d:.3} {d:.3}], {d:.3} wide by {d:.3} tall, ratio {d:.3}",
        .{ min[0], max[0], min[1], max[1], width, height, height / width },
    );
    // The viewport turns normalized device coordinates into pixels, so a shape
    // that is square on screen is one whose device box has the viewport's
    // inverse ratio.
    const pixels_wide = width * 0.5 * @as(f32, @floatFromInt(viewport.width));
    const pixels_tall = height * 0.5 * @as(f32, @floatFromInt(viewport.height));
    log.info("on screen: {d:.0} by {d:.0} pixels, aspect {d:.4}, ratio {d:.3}", .{
        pixels_wide, pixels_tall, aspect, pixels_tall / pixels_wide,
    });
}

// Where the model's highest world point lands, in framebuffer rows.
//
// Vulkan specification, vkCmdSetViewport: with a positive height the viewport
// maps device y of -1 to the first row and +1 to the last, so a row near zero
// is the top of the window. The matrix passed in is the one the shader gets,
// flip included, which is what makes the answer the one on screen.
fn reportUpAxis(view_projection: zm.Mat, box: res.Aabb, viewport: platform.Extent2D) void {
    var highest = aabbCorners(box)[0];
    for (aabbCorners(box)) |corner| {
        if (corner[1] > highest[1]) highest = corner;
    }
    const clip = zm.mul(zm.f32x4(highest[0], highest[1], highest[2], 1), view_projection);
    const ndc_y = clip[1] / clip[3];
    const rows: f32 = @floatFromInt(viewport.height);
    const row = (ndc_y * 0.5 + 0.5) * rows;

    log.info("up axis: the highest point of the model lands at row {d:.0} of {d}", .{ row, viewport.height });
    log.info("up axis: {s}", .{
        if (row > rows * 0.5)
            "in the lower half, so the picture is upside down and the Y flip is unplaced"
        else
            "in the upper half, so the picture is the right way up",
    });
}
