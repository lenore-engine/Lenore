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
// Usage: run-validation_app -- <model.glb>
//
// The first mesh's base-colour image is decoded when it is an embedded PNG or
// JPEG, as in the Khronos glTF-Binary samples. Other slots keep their neutral
// fallbacks; the draw still has one material, so this exercises source-image
// decoding and RGBA8 upload without pretending the material loop already exists.

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

const frames_in_flight = 2;

// What one frame's rings hold. The joint bound is what the scene's joint plan
// packs into, so the two are the same number and not two copies of it.
const frame_capacity: gpu.FrameCapacity = .{ .instances = 64, .joints = 256 };

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

// One instance, placed at the origin. The camera is what moves.
const instance_count = 1;

// One skinned instance's playback: the pose the frame samples into, and the
// clip it samples from. The pose borrows the skin's template, which the model
// owns.
const Skin = struct {
    pose: res.SkeletonPose,
    clip: ?*const res.Animation,
};

// The base an instance record carries, from what the plan assigned it. The plan
// marks an entity with no pose, and that marker is not an index: the unskinned
// pipeline reads no joint array, so it becomes zero here rather than reaching
// the device.
fn jointBase(planned: u32) u32 {
    return if (planned == scene.no_joint_base) 0 else planned;
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
    const model_directory = std.fs.path.dirname(model_path) orelse ".";
    const model_name = std.fs.path.basename(model_path);
    var root = try std.Io.Dir.cwd().openDir(io, model_directory, .{});
    defer root.close(io);

    // Host side first, so a broken asset is reported before a device is touched.
    var loaded = try gltf.loader.open(gpa, io, root, model_name);
    defer loaded.deinit(gpa);
    var model = try gltf.importer.build(gpa, &loaded.document, loaded.directory);
    defer model.deinit(gpa);

    check(model.meshes.len > 0, "model holds {d} mesh(es), {d} material(s)", .{
        model.meshes.len,
        model.materials.len,
    });
    if (model.meshes.len == 0) return error.NoGeometry;

    const source = &model.meshes[0];
    log.info("mesh: {d} vertices, {d} indices, streams {any}", .{
        source.vertices.len,
        source.indices.len,
        source.streams,
    });
    check(source.indices.len % 3 == 0, "index count {d} is whole triangles", .{source.indices.len});
    check(source.indices.len > 0, "geometry is indexed", .{});

    // Skeletal playback, for a mesh that declares the skinning stream.
    //
    // The pose borrows the skin's template, so both outlive the frame loop.
    var skin: ?Skin = null;
    defer if (skin) |*owned| owned.pose.deinit(gpa);
    if (source.streams.skinned) {
        // glTF 2.0 specification, 3.7.3.3: a skinned mesh primitive has the
        // attributes that skinning reads, and a node referencing it names the
        // skin they index. A document with the stream and no skin gives the
        // shader an array to index and nothing to fill it with, so this stops
        // before anything reaches the device.
        if (model.skins.len == 0) return error.SkinnedMeshWithoutSkin;

        const template = &model.skins[0].skeleton;
        skin = .{
            .pose = try res.SkeletonPose.init(gpa, template),
            // The first clip. Which one plays is playback's choice and this app
            // has no way to make it; a skin with none holds its bind pose.
            .clip = if (model.skins[0].clips.len > 0) &model.skins[0].clips[0] else null,
        };
        skin.?.pose.evaluate();

        log.info("skin: {d} joints over {d} slots, {d} clip(s)", .{
            template.jointCount(),
            template.slotCount(),
            model.skins[0].clips.len,
        });
        if (skin.?.clip) |clip|
            log.info("clip: keyed from {d:.3} to {d:.3} s over {d} slots", .{
                clip.start_time,
                clip.duration,
                clip.slot_count,
            });

        // The shader indexes the joint array with what a vertex carries and no
        // bound of its own, so the one that would read outside it is answered
        // here, once, over the whole mesh.
        const joint_count = template.jointCount();
        var highest: u32 = 0;
        for (source.vertices) |vertex| {
            // Indexing a vector needs a comptime index, so the lanes are unrolled.
            inline for (0..4) |lane| highest = @max(highest, vertex.joints[lane]);
        }
        if (highest >= joint_count) return error.JointIndexOutOfRange;
        check(true, "every vertex joint index is under the skin's {d}", .{joint_count});

        // The whole chain from the asset's inverse bind matrices to the
        // matrices the shader reads, answered without a device.
        //
        // glTF 2.0 specification, 3.7.3.3: an inverse bind matrix takes the
        // mesh into a joint's own space, and the joint's global transform takes
        // it back out into the scene. Their product is the mesh's placement in
        // the scene and does not depend on which joint it was taken from, so at
        // the bind pose every joint matrix is the same matrix.
        //
        // That is the invariant, and it holds whatever the asset's placement
        // is. A transposed matrix, a joint bound to the wrong slot or an
        // inverse bind applied on the wrong side each break it, and on screen
        // all three are the same torn mesh.
        const bind_placement = skin.?.pose.joint_transforms[0];
        var worst_joint: f32 = 0;
        for (skin.?.pose.joint_transforms) |joint| {
            inline for (0..4) |row| {
                inline for (0..4) |column| {
                    worst_joint = @max(worst_joint, @abs(joint[row][column] - bind_placement[row][column]));
                }
            }
        }
        log.info("bind pose: joints disagree by at most {d:.6}", .{worst_joint});
        check(worst_joint < 1e-4, "every joint carries the same bind placement", .{});

        // And what that one placement does to the mesh, which is where the
        // asset's own orientation shows. Not a failure: a skeleton under a
        // rotated node legitimately moves every vertex, and this says by how
        // much before the device is asked to do the same.
        var worst_vertex_drift: f32 = 0;
        for (source.vertices) |vertex| {
            const skinned = skinnedPosition(&skin.?.pose, vertex);
            const placed = zm.mul(
                zm.f32x4(vertex.position[0], vertex.position[1], vertex.position[2], 1),
                bind_placement,
            );
            // Against the placement rather than against the source: this is
            // the per-vertex half, and it fails where a weight set does not sum
            // to one or a joint index reaches the wrong matrix.
            inline for (0..3) |axis|
                worst_vertex_drift = @max(worst_vertex_drift, @abs(skinned[axis] - placed[axis]));
        }
        check(worst_vertex_drift < 1e-3, "the bind pose moves every vertex by that placement alone", .{});
        log.info("bind pose: the placement moves the model's height axis to [{d:.3} {d:.3} {d:.3}]", .{
            bind_placement[2][0], bind_placement[2][1], bind_placement[2][2],
        });
    }

    const material = &model.materials[source.material];
    const base_colour_key = material.textures.base_colour.path;
    var base_colour: ?DecodedImage = if (base_colour_key) |key|
        try decodeEmbeddedImage(gpa, &model, key)
    else
        null;
    defer if (base_colour) |*decoded| decoded.deinit(gpa);
    if (base_colour) |decoded| {
        log.info("base colour: decoded embedded image {d}x{d}", .{ decoded.cols, decoded.rows });
    }

    const local = localBounds(source.vertices);
    log.info("local bounds: [{d:.3} {d:.3} {d:.3}] to [{d:.3} {d:.3} {d:.3}]", .{
        local.min[0], local.min[1], local.min[2],
        local.max[0], local.max[1], local.max[2],
    });
    check(
        local.min[0] <= local.max[0] and local.min[1] <= local.max[1] and local.min[2] <= local.max[2],
        "bounds are not inverted",
        .{},
    );

    // The transform the whole model is drawn with. Identity here: the importer
    // has already baked every node's world matrix into the vertices, so a
    // second one would apply the placement twice.
    const model_matrix = zm.identity();
    const world = scene.worldAabb(local, model_matrix);
    check(
        approxEqual(world.min, local.min) and approxEqual(world.max, local.max),
        "an identity transform leaves the bounds where they were",
        .{},
    );

    const sphere = scene.sphereAroundAabb(world);
    log.info("bounding sphere: centre [{d:.3} {d:.3} {d:.3}] radius {d:.4}", .{
        sphere.centre[0], sphere.centre[1], sphere.centre[2], sphere.radius,
    });
    check(sphere.radius > 0, "the bounding sphere has a radius", .{});

    // Platform and device.
    var host: platform.Platform = try .init();
    defer host.deinit();
    var window = try host.createWindow(.{ .width = 1280, .height = 720 }, "Lenore validation");
    defer window.deinit();

    // The compositor tells a client its size; nothing here can ask. Without
    // reading those events the swapchain keeps the extent it was created with
    // and the camera's aspect stops matching the window the moment it is
    // resized.
    const clock: platform.Clock = .init(io);
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

    var staging: gpu.StagingArena = try .init(&context, &memory, 32 << 20);
    defer staging.deinit();

    var setup_pool: gpu.OneShotPool = try .init(&context);
    defer setup_pool.deinit(&context);

    const cache_setup = try gpu.beginOneShot(&context, setup_pool.handle);
    var textures: gpu.TextureCache = try .init(&context, &memory, gpa, &staging, cache_setup);
    defer if (textures.deinit() == .leak) log.err("texture references outstanding", .{});
    try gpu.submitOneShotAndWait(&context, setup_pool.handle, cache_setup);

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
        const mesh_handle = try batch.addMesh(u32, .{
            .vertices = source.vertices,
            .indices = source.indices,
            .streams = source.streams,
        });
        const set_handle = try batch.addTextureSet(.{
            .base_colour = if (base_colour) |decoded| .{
                .key = base_colour_key.?,
                .source = .{ .rgba8 = .{
                    .width = decoded.cols,
                    .height = decoded.rows,
                    .bytes = std.mem.sliceAsBytes(decoded.data),
                } },
                .sampler = material.textures.base_colour.sampler,
            } else null,
        });
        _ = mesh_handle;
        _ = set_handle;
        break :uploaded try batch.finish();
    };
    defer uploaded.deinit(gpa, &storage, &textures);

    const mesh = storage.mesh(uploaded.meshes.items[0]).?;
    const texture_set = storage.textureSet(uploaded.texture_sets.items[0]).?;
    check(
        mesh.vertex_count == source.vertices.len,
        "uploaded {d} vertices, the asset had {d}",
        .{ mesh.vertex_count, source.vertices.len },
    );
    check(
        mesh.index_count == source.indices.len,
        "uploaded {d} indices, the asset had {d}",
        .{ mesh.index_count, source.indices.len },
    );
    log.info("bounds spheres: vertices {d:.4}, enclosing box {d:.4}", .{
        mesh.bounds.sphere.radius,
        sphere.radius,
    });
    check(
        approxEqual(mesh.bounds.box.min, local.min) and approxEqual(mesh.bounds.box.max, local.max),
        "uploaded bounds reproduce the source box",
        .{},
    );

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
        swapchain.surface_format.format,
        post_sampler,
    );
    defer renderer.deinit();
    renderer.bindMaterial(texture_set);

    // A clear that is not black, so a window showing it proves the whole post
    // chain carried the main pass's target to the screen, and a window that is
    // still black narrows the fault to that chain rather than to the geometry.
    renderer.clear_colour = .{ 0.05, 0.10, 0.20, 1 };

    log.info("diagnostic: clear [{d:.2} {d:.2} {d:.2}], culling {s}", .{
        renderer.clear_colour[0],
        renderer.clear_colour[1],
        renderer.clear_colour[2],
        if (renderer.cull_mode.back_bit) "back faces" else "none",
    });

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
    const distance = sphere.radius * 3.0 + 1.0;
    var camera: scene.Camera = .{
        .anchor = .{ .orbit = .{
            .target = .{ sphere.centre[0], sphere.centre[1], sphere.centre[2] },
            .distance = distance,
        } },
        .yaw = -std.math.pi / 4.0,
        .pitch = -0.35,
    };
    camera.projection = .{ .perspective = .{
        .near = @max(distance * 0.01, 0.01),
        .far = distance * 10.0,
    } };

    const aspect = @as(f32, @floatFromInt(extent.width)) / @as(f32, @floatFromInt(extent.height));
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
    const poses = [instance_count]?*const res.SkeletonPose{
        if (skin) |*owned| &owned.pose else null,
    };
    var joint_bases: [instance_count]u32 = undefined;
    const joint_total = try scene.assignJointOffsets(&poses, &joint_bases, @intCast(frame_capacity.joints));
    check(
        joint_total <= frame_capacity.joints,
        "the frame's {d} joint(s) fit an array of {d}",
        .{ joint_total, frame_capacity.joints },
    );

    const instances = [instance_count]gpu.Instance{.{
        .model = model_matrix,
        .joint_base = jointBase(joint_bases[0]),
    }};

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
    reportLitFraction(source.vertices, live);

    // The normal as the device will read it, not as the asset stated it. It is
    // packed 10-10-10-2 on upload and expanded by the vertex input stage, and
    // until something shaded with it no run had ever looked at what comes back
    // out.
    reportPackedNormals(source.vertices);

    log.info("checks before the first frame: {d} failed", .{failures});
    log.info("validation errors before the first frame: {d}", .{gpu.validationErrorCount()});

    // The frame loop.
    var frame_index: usize = 0;
    var surface_extent = swapchain.currentExtent();
    var stale = false;
    var presented_frames: u64 = 0;
    var reported_validation: u32 = 0;

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
            renderer.bindMaterial(texture_set);
            stale = false;
            const resized_aspect = @as(f32, @floatFromInt(size.width)) / @as(f32, @floatFromInt(size.height));
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

        // The pose this frame draws. Sampling writes the clip's channels over
        // the locals and leaves every slot it does not target at its bind
        // value, so a clip that moves part of a skeleton is not a skeleton
        // half undefined.
        var joint_matrices: []const gpu.Joint = &.{};
        if (skin) |*owned| {
            if (owned.clip) |clip| {
                const elapsed = @as(f32, @floatFromInt(clock.now() - started)) / std.time.ns_per_s;
                try clip.sample(
                    clip.cursorAt(elapsed),
                    owned.pose.local_translations,
                    owned.pose.local_rotations,
                    owned.pose.local_scales,
                );
            }
            owned.pose.evaluate();
            joint_matrices = owned.pose.joint_transforms;
        }

        try renderer.update(frame_index, .{
            .camera = .{
                .view_projection = frame_view_projection,
                .position = .{ eye[0], eye[1], eye[2], 1 },
            },
            .models = &instances,
            .joints = joint_matrices,
            .lights = live,
        });

        const commands = try frame.beginCommands(&context);
        renderer.record(commands, frame_index, .{
            .image = swapchain.images[acquired.image_index].image,
            .view = swapchain.images[acquired.image_index].view,
            .extent = .{ .width = current.width, .height = current.height },
        }, .{
            .mesh = mesh,
            .textures = texture_set,
            .instances = &instances,
        });
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
// Intensity is pi because the shader's Lambertian BRDF divides by pi: a white
// surface facing this light returns exactly its albedo, so a fallback texture
// still comes out white and the shading is visible only in how it falls off.
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

// How much of the mesh faces a light, counted the way the fragment shader
// decides it: a vertex is lit when its normal has a positive dot with the
// direction toward some light.
//
// Vertex normals rather than fragments, so this is an estimate. It answers the
// only question a black window raises here, which is whether the geometry is
// pointing at the light at all.
fn reportLitFraction(vertices: []const res.Vertex3D, lights: []const gpu.LightUniform) void {
    var lit: usize = 0;
    for (vertices) |vertex| {
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
    log.info("lighting: {d} of {d} vertices face a light", .{ lit, vertices.len });
    check(lit > 0, "some of the model faces a light", .{});
}

// The largest distance between a normal as authored and the same normal after
// the round trip through the packed vertex.
//
// Vulkan specification, Fixed-Point Data Conversion: a 10-bit snorm reads back
// as max(c / 511, -1), so a component resolves to about 1/511 and the worst a
// direction can move is a little over that.
fn reportPackedNormals(vertices: []const res.Vertex3D) void {
    var worst: f32 = 0;
    var worst_at: usize = 0;
    for (vertices, 0..) |vertex, index| {
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
            worst_at = index;
        }
    }
    log.info("normals: vertex 0 source [{d:.3} {d:.3} {d:.3}] packs to 0x{X:0>8} at byte {d} of {d}", .{
        vertices[0].normal[0],
        vertices[0].normal[1],
        vertices[0].normal[2],
        gpu.packVertex(&vertices[0]).normal,
        @offsetOf(gpu.GpuVertex, "normal"),
        @sizeOf(gpu.GpuVertex),
    });
    log.info("normals: worst packing drift {d:.5} at vertex {d}, source [{d:.3} {d:.3} {d:.3}]", .{
        worst,                        worst_at,
        vertices[worst_at].normal[0], vertices[worst_at].normal[1],
        vertices[worst_at].normal[2],
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

// Resolve a material's image identity in the document's image table and decode
// it to the byte layout the GPU upload path accepts. A file-backed image is an
// explicit error: reading it must go through the loader's no-symlink confinement
// path rather than reopening its normalized key less safely here.
fn decodeEmbeddedImage(
    allocator: Allocator,
    model: *const gltf.importer.Model,
    key: []const u8,
) !DecodedImage {
    const source = for (model.images) |*image| {
        if (std.mem.eql(u8, image.key, key)) break image;
    } else return error.ImageNotFound;

    const bytes = source.bytes orelse return error.ExternalImageNotLoaded;
    var decoded = DecodedImage.loadFromBytes(allocator, bytes) catch |err| switch (err) {
        error.UnsupportedImageFormat => return error.UnsupportedImageEncoding,
        else => return err,
    };
    errdefer decoded.deinit(allocator);

    if (decoded.stride != decoded.cols) return error.NonContiguousDecodedImage;
    return decoded;
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
