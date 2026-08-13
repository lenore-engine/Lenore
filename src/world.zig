const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");
const zm = @import("zmath");

const time_module = @import("time.zig");
const translate = @import("translate.zig");

const Allocator = std.mem.Allocator;
const FrameTime = time_module.FrameTime;
const RecordPlan = translate.RecordPlan;

// What is being drawn, and the one update that turns a moment into the record a
// frame is submitted from.
//
// The state here was a scope full of parallel arrays that had to be advanced,
// rebuilt and packed in one order, with nothing but reading order to say so.
// Four of those orderings are structural now: `update` is the only thing that
// advances an animator, the only thing that fills a frame's vertex sources, the
// only thing that rebuilds the plan, and the only producer of the `FrameState`
// that the renderer is handed.

pub const WorldError = error{
    NoGeometry,
    MeshCountMismatch,
    MaterialIndexOutOfRange,
    InstanceCapacityExceeded,
    SkinnedMeshWithoutSkin,
    EmptySkin,
    JointIndexOutOfRange,
    AnchoredMeshWithoutAnimator,
};

// The one definition of a draw's object-to-world transform. Everything that
// needs it comes through here: the instance record the shader reads, the depth
// key the blended run is ordered by, and the bounds the camera is framed on.
//
// Rigid node animation is the only thing that moves it. Static geometry is baked
// in world space and a skinned mesh's motion is in its joint matrices, so both
// take `placement` alone.
//
// Row vector convention, so the anchor's motion applies before the placement: an
// anchored mesh's vertices are baked in its anchor's space and reach the world
// only through it.
pub fn instanceMatrix(anchor: ?res.Slot, animator: ?*const res.NodeAnimator, placement: zm.Mat) zm.Mat {
    const slot = anchor orelse return placement;
    // An anchor exists only where the importer produced an animation to resolve
    // it against, and `World.init` refuses a model where it does not, so this is
    // not reachable with a missing animator.
    return zm.mul(animator.?.world_transforms[slot], placement);
}

// A point carried into world space by an instance matrix. The w lane is one, so
// the matrix's translation applies.
pub fn placePoint(point: res.Vec3, matrix: zm.Mat) res.Vec3 {
    const carried = zm.mul(zm.f32x4(point[0], point[1], point[2], 1.0), matrix);
    return .{ carried[0], carried[1], carried[2] };
}

// One imported skeleton bound to its playback.
//
// One per connected component of the document's joint forest rather than one per
// skin, which is the importer's shape and not a choice made here: joints that
// share a tree have to be posed together, because one skin's joint may be what
// carries another's. A mesh names the run of joint transforms it reads.
//
// Nothing of glTF is left in it. The animator, its cursor and the pose it writes
// all live in `lenore-resources` beside the node and morph animators, and this
// is now the whole of the engine's side.
pub const Skeleton = struct {
    animator: res.SkeletonAnimator,
    // The root slots whose transform above them the rigid hierarchy carries.
    // Borrowed from the model, which outlives the world for the same reason the
    // meshes are.
    prefix_links: []const gltf.importer.PrefixLink,
};

// Where every draw is this frame, what order they are recorded in, and the
// instance record each one reads.
//
// Rebuilt every frame rather than once. The blended run is ordered by each
// draw's distance from the eye, and rigid node animation moves a draw between
// one frame and the next, so an order computed at load describes the bind pose
// and nothing after it.
//
// Nothing here allocates. Every buffer belongs to the `World` that owns the
// plan, which is what lets this sit in the frame loop.
pub const DrawPlan = struct {
    // In mesh order. Everything but `front_face` is settled when the level is
    // built; that one follows the sign of the instance matrix and is rewritten
    // every frame beside the matrix it comes from.
    candidates: []RecordPlan.Draw,
    // The centre of each draw in the space its instance matrix maps from. A
    // skinned mesh carries its bind placement here instead, because its instance
    // matrix is identity and the joint matrices hold its motion.
    centres: []const res.Vec3,
    // The box the centre above was taken from, in the same space, kept whole for
    // the frustum test. A box rather than the sphere the centre came from: see
    // `worldAabb`, which is exact for any linear map where scaling a radius is
    // an underestimate under shear.
    local_bounds: []const res.Aabb,
    // Whether a mesh's box describes where its vertices are this frame. False
    // for a skinned or morphed mesh, whose box is its bind pose while the
    // vertices have moved, and such a mesh is never culled.
    //
    // The alternative is to bound the animation, which needs the extreme of
    // every pose and every weight rather than one frame's. A false negative here
    // is a limb that vanishes, so the conservative answer is the only one.
    cullable: []const bool,
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

    // By mesh index, rewritten whenever the frame slot changes.
    vertex_sources: []?gpu.MeshVertexSource,

    // The last rebuild's output.
    ordered: []const u32 = &.{},
    batches: []const RecordPlan.Batch = &.{},
    records: []const gpu.RecordBatch = &.{},
    // How many of `records` the camera can see. They are a prefix, so a camera
    // pass records that much and a shadow bake records all of them: a caster
    // outside the view still casts into it, and the sun's fit is built around
    // the whole scene rather than around what is on screen.
    visible_records: usize = 0,

    pub fn rebuild(
        self: *DrawPlan,
        meshes: []const gltf.importer.Mesh,
        animator: ?*const res.NodeAnimator,
        placement: zm.Mat,
        eye: res.Vec3,
        view_projection: zm.Mat,
    ) !void {
        // One extraction for the whole list, because the camera is a property of
        // the frame and not of a draw.
        const frustum: scene.Frustum = .fromViewProj(view_projection);

        for (
            meshes,
            self.centres,
            self.local_bounds,
            self.cullable,
            self.matrices,
            self.keys,
            self.candidates,
        ) |*mesh, centre, bounds, cullable, *matrix, *key, *candidate| {
            matrix.* = instanceMatrix(mesh.anchor, animator, placement);
            // glTF 2.0, section 3.7.4: the determinant of the node's global
            // transform decides the winding. The importer settled the part it
            // baked into the vertices, so what is left is exactly this matrix,
            // and it is per instance and per frame because an animated node or
            // the application's own placement can mirror.
            //
            // A skinned draw's joint matrices can mirror it again per vertex,
            // which no plan can answer for. That is why such a draw keeps its
            // culling off rather than being trusted to one winding.
            candidate.front_face = if (zm.determinant(matrix.*)[0] < 0)
                .clockwise
            else
                .counter_clockwise;
            key.depth = scene.depthOf(eye, placePoint(centre, matrix.*));
            // The box is carried into world space rather than the frustum into
            // the mesh's: six planes against one box either way, and this
            // direction needs no inverse.
            key.visible = !cullable or
                frustum.intersectsAabb(scene.worldAabb(bounds, matrix.*));
        }

        // Every draw, with the visible ones first. The culled half is kept
        // because the shadow bake reads it and because an instance record has to
        // exist for anything that bake draws.
        const ordered = try scene.orderDraws(self.keys, self.order);
        self.ordered = ordered.draws;

        for (ordered.draws, self.instances) |draw_index, *instance| {
            instance.* = .{
                .model = self.matrices[draw_index],
                .joint_base = translate.jointBase(self.joint_bases[draw_index]),
                .material_index = meshes[draw_index].material,
            };
        }

        const built = try RecordPlan.build(
            self.candidates,
            ordered.draws,
            ordered.visible,
            self.batch_storage,
        );
        self.batches = built.batches;
        self.visible_records = built.visible;
        self.records = try translate.recordBatches(
            built.batches,
            ordered.draws,
            self.vertex_sources,
            self.record_storage,
        );
    }
};

// What one frame draws, and the only value `Renderer.update` and `Renderer.plan`
// are filled from.
//
// It exists so that the three slices cannot come from three different moments.
// Only `World.update` returns one, and it returns one only after it has advanced
// every animator, resolved this frame's vertex sources and rebuilt the plan.
pub const FrameState = struct {
    instances: []const gpu.Instance,
    joints: []const gpu.Joint,
    // Every batch, visible ones first. A camera pass takes `visibleRecords`
    // and a shadow bake takes the whole slice.
    records: []const gpu.RecordBatch,
    visible_records: usize,

    pub fn visibleRecords(self: FrameState) []const gpu.RecordBatch {
        return self.records[0..self.visible_records];
    }
};

pub const World = struct {
    allocator: Allocator,

    // Borrowed from the model, which outlives the world.
    meshes: []const gltf.importer.Mesh,

    skeletons: []Skeleton,
    // How many of `skeletons` hold a pose that must be released. The slice is
    // allocated whole and filled one at a time, so a failure part way through
    // init leaves these two disagreeing, and teardown has to follow the counter
    // rather than the length.
    skeletons_ready: usize,
    // Where each mesh reads its joints: which skeleton, and the run inside it.
    // The importer resolved it, because a skeleton may carry several skins and
    // the document's skin index alone no longer says where to read. Values
    // rather than pointers: a pointer into `skeletons` would make the world
    // unmovable after init, for a lookup that costs one load either way.
    skin_of_mesh: []?gltf.importer.SkinPlacement,

    node_animator: ?res.NodeAnimator,
    morph_animators: []res.MorphAnimator,
    morph_ready: usize,
    // Which prepass registration each mesh draws through, by mesh index. Null is
    // a mesh with no shape targets, which is every mesh of most assets.
    morph_registrations: []?u32,

    plan: DrawPlan,
    joint_storage: []gpu.Joint,
    joint_total: u32,

    // Where this world puts the model. Identity unless a caller says otherwise:
    // static vertices are already in world space, a skinned mesh's bind
    // placement is in its joint matrices, and an anchored mesh's is in its
    // animator slot. It is kept rather than dropped because it is what an
    // instance matrix composes against, and only one order is correct.
    placement: zm.Mat,

    // The bind pose's extent, which is what a camera frames on and a shadow fit
    // is built from. An animation can carry a draw outside it.
    bounds: res.Aabb,
    sphere: res.Sphere,

    pub const Options = struct {
        capacity: gpu.FrameCapacity,
        placement: zm.Mat = zm.identity(),
        // Which document animation everything starts on, or null to open at the
        // bind pose with nothing playing.
        //
        // A document index, which is what a skin's and the rigid hierarchy's
        // clip lists are addressed by: both hold one entry per animation in the
        // document, so the same index is the same authored animation in both.
        // Morph templates are the exception and are described at `playClip`.
        clip: ?u16 = 0,
    };

    // Everything an asset got wrong is answered here, once, before a frame is
    // ever built. Past this point the per-frame path carries no check on
    // document data at all: a joint index, an anchor slot and a material index
    // are all indices into arrays this has already measured.
    // A world with nothing in it, for an application whose picture is not made
    // of meshes.
    //
    // Not a degenerate case of `init`: that one takes a document and refuses an
    // empty one, because for an asset zero meshes is a model that failed to
    // parse rather than a scene. This is the other thing, and the two are
    // different enough to be different constructors.
    //
    // Every field is the empty value of its own type and none is `undefined`,
    // so `deinit` runs over this unchanged. Freeing a zero-length slice returns
    // immediately (std 0.16, `mem/Allocator.zig`, `Allocator.free`), which is
    // what makes that true without a branch anywhere in teardown.
    //
    // The bounds are a degenerate box at the origin, which is the convention
    // `res.Aabb.compute` already states for a mesh with no vertices.
    pub fn empty(allocator: Allocator) World {
        return .{
            .allocator = allocator,
            .meshes = &.{},
            .skeletons = &.{},
            .skeletons_ready = 0,
            .skin_of_mesh = &.{},
            .node_animator = null,
            .morph_animators = &.{},
            .morph_ready = 0,
            .morph_registrations = &.{},
            .plan = .{
                .candidates = &.{},
                .centres = &.{},
                .local_bounds = &.{},
                .cullable = &.{},
                .joint_bases = &.{},
                .matrices = &.{},
                .keys = &.{},
                .order = &.{},
                .instances = &.{},
                .batch_storage = &.{},
                .record_storage = &.{},
                .vertex_sources = &.{},
            },
            .joint_storage = &.{},
            .joint_total = 0,
            .placement = zm.identity(),
            .bounds = .{ .min = @splat(0), .max = @splat(0) },
            .sphere = .{ .centre = @splat(0), .radius = 0 },
        };
    }

    pub fn init(
        allocator: Allocator,
        model: *const gltf.importer.Model,
        meshes: []const *const gpu.Mesh,
        morph_registrations: []const ?u32,
        options: Options,
    ) !World {
        if (model.meshes.len == 0) return error.NoGeometry;
        if (meshes.len != model.meshes.len) return error.MeshCountMismatch;
        if (morph_registrations.len != model.meshes.len) return error.MeshCountMismatch;
        if (model.meshes.len > options.capacity.instances) return error.InstanceCapacityExceeded;

        for (model.meshes) |*mesh| {
            if (mesh.material >= model.materials.len) return error.MaterialIndexOutOfRange;
        }

        // Every owning field starts empty rather than undefined, because the
        // errdefer below is `deinit` and it reads all of them. A field left
        // undefined here is one that teardown frees on the first failure.
        var self: World = .{
            .allocator = allocator,
            .meshes = model.meshes,
            .skeletons = &.{},
            .skeletons_ready = 0,
            .skin_of_mesh = &.{},
            .node_animator = null,
            .morph_animators = &.{},
            .morph_ready = 0,
            .morph_registrations = &.{},
            .plan = .{
                .candidates = &.{},
                .centres = &.{},
                .local_bounds = &.{},
                .cullable = &.{},
                .joint_bases = &.{},
                .matrices = &.{},
                .keys = &.{},
                .order = &.{},
                .instances = &.{},
                .batch_storage = &.{},
                .record_storage = &.{},
                .vertex_sources = &.{},
            },
            .joint_storage = &.{},
            .joint_total = 0,
            .placement = options.placement,
            .bounds = .{ .min = @splat(0), .max = @splat(0) },
            .sphere = .{ .centre = @splat(0), .radius = 0 },
        };
        errdefer self.deinit();

        try self.buildSkeletons(model);
        try self.buildAnimators(model);
        self.morph_registrations = try allocator.dupe(?u32, morph_registrations);

        // An anchor is a slot inside the rigid template, so geometry cannot
        // carry one unless the importer produced a template to resolve it
        // against. Answered here rather than at the first frame, where a missing
        // animator reads as an asset that simply does not move.
        for (model.meshes) |*mesh| {
            if (mesh.anchor != null and self.node_animator == null)
                return error.AnchoredMeshWithoutAnimator;
        }

        try self.buildPlan(model, meshes, options.capacity);

        // Started last, so nothing is playing while the bounds above are taken:
        // those describe the bind pose, and a camera framed on a mid-clip pose
        // would frame on whichever moment init happened to reach.
        if (options.clip) |clip| {
            _ = try self.playClip(clip);
            // Morph weights are addressed by their own compacted list, so the
            // index above means nothing to them. The first clip is the only
            // one that can be named without the mapping `playClip` describes.
            for (self.morph_animators) |*animator| {
                if (animator.template.clips.len > 0) try animator.play(0);
            }
        }
        return self;
    }

    pub fn deinit(self: *World) void {
        const allocator = self.allocator;

        allocator.free(self.plan.record_storage);
        allocator.free(self.plan.batch_storage);
        allocator.free(self.plan.vertex_sources);
        allocator.free(self.plan.instances);
        allocator.free(self.plan.order);
        allocator.free(self.plan.keys);
        allocator.free(self.plan.matrices);
        allocator.free(self.plan.joint_bases);
        allocator.free(self.plan.cullable);
        allocator.free(self.plan.local_bounds);
        allocator.free(self.plan.centres);
        allocator.free(self.plan.candidates);
        allocator.free(self.joint_storage);

        allocator.free(self.morph_registrations);
        for (self.morph_animators[0..self.morph_ready]) |*animator| animator.deinit(allocator);
        allocator.free(self.morph_animators);
        if (self.node_animator) |*animator| animator.deinit(allocator);

        allocator.free(self.skin_of_mesh);
        for (self.skeletons[0..self.skeletons_ready]) |*skeleton| skeleton.animator.deinit(allocator);
        allocator.free(self.skeletons);

        self.* = undefined;
    }

    // The frame's record, and the only place any of it is produced.
    //
    // The order below is the reason this is one function. Animation writes the
    // world transforms the plan reads; the plan produces the instance records
    // and batches; the joint array is packed from the poses animation just
    // evaluated. Any two of those swapped puts a frame on screen built from two
    // different moments, and every check that runs before the first frame stays
    // green while it happens.
    pub fn update(
        self: *World,
        morph_pass: *gpu.MorphPass,
        frame_index: usize,
        frame_time: FrameTime,
        eye: res.Vec3,
        // The same matrix the frame's camera block is filled from. Taking it
        // rather than deriving one here is what keeps the frustum a draw is
        // culled against and the clip space it is drawn into the same camera.
        view_projection: zm.Mat,
    ) !FrameState {
        if (self.node_animator) |*animator| animator.update(frame_time.delta);
        for (self.morph_animators) |*animator| animator.update(frame_time.delta);

        // Between the two, and that is the whole reason the order is written
        // out here. A skeleton hanging below an animated node reads that node's
        // world transform from the rigid hierarchy, which the line above has
        // just advanced, and its own pass below composes against it. Advancing
        // the skeletons first would pose them against last frame's carrier.
        self.carrySkeletons();
        for (self.skeletons) |*skeleton| skeleton.animator.update(frame_time.delta);

        // The weights reach the device through the prepass rather than through
        // an instance record, so this is their only path and it has to follow
        // the advance above.
        for (self.meshes, self.morph_registrations) |*mesh, registration| {
            const index = registration orelse continue;
            const template = mesh.morph_template orelse continue;
            try morph_pass.writeWeights(frame_index, index, self.morph_animators[template].weights);
        }

        // Which slot the prepass writes and the draws read moves with the frame
        // index, so this precedes the rebuild that translates it into batches.
        try translate.fillVertexSources(
            morph_pass,
            self.morph_registrations,
            frame_index,
            self.plan.vertex_sources,
        );

        try self.plan.rebuild(
            self.meshes,
            if (self.node_animator) |*animator| animator else null,
            self.placement,
            eye,
            view_projection,
        );

        self.packJoints();
        return .{
            .instances = self.plan.instances,
            .joints = self.joint_storage[0..self.joint_total],
            .records = self.plan.records,
            .visible_records = self.plan.visible_records,
        };
    }

    // The transform above each skeleton that hangs below something the rigid
    // hierarchy moves.
    //
    // glTF 2.0, 3.7.3.2 keeps a skinned mesh node's own transform out of the
    // draw, and that is the rule everyone implements. It says nothing about the
    // chain above the joints, which does move the skeleton: `BrainStem` parents
    // its eighteen joints under a node carrying 1309 keys of translation,
    // rotation and scale, and freezing that node was this model animating its
    // limbs while standing still.
    //
    // Nothing to do for a document with no rigid animation, and nothing to do
    // for the skeletons whose chain really is static: they carry no links.
    fn carrySkeletons(self: *World) void {
        const animator = if (self.node_animator) |*node| node else return;
        for (self.skeletons) |*skeleton| {
            for (skeleton.prefix_links) |link| {
                skeleton.animator.pose.setRootPrefix(
                    link.slot,
                    animator.world_transforms[link.node_slot],
                );
            }
        }
    }

    // Copies each posed mesh's joint transforms into the run the offsets
    // assigned it. The offsets were planned once: they are a pure function of
    // the joint counts being drawn, and no draw is added or removed here.
    //
    // Public because `update` is the only caller and `update` needs a device,
    // which would leave the one place three arrays have to agree reachable by
    // nothing a test can run.
    pub fn packJoints(self: *World) void {
        for (self.skin_of_mesh, self.plan.joint_bases) |maybe_skin, base| {
            const skin = maybe_skin orelse continue;
            // The mesh's own run of its skeleton's joints, not the whole of
            // them: a skeleton shared by several skins holds all of their joints
            // end to end, and the vertex attribute indexes one run.
            const transforms = self.skeletons[skin.skeleton].animator.jointTransforms();
            @memcpy(
                self.joint_storage[base..][0..skin.joint_count],
                transforms[skin.joint_offset..][0..skin.joint_count],
            );
        }
    }

    // Starts one document animation everywhere it means something, and returns
    // how many animators took it.
    //
    // Skins and the rigid hierarchy share this index: each holds one clip per
    // animation in the document, so switching them together is what keeps a
    // character's limbs and the prop anchored to its hand on the same authored
    // animation. Switching only one of them was the state this replaces.
    //
    // Morph weights cannot follow it. The importer keeps only the animations
    // that drive a given node's weights and discards which document animation
    // each of them was, so there is no index to match against and the blend
    // stays on whatever it was started with. Aligning the two needs that
    // mapping to survive the import.
    pub fn playClip(self: *World, index: u16) !usize {
        var started: usize = 0;
        if (self.node_animator) |*animator| {
            if (index < animator.template.clips.len) {
                try animator.play(index);
                started += 1;
            }
        }
        for (self.skeletons) |*skeleton| {
            if (index < skeleton.animator.clips.len) {
                try skeleton.animator.play(index);
                started += 1;
            }
        }
        return started;
    }

    // How many document animations anything in this world can play. Skeletons
    // and the rigid hierarchy agree on it, so either one answers.
    pub fn clipCount(self: *const World) usize {
        if (self.node_animator) |*animator| return animator.template.clips.len;
        if (self.skeletons.len > 0) return self.skeletons[0].animator.clips.len;
        return 0;
    }

    pub fn activeClip(self: *const World) ?u16 {
        if (self.node_animator) |*animator| return animator.active_clip;
        if (self.skeletons.len > 0) return self.skeletons[0].animator.active_clip;
        return null;
    }

    // Whether anything the frame draws can move. A world where nothing does lets
    // the renderer bake its shadow map once instead of every frame.
    pub fn castersMove(self: *const World) bool {
        if (self.node_animator != null) return true;
        if (self.morph_animators.len > 0) return true;
        for (self.skeletons) |*skeleton| {
            if (skeleton.animator.clips.len > 0) return true;
        }
        return false;
    }

    fn buildSkeletons(self: *World, model: *const gltf.importer.Model) !void {
        const allocator = self.allocator;

        // The whole slice at once, so teardown frees the allocation it was
        // given. `skeletons_ready` is what says how much of it holds an
        // animator.
        self.skeletons = try allocator.alloc(Skeleton, model.skins.len);
        for (model.skins, self.skeletons) |*source, *skeleton| {
            skeleton.* = .{
                // Every clip the document gave this skeleton, not the first:
                // which one plays is playback's choice and the world offers no
                // way to make it here.
                .animator = try .init(allocator, &source.skeleton, source.clips),
                .prefix_links = source.prefix_links,
            };
            self.skeletons_ready += 1;
        }

        // glTF 2.0 specification, 3.7.3.3: a skinned mesh primitive carries the
        // attributes skinning reads, and the node referencing it names the skin
        // they index. A document with the stream and no matching skin hands the
        // shader an array to index and nothing to fill it with.
        self.skin_of_mesh = try allocator.alloc(?gltf.importer.SkinPlacement, model.meshes.len);
        for (model.meshes, self.skin_of_mesh) |*mesh, *slot| {
            if (!mesh.streams.skinned) {
                slot.* = null;
                continue;
            }
            const placement = mesh.skin orelse return error.SkinnedMeshWithoutSkin;
            if (placement.skeleton >= self.skeletons.len) return error.SkinnedMeshWithoutSkin;
            if (placement.joint_count == 0) return error.EmptySkin;

            // The run has to be inside the skeleton that holds it. Checked here
            // because `packJoints` copies it every frame with no bound of its
            // own, and because the importer's arithmetic is the only thing
            // between an asset and this.
            const joints = self.skeletons[placement.skeleton].animator.pose.jointCount();
            if (placement.joint_offset + placement.joint_count > joints)
                return error.JointIndexOutOfRange;

            // The shader has no bound of its own, and the indices came from a
            // file. Answered once here, so the vertex path carries no check.
            // Against the mesh's own run, which is the space the attribute
            // addresses rather than the whole skeleton.
            for (mesh.vertices) |vertex| {
                inline for (0..4) |lane| {
                    if (vertex.joints[lane] >= placement.joint_count)
                        return error.JointIndexOutOfRange;
                }
            }
            slot.* = placement;
        }
    }

    fn buildAnimators(self: *World, model: *const gltf.importer.Model) !void {
        const allocator = self.allocator;

        // One rigid animator over the whole document: the template is one
        // hierarchy holding every dynamic node the default scene reaches, and a
        // mesh names the slot inside it that moves the mesh. Init propagates the
        // bind pose, so the world transforms are readable before a clip is ever
        // played.
        // The animator keeps this address for as long as it runs, so the model
        // has to outlive the world. It is the model that owns the box, which is
        // why passing the pointer on is enough and copying it would not be.
        if (model.node_animation) |template| {
            self.node_animator = try res.NodeAnimator.init(allocator, template);
        }

        self.morph_animators = try allocator.alloc(res.MorphAnimator, model.morph_templates.len);
        for (model.morph_templates, self.morph_animators) |*template, *animator| {
            animator.* = try res.MorphAnimator.init(allocator, template);
            self.morph_ready += 1;
        }
    }

    fn buildPlan(
        self: *World,
        model: *const gltf.importer.Model,
        meshes: []const *const gpu.Mesh,
        capacity: gpu.FrameCapacity,
    ) !void {
        const allocator = self.allocator;
        const count = model.meshes.len;

        const candidates = try allocator.alloc(RecordPlan.Draw, count);
        self.plan.candidates = candidates;
        const centres = try allocator.alloc(res.Vec3, count);
        self.plan.centres = centres;
        const local_bounds = try allocator.alloc(res.Aabb, count);
        self.plan.local_bounds = local_bounds;
        const cullable = try allocator.alloc(bool, count);
        self.plan.cullable = cullable;
        const joint_bases = try allocator.alloc(u32, count);
        self.plan.joint_bases = joint_bases;
        self.plan.matrices = try allocator.alloc(zm.Mat, count);
        self.plan.keys = try allocator.alloc(scene.DrawKey, count);
        self.plan.order = try allocator.alloc(u32, count);
        self.plan.instances = try allocator.alloc(gpu.Instance, count);
        self.plan.vertex_sources = try allocator.alloc(?gpu.MeshVertexSource, count);
        self.plan.batch_storage = try allocator.alloc(RecordPlan.Batch, count);
        self.plan.record_storage = try allocator.alloc(gpu.RecordBatch, count);
        self.plan.ordered = &.{};
        self.plan.batches = &.{};
        self.plan.records = &.{};
        @memset(self.plan.vertex_sources, null);

        for (model.meshes, meshes, candidates, self.plan.keys) |*mesh, resident, *candidate, *key| {
            candidate.* = .{
                .mesh = resident,
                .material = mesh.material,
                // Section 3.9.5: doubleSided false means the back faces are
                // culled, and true means neither side is. What makes reading it
                // safe is that the winding is now settled on both sides of the
                // load: the importer reverses a mirrored primitive's corners,
                // and `rebuild` carries the instance matrix's sign into
                // `front_face` for the recorder to set.
                //
                // Skinned geometry is the exception and keeps both sides. Its
                // joint matrices can mirror a vertex without the instance
                // matrix saying so, and they are per vertex, so no state on
                // this draw can answer for its winding. Dropping a side there
                // would drop parts of a character rather than the back of a
                // surface.
                .face_culling = if (mesh.skin != null or
                    model.materials[mesh.material].rendering.double_sided)
                    .none
                else
                    .back,
                // Overwritten every frame by `rebuild` before anything reads
                // it. Named rather than left undefined so that a plan built and
                // never rebuilt is still a plan and not a trap.
                .front_face = .counter_clockwise,
            };
            key.* = .{
                .layer = switch (model.materials[mesh.material].rendering.alpha_mode) {
                    .@"opaque", .mask => .solid,
                    .blend => .blended,
                },
                .depth = undefined,
            };
        }

        // Kept per mesh as well as unioned, because the blended run is ordered
        // by each draw's own distance from the eye and recomputing these beside
        // the plan would walk every vertex a second time.
        for (model.meshes, centres, local_bounds, cullable, 0..) |
            *mesh,
            *centre,
            *box,
            *may_cull,
            index,
        | {
            var placed: res.Aabb = .compute(mesh.vertices);
            if (self.skin_of_mesh[index]) |skin|
                placed = scene.worldAabb(
                    placed,
                    self.skeletons[skin.skeleton].animator.jointTransforms()[skin.joint_offset],
                );
            centre.* = scene.sphereAroundAabb(placed).centre;
            box.* = placed;
            // A skinned mesh's box is its bind pose and a morphed mesh's is its
            // base shape, and in both cases the vertices leave it. Neither is
            // culled, so neither can vanish while it is on screen.
            may_cull.* = self.skin_of_mesh[index] == null and
                self.morph_registrations[index] == null;

            // The bind pose is what the bounds describe. An anchored mesh
            // reaches world space only through its animator slot, so unioning
            // its baked vertices directly would measure geometry sitting at the
            // anchor's origin instead of where the asset puts it.
            const bind = instanceMatrix(
                mesh.anchor,
                if (self.node_animator) |*animator| animator else null,
                self.placement,
            );
            const in_world = scene.worldAabb(placed, bind);
            self.bounds = if (index == 0) in_world else scene.unionAabb(self.bounds, in_world);
        }
        self.sphere = scene.sphereAroundAabb(self.bounds);

        // Planned once: the offsets are a pure function of the joint counts
        // being drawn, and no draw is added or removed after this.
        //
        // Each mesh's own count, not its skeleton's. A skeleton shared by
        // several skins holds every one of their joints, and a mesh uploads only
        // the run it reads.
        const joint_counts = try allocator.alloc(?u32, count);
        defer allocator.free(joint_counts);
        for (self.skin_of_mesh, joint_counts) |maybe_skin, *joints| {
            joints.* = if (maybe_skin) |skin| skin.joint_count else null;
        }
        self.joint_total = try scene.assignJointOffsets(
            joint_counts,
            joint_bases,
            @intCast(capacity.joints),
        );
        self.joint_storage = try allocator.alloc(gpu.Joint, capacity.joints);
    }
};
