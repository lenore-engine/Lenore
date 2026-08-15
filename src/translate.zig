const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const imui = @import("lenore-imui");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const log = std.log.scoped(.lenore);

// Where a value crosses from one lenore module into another.
//
// This file exists because of a rule rather than a preference: no module names a
// sibling, so nothing but the engine may hold a value from two of them at once.
// `lenore-gltf` cannot produce a `scene.Light` and `lenore-gpu` cannot accept
// one, so the conversion between them has exactly one possible home. The
// importer says the same thing from its own side: what turns a document into
// draws is composition, and composition is the engine's.
//
// The consequence is worth stating plainly. Anyone who builds their own engine
// on these modules writes this file again. It is the price of the modules not
// knowing each other, and it is paid here once.

// The engine's instantiation of the scene's batching over resident meshes. The
// scene decides order and face policy without naming a device; the pointer it
// carries is the engine's business, and this is where the two are joined.
pub const RecordPlan = scene.DrawBatches(*const gpu.Mesh, u32);

pub const BatchError = error{
    BatchDestinationTooSmall,
    BatchDrawOutOfRange,
};

// The deliberate module-boundary copy: scene owns ordering and face policy, GPU
// owns resource pointers and Vulkan state. The destination is preallocated by
// the caller, so nothing here allocates and this may sit in the frame loop.
//
// Four slices arrive independently and are indexed through one another, so the
// ranges are checked rather than assumed. They cannot be justified by
// construction at this level: `ordered` and `sources` are the caller's, and
// nothing in the signature says they came from the same rebuild. The cost is two
// comparisons per batch, against a count bounded by the draws in a frame.
pub fn recordBatches(
    batches: []const RecordPlan.Batch,
    ordered: []const u32,
    sources: []const ?gpu.MeshVertexSource,
    destination: []gpu.RecordBatch,
) BatchError![]gpu.RecordBatch {
    if (destination.len < batches.len) return error.BatchDestinationTooSmall;

    for (batches, destination[0..batches.len]) |batch, *record| {
        // `first_instance` is a position within `ordered`, so this resolves to
        // the first draw the batch coalesced. Any draw of the batch would do:
        // batching keys on the mesh, so every draw in one resolves to a single
        // vertex source.
        if (batch.first_instance >= ordered.len) return error.BatchDrawOutOfRange;
        const draw = ordered[batch.first_instance];
        if (draw >= sources.len) return error.BatchDrawOutOfRange;

        record.* = .{
            .mesh = batch.mesh,
            .material_index = batch.material,
            .cull_mode = switch (batch.face_culling) {
                .none => .{},
                .back => .{ .back_bit = true },
                .front => .{ .front_bit = true },
            },
            // The two enumerations name the same two windings, and the sense of
            // `counter_clockwise` is the same on both sides: `rasterizationState`
            // records why the glTF convention survives `vulkanClip` unchanged.
            .front_face = switch (batch.front_face) {
                .counter_clockwise => .counter_clockwise,
                .clockwise => .clockwise,
            },
            .first_instance = batch.first_instance,
            .instance_count = batch.instance_count,
            .vertex_source = sources[draw],
        };
    }
    return destination[0..batches.len];
}

// The base an instance record carries, from what the scene's plan assigned it.
//
// The plan marks an entity with no pose, and that marker is not an index: the
// unskinned pipeline reads no joint array, so the marker becomes zero here
// rather than reaching the device as a number that means something else.
pub fn jointBase(planned: u32) u32 {
    return if (planned == scene.no_joint_base) 0 else planned;
}

pub const VertexSourceError = error{VertexSourceCountMismatch};

// Where each mesh fetches its vertices this frame, by mesh index.
//
// A morphed mesh reads the slot the prepass is about to write for `frame`, and
// every other one reads its own buffer, which `RecordBatch` spells as null.
pub fn fillVertexSources(
    pass: *const gpu.MorphPass,
    registrations: []const ?u32,
    frame: usize,
    destination: []?gpu.MeshVertexSource,
) VertexSourceError!void {
    if (registrations.len != destination.len) return error.VertexSourceCountMismatch;
    for (registrations, destination) |registration, *source| {
        source.* = if (registration) |index| pass.vertexSource(index, frame) else null;
    }
}

// One document light, placed by the transform of the node that references it.
//
// KHR_lights_punctual, "Light Shared Properties": a light with a direction emits
// along the 3-vector (0, 0, -1) in its own space, and the node's rotation orients
// it. That local axis is what goes to the constructor; the world matrix turns it
// into where the light points.
//
// Every light goes through `lenore-scene` rather than being packed straight from
// the document. That is where a direction is normalized, a cone is ordered and a
// range is derived, and it is the only place that rejects a light instead of
// drawing a wrong one.
pub fn sceneLight(entry: gltf.importer.Light) scene.LightError!scene.Light {
    const source = entry.source;
    const local_axis = res.Vec3{ 0, 0, -1 };
    const colour = res.Vec3{ source.color[0], source.color[1], source.color[2] };

    // KHR_lights_punctual, "Range Property": range is optional and undefined
    // means infinite. The shader has no infinite case, so the distance at which
    // the falloff is spent stands in for it, which is what `rangeFor` computes.
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

// A scene light in the layout the shader reads. The two types are separate
// because `lenore-gpu` does not know `lenore-scene`, which is what keeps a
// device layout out of the module that has no device in it.
pub fn packLight(light: scene.Light) gpu.LightUniform {
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

// A document's lights in the block the shader reads, returned as the prefix of
// `out` that holds one.
//
// A light the scene module rejects is dropped and the rest still draw: one light
// a document got wrong is not the frame. Lights past the block's capacity are
// dropped too, which the caller can see by comparing the length it gets back
// against the length it passed in.
pub fn packDocumentLights(
    document_lights: []const gltf.importer.Light,
    out: *[gpu.max_lights]gpu.LightUniform,
) []const gpu.LightUniform {
    var count: usize = 0;
    for (document_lights, 0..) |entry, index| {
        if (count == out.len) break;
        const placed = sceneLight(entry) catch |err| {
            log.warn("light {d} rejected: {t}", .{ index, err });
            continue;
        };
        out[count] = packLight(placed);
        count += 1;
    }
    return out[0..count];
}

// What the window says, in the vocabulary the UI routes.
//
// Two enumerations that name overlapping sets of things, so every arm is
// written out and the ones with no counterpart are named rather than caught by
// an else. The window speaks a keyboard and a mouse; the UI speaks the few
// transitions a widget can act on, and what it has no word for is not an
// omission to be worked around here.
//
// The surface configuration is folded as the batch is walked rather than taken
// once for the whole batch. A resize submits new metrics in the middle of the
// events it affects, and a position is only meaningful under the configuration
// it was measured against.
// What a logical rectangle is multiplied by to land on the raster.
//
// The framebuffer/logical ratio and not `metrics.scale`, for the reason
// `lenore-platform` gives where a position is converted: a fractional content
// scale has already been rounded into the integer extent. Null where the window
// has reported no configuration a ratio can be taken from.
//
// Positions and rectangles therefore reach the UI through the same number,
// which is what keeps a widget's geometry and the pointer that hits it in one
// space.
pub fn uiScale(metrics: platform.SurfaceMetrics) ?imui.ScaleFactor {
    if (metrics.logical_size[0] <= 0 or metrics.logical_size[1] <= 0) return null;
    return imui.ScaleFactor.init(
        @as(f32, @floatFromInt(metrics.framebuffer_extent.width)) / metrics.logical_size[0],
        @as(f32, @floatFromInt(metrics.framebuffer_extent.height)) / metrics.logical_size[1],
    ) catch null;
}

pub const UiEvents = struct {
    metrics: ?platform.SurfaceMetrics,

    // Seeded from what the window has already reported, which the polled input
    // state holds from the moment the window is created. A translator that
    // started empty would place nothing until the next resize.
    pub fn init(metrics: ?platform.SurfaceMetrics) UiEvents {
        return .{ .metrics = metrics };
    }

    // One event in the UI's terms, or null where the UI has no word for it.
    pub fn translate(self: *UiEvents, event: platform.Event) ?imui.Event {
        switch (event.payload) {
            .surface_metrics => |metrics| {
                self.metrics = metrics;
                return null;
            },
            .cursor => |cursor| {
                const position = self.place(cursor.logical_position, cursor.metrics_generation) orelse
                    // A motion that cannot be placed is a motion nothing knows
                    // about, and the pointer keeps the last position it was
                    // known at. That is the module's own rule for a position it
                    // refuses, and the next motion answers.
                    return null;
                return .{ .pointer_move = position };
            },
            .mouse_button => |button| {
                const mapped: imui.PointerButton = switch (button.button) {
                    .left => .primary,
                    .right => .secondary,
                    .middle => .middle,
                    // Nothing a widget acts on. A back or forward button is
                    // navigation, which belongs to whoever owns a history.
                    .back, .forward, .other => return null,
                };
                const action: imui.ButtonAction = switch (button.action) {
                    .press => .press,
                    .release => .release,
                    // A button transition is one or the other. The action type
                    // is shared with the keyboard, where a held key repeats.
                    .repeat => return null,
                };
                const position = self.place(button.logical_position, button.metrics_generation) orelse
                    // Unlike a motion, a transition that is dropped can strand
                    // a gesture: the release that would have ended it is the
                    // one lost, and the widget stays held for as long as the
                    // frame lasts. Abandoning the gesture is what `cancel` is
                    // for, and it does nothing when none is in progress.
                    return .cancel;
                return .{ .pointer_button = .{
                    .position = position,
                    .button = mapped,
                    .action = action,
                    .shift = button.modifiers.shift,
                } };
            },
            .key => |key| {
                const mapped: imui.Key = switch (key.physical) {
                    .tab => .tab,
                    // Both enters actuate. Which key was struck is a difference
                    // no widget has ever asked about.
                    .enter, .numpad_enter => .enter,
                    .space => .space,
                    .escape => .escape,
                    .arrow_left => .left,
                    .arrow_right => .right,
                    .arrow_up => .up,
                    .arrow_down => .down,
                    else => return null,
                };
                return .{ .key = .{
                    .key = mapped,
                    .action = switch (key.action) {
                        .press => .press,
                        .repeat => .repeat,
                        .release => .release,
                    },
                    .shift = key.modifiers.shift,
                } };
            },
            .focus => |focus| return .{ .focus = focus.focused },
            // The UI declares no event for either. A wheel needs a scrollable
            // region and a character needs a text target, and a translation
            // that invented one here would be answering for a widget that does
            // not exist.
            .text, .scroll => return null,
        }
    }

    fn place(self: *const UiEvents, logical: [2]f32, generation: u32) ?imui.Point {
        const metrics = self.metrics orelse return null;
        const position = platform.framebufferPosition(metrics, logical, generation) orelse
            return null;
        return .{ .x = position[0], .y = position[1] };
    }
};
