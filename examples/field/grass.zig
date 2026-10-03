// The foliage pass: three buffers, three pipelines, and no blade on the host.
//
// What this owns is the arrangement, not the field. Where a blade stands is
// decided in `shaders/grass.slang` from the coordinates of the cell it grows in;
// what is here is the storage those decisions are written into, the terrain map
// they are read from, and the two places in the frame they happen: a dispatch
// before the main pass and an indirect draw inside it.
//
// It adds nothing to `lenore-gpu`. The module already creates buffers with any
// usage the caller names, already carries `cmdDrawIndirect` in its bindings, and
// already hands a driver the command buffer of the main pass between the scene's
// draws and its end. That the engine had never drawn anything indirect was a
// fact about its consumers rather than about its surface.

const std = @import("std");
const gpu = @import("lenore-gpu");
const lenore = @import("lenore");

const terrain = @import("terrain.zig");

const Allocator = std.mem.Allocator;

// Mirrors `Blade` in the shader. Measured from the compiled module's reflection
// rather than derived from the packing rules: `position` at 0, `yaw` at 12,
// `height` at 16, `width` at 20, `tint` at 24, `kind` at 28.
pub const Blade = extern struct {
    position: [3]f32,
    yaw: f32,
    height: f32,
    width: f32,
    tint: f32,
    kind: u32,
};

// Mirrors `PlantPush`. Both blocks below were read off the same reflection, and
// both begin at offset zero: Slang emits one push constant block per entry point
// rather than concatenating them, which is what makes two blocks in one module
// legal here and what would silently corrupt every draw if it were not so.
const PlantPush = extern struct {
    camera_position: [3]f32,
    time: f32,

    camera_forward: [3]f32,
    base_spacing: f32,

    extent: f32,
    relief: f32,
    map_resolution: u32,
    capacity: u32,

    grass_height: f32,
    wheat_height: f32,
    height_spread: f32,
    ring_side: u32,

    cull_cos: f32,
    density_scale: f32,
    blade_vertices: u32,
    padding: u32 = 0,
};

const DrawPush = extern struct {
    view_projection: [16]f32,

    camera_position: [3]f32,
    time: f32,

    sun_direction: [3]f32,
    lean: f32,

    sun_colour: [3]f32,
    ambient: f32,
};

comptime {
    // Held against the shader's own reflection. A field inserted above without
    // the same field in the `.slang` moves everything after it, and the result
    // is a plausible picture drawn from the wrong numbers rather than an error.
    std.debug.assert(@sizeOf(Blade) == 32);
    std.debug.assert(@offsetOf(Blade, "yaw") == 12);
    std.debug.assert(@offsetOf(Blade, "height") == 16);
    std.debug.assert(@offsetOf(Blade, "width") == 20);
    std.debug.assert(@offsetOf(Blade, "tint") == 24);
    std.debug.assert(@offsetOf(Blade, "kind") == 28);

    std.debug.assert(@sizeOf(PlantPush) == 80);
    std.debug.assert(@offsetOf(PlantPush, "camera_forward") == 16);
    std.debug.assert(@offsetOf(PlantPush, "extent") == 32);
    std.debug.assert(@offsetOf(PlantPush, "grass_height") == 48);
    std.debug.assert(@offsetOf(PlantPush, "cull_cos") == 64);

    std.debug.assert(@sizeOf(DrawPush) == 112);
    std.debug.assert(@offsetOf(DrawPush, "camera_position") == 64);
    std.debug.assert(@offsetOf(DrawPush, "sun_direction") == 80);
    std.debug.assert(@offsetOf(DrawPush, "sun_colour") == 96);

    // `VkDrawIndirectCommand` is four 32-bit words, and the reset entry point
    // writes exactly that many.
    std.debug.assert(@sizeOf(IndirectCommand) == 16);
}

// `VkDrawIndirectCommand`, in the layout the specification fixes. Declared here
// rather than taken from the bindings because the host never fills it: the
// device writes all four words and the host only reads the count back for the
// panel.
pub const IndirectCommand = extern struct {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

// The blade's vertex count, which the reset entry point writes into the command
// and which has to be the number `blade_vertex_count` states in the shader.
// Three quads of six vertices and a tip of three.
pub const blade_vertices: u32 = 3 * 6 + 3;

pub const Settings = struct {
    // Metres between planting candidates in the nearest ring. Every ring out
    // doubles it.
    base_spacing: f32 = 0.085,
    // Cells along one edge of a ring. Each ring dispatches this squared threads,
    // whatever its spacing, so this alone sets the cost of planting.
    ring_side: u32 = 256,
    // How many rings. Ring k reaches `base_spacing * 2^k * ring_side / 2` metres,
    // so four rings at the defaults reach 87 m and the fifth would reach 174,
    // which is past the domain.
    rings: u32 = 4,

    // The most blades one frame may hold. The planter counts past it and the
    // draw clamps, so this bounds memory and not correctness.
    capacity: u32 = 160_000,

    // Multiplies the density the map carries. One plants the field as the
    // terrain describes it; less thins it without changing where the field is.
    density: f32 = 1.0,

    // The cosine of the half angle planted around the view direction. A cone
    // rather than the frustum: it has no aspect ratio in it, and what it is for
    // is discarding the three quarters of every ring behind the camera. Wide
    // enough that a blade entering from the side is already there.
    cull_cos: f32 = -0.35,

    // How far a blade leans at rest, as a fraction of its own height.
    lean: f32 = 0.22,

    // Metres to the tip, and how far one blade may differ from its kind's
    // nominal height either way.
    //
    // Here and not in the terrain's settings, which describe where the field is
    // and not what stands in it. The terrain answers how much grows at a point;
    // how tall the thing that grows there is belongs to whatever draws it.
    grass_height: f32 = 0.42,
    wheat_height: f32 = 0.95,
    height_spread: f32 = 0.34,

    // What the blades are lit by beyond the sun, as a fraction of the sunlight.
    //
    // A number and not a sky, and that is the honest statement of what is
    // missing: with no environment loaded the engine's image-based terms are
    // exactly zero, so a blade facing away from the sun has nothing at all on
    // it. This stands in until the sky is something the shading can sample.
    ambient: f32 = 0.35,
};

pub const Grass = struct {
    context: *const gpu.Context,
    allocator: Allocator,
    settings: Settings,

    map: gpu.Buffer,
    blades: gpu.Buffer,
    command: gpu.Buffer,
    // One per frame in flight. The count is read a frame late, which is what
    // makes reading it free: the buffer a frame wrote is not the one the next
    // frame is writing.
    counts: []gpu.Buffer,
    // Whether a slot holds a completed frame's copy. Reading one before any
    // frame has written it would report whatever the allocation happened to
    // contain, which is a number and therefore indistinguishable from an answer.
    // The engine's own timer keeps the same flag for the same reason.
    counted: []bool,

    sets: Sets,
    shaders: Shaders,

    map_resolution: u32,
    extent: f32,
    relief: f32,

    // What the last completed frame planted, for the panel. Null until one has
    // come back.
    planted: ?u32 = null,

    const bindings = [_]gpu.DescriptorBinding{
        .{
            .slot = 0,
            .name = "map",
            .kind = .storage_buffer,
            .stages = .{ .compute_bit = true },
        },
        .{
            .slot = 1,
            .name = "blades",
            .kind = .storage_buffer,
            .stages = .{ .compute_bit = true, .vertex_bit = true },
        },
        .{
            .slot = 2,
            .name = "command",
            .kind = .storage_buffer,
            .stages = .{ .compute_bit = true },
        },
    };

    const Sets = gpu.DescriptorSets(&bindings);
    const LayoutConfig = gpu.PipelineLayoutConfig;
    const Spec = gpu.ShaderEffectSpec;

    const Shaders = gpu.ShaderEffect(.{
        .modules = .{"grass"},
        .layouts = .{ "plant", "draw" },
        .pipelines = .{
            .reset = Spec{
                .module = "grass",
                .layout = "plant",
                .stage = .{ .compute = "resetMain" },
            },
            .plant = Spec{
                .module = "grass",
                .layout = "plant",
                .stage = .{ .compute = "plantMain" },
            },
            .blade = Spec{
                .module = "grass",
                .layout = "draw",
                .stage = .{
                    .graphics = .{
                        .vertex = "bladeVertex",
                        .fragment = "bladeFragment",
                        // Depth tested and written, no blending: a blade is opaque
                        // where it covers a pixel and absent where it does not, and
                        // it has to occlude the blades behind it.
                        .mode = .solid,
                        // Empty flags, which is no culling. A blade is a surface with
                        // two sides and both are seen; the fragment stage flips the
                        // normal for the far one.
                        .culling = .{ .fixed = .{} },
                    },
                },
            },
        },
    });

    pub fn init(
        engine: *lenore.Engine,
        allocator: Allocator,
        field: terrain.Terrain,
        settings: Settings,
    ) !Grass {
        const context = &engine.context;

        // The map, baked on the host and uploaded once. This is the whole of
        // what the device knows about the field: where the ground is and how
        // much grows there, in one image's worth of bytes, rather than a second
        // implementation of the terrain noise in Slang.
        var map_data = try terrain.bake(allocator, field);
        defer map_data.deinit(allocator);

        // Host visible, and written directly rather than staged. `upload` is a
        // memcpy into mapped memory and refuses a class that is not mapped, so
        // this class is what makes the line below legal. It is also what the
        // target wants: the reference part shares one memory bus between the
        // processor and the graphics, so a staging copy of eight megabytes would
        // be the same bytes crossing it twice to end up where they started.
        //
        // Written once, before any frame is submitted, so nothing synchronises
        // it. The specification puts that on the caller and here the caller has
        // nothing to synchronise against.
        var map = try gpu.Buffer.init(
            context,
            &engine.memory,
            map_data.texels.len,
            .{ .storage_buffer_bit = true },
            .upload,
        );
        errdefer map.deinit();
        try map.upload(map_data.texels);

        var blades = try gpu.Buffer.init(
            context,
            &engine.memory,
            @as(u64, settings.capacity) * @sizeOf(Blade),
            .{ .storage_buffer_bit = true },
            .device,
        );
        errdefer blades.deinit();

        var command = try gpu.Buffer.init(
            context,
            &engine.memory,
            @sizeOf(IndirectCommand),
            .{
                .storage_buffer_bit = true,
                .indirect_buffer_bit = true,
                .transfer_src_bit = true,
            },
            .device,
        );
        errdefer command.deinit();

        const counts = try allocator.alloc(gpu.Buffer, engine.frames.len);
        errdefer allocator.free(counts);
        var made: usize = 0;
        errdefer for (counts[0..made]) |*buffer| buffer.deinit();
        for (counts) |*buffer| {
            buffer.* = try gpu.Buffer.init(
                context,
                &engine.memory,
                @sizeOf(IndirectCommand),
                .{ .transfer_dst_bit = true },
                .readback,
            );
            made += 1;
        }

        const counted = try allocator.alloc(bool, engine.frames.len);
        errdefer allocator.free(counted);
        @memset(counted, false);

        var sets = try Sets.init(context, allocator, 1);
        errdefer sets.deinit(context, allocator);

        var shaders = try Shaders.init(context, .{
            .modules = .{ .grass = gpu.spirvWords(@embedFile("grass")) },
            .layouts = .{
                .plant = LayoutConfig{
                    .descriptor_sets = &.{sets.layout},
                    .push_constants = &.{.{
                        .stage_flags = .{ .compute_bit = true },
                        .offset = 0,
                        .size = @sizeOf(PlantPush),
                    }},
                },
                .draw = LayoutConfig{
                    .descriptor_sets = &.{sets.layout},
                    .push_constants = &.{.{
                        .stage_flags = .{ .vertex_bit = true, .fragment_bit = true },
                        .offset = 0,
                        .size = @sizeOf(DrawPush),
                    }},
                },
            },
            .formats = engine.renderer.mainPassFormats(),
        });
        errdefer shaders.deinit(context);

        const self: Grass = .{
            .context = context,
            .allocator = allocator,
            .settings = settings,
            .map = map,
            .blades = blades,
            .command = command,
            .counts = counts,
            .counted = counted,
            .sets = sets,
            .shaders = shaders,
            .map_resolution = map_data.resolution,
            .extent = field.settings.extent(),
            .relief = field.settings.relief,
        };

        // Keyed by binding name rather than by position, so two of these
        // exchanged is a compile error and not a shader reading the wrong
        // buffer.
        self.sets.writeBuffers(context, 0, .{
            .map = &self.map,
            .blades = &self.blades,
            .command = &self.command,
        });

        return self;
    }

    pub fn deinit(self: *Grass) void {
        self.shaders.deinit(self.context);
        self.sets.deinit(self.context, self.allocator);
        for (self.counts) |*buffer| buffer.deinit();
        self.allocator.free(self.counts);
        self.allocator.free(self.counted);
        self.command.deinit();
        self.blades.deinit();
        self.map.deinit();
        self.* = undefined;
    }

    // How many blades one frame plants at most, which is the cost of the
    // dispatch whatever the field does with it.
    pub fn candidates(self: Grass) u32 {
        return self.settings.ring_side * self.settings.ring_side * self.settings.rings;
    }

    // Recorded before the main pass opens.
    //
    // Three steps and two barriers, and the barriers are the whole of the
    // ordering: the reset writes the count the planter increments, and the
    // planter writes the blades and the count the draw reads. Neither is implied
    // by submission order inside one command buffer.
    pub fn plant(
        self: *Grass,
        engine: *lenore.Engine,
        commands: gpu.vk.CommandBuffer,
        time: f32,
    ) void {
        const device = &self.context.device;
        const placement = engine.camera.placement();

        const push: PlantPush = .{
            .camera_position = .{ placement.position[0], placement.position[1], placement.position[2] },
            .time = time,
            .camera_forward = .{ placement.front[0], placement.front[1], placement.front[2] },
            .base_spacing = self.settings.base_spacing,
            .extent = self.extent,
            .relief = self.relief,
            .map_resolution = self.map_resolution,
            .capacity = self.settings.capacity,
            .grass_height = self.settings.grass_height,
            .wheat_height = self.settings.wheat_height,
            .height_spread = self.settings.height_spread,
            .ring_side = self.settings.ring_side,
            .cull_cos = self.settings.cull_cos,
            .density_scale = self.settings.density,
            .blade_vertices = blade_vertices,
        };

        const layout = self.shaders.layoutFor(.reset);
        device.cmdBindDescriptorSets(commands, .compute, layout, 0, &.{self.sets.set(0)}, &.{});
        var values = push;
        device.cmdPushConstants(
            commands,
            layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PlantPush),
            @ptrCast(&values),
        );

        device.cmdBindPipeline(commands, .compute, self.shaders.get(.reset));
        device.cmdDispatch(commands, 1, 1, 1);

        gpu.recordMemoryBarrier(self.context, commands, .{
            .src_stage = .{ .compute_shader_bit = true },
            .src_access = .{ .shader_storage_write_bit = true },
            .dst_stage = .{ .compute_shader_bit = true },
            .dst_access = .{
                .shader_storage_read_bit = true,
                .shader_storage_write_bit = true,
            },
        });

        device.cmdBindPipeline(commands, .compute, self.shaders.get(.plant));
        // The workgroup is eight by eight and the third dimension is the ring,
        // so one dispatch covers every ring and the shader reads which it is in
        // from its own z.
        const groups = (self.settings.ring_side + 7) / 8;
        device.cmdDispatch(commands, groups, groups, self.settings.rings);

        // Two consumers, and they are reached at different points: the vertex
        // stage reads the blades, and the indirect stage reads the count before
        // any shader runs at all. A barrier naming only the shader stage would
        // let the draw fetch a command the planter had not finished writing.
        gpu.recordMemoryBarrier(self.context, commands, .{
            .src_stage = .{ .compute_shader_bit = true },
            .src_access = .{ .shader_storage_write_bit = true },
            .dst_stage = .{
                .vertex_shader_bit = true,
                .draw_indirect_bit = true,
                // The readback copy below reads the same word the planter just
                // wrote. Leaving the transfer stage out of this dependency is a
                // race the picture never shows: the draw would still be correct,
                // and only the number reported beside it would be a word from
                // some other moment.
                .copy_bit = true,
            },
            .dst_access = .{
                .shader_storage_read_bit = true,
                .indirect_command_read_bit = true,
                .transfer_read_bit = true,
            },
        });

        // The count this frame produced, copied where the host can read it once
        // the frame comes back. A copy and not a map of the command buffer
        // itself: that one is device local so the draw reads it at full speed,
        // and a readback allocation is host visible by class.
        device.cmdCopyBuffer(
            commands,
            self.command.handle,
            self.counts[engine.frame_index].handle,
            &.{.{ .src_offset = 0, .dst_offset = 0, .size = @sizeOf(IndirectCommand) }},
        );
        self.counted[engine.frame_index] = true;
    }

    // Recorded inside the main pass, after the scene's draws.
    //
    // The draw count is one and the instance count is a word the host has never
    // seen. That is the whole of what indirect buys here: how much grass is in
    // front of the camera is decided on the device, in the same dispatch that
    // decided where it is.
    pub fn draw(
        self: *Grass,
        engine: *lenore.Engine,
        commands: gpu.vk.CommandBuffer,
        time: f32,
        sun_direction: [3]f32,
        sun_colour: [3]f32,
    ) !void {
        const device = &self.context.device;
        const placement = engine.camera.placement();
        const view_projection = try engine.camera.viewProjection(engine.aspect());

        var push: DrawPush = .{
            .view_projection = undefined,
            .camera_position = .{ placement.position[0], placement.position[1], placement.position[2] },
            .time = time,
            .sun_direction = sun_direction,
            .lean = self.settings.lean,
            .sun_colour = sun_colour,
            .ambient = self.settings.ambient,
        };
        // Through the clip correction, not raw. Vulkan's clip space has Y
        // downward where the projection this camera builds has it upward, and
        // the scene the grass stands in went through the same flip on its way
        // into the frame's camera block. Without it the field would be drawn
        // mirrored about the horizon and would still look like grass.
        push.view_projection = @bitCast(gpu.Uniforms.vulkanClip(view_projection));

        const layout = self.shaders.layoutFor(.blade);
        device.cmdBindDescriptorSets(commands, .graphics, layout, 0, &.{self.sets.set(0)}, &.{});
        device.cmdPushConstants(
            commands,
            layout,
            .{ .vertex_bit = true, .fragment_bit = true },
            0,
            @sizeOf(DrawPush),
            @ptrCast(&push),
        );
        device.cmdBindPipeline(commands, .graphics, self.shaders.get(.blade));
        device.cmdDrawIndirect(commands, self.command.handle, 0, 1, @sizeOf(IndirectCommand));
    }

    // What the frame that has just come back planted. Read where its fence has
    // signalled, so the copy has landed and this waits for nothing.
    pub fn readCount(self: *Grass, frame_index: usize) void {
        if (!self.counted[frame_index]) return;
        const bytes = self.counts[frame_index].mapped() orelse return;
        const value = std.mem.bytesToValue(IndirectCommand, bytes[0..@sizeOf(IndirectCommand)]);
        // The planter counts every candidate it accepted, including the ones past
        // the capacity that it then declined to write. Reported as it stands,
        // because a number silently clamped to the capacity is a field that looks
        // full and is not.
        self.planted = value.instance_count;
    }
};
