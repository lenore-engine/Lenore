const std = @import("std");
const gpu = @import("lenore-gpu");

// The shading the engine ships, and the only place that embeds any SPIR-V.
//
// One binary per Slang source in `assets/shaders/`, carrying every entry point
// that file declares. A stage is selected by entry point name at pipeline
// creation, which is why the names are here beside the words rather than
// spelled at each call site.
//
// This is the engine's answer to what a picture looks like, not `lenore-gpu`'s.
// That module states the shape each pass needs, creates modules from words it
// is handed and indexes the entry points it is given; which BRDF, which
// background and which tone operator those words compute is decided here. An
// application wanting its own shading fills the same structures and never
// touches this file.

// `@embedFile` yields bytes, and SPIR-V is words. The array is copied into an
// aligned constant so the reinterpretation below is valid rather than merely
// likely: an embedded file has no alignment of its own.
fn words(comptime bytes: anytype) []const u32 {
    const aligned: [bytes.len]u8 align(@alignOf(u32)) = bytes;
    const count = aligned.len / @sizeOf(u32);
    // Vulkan specification, VkShaderModuleCreateInfo: codeSize is a multiple of
    // four. Dividing would otherwise drop a partial word and hand the driver a
    // module shorter than the file, which reads as a corrupt shader rather than
    // as a truncated build output.
    comptime std.debug.assert(count * @sizeOf(u32) == aligned.len);
    return @as([*]const u32, @ptrCast(&aligned))[0..count];
}

// The main pass: one instanced mesh, transformed, lit and textured.
//
// The four vertex entry points are the product of the two optional streams a
// shader path exists for, the skin and the second UV set, and they are listed
// in the order `gpu.sceneVariantIndex` computes: skinning is the significant
// axis, so the pair without a skin comes first.
pub const scene: gpu.SceneShader = .{
    .spirv = words(@embedFile("scene").*),
    .vertex_entry = .{
        "vertexMain",
        "uv1VertexMain",
        "skinnedVertexMain",
        "skinnedUv1VertexMain",
    },
    .fragment_entry = "fragmentMain",
};

// The background: one screen-covering triangle sampling the environment cube
// along the view ray. It reads the camera and the cube and declares nothing
// else, which is what lets it be created against the pipeline layout the scene
// draws through.
pub const sky: gpu.SkyShader = .{
    .spirv = words(@embedFile("sky").*),
    .vertex_entry = "vertexMain",
    .fragment_entry = "fragmentMain",
};

// The post pass: a screen-covering triangle that samples the HDR target and
// tone maps it. The second fragment stage composites the bloom chain's finest
// level before the operator; the first never names that binding, which is what
// a recording with no chain behind it is drawn with.
pub const post: gpu.PostShader = .{
    .spirv = words(@embedFile("fullscreen").*),
    .vertex_entry = "vertexMain",
    .fragment_entry = "fragmentMain",
    .bloom_fragment_entry = "bloomFragmentMain",
};

// The bloom chain: one covering triangle per level, reducing the HDR target
// down the chain and adding it back up. Both directions read the level below
// through one binding and write one level, so the whole difference between them
// is the fragment stage.
pub const bloom: gpu.BloomShader = .{
    .spirv = words(@embedFile("bloom").*),
    .vertex_entry = "vertexMain",
    .downsample_entry = "downsampleMain",
    .upsample_entry = "upsampleMain",
};

// The sun shadow bake: casters drawn depth-only through the fit's matrix. The
// fragment entry point is the masked variant's alone, and an opaque caster is
// drawn with no fragment stage at all.
pub const shadow: gpu.ShadowShader = .{
    .spirv = words(@embedFile("shadow").*),
    .vertex_entry = "vertexMain",
    .skinned_vertex_entry = "skinnedVertexMain",
    .masked_fragment_entry = "fragmentMain",
};

// The morph prepass: shape targets resolved into a vertex buffer the main pass
// draws in place of the mesh's own. Compute only, and it shares no binding with
// any pass above.
pub const morph: gpu.MorphShader = .{
    .spirv = words(@embedFile("morph").*),
    .compute_entry = "morphMain",
};

// The five the renderer builds pipelines from, in the one value its `init`
// takes. The prepass is not among them: the engine owns that pass directly and
// hands it `morph` above.
pub const renderer: gpu.Shaders = .{
    .scene = scene,
    .sky = sky,
    .post = post,
    .bloom = bloom,
    .shadow = shadow,
};

// Below: a uniform view of the table for the checks that walk it rather than
// name one module. Nothing on a frame path reads any of it.

// The tag names are the strings Slang's reflection reports for a stage, so a
// check can hold one against the other with `@tagName` and no second table.
pub const Stage = enum { vertex, fragment, compute };

pub const Declared = struct {
    name: [*:0]const u8,
    stage: Stage,
};

// One shader file's output: its words, the compiler's account of them, and
// every entry point this table names in it.
//
// The reflection travels with the words rather than in a list beside them. Two
// lists is a state where a module has no reflection or a reflection has no
// module, and the check that caught it is a check this cannot need.
pub const Module = struct {
    name: []const u8,
    spirv: []const u32,
    // The compiler's own account of the layout it produced, emitted beside the
    // words by the same invocation. It is here so a test can hold the
    // hand-written mirrors in `lenore-gpu` against it, which is the only thing
    // that keeps the two from drifting. Nothing is generated from it.
    reflection: []const u8,
    entry_points: []const Declared,
};

// Every module above, with the entry points read out of the values themselves
// rather than spelled a second time. A name written twice is a name that can
// disagree with itself, and the check walking this list is what would then pass
// while a pipeline could not be created.
pub const all = [_]Module{
    .{
        .name = "scene",
        .spirv = scene.spirv,
        .reflection = @embedFile("scene_reflection"),
        .entry_points = &.{
            .{ .name = scene.vertex_entry[0], .stage = .vertex },
            .{ .name = scene.vertex_entry[1], .stage = .vertex },
            .{ .name = scene.vertex_entry[2], .stage = .vertex },
            .{ .name = scene.vertex_entry[3], .stage = .vertex },
            .{ .name = scene.fragment_entry, .stage = .fragment },
        },
    },
    .{
        .name = "sky",
        .spirv = sky.spirv,
        .reflection = @embedFile("sky_reflection"),
        .entry_points = &.{
            .{ .name = sky.vertex_entry, .stage = .vertex },
            .{ .name = sky.fragment_entry, .stage = .fragment },
        },
    },
    .{
        .name = "fullscreen",
        .spirv = post.spirv,
        .reflection = @embedFile("fullscreen_reflection"),
        .entry_points = &.{
            .{ .name = post.vertex_entry, .stage = .vertex },
            .{ .name = post.fragment_entry, .stage = .fragment },
            .{ .name = post.bloom_fragment_entry, .stage = .fragment },
        },
    },
    .{
        .name = "bloom",
        .spirv = bloom.spirv,
        .reflection = @embedFile("bloom_reflection"),
        .entry_points = &.{
            .{ .name = bloom.vertex_entry, .stage = .vertex },
            .{ .name = bloom.downsample_entry, .stage = .fragment },
            .{ .name = bloom.upsample_entry, .stage = .fragment },
        },
    },
    .{
        .name = "shadow",
        .spirv = shadow.spirv,
        .reflection = @embedFile("shadow_reflection"),
        .entry_points = &.{
            .{ .name = shadow.vertex_entry, .stage = .vertex },
            .{ .name = shadow.skinned_vertex_entry, .stage = .vertex },
            .{ .name = shadow.masked_fragment_entry, .stage = .fragment },
        },
    },
    .{
        .name = "morph",
        .spirv = morph.spirv,
        .reflection = @embedFile("morph_reflection"),
        .entry_points = &.{
            .{ .name = morph.compute_entry, .stage = .compute },
        },
    },
};

comptime {
    // Every module carries words. An empty one names the shader it came from
    // here, where a zero-length module handed to pipeline creation names only a
    // handle.
    for (all) |module| std.debug.assert(module.spirv.len > 0);
}
