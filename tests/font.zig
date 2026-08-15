const std = @import("std");
const lenore = @import("lenore");
const imui = @import("lenore-imui");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const text = @import("lenore-text");

const testing = std.testing;

const ahem = @embedFile("test-font");
const pixel_size = 16;

// Ahem's em is 1000 units at an ascent of 800 and a descent of -200, and every
// glyph advances exactly one em. At sixteen pixels that makes an advance of 16
// and a line of 16, both exact in f32, and a lettered glyph a solid rectangle
// sixteen texels across.
const em: f32 = pixel_size;

// Room for eight glyphs at every bucket any face may want.
//
// A destination is an array and its length is decided at compile time, while
// how many buckets a face has is decided when the face is opened, so the way to
// be sure is to reserve the most there are. A caller that guessed low would get
// a run truncated to what its destination held, with nothing to say so.
const run_glyphs = 8;
const run_placements = run_glyphs * res.SubpixelBuckets.most.count();

fn loaded(store: *lenore.Fonts) !lenore.FontId {
    store.* = try .init(testing.allocator, .{});
    return store.add(testing.allocator, ahem, 0, pixel_size, .{});
}

test "a face opened over borrowed bytes reports the font's line metrics" {
    var store: lenore.Fonts = undefined;
    const id = try loaded(&store);
    defer store.deinit(testing.allocator);

    const metrics = try store.metrics(id);
    // A sixty-fourth of a pixel is the grid HarfBuzz reports on, so the
    // comparison is to within one step of it rather than exact.
    const grid = 1.0 / 64.0;
    try testing.expectApproxEqAbs(@as(f32, 12.8), metrics.ascent, grid);
    try testing.expectApproxEqAbs(@as(f32, -3.2), metrics.descent, grid);
    try testing.expectApproxEqAbs(em, metrics.lineHeight(), 2 * grid);
    try testing.expectEqual(@as(u32, pixel_size), try store.pixelSize(id));

    try testing.expectError(error.NoSuchFont, store.metrics(id + 1));
}

test "loading a face rasterises the printable range and leaves it to be uploaded" {
    var store: lenore.Fonts = undefined;
    const id = try loaded(&store);
    defer store.deinit(testing.allocator);
    _ = id;

    // Ninety-five printable characters went in, so the dirty box spans several
    // shelves and is taken exactly once.
    const dirty = store.takeDirty() orelse return error.NothingWasRasterised;
    try testing.expect(dirty.height >= 2 * (pixel_size + 1));
    try testing.expectEqual(@as(?text.AtlasBox, null), store.takeDirty());

    // Coverage was written, and it is eight-bit: a lettered glyph of this face
    // is a solid rectangle, so full coverage appears somewhere in the atlas.
    var full: usize = 0;
    for (store.coverage()) |value| {
        if (value == 255) full += 1;
    }
    try testing.expect(full > 0);
}

test "a run that was prewarmed shapes and places without touching the atlas" {
    var store: lenore.Fonts = undefined;
    const id = try loaded(&store);
    defer store.deinit(testing.allocator);
    _ = store.takeDirty();

    var glyphs: [run_glyphs]res.ShapedGlyph = undefined;
    var placements: [run_placements]res.GlyphPlacement = undefined;
    const run = try store.shape(testing.allocator, id, "abc", &glyphs, &placements);

    try testing.expect(run.isValid());
    try testing.expectEqual(@as(usize, 3), run.glyphs.len);
    for (run.glyphs) |glyph| try testing.expectEqual(em, glyph.x_advance);
    // Every bucket of every glyph, which is what the run carries and what the
    // drawing side chooses one of. Ahem's rectangle is a whole em across, and
    // a box is never narrower than that: a subpixel bucket's shift only ever
    // reaches into the pixel beyond an edge.
    for (run.placements) |placement| {
        try testing.expect(!placement.isBlank());
        try testing.expect(placement.width >= em);
    }
    // Bucket zero is the unshifted glyph, and the engine measures coverage per
    // colour stripe, so its box is the em plus the pixel of padding FreeType
    // adds on each side for the stripe offsets. `lenore-text` derives that.
    for (0..run.glyphs.len) |index| {
        try testing.expectEqual(em + 2, run.placement(index, 0).width);
    }

    // Nothing new was rasterised, which is the property the prewarming exists
    // for: a frame drawing Latin text uploads nothing.
    try testing.expectEqual(@as(?text.AtlasBox, null), store.takeDirty());
}

test "a space is a placement with no ink and still advances" {
    var store: lenore.Fonts = undefined;
    const id = try loaded(&store);
    defer store.deinit(testing.allocator);

    var glyphs: [run_glyphs]res.ShapedGlyph = undefined;
    var placements: [run_placements]res.GlyphPlacement = undefined;
    const run = try store.shape(testing.allocator, id, "a b", &glyphs, &placements);

    try testing.expectEqual(@as(usize, 3), run.glyphs.len);
    // Blank at every bucket: a space has no ink to shift.
    for (0..run.buckets.count()) |bucket| {
        try testing.expect(run.placement(1, @intCast(bucket)).isBlank());
    }
    try testing.expectEqual(em, run.glyphs[1].x_advance);
}

test "the store holds the faces it was given room for and no more" {
    var store: lenore.Fonts = try .init(testing.allocator, .{ .faces = 1 });
    defer store.deinit(testing.allocator);

    _ = try store.add(testing.allocator, ahem, 0, pixel_size, .{});
    try testing.expectError(
        error.TooManyFaces,
        store.add(testing.allocator, ahem, 0, 2 * pixel_size, .{}),
    );
}

test "a face that will not open leaves the store as it was" {
    var store: lenore.Fonts = try .init(testing.allocator, .{});
    defer store.deinit(testing.allocator);

    // The bytes are not a font. What matters is that the failure is reported
    // and that a face opened afterwards is still the first one, so the numbers
    // the atlas keys by have not been spent on a face that does not exist.
    try testing.expectError(
        error.FontUnreadable,
        store.add(testing.allocator, "not a font at all", 0, pixel_size, .{}),
    );
    try testing.expectEqual(@as(lenore.FontId, 0), try store.add(testing.allocator, ahem, 0, pixel_size, .{}));
}

// The other half of that, and the harder half: a face that opens and then fails
// to rasterise. Its number goes back to be given to the next face, so what it
// put in the atlas has to go back with it. Otherwise the next face is answered
// with the coverage of one that no longer exists, and every letter it draws is
// the dead face's picture at the dead face's size.
test "a face whose prewarm fails takes its glyphs back out of the atlas" {
    // An atlas too small for the printable range at this size, so the prewarm
    // fills it and stops partway instead of failing on the first glyph.
    var store: lenore.Fonts = try .init(testing.allocator, .{ .atlas_size = 64 });
    defer store.deinit(testing.allocator);

    try testing.expectError(
        error.AtlasFull,
        store.add(testing.allocator, ahem, 0, pixel_size, .{}),
    );
    // Coverage was written before it stopped, which is what makes this the
    // partial case and not the empty one.
    try testing.expect(store.takeDirty() != null);

    // Face zero was the only one, so nothing at all is left keyed by a number
    // that is about to be handed out again.
    try testing.expectEqual(@as(u32, 0), store.atlas.entries.count());
}

// A caption measured before it is drawn, which is the whole reason this exists:
// a layout node is sized from a caption before there is a rectangle to draw it
// into, and shaping the run a second time at drawing would be the same work
// done twice with two chances to disagree.
//
// It is `Fonts` rather than `Engine` here because the store is what shapes; the
// engine's `shapeLabel` adds the atlas handle to the same three numbers, and it
// needs a device.
test "a caption carries its own measurement" {
    var store: lenore.Fonts = undefined;
    const id = try loaded(&store);
    defer store.deinit(testing.allocator);

    var glyphs: [run_glyphs]res.ShapedGlyph = undefined;
    var placements: [run_placements]res.GlyphPlacement = undefined;
    const run = try store.shape(testing.allocator, id, "abc", &glyphs, &placements);
    const metrics = try store.metrics(id);

    const caption: imui.Label = .{
        .run = run,
        .metrics = metrics,
        .advance = lenore.fontAdvance(run.glyphs),
        .atlas = @enumFromInt(1),
    };

    // Three glyphs of a face whose every advance is one em, so the width is the
    // em three times over and it is exact.
    try testing.expectEqual(3 * em, caption.advance);

    // The height is the line's own box and not `lineHeight`, so it is the
    // ascent above the baseline plus the descent below it. Ahem asks for no
    // leading, so the two agree here and the test that tells them apart is
    // `lenore-imui`'s, against a face with a gap.
    try testing.expectApproxEqAbs(metrics.ascent - metrics.descent, caption.height(), 1e-4);
    try testing.expectApproxEqAbs(em, caption.height(), 2.0 / 64.0);
}

// The host's answer turned into what this engine can act on.
//
// Worth its own test because it is arithmetic over two enumerations with no
// picture attached: a wrong pairing here makes every letter worse on every
// machine that configured the property, and nothing on the screen says which
// step was mistranslated.
test "the host's preferences map onto the modes this engine has" {
    // Four steps of hinting fold onto two. No mode in this build moves a glyph
    // horizontally, so the two steps this engine does not have would be the
    // same picture as the one it does.
    for ([_]platform.SystemFontHintStyle{ .slight, .medium, .full }) |style| {
        const rendering: lenore.FontRendering = .fromHost(.{ .hint_style = style });
        try testing.expectEqual(text.Hinting.light, rendering.hinting);
    }
    {
        const rendering: lenore.FontRendering = .fromHost(.{ .hint_style = .none });
        try testing.expectEqual(text.Hinting.none, rendering.hinting);
    }

    // Only the layout FreeType is left on can be measured per stripe. Every
    // other one is answered with grayscale rather than with the stripes in the
    // wrong order, which would put coverage on the wrong colour.
    {
        const rendering: lenore.FontRendering = .fromHost(.{ .subpixel = .rgb });
        try testing.expectEqual(text.Antialias.subpixel, rendering.antialias);
    }
    for ([_]platform.SystemFontSubpixel{ .bgr, .vrgb, .vbgr, .none }) |layout| {
        const rendering: lenore.FontRendering = .fromHost(.{ .subpixel = layout });
        try testing.expectEqual(text.Antialias.grayscale, rendering.antialias);
    }
    // Nobody having said is not a display saying it has no stripe order, so it
    // is not answered the same way.
    {
        const rendering: lenore.FontRendering = .fromHost(.{ .subpixel = .unknown });
        try testing.expectEqual(text.Antialias.subpixel, rendering.antialias);
    }

    // Antialiasing off outranks any layout. It is not what the host asked for
    // either, this project having no monochrome rasterisation, but it is the
    // closest thing it produces.
    {
        const rendering: lenore.FontRendering = .fromHost(.{ .antialias = false, .subpixel = .rgb });
        try testing.expectEqual(text.Antialias.grayscale, rendering.antialias);
    }
}

// The policy, read back through the face it was applied to.
//
// The thresholds themselves are a judgement rather than a measurement, so what
// is worth pinning is that a face gets one at all and that it falls with size:
// a change that made every face take four buckets would multiply the atlas by
// four with nothing on the screen to say so.
test "the subpixel count falls as a face grows" {
    // A larger atlas than the default, because three faces at once do not fit
    // in it: measured, one face at sixteen pixels with its four buckets takes
    // 256 of the default's 512 texel rows. That is the packing rather than the
    // policy, and the policy is what this test is about.
    var store: lenore.Fonts = try .init(testing.allocator, .{ .atlas_size = 1024 });
    defer store.deinit(testing.allocator);

    const small = try store.add(testing.allocator, ahem, 0, 16, .{});
    const middle = try store.add(testing.allocator, ahem, 0, 20, .{});
    const large = try store.add(testing.allocator, ahem, 0, 24, .{});

    try testing.expectEqual(res.SubpixelBuckets.quarter, try store.subpixelBuckets(small));
    try testing.expectEqual(res.SubpixelBuckets.half, try store.subpixelBuckets(middle));
    try testing.expectEqual(res.SubpixelBuckets.whole, try store.subpixelBuckets(large));
}

// A run of the largest face carries one placement per glyph, which is the
// arrangement everything above the store had before there were buckets. It is
// worth its own test because it is the case where the general path has to cost
// nothing: one lookup per glyph, and a drawing side that rounds to the nearest
// pixel and finds the only rasterisation there is.
test "a face of one bucket carries one placement per glyph" {
    var store: lenore.Fonts = try .init(testing.allocator, .{});
    defer store.deinit(testing.allocator);
    const id = try store.add(testing.allocator, ahem, 0, 24, .{});

    var glyphs: [run_glyphs]res.ShapedGlyph = undefined;
    var placements: [run_placements]res.GlyphPlacement = undefined;
    const run = try store.shape(testing.allocator, id, "abc", &glyphs, &placements);

    try testing.expect(run.isValid());
    try testing.expectEqual(res.SubpixelBuckets.whole, run.buckets);
    try testing.expectEqual(run.glyphs.len, run.placements.len);
}

// Both destinations bound the run, each in its own units, and a run that will
// not fit is refused rather than cut short: a line with its end silently
// missing looks like a line. The placements are the tighter of the two here,
// which is the case a caller gets wrong by sizing them per glyph.
test "placements too small for the glyphs refuse the run" {
    var store: lenore.Fonts = undefined;
    const id = try loaded(&store);
    defer store.deinit(testing.allocator);

    var glyphs: [run_glyphs]res.ShapedGlyph = undefined;
    // Room for one glyph at this face's four buckets, and three asked for.
    var placements: [4]res.GlyphPlacement = undefined;
    try testing.expectError(
        error.GlyphsDoNotFit,
        store.shape(testing.allocator, id, "abc", &glyphs, &placements),
    );

    // One character does fit, so the refusal is the room and not the call.
    const run = try store.shape(testing.allocator, id, "a", &glyphs, &placements);
    try testing.expect(run.isValid());
    try testing.expectEqual(@as(usize, 1), run.glyphs.len);
}
