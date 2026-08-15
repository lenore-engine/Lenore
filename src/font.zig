const std = @import("std");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const text = @import("lenore-text");

const Allocator = std.mem.Allocator;

// The fonts an application draws with: the library, the faces, the shaper and
// the coverage atlas they all write into.
//
// It names no device. What it produces is a run of glyphs and a rectangle of
// eight-bit coverage, and whoever owns the device decides when that rectangle
// reaches an image. That split is what lets the whole of this run in a test.
//
// **Two kinds of caller, and the difference is who owns the bytes.** A game's
// font is in its binary or in its archive, so the bytes already exist for the
// life of the program and are borrowed. A system font is a file on the host
// that has to be read and kept resident, because FreeType parses it in place
// and keeps the pointer. `add` takes the first case and `addFile` the second;
// nothing else about a face differs between them.

// A face of this store, and the number the atlas keys its glyphs by.
//
// It is an index, and no two faces ever hold one at the same time. Faces are
// appended, and the one removal there is takes the last of them together with
// everything it put in the atlas. The atlas never evicts on its own, so a
// number given to a second face while the first still had glyphs under it would
// draw the first face's pictures for the second.
pub const FontId = u32;

// Bytes per texel of the coverage the store hands out, which is what the image
// it is uploaded into has to be built with. Re-exported rather than reached for
// through `lenore-text` by whoever owns the device, so that the store and the
// image cannot be built from two readings of it.
pub const atlas_channels = text.atlas_channels;

// How wide a shaped run is, re-exported so that the engine above this file can
// measure one without naming `lenore-text`. This module is the only thing in
// the engine that does, which is what keeps the shaper's surface reachable from
// one place rather than from wherever a caller happened to need a sum.
pub const advance = text.advance;

pub const Capacity = struct {
    // How many faces may be open at once. A face is a face at one size, so an
    // application showing one typeface at three sizes needs three.
    faces: u32 = 4,

    // How many glyphs one run is expected to reach before HarfBuzz's buffer
    // stops growing. It is a high-water mark and not a limit: a longer run
    // costs one allocation inside the shaper and nothing else.
    run_glyphs: u32 = 64,

    // The square coverage atlas, in texels. Four bytes to a texel, so 1024 is
    // 4 MB of host memory, and the staging that carries it is that again per
    // frame in flight.
    //
    // **A face costs a quarter of it at the smallest size.** Measured against a
    // face whose every glyph is a full em, which is the worst a face packs: the
    // printable range at sixteen pixels takes 256 texel rows, at twenty pixels
    // 188, and at twenty-four 124. The largest is the cheapest because the
    // count of subpixel buckets falls faster than a glyph grows.
    //
    // 1024 rather than 512 because a subpixel glyph is one pixel wider on each
    // side than the outline, and because 512 held one small face and no more.
    // It is a budget rather than a limit, and it is a field so that an
    // application drawing one face at one size can say so. Zero and a size
    // whose bytes do not number in a `u32` are refused by `init` rather than
    // carried into an atlas that could only fail at the first glyph.
    atlas_size: u32 = 1024,
};

// The range rasterised when a face is loaded.
//
// Printable ASCII, which is what makes the per-frame cost of a Latin interface
// nothing: every glyph it draws has been seen before the first frame. A
// codepoint outside it still works and is rasterised the frame it appears in,
// which is what the in-frame upload path exists for.
const prewarm_first: u8 = ' ';
const prewarm_last: u8 = '~';

// What a font file may weigh. A face carries its bytes for the life of the
// store, so this is a resident cost and not a transient one. Twenty megabytes
// admits a CJK face and refuses a file that is not a font before FreeType
// parses it.
const max_font_bytes: usize = 20 << 20;

// How a face is rasterised, which is a property of the display and not of the
// font file.
//
// It is a parameter rather than a constant because the machine has an opinion
// and it is a better one than any default here: both of these are judgements
// about physical pixels, and the person using the machine is the one who can
// see them. `fromHost` is where the answer arrives.
//
// The defaults are what a face gets when nobody said, and they are the
// reference target's: a striped RGB panel of ordinary density, where an
// unsnapped horizontal stem lands across two rows at half coverage and reads as
// a blur. Light hinting costs nothing that matters, a light-hinted glyph having
// the horizontal extent and bearing of the unhinted one, which `lenore-text`
// pins over the printable range.
//
// Two faces of one store may differ, and the atlas does not mind: it keys a
// glyph by the face it came from, so neither ever answers for the other.
pub const Rendering = struct {
    hinting: text.Hinting = .light,
    antialias: text.Antialias = .subpixel,

    // What the host asked for, in the terms this engine can act on.
    //
    // Two of the three properties do not map one to one, and neither gap is
    // hidden by picking the nearest name.
    //
    // **A stripe order this build cannot render is answered with grayscale
    // rather than with the wrong order.** FreeType's stripe geometry is left at
    // its default, which is horizontal RGB, and `FT_Library_SetLcdGeometry` is
    // what would change it; nothing calls it. So a BGR or vertical panel would
    // get its stripes inverted, which is worse than not measuring per stripe at
    // all: coverage would land on the wrong colour and the text would fringe.
    //
    // **A host that asked for no antialiasing gets grayscale**, which is not
    // what it asked for. This project produces coverage and has no monochrome
    // rasterisation, so the honest answers are this one or refusing the face,
    // and refusing to draw text because a preference cannot be honoured
    // exactly is the worse trade.
    pub fn fromHost(host: platform.SystemFontRendering) Rendering {
        return .{
            // fontconfig has four steps and this has two, and the two it does
            // not have would be the same picture: no mode in this build moves
            // a glyph horizontally, which `lenore-text` records beside its own
            // enumeration with the measurement behind it. So `medium` and
            // `full` fold onto `light` without losing anything.
            .hinting = switch (host.hint_style) {
                .none => .none,
                .slight, .medium, .full => .light,
            },
            .antialias = if (!host.antialias) .grayscale else switch (host.subpixel) {
                .rgb => .subpixel,
                .bgr, .vrgb, .vbgr, .none => .grayscale,
                // Nobody said, which is not the same as a host saying its
                // display has no stripe order. Godot answers the same question
                // with horizontal RGB by default (`servers/text/text_server.cpp`,
                // the `gui/theme/lcd_subpixel_layout` setting), and a striped
                // RGB panel is what most displays are.
                .unknown => .subpixel,
            },
        };
    }
};

// How many horizontal rasterisations a face of this size is given.
//
// A shaped advance is fractional and a glyph has to be drawn at a whole pixel,
// so without this the drawn gap between two letters is the true one rounded,
// and a rhythm the face designed at 10.5 pixels comes out as ten and eleven
// alternating. Rasterising the glyph at the fraction it will be drawn at is
// what removes that, and this is how finely the fraction is resolved.
//
// The two thresholds are Godot's, which is a shipping engine's judgement
// against real content rather than a measurement of ours
// (`servers/text/text_server.h`, `SUBPIXEL_POSITIONING_ONE_HALF_MAX_SIZE` and
// `SUBPIXEL_POSITIONING_ONE_QUARTER_MAX_SIZE`). The shape of the answer is the
// part that is derivable: half a pixel of error is a fixed fraction of a pixel
// and a shrinking fraction of the glyph, so the larger the face the less there
// is to buy, while the atlas an extra bucket costs grows with the glyph.
//
// The cost is that many entries per glyph. At sixteen pixels the printable
// range is four rows of the atlas, so four buckets of it is sixteen, and the
// default atlas is thirty-two rows deep.
fn subpixelFor(pixel_size: u32) res.SubpixelBuckets {
    if (pixel_size <= 16) return .quarter;
    if (pixel_size <= 20) return .half;
    return .whole;
}

pub const Error = error{
    // Every face this store can hold is taken. Faces are never removed, so it
    // is a budget rather than a moment.
    TooManyFaces,
    // A face number no `add` returned.
    NoSuchFont,
} || text.Error || text.ShapeError || text.AtlasError || text.AtlasSizeError || Allocator.Error;

const Entry = struct {
    face: text.Face,
    // The bytes this store read and must free. Null for a face opened over
    // memory the caller keeps, which is a game's own embedded font.
    owned: ?[]u8,
    pixel_size: u32,
};

pub const Fonts = struct {
    library: text.Library,
    shaper: text.Shaper,
    atlas: text.Atlas,
    entries: []Entry,
    count: u32,

    pub fn init(allocator: Allocator, capacity: Capacity) Error!Fonts {
        var library: text.Library = try .init();
        errdefer library.deinit();

        var shaper: text.Shaper = try .init(capacity.run_glyphs);
        errdefer shaper.deinit();

        var atlas: text.Atlas = try .init(allocator, capacity.atlas_size, capacity.atlas_size);
        errdefer atlas.deinit(allocator);

        const entries = try allocator.alloc(Entry, capacity.faces);
        return .{
            .library = library,
            .shaper = shaper,
            .atlas = atlas,
            .entries = entries,
            .count = 0,
        };
    }

    pub fn deinit(self: *Fonts, allocator: Allocator) void {
        for (self.entries[0..self.count]) |*entry| {
            entry.face.deinit();
            if (entry.owned) |bytes| allocator.free(bytes);
        }
        allocator.free(self.entries);
        self.atlas.deinit(allocator);
        self.shaper.deinit();
        self.library.deinit();
        self.* = undefined;
    }

    // Opens a face over bytes the caller keeps.
    //
    // **The bytes must outlive this store.** FreeType parses them in place and
    // holds the pointer, which is the contract `lenore-text` states and this
    // does not soften: nothing is copied here. An `@embedFile` and a slice of a
    // loaded archive both satisfy it without an allocation.
    pub fn add(
        self: *Fonts,
        allocator: Allocator,
        bytes: []const u8,
        face_index: u32,
        pixel_size: u32,
        rendering: Rendering,
    ) Error!FontId {
        const id = try self.open(bytes, face_index, pixel_size, rendering, null);
        errdefer self.dropLast(allocator);
        try self.prewarm(allocator, id);
        return id;
    }

    // Reads a font file and opens a face over what it read, keeping the bytes
    // for as long as the face exists. This is the system-font path: the file is
    // the host's and nothing else in the program is holding it.
    pub fn addFile(
        self: *Fonts,
        allocator: Allocator,
        io: std.Io,
        directory: std.Io.Dir,
        path: []const u8,
        face_index: u32,
        pixel_size: u32,
        rendering: Rendering,
    ) !FontId {
        if (self.count == self.entries.len) return error.TooManyFaces;

        const bytes = try directory.readFileAlloc(io, path, allocator, .limited(max_font_bytes));
        // The bytes become the store's the moment a face is opened over them,
        // and `dropLast` frees them from then on. An `errdefer` covering both
        // would free them twice.
        const id = self.open(bytes, face_index, pixel_size, rendering, bytes) catch |err| {
            allocator.free(bytes);
            return err;
        };
        errdefer self.dropLast(allocator);
        try self.prewarm(allocator, id);
        return id;
    }

    pub fn metrics(self: *const Fonts, id: FontId) Error!res.FontMetrics {
        const entry = try self.entryOf(id);
        return entry.face.metrics();
    }

    pub fn pixelSize(self: *const Fonts, id: FontId) Error!u32 {
        const entry = try self.entryOf(id);
        return entry.pixel_size;
    }

    // Shapes `source` and resolves every glyph against the atlas.
    //
    // **`placements` needs room for `buckets` entries per glyph**, so a caller
    // sizes it from `subpixelBuckets` rather than from `glyphs`. Every bucket
    // is resolved because which one a glyph wants is decided by the pen, and
    // the pen is decided by whoever aligns the run inside a rectangle, which is
    // past the last point a font was in reach. Resolving one there would mean
    // handing the drawing side a font, which is the dependency that module is
    // built not to have.
    //
    // The glyph count is therefore bounded by both destinations, each in its
    // own units, and the run handed back is the prefix of them that was
    // written.
    //
    // Allocates only for a glyph this store has not rasterised before. After a
    // face is loaded that is the empty set for anything in the printable range,
    // which is what makes this callable from a frame.
    pub fn shape(
        self: *Fonts,
        allocator: Allocator,
        id: FontId,
        source: []const u8,
        glyphs: []res.ShapedGlyph,
        placements: []res.GlyphPlacement,
    ) Error!res.GlyphRun {
        const entry = try self.entryOf(id);
        const buckets = entry.face.subpixel;
        const width = buckets.count();
        const room = @min(glyphs.len, placements.len / width);
        const count = try self.shaper.shape(entry.face, source, glyphs[0..room]);

        for (glyphs[0..count], 0..) |glyph, index| {
            for (0..width) |bucket| {
                placements[index * width + bucket] = try self.atlas.glyph(
                    allocator,
                    .{ .font = id, .glyph = glyph.index, .bucket = @intCast(bucket) },
                    entry.face,
                );
            }
        }
        return .{
            .glyphs = glyphs[0..count],
            .placements = placements[0 .. count * width],
            .buckets = buckets,
        };
    }

    // How many placements one glyph of this face needs. A caller sizes its
    // destination by it, and it is fixed for the life of the face.
    pub fn subpixelBuckets(self: *const Fonts, id: FontId) Error!res.SubpixelBuckets {
        const entry = try self.entryOf(id);
        return entry.face.subpixel;
    }

    // The coverage, and what of it has changed since this was last asked.
    //
    // The box is cleared by the asking, so a caller that takes it and then
    // fails to upload has lost the record. Whoever uploads it copies whole rows
    // rather than the box's columns: the atlas is stored row by row at its full
    // width, so a run of rows is one contiguous slice and a rectangle is not.
    pub fn takeDirty(self: *Fonts) ?text.AtlasBox {
        return self.atlas.takeDirty();
    }

    pub fn coverage(self: *const Fonts) []const u8 {
        return self.atlas.pixels;
    }

    pub fn atlasSize(self: *const Fonts) u32 {
        return self.atlas.width;
    }

    fn open(
        self: *Fonts,
        bytes: []const u8,
        face_index: u32,
        pixel_size: u32,
        rendering: Rendering,
        owned: ?[]u8,
    ) Error!FontId {
        if (self.count == self.entries.len) return error.TooManyFaces;

        const face: text.Face = try .open(self.library, bytes, .{
            .face_index = face_index,
            .pixel_size = pixel_size,
            .hinting = rendering.hinting,
            .subpixel = subpixelFor(pixel_size),
            .antialias = rendering.antialias,
        });
        self.entries[self.count] = .{ .face = face, .owned = owned, .pixel_size = pixel_size };
        self.count += 1;
        return self.count - 1;
    }

    // Undoes the last `open`, for a face that was opened and then failed to
    // rasterise. It is the only removal there is.
    //
    // The number goes back, so what the face put in the atlas has to go with
    // it. A prewarm that stops partway is the ordinary case here rather than a
    // remote one: it is what `AtlasFull` and a glyph FreeType will not render
    // both look like, and every entry written before that point is keyed by a
    // number the next face to be added will be given.
    fn dropLast(self: *Fonts, allocator: Allocator) void {
        self.count -= 1;
        self.atlas.forget(self.count);
        const last = &self.entries[self.count];
        last.face.deinit();
        if (last.owned) |bytes| allocator.free(bytes);
    }

    // Rasterises the printable range, so that a frame drawing Latin text
    // rasterises nothing and uploads nothing.
    //
    // A codepoint the face has no glyph for maps to glyph zero, which is the
    // face's own notdef box and is cached like any other glyph. So the loop
    // needs no branch for coverage: what it costs is one entry per missing
    // character, all of them naming the same box.
    //
    // One character at a time, so no ligature or contextual form is reached
    // here. Those are glyph indices of their own and are rasterised the first
    // frame a run produces them, which is the same path any other unseen glyph
    // takes.
    //
    // Every bucket of every glyph, because which one a frame asks for depends
    // on where the run lands and a frame that had to rasterise would allocate.
    // That is what multiplies the atlas by the bucket count, and it is the
    // whole of what subpixel rasterisation costs at run time.
    fn prewarm(self: *Fonts, allocator: Allocator, id: FontId) Error!void {
        const entry = try self.entryOf(id);
        const width = entry.face.subpixel.count();
        var glyph: [4]res.ShapedGlyph = undefined;

        var character: u8 = prewarm_first;
        while (character <= prewarm_last) : (character += 1) {
            const source = [_]u8{character};
            const count = try self.shaper.shape(entry.face, &source, &glyph);
            for (glyph[0..count]) |shaped| {
                for (0..width) |bucket| {
                    _ = try self.atlas.glyph(
                        allocator,
                        .{ .font = id, .glyph = shaped.index, .bucket = @intCast(bucket) },
                        entry.face,
                    );
                }
            }
        }
    }

    fn entryOf(self: *const Fonts, id: FontId) Error!*Entry {
        if (id >= self.count) return error.NoSuchFont;
        return &self.entries[id];
    }
};
