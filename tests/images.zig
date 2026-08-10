const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const lenore = @import("lenore");
const res = @import("lenore-resources");

const testing = std.testing;

// A 2x3 RGBA8 PNG, written by hand so the decode paths have something real to
// read without an asset on disk. Deliberately not square: a width and a height
// that differ is what tells the two apart when one is read for the other.
const png_2x3 = [_]u8{
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    0x08, 0x06, 0x00, 0x00, 0x00, 0xb9, 0xea, 0xde, 0x81, 0x00, 0x00, 0x00,
    0x1b, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x60, 0x60, 0x60, 0xf8,
    0x0f, 0xc6, 0x5c, 0x22, 0x72, 0x5f, 0x41, 0x98, 0x41, 0x44, 0xc3, 0xe6,
    0x35, 0x08, 0x03, 0x00, 0x51, 0x33, 0x07, 0x27, 0x8e, 0xa5, 0xa2, 0x99,
    0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

fn image(key: []const u8) gltf.importer.Image {
    return .{ .key = key, .bytes = &png_2x3, .mime = null };
}

fn slotWith(path: ?[]const u8) res.MaterialInfo.TextureMaps.Slot {
    return .{ .path = path };
}

fn material(maps: res.MaterialInfo.TextureMaps) res.MaterialInfo {
    return .{ .name = "", .textures = maps, .factors = .{}, .rendering = .{} };
}

fn modelWith(
    images: []gltf.importer.Image,
    materials: []res.MaterialInfo,
) gltf.importer.Model {
    return .{
        .meshes = &.{},
        .materials = materials,
        .images = images,
        .lights = &.{},
        .skins = &.{},
        .node_animation = null,
        .morph_templates = &.{},
    };
}

test "the decode size comes from the header, four bytes a pixel" {
    // Six pixels of RGBA8. The header is read rather than the file decoded,
    // which is what makes the window's reservation exact before any work runs.
    try testing.expectEqual(@as(u64, 2 * 3 * 4), try lenore.decodedByteSize(&png_2x3));
}

test "a source in no format the decoder knows is refused" {
    const nonsense = [_]u8{ 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 };
    try testing.expectError(
        error.UnsupportedImageEncoding,
        lenore.decodedByteSize(&nonsense),
    );
}

test "an image is found by its key and a key nothing carries is refused" {
    var images = [_]gltf.importer.Image{ image("first"), image("second"), image("third") };
    var materials = [_]res.MaterialInfo{material(.{})};
    const model = modelWith(&images, &materials);

    // Not position zero, so a lookup that returned the first match of anything
    // fails here.
    try testing.expectEqual(@as(usize, 1), try lenore.imageIndex(&model, "second"));
    try testing.expectEqual(@as(usize, 2), try lenore.imageIndex(&model, "third"));
    try testing.expectError(error.ImageNotFound, lenore.imageIndex(&model, "absent"));
}

test "an image is claimed by every slot that names it, once each" {
    const allocator = testing.allocator;

    // One image in two slots of one material, and the same image in a third
    // slot of another. What the loader must produce is one entry per slot
    // interpretation, not one per naming.
    var images = [_]gltf.importer.Image{ image("shared"), image("lonely") };
    var materials = [_]res.MaterialInfo{
        material(.{
            .base_colour = slotWith("shared"),
            .emissive = slotWith("shared"),
        }),
        material(.{
            .normal = slotWith("shared"),
            .occlusion = slotWith("lonely"),
        }),
    };
    const model = modelWith(&images, &materials);

    var loader = try lenore.ImageLoader.init(allocator, &model);
    defer loader.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), loader.uses[0].count());
    try testing.expect(loader.uses[0].contains(.base_colour));
    try testing.expect(loader.uses[0].contains(.emissive));
    try testing.expect(loader.uses[0].contains(.normal));
    try testing.expect(!loader.uses[0].contains(.occlusion));

    try testing.expectEqual(@as(usize, 1), loader.uses[1].count());
    try testing.expect(loader.uses[1].contains(.occlusion));
}

test "an image nothing names is claimed by nothing" {
    const allocator = testing.allocator;

    var images = [_]gltf.importer.Image{ image("used"), image("unused") };
    var materials = [_]res.MaterialInfo{material(.{ .base_colour = slotWith("used") })};
    const model = modelWith(&images, &materials);

    var loader = try lenore.ImageLoader.init(allocator, &model);
    defer loader.deinit(allocator);

    // A count of zero is what makes the load pass skip it entirely, so an
    // asset carrying images no material reads costs nothing to open.
    try testing.expectEqual(@as(usize, 0), loader.uses[1].count());
}

test "a material naming an image the document does not carry is refused" {
    const allocator = testing.allocator;

    var images = [_]gltf.importer.Image{image("present")};
    var materials = [_]res.MaterialInfo{material(.{ .base_colour = slotWith("missing") })};
    const model = modelWith(&images, &materials);

    // Before a device exists, which is the point of answering it here.
    try testing.expectError(error.ImageNotFound, lenore.ImageLoader.init(allocator, &model));
}

test "a slot whose image was never uploaded is refused rather than left neutral" {
    const allocator = testing.allocator;

    var images = [_]gltf.importer.Image{image("named")};
    var materials = [_]res.MaterialInfo{material(.{ .base_colour = slotWith("named") })};
    const model = modelWith(&images, &materials);

    var loader = try lenore.ImageLoader.init(allocator, &model);
    defer loader.deinit(allocator);

    // Nothing has been loaded, so the slot has no resident image behind it. A
    // fallback here would draw a plausible picture of the wrong material.
    try testing.expectError(
        error.TextureNotUploaded,
        loader.slot(&model, materials[0].textures.base_colour, .base_colour),
    );

    // A slot the material leaves empty is null rather than an error: that is a
    // material falling back to its factors, which is a legitimate asset.
    try testing.expectEqual(
        @as(?gpu.TextureSlot, null),
        try loader.slot(&model, materials[0].textures.normal, .normal),
    );
}

test "the report counts decodes per image and totals them" {
    var records = [_]lenore.ImageRecord{
        .{ .decodes = 2 },
        .{ .decodes = 0 },
        .{ .decodes = 1 },
    };
    const report: lenore.ImageReport = .{ .records = &records };

    // Three calls over two distinct images. Equal counts would let one stand in
    // for the other, and the difference is exactly what says an image was
    // decoded twice.
    try testing.expectEqual(@as(u32, 3), report.calls());
    try testing.expectEqual(@as(u32, 2), report.distinct());
}

test "a report with a record for every image and none decoded is empty, not wrong" {
    var records = [_]lenore.ImageRecord{ .{}, .{} };
    const report: lenore.ImageReport = .{ .records = &records };

    try testing.expectEqual(@as(u32, 0), report.calls());
    try testing.expectEqual(@as(u32, 0), report.distinct());
}

test "a report sized for the wrong document is refused before the device is touched" {
    // The guard returns before the batch, the clock and the Io are read, which
    // is what makes undefined ones safe here and is the property under test.
    var batch: gpu.UploadBatch = undefined;
    var images = [_]gltf.importer.Image{ image("a"), image("b") };
    var materials = [_]res.MaterialInfo{material(.{})};
    const model = modelWith(&images, &materials);

    var loader = try lenore.ImageLoader.init(testing.allocator, &model);
    defer loader.deinit(testing.allocator);

    var records = [_]lenore.ImageRecord{.{}};
    var report: lenore.ImageReport = .{ .records = &records };

    try testing.expectError(error.ImageRecordCountMismatch, loader.load(
        testing.allocator,
        undefined,
        undefined,
        &batch,
        &model,
        &report,
    ));
}

test "the decode window admits one image however large, and bounds the rest" {
    const window = lenore.decode_window_bytes;

    // Nothing outstanding: an image larger than the whole window is still let
    // through, because refusing it would be an asset the viewer cannot open.
    try testing.expectEqual(false, lenore.windowWouldExceed(0, 0, window * 4));

    // With something outstanding the sum decides. Exactly at the window is not
    // over it, and one byte past is, which is the only place the comparison
    // itself is observable.
    try testing.expectEqual(false, lenore.windowWouldExceed(1, window - 100, 100));
    try testing.expectEqual(true, lenore.windowWouldExceed(1, window - 100, 101));

    // And the charge is what counts, not the number outstanding: many small
    // decodes stay admissible.
    try testing.expectEqual(false, lenore.windowWouldExceed(9, 1024, 1024));
}
