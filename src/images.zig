const std = @import("std");
const gltf = @import("lenore-gltf");
const gpu = @import("lenore-gpu");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");
const zignal = @import("zignal");

const Allocator = std.mem.Allocator;

// Turning a document's images into resident textures.
//
// glTF's unit is the image and the renderer's unit is a material slot, and the
// two do not correspond. Several materials name one image, and the slot decides
// how it is interpreted, so an image is decoded once and uploaded once per slot
// interpretation. A pass over materials instead of images produces the same
// picture for one decode per naming, which on a model whose materials share
// their maps is most of the load.

pub const SourcePixel = zignal.Rgba(u8);
pub const DecodedImage = zignal.Image(SourcePixel);

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

// The slots a material set binds, which is what this decodes for. A slot left
// out of this list and still named by a material stops the load with
// `TextureNotUploaded` rather than binding a fallback, so the two lists
// disagreeing is loud rather than silent.
pub const decoded_slots = [_]gpu.MaterialSlot{
    .base_colour,
    .metallic_roughness,
    .normal,
    .emissive,
    .occlusion,
};

pub const ImageError = error{
    ImageNotFound,
    TextureNotUploaded,
    ExternalImageNotLoaded,
    UnsupportedImageEncoding,
    NonContiguousDecodedImage,
    DecodeSizeMismatch,
    ImageRecordCountMismatch,
};

// What one document image cost and what it turned out to hold.
//
// The alpha range is here because this is the only place the pixels exist, and
// it is what the alpha modes read: an image whose alpha is 255 everywhere makes
// MASK and BLEND indistinguishable from OPAQUE, which is a property of the asset
// rather than a fault in the load.
pub const ImageRecord = struct {
    width: usize = 0,
    height: usize = 0,
    // How many material slots this image was uploaded under. Two slots of the
    // same interpretation resolve to one cached image, so this counts
    // interpretations and not references.
    slots: usize = 0,
    // Above one means the image pass decoded it more than once, which is the
    // property walking images first exists to hold at one.
    decodes: u32 = 0,
    alpha_low: u8 = 255,
    alpha_high: u8 = 0,
    // Measured on the thread that did the work, because the recording thread
    // cannot see when it started.
    decode_ns: u64 = 0,
};

// Where a load spent itself. The caller owns `records`, one per document image.
//
// `decode_ns` is summed across the decode threads and `wall_ns` is the pass that
// contains them, so the two are equal only when the pool contributed nothing.
// Their ratio is the concurrency the run actually got, and the sum alone reads
// as a regression.
pub const Report = struct {
    records: []ImageRecord,
    decode_ns: u64 = 0,
    wall_ns: u64 = 0,
    pixels: u64 = 0,
    redundant_pixels: u64 = 0,
    source_bytes: u64 = 0,
    peak_window_bytes: u64 = 0,

    pub fn calls(self: *const Report) u32 {
        var sum: u32 = 0;
        for (self.records) |record| sum += record.decodes;
        return sum;
    }

    pub fn distinct(self: *const Report) u32 {
        var sum: u32 = 0;
        for (self.records) |record| sum += @intFromBool(record.decodes > 0);
        return sum;
    }
};

// The decoded pixels the image pass may hold at once, over every thread.
//
// Bytes rather than a count of images: the images are not the same size, so a
// count bounds nothing.
//
// It is not simply "large enough". The window sets how many decodes run at once,
// and past a point that costs more than it buys, because the decoded pixels
// leave cache and the host and the integrated GPU share one memory bus.
// Measured on ABeautifulGame, 33 images of 2048 square, ReleaseFast on a
// sixteen-thread host at low power with the powersave governor, as decode wall
// against summed pool CPU: 128 MiB gives 1.38 s for 9.1 s, 192 MiB gives 1.24 s
// for 11.4 s, 256 MiB gives 1.17 s for 14.5 s, and 768 MiB is worse on both at
// 1.36 s for 16.9 s. This is the knee. Buying the remaining 0.07 s costs three
// more CPU-seconds, and energy is the priority the engine states first.
//
// A load reports where the window actually peaked, so an asset that never
// reaches it says so.
pub const decode_window_bytes: u64 = 192 << 20;

// Whether admitting one more decode of `size` would put the window over, given
// how many are outstanding and what they are charged.
//
// An empty queue admits anything, so an image larger than the whole window still
// loads: refusing it would mean an asset the viewer cannot open, and the peak is
// then that image rather than the window, which a load reports.
//
// Separate from the queue because it is the whole of the admission policy and
// the queue around it needs a thread pool to exercise. The measured knee above
// is only a number until something can be held against it.
pub fn windowWouldExceed(outstanding: usize, charged: u64, size: u64) bool {
    return outstanding > 0 and charged + size > decode_window_bytes;
}

// What one decode produces.
const DecodedSource = struct {
    image: DecodedImage,
    nanoseconds: u64,
};

// One work item, and the reason this pass can be spread at all: no device, no
// staging block, no shared mutable state. The source bytes are immutable for the
// whole pass and the pixels belong to whoever awaits this.
//
// The allocator has to be threadsafe. Both of the engine's are: DebugAllocator's
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

// What the decode will allocate, without decoding. Both codecs answer from the
// header, and every decode targets `SourcePixel`, so four bytes a pixel is the
// whole cost whatever the source encoding was.
//
// This is what makes the window exact rather than an estimate: the reservation
// is charged before the work is spawned. Admitting on current occupancy alone
// overshoots by one image per worker.
pub fn decodedByteSize(bytes: []const u8) !u64 {
    var reader: std.Io.Reader = .fixed(bytes);
    const pixels: u64 = switch (zignal.ImageFormat.detectFromBytes(bytes) orelse
        return error.UnsupportedImageEncoding) {
        .png => (try zignal.png.getInfo(&reader, .{})).totalPixels(),
        .jpeg => (try zignal.jpeg.getInfo(&reader, .{})).totalPixels(),
    };
    return pixels * @sizeOf(SourcePixel);
}

// The engine owns this pipeline, and it is the one place where the engine owns
// work that runs concurrently. The project's rule that threading is the
// application's is about the frame loop: no engine-owned thread drives frames,
// and nothing below names a thread. It names an `std.Io`, and the caller decides
// what that is. Hand it a single-threaded one and the same window arithmetic
// runs the pass serially.
//
// The alternative is that the policy moves out and the ring stays in the
// application, and it does not cut cleanly: the byte window exists only because
// decodes run ahead of the upload, and its measured knee is the part most worth
// holding under test. Godot places threaded loading in the engine for the same
// reason (ResourceLoader::load_threaded_request).
//
// What would move it out: an application wanting a different admission policy,
// or a second consumer for which the window is wrong.
//
// Admission is the recording thread's and never a worker's. `Io.async` runs the
// task inline on the calling thread once the pool is at its limit, and also when
// the future cannot be allocated or a thread cannot be spawned
// (std/Io/Threaded.zig, `async`). A worker that blocked waiting for the
// recording thread to free window space would deadlock the moment it ran inline.
// Charging before the spawn removes that case entirely.
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
    // run, and a ring that could not hold them all would be a second limit with
    // no reason behind it.
    fn init(allocator: Allocator, io: std.Io, clock: platform.Clock, capacity: usize) !DecodeQueue {
        return .{ .io = io, .clock = clock, .ring = try allocator.alloc(Pending, capacity) };
    }

    // Awaits everything outstanding and frees what it produced. This is the
    // normal path's release as well as the failure path's: a future whose task
    // is still running owns an allocation and possibly a thread, so abandoning
    // one leaks both.
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

    fn wouldExceed(self: *const DecodeQueue, size: u64) bool {
        return windowWouldExceed(self.len, self.charged, size);
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

// What the load has to do, in the order the upload path wants it: which images
// are named at all, and in which slots.
pub const ImageLoader = struct {
    // Parallel to model.images. A slot present in `uses` gets an upload, and the
    // resident image it produced lands in `bound` under the same slot.
    uses: []std.EnumSet(gpu.MaterialSlot),
    bound: []std.EnumArray(gpu.MaterialSlot, ?gpu.ResidentTexture),

    pub fn init(allocator: Allocator, model: *const gltf.importer.Model) !ImageLoader {
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
            inline for (decoded_slots) |which| {
                const reference = @field(material.textures, @tagName(which));
                if (reference.path) |key| {
                    // Before a device exists, so a document naming an image it
                    // does not carry fails while the failure is still cheap.
                    uses[try imageIndex(model, key)].insert(which);
                }
            }
        }
        return .{ .uses = uses, .bound = bound };
    }

    pub fn deinit(self: *ImageLoader, allocator: Allocator) void {
        allocator.free(self.bound);
        allocator.free(self.uses);
        self.* = undefined;
    }

    // Decodes every image a material names and uploads it under each slot that
    // names it, holding at most one window of decoded pixels at a time.
    //
    // Draining the oldest entry is what makes room, and it is also the upload,
    // so the window doubles as the pipeline's depth. The upload stays on the
    // calling thread: the transfer owns a command buffer and the staging pool,
    // and nothing about spreading the decode makes either threadsafe.
    pub fn load(
        self: *ImageLoader,
        allocator: Allocator,
        io: std.Io,
        clock: platform.Clock,
        batch: *gpu.UploadBatch,
        model: *const gltf.importer.Model,
        report: *Report,
    ) !void {
        if (report.records.len != model.images.len) return error.ImageRecordCountMismatch;

        var queue: DecodeQueue = try .init(allocator, io, clock, model.images.len);
        defer queue.deinit(allocator);

        const started = clock.now();
        for (model.images, 0..) |*source, index| {
            if (self.uses[index].count() == 0) continue;

            const bytes = source.bytes orelse return error.ExternalImageNotLoaded;
            report.source_bytes += bytes.len;

            // Already in a form the device samples. Nothing to decode, nothing
            // to charge the window for, and nothing this pass can improve: the
            // file carries its own format and its own mip chain.
            //
            // Where such bytes come from is not decided here. They are whatever
            // the caller put in the model, which is what lets a converted asset
            // tree and a source one go through the same load.
            if (gpu.isKtx2(bytes)) {
                try self.uploadResident(index, bytes, batch, model, report);
                continue;
            }

            const size = try decodedByteSize(bytes);
            while (queue.wouldExceed(size))
                try self.drain(allocator, &queue, batch, model, report);
            queue.spawn(allocator, index, bytes, size);
        }
        while (queue.len > 0)
            try self.drain(allocator, &queue, batch, model, report);

        report.wall_ns = clock.now() - started;
        report.peak_window_bytes = queue.peak;
    }

    // Uploads bytes that are already a GPU format, under every slot that names
    // the image.
    //
    // The record says what the file is rather than what a decode cost: there
    // was no decode, so the counters that measure one stay at zero and a report
    // that shows a distinct image with no decode is telling the truth about it.
    fn uploadResident(
        self: *ImageLoader,
        index: usize,
        bytes: []const u8,
        batch: *gpu.UploadBatch,
        model: *const gltf.importer.Model,
        report: *Report,
    ) !void {
        const record = &report.records[index];
        record.slots = self.uses[index].count();
        record.width = 0;
        record.height = 0;

        const request: gpu.TextureRequest = .{
            .key = model.images[index].key,
            .source = .{ .ktx2 = bytes },
        };
        inline for (decoded_slots) |which| {
            if (self.uses[index].contains(which))
                self.bound[index].set(which, (try batch.addTexture(which, request)).resident());
        }
    }

    // Awaits the oldest decode, uploads it under every slot that names it, and
    // gives the window its bytes back.
    //
    // The charge is released whether or not the decode succeeded, so a failing
    // image cannot wedge the pass behind a reservation nothing will ever free.
    fn drain(
        self: *ImageLoader,
        allocator: Allocator,
        queue: *DecodeQueue,
        batch: *gpu.UploadBatch,
        model: *const gltf.importer.Model,
        report: *Report,
    ) !void {
        var pending = queue.take();
        defer queue.release(pending);

        var produced = try pending.future.await(queue.io);
        defer produced.image.deinit(allocator);

        const index = pending.index;
        const record = &report.records[index];
        const pixels: u64 = @as(u64, produced.image.cols) * produced.image.rows;

        report.decode_ns += produced.nanoseconds;
        report.pixels += pixels;
        if (record.decodes > 0) report.redundant_pixels += pixels;

        record.decodes += 1;
        record.decode_ns += produced.nanoseconds;
        record.width = produced.image.cols;
        record.height = produced.image.rows;
        record.slots = self.uses[index].count();

        // The header said what the decode would allocate and the window was
        // charged for exactly that. A disagreement means the reservation was
        // never the size it claimed, which is a defect in the accounting rather
        // than in the asset.
        if (pixels * @sizeOf(SourcePixel) != pending.charged) return error.DecodeSizeMismatch;

        for (produced.image.data) |texel| {
            record.alpha_low = @min(record.alpha_low, texel.a);
            record.alpha_high = @max(record.alpha_high, texel.a);
        }

        const request: gpu.TextureRequest = .{
            .key = model.images[index].key,
            .source = .{ .rgba8 = .{
                .width = produced.image.cols,
                .height = produced.image.rows,
                .bytes = std.mem.sliceAsBytes(produced.image.data),
            } },
        };
        // Two slots of the same interpretation resolve to one cached image, so
        // what this costs beyond the first is a reference, not an upload.
        inline for (decoded_slots) |which| {
            if (self.uses[index].contains(which))
                self.bound[index].set(which, (try batch.addTexture(which, request)).resident());
        }
    }

    // What a material's slot binds: the image this loader already uploaded,
    // paired with the sampler of this reference. The sampler belongs to the glTF
    // texture rather than to the image, so two references to one image may want
    // different filtering and neither takes a second upload.
    pub fn slot(
        self: *const ImageLoader,
        model: *const gltf.importer.Model,
        reference: anytype,
        which: gpu.MaterialSlot,
    ) !?gpu.TextureSlot {
        const key = reference.path orelse return null;
        const index = try imageIndex(model, key);
        const held = self.bound[index].get(which) orelse return error.TextureNotUploaded;
        return .{ .resident = .{ .texture = held, .sampler = reference.sampler } };
    }

    // Every slot of one material, which is what `addTextureSet` takes.
    pub fn textureSet(
        self: *const ImageLoader,
        model: *const gltf.importer.Model,
        material: *const res.MaterialInfo,
    ) !gpu.TextureSetRequest {
        return .{
            .base_colour = try self.slot(model, material.textures.base_colour, .base_colour),
            .metallic_roughness = try self.slot(
                model,
                material.textures.metallic_roughness,
                .metallic_roughness,
            ),
            .normal = try self.slot(model, material.textures.normal, .normal),
            .emissive = try self.slot(model, material.textures.emissive, .emissive),
            .occlusion = try self.slot(model, material.textures.occlusion, .occlusion),
        };
    }
};

pub fn imageIndex(model: *const gltf.importer.Model, key: []const u8) ImageError!usize {
    for (model.images, 0..) |*image, index| {
        if (std.mem.eql(u8, image.key, key)) return index;
    }
    return error.ImageNotFound;
}
