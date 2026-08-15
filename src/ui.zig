const std = @import("std");
const imui = @import("lenore-imui");
const platform = @import("lenore-platform");
const res = @import("lenore-resources");

const translate = @import("translate.zig");

const Allocator = std.mem.Allocator;

// The host side of the overlay: the arrays one frame's widget tree needs, the
// façade over them, and the frame protocol driven as the loop reaches it.
//
// It owns no device object and names none. The memory a frame's geometry is
// written into arrives per frame from whoever owns the rings, so everything
// here runs over a stack array with no device and no window, which is what
// makes the engine's UI composition testable at all.
//
// What it is not is a place for policy. Which widgets exist and where they sit
// is the application's, and the order the three passes run in is the frame
// loop's. This holds the state those two need and the one translation between
// them.

// How much of one frame's widget tree there is room for.
//
// A frame registers its regions and forgets them, so this is a per-frame
// ceiling rather than a total. Exceeding it is an error from the UI module and
// not a reallocation, because the frame loop allocates nothing.
pub const Capacity = struct {
    // Interactive areas one frame may register. A panelled application
    // registers one per control plus one per panel background.
    regions: u32 = 128,
    // How deeply identity scopes and clips may nest. Nesting, not count.
    scope_depth: u32 = 16,
};

// What the engine's widget identities are derived against. One surface of UI,
// so one seed; a second surface built from the same widget code would take
// another to keep the two apart, which is what the parameter is for.
const root_seed: u64 = 0;

pub const Host = struct {
    regions: []imui.Region,
    interactions: []imui.Interaction,
    lookup: []imui.LookupSlot,
    scopes: []imui.Id,

    input: imui.InputContext,
    // Re-pointed at the slot being filled at the top of every frame, so it
    // cannot outlive the storage it writes into.
    canvas: imui.Canvas,
    widgets: imui.WidgetContext,
    // The surface configuration positions are converted under, folded as a
    // batch is walked.
    events: translate.UiEvents,

    // In place, because `widgets` holds pointers into two of the fields beside
    // it. A host returned by value would carry pointers into the temporary it
    // was built in.
    pub fn init(
        self: *Host,
        allocator: Allocator,
        capacity: Capacity,
        storage: res.DrawListStorage,
        metrics: ?platform.SurfaceMetrics,
        image: res.ImageHandle,
    ) !void {
        self.regions = try allocator.alloc(imui.Region, capacity.regions);
        errdefer allocator.free(self.regions);
        self.interactions = try allocator.alloc(imui.Interaction, capacity.regions);
        errdefer allocator.free(self.interactions);
        // Twice the regions, which is what the identity index is written
        // against: at that load factor a linear probe walks under three slots,
        // and a free slot exists whenever a region does, so insertion has no
        // failure path at all. Derived rather than configured, because the two
        // numbers agreeing is what that guarantee rests on.
        self.lookup = try allocator.alloc(imui.LookupSlot, capacity.regions * 2);
        errdefer allocator.free(self.lookup);
        self.scopes = try allocator.alloc(imui.Id, capacity.scope_depth);
        errdefer allocator.free(self.scopes);

        self.input = try .initBuffers(self.regions, self.interactions, self.lookup);
        self.canvas = try .init(storage);
        self.widgets = try .init(&self.input, &self.canvas, self.scopes, root_seed, image);
        self.events = .init(metrics);
    }

    pub fn deinit(self: *Host, allocator: Allocator) void {
        allocator.free(self.scopes);
        allocator.free(self.lookup);
        allocator.free(self.interactions);
        allocator.free(self.regions);
        self.* = undefined;
    }

    // Opens the registering pass over the frame slot the geometry will go into.
    //
    // The storage is taken per frame rather than held, so a host cannot write
    // the slot it was built against after the loop has moved on to another.
    pub fn beginFrame(self: *Host, storage: res.DrawListStorage, root_clip: res.Rect) !void {
        self.canvas = try .init(storage);
        try self.widgets.beginFrame(root_clip);
    }

    pub fn beginRouting(self: *Host) !void {
        return self.widgets.beginRouting();
    }

    // One window event to the UI, and whether the UI took it.
    //
    // False for everything the UI has no word for, which is most of what a
    // window says. A caller suppressing its own binding on this is suppressing
    // it on what a widget actually took.
    pub fn route(self: *Host, event: platform.Event) !bool {
        const ui_event = self.events.translate(event) orelse return false;
        return self.widgets.routeEvent(ui_event);
    }

    // Abandons whatever gesture is in progress, for a caller that has lost
    // events rather than delivered them. The release that would have ended a
    // gesture may be among the lost ones, and a widget waiting for one that
    // will never arrive stays held.
    pub fn cancel(self: *Host) !void {
        _ = try self.widgets.routeEvent(.cancel);
    }

    pub fn finishRouting(self: *Host) !void {
        return self.widgets.finishRouting();
    }

    // Drops a frame that failed part-way, so that the next one may begin. What
    // it costs and why nothing else is reset is in `lenore-imui`.
    pub fn abandonFrame(self: *Host) void {
        self.widgets.abandonFrame();
    }

    // The draws this frame produced, for whoever records them.
    pub fn commands(self: *const Host) []const res.DrawCommand {
        return self.canvas.commands();
    }
};
