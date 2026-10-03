const std = @import("std");
const imui = @import("lenore-imui");
const res = @import("lenore-resources");

// The editor's frame, as a layout tree solved once per frame.
//
// Written as a literal rather than as arithmetic. The module below already has
// a flex solver with padding, gaps, growth and alignment, and a panel described
// by hand is a panel whose edges are kept in step by care: two rectangles that
// have to meet are two expressions that have to agree, and the one that drifts
// is whichever nobody looked at. The solver makes them meet because they are
// siblings.
//
// **Everything here is logical units.** The tree is solved in them and the one
// conversion to the raster is `Frame.rect`, at the edge, which is what keeps a
// chain of nested panels from accumulating four roundings. The widgets take
// framebuffer pixels and get them there.
//
// The shape is the one a person who has used an editor expects: a menu bar, a
// dock down each side, a centre column holding the document's tabs and its
// viewport over a panel of output, and a status line. Every panel opens with a
// strip of tabs rather than a title, because a panel that can only ever show
// one thing is a panel that will be rebuilt when it has to show two.
//
// The tree runs without a device, so the `test` blocks below are the check on
// it rather than a look at the screen.

// The fixed measurements of the frame, in logical units.
pub const metrics = struct {
    pub const menu_bar: f32 = 30;
    pub const status_bar: f32 = 22;
    // A strip of tabs, which is what every panel opens with.
    pub const tabs: f32 = 26;
    // A row of controls under a strip of tabs: the scene panel's buttons, and
    // the viewport's tools.
    pub const toolbar: f32 = 28;
    pub const splitter: f32 = 4;
    pub const gutter: f32 = 6;
    pub const row: f32 = 22;
    pub const field: f32 = 24;

    // How narrow a dock may be dragged, and how wide. Below the first the tabs
    // stop being readable and the dock is worth closing rather than shrinking,
    // which is a thing to add when there is somewhere to close it to.
    pub const dock_minimum: f32 = 180;
    pub const dock_maximum: f32 = 560;

    // What each panel measures to with nothing in its body, derived from the
    // tree below: a panel is its tabs, whatever row sits under them, and its
    // body's padding.
    //
    // **A fixed dimension is honoured exactly**, including one smaller than the
    // content, which then draws outside the panel it was given. And a panel
    // dragged larger than there is room for overflows its parent, and an
    // overflow grows that parent and its parent in turn, until the window
    // itself is the wrong size. `clampSplit` stops the first with a floor and
    // the second by leaving the panel opposite its own measurement.
    //
    // A panel that gains a row moves its floor. The tests below pin every one
    // of them against what the solver measures, so the drift is caught here
    // rather than by a window that grew.
    pub const scene_minimum: f32 = tabs + toolbar + 2 * gutter;
    pub const files_minimum: f32 = tabs + 2 * gutter;
    pub const output_minimum: f32 = tabs + 2 * gutter;
    // The centre keeps its own tabs and its tools whatever the output panel is
    // dragged to, because those are how the viewport is reached at all.
    pub const centre_minimum: f32 = tabs + toolbar;
};

// The four splitters, as identities, because the shell registers them and the
// shell reads them back.
pub const left_edge_key: imui.IdKey = .{ .string = "shell.left-edge" };
pub const right_edge_key: imui.IdKey = .{ .string = "shell.right-edge" };
pub const left_split_key: imui.IdKey = .{ .string = "shell.left-split" };
pub const centre_split_key: imui.IdKey = .{ .string = "shell.centre-split" };

pub const Tree = imui.Layout.define(imui.Layout.branch(.{
    .arrangement = .column,
    .cross_alignment = .stretch,
}, .{
    .menu_bar = imui.Layout.leaf(.{ .height = .{ .fixed = metrics.menu_bar } }),
    .body = imui.Layout.branch(.{
        .arrangement = .row,
        .flex_grow = 1,
        .cross_alignment = .stretch,
    }, .{
        .left_dock = imui.Layout.branch(.{
            .arrangement = .column,
            .width = .{ .fixed = metrics.dock_minimum },
            .cross_alignment = .stretch,
        }, .{
            // The scene, which is what the editor is looking at. Its toolbar
            // holds the filter, because filtering is done to the list rather
            // than to a selection in it.
            .scene = imui.Layout.branch(.{
                .arrangement = .column,
                .height = .{ .fixed = metrics.scene_minimum },
                .cross_alignment = .stretch,
            }, .{
                .scene_tabs = imui.Layout.leaf(.{ .height = .{ .fixed = metrics.tabs } }),
                .scene_toolbar = imui.Layout.branch(.{
                    .arrangement = .row,
                    .height = .{ .fixed = metrics.toolbar },
                    .cross_alignment = .center,
                    .padding = padX(),
                }, .{
                    .scene_filter = imui.Layout.leaf(.{
                        .flex_grow = 1,
                        .height = .{ .fixed = metrics.field },
                    }),
                }),
                .scene_body = imui.Layout.branch(.{
                    .arrangement = .column,
                    .flex_grow = 1,
                    .cross_alignment = .stretch,
                    .padding = pad(),
                }, .{
                    .scene_content = imui.Layout.leaf(.{ .flex_grow = 1 }),
                }),
            }),
            .left_split = imui.Layout.leaf(.{ .height = .{ .fixed = metrics.splitter } }),
            .files = imui.Layout.branch(.{
                .arrangement = .column,
                .flex_grow = 1,
                .cross_alignment = .stretch,
            }, .{
                .files_tabs = imui.Layout.leaf(.{ .height = .{ .fixed = metrics.tabs } }),
                .files_body = imui.Layout.branch(.{
                    .arrangement = .column,
                    .flex_grow = 1,
                    .cross_alignment = .stretch,
                    .padding = pad(),
                }, .{
                    .files_content = imui.Layout.leaf(.{ .flex_grow = 1 }),
                }),
            }),
        }),
        .left_edge = imui.Layout.leaf(.{ .width = .{ .fixed = metrics.splitter } }),
        .centre = imui.Layout.branch(.{
            .arrangement = .column,
            .flex_grow = 1,
            .cross_alignment = .stretch,
        }, .{
            // The documents that are open, which is a strip of tabs and not a
            // title: an editor showing one scene at a time is one that will be
            // rebuilt when it shows two.
            .document_tabs = imui.Layout.leaf(.{ .height = .{ .fixed = metrics.tabs } }),
            .viewport_toolbar = imui.Layout.leaf(.{ .height = .{ .fixed = metrics.toolbar } }),
            // No padding and no children. What is drawn over it is an overlay
            // the editor places itself, because a gizmo and a mode chip sit on
            // the picture rather than beside it.
            .viewport = imui.Layout.leaf(.{ .flex_grow = 1 }),
            .centre_split = imui.Layout.leaf(.{ .height = .{ .fixed = metrics.splitter } }),
            .output = imui.Layout.branch(.{
                .arrangement = .column,
                .height = .{ .fixed = metrics.output_minimum },
                .cross_alignment = .stretch,
            }, .{
                .output_tabs = imui.Layout.leaf(.{ .height = .{ .fixed = metrics.tabs } }),
                .output_body = imui.Layout.branch(.{
                    .arrangement = .column,
                    .flex_grow = 1,
                    .cross_alignment = .stretch,
                    .padding = pad(),
                }, .{
                    .output_content = imui.Layout.leaf(.{ .flex_grow = 1 }),
                }),
            }),
        }),
        .right_edge = imui.Layout.leaf(.{ .width = .{ .fixed = metrics.splitter } }),
        .right_dock = imui.Layout.branch(.{
            .arrangement = .column,
            .width = .{ .fixed = metrics.dock_minimum },
            .cross_alignment = .stretch,
        }, .{
            .inspector_tabs = imui.Layout.leaf(.{ .height = .{ .fixed = metrics.tabs } }),
            .inspector_body = imui.Layout.branch(.{
                .arrangement = .column,
                .flex_grow = 1,
                .cross_alignment = .stretch,
                .padding = pad(),
            }, .{
                .inspector_content = imui.Layout.leaf(.{ .flex_grow = 1 }),
            }),
        }),
    }),
    .status_bar = imui.Layout.leaf(.{ .height = .{ .fixed = metrics.status_bar } }),
}));

pub const Name = Tree.Name;

// A panel's body is a branch and not a leaf, because padding insets the origin
// of a node's children and a leaf has none: on a leaf it inflates what the node
// measures to and moves nothing. The content leaf inside each body is what the
// padding is applied against, and it is what the editor draws in.
// A row is inset at its ends and not above and below: a toolbar of twenty-eight
// padded six on four sides leaves sixteen, and a field of twenty-four centred in
// sixteen is a field drawn outside its own row.
fn padX() imui.Padding {
    return .{ .left = metrics.gutter, .right = metrics.gutter };
}

fn pad() imui.Padding {
    return .{
        .left = metrics.gutter,
        .top = metrics.gutter,
        .right = metrics.gutter,
        .bottom = metrics.gutter,
    };
}

// One dragged edge: what it is worth now, and where inside the splitter it was
// grabbed.
//
// The anchor is the difference between the value and what the pointer alone
// implies. Without it a splitter grabbed anywhere but its exact centre jumps
// that far on the first frame, because the edge would be placed at the pointer
// rather than moved by it.
pub const Split = struct {
    value: f32,
    grab: f32 = 0,
};

// What the user has dragged. Everything else about the frame is derived.
//
// All four are logical units, so a window moved to another output keeps its
// panels the same size on the desk rather than the same size in pixels.
pub const Shell = struct {
    left_dock: Split = .{ .value = 260 },
    right_dock: Split = .{ .value = 300 },
    scene: Split = .{ .value = 300 },
    output: Split = .{ .value = 180 },

    // A dock's width, measured from the window's edge it is anchored to.
    pub fn dragLeftEdge(self: *Shell, frame: Frame, state: imui.Interaction) void {
        self.dragDock(&self.left_dock, frame, state, .left);
    }

    pub fn dragRightEdge(self: *Shell, frame: Frame, state: imui.Interaction) void {
        self.dragDock(&self.right_dock, frame, state, .right);
    }

    const Side = enum { left, right };

    fn dragDock(
        _: *Shell,
        split: *Split,
        frame: Frame,
        state: imui.Interaction,
        side: Side,
    ) void {
        const pointer = state.pointer orelse return;
        const along = pointer.x / frame.scale.x;
        const implied = switch (side) {
            .left => along,
            .right => frame.root.width - along,
        };
        if (state.capture_began == .primary) split.grab = split.value - implied;
        if (state.capture != .primary) return;

        split.value = clampDock(implied + split.grab, frame.root.width);
    }

    // The scene panel's height, out of the left dock.
    pub fn dragLeftSplit(self: *Shell, frame: Frame, state: imui.Interaction) void {
        const pointer = state.pointer orelse return;
        const dock = frame.logical(.left_dock);
        if (dock.height <= 0) return;

        const implied = pointer.y / frame.scale.y - dock.y;
        if (state.capture_began == .primary) self.scene.grab = self.scene.value - implied;
        if (state.capture != .primary) return;

        self.scene.value = clampSplit(
            implied + self.scene.grab,
            dock.height,
            metrics.scene_minimum,
            metrics.files_minimum,
        );
    }

    // The output panel's height, out of the centre column, measured from its
    // bottom edge because that is the edge it keeps.
    pub fn dragCentreSplit(self: *Shell, frame: Frame, state: imui.Interaction) void {
        const pointer = state.pointer orelse return;
        const centre = frame.logical(.centre);
        if (centre.height <= 0) return;

        const bottom = centre.y + centre.height;
        const implied = bottom - pointer.y / frame.scale.y;
        if (state.capture_began == .primary) self.output.grab = self.output.value - implied;
        if (state.capture != .primary) return;

        self.output.value = clampSplit(
            implied + self.output.grab,
            centre.height,
            metrics.output_minimum,
            metrics.centre_minimum,
        );
    }
};

// A dock's width, kept between the two limits and inside the window.
//
// Written once because the drag and the solve have to agree: a dock the drag
// stopped at one width and the solve gave another would leave the edge somewhere
// the user did not put it.
fn clampDock(value: f32, available: f32) f32 {
    const widest = @min(metrics.dock_maximum, @max(available, 0));
    return std.math.clamp(value, @min(metrics.dock_minimum, widest), widest);
}

// A dragged size, kept inside what its parent can hold.
//
// Written once because the drag and the solve have to agree: a splitter that
// stopped at one bound and a layout that stopped at another would leave the
// edge somewhere the user did not put it.
//
// `opposite` is what the panel on the other side of the splitter measures to.
// Leaving it that much is what keeps an overflow from growing the window.
fn clampSplit(value: f32, available: f32, own_minimum: f32, opposite: f32) f32 {
    const largest = @max(available - metrics.splitter - opposite, 0);
    return std.math.clamp(value, @min(own_minimum, largest), largest);
}

// One frame's solved tree, and the scale its rectangles are read out through.
pub const Frame = struct {
    tree: Tree = .{},
    scale: imui.ScaleFactor = .identity,
    root: imui.LogicalRect = .{ .x = 0, .y = 0, .width = 0, .height = 0 },

    // Scratch for one solve. It is meaningless between solves, so it lives with
    // the frame rather than being handed in by every caller.
    parents: [Tree.node_count]imui.NodeIndex = undefined,
    order: [Tree.node_count]imui.NodeIndex = undefined,
    measured: [Tree.node_count]imui.LogicalSize = undefined,
    scratch: [Tree.node_count]imui.LogicalRect = undefined,

    // Solves the frame for this window and this shell state.
    //
    // The four dragged values are written into the tree here and nowhere else,
    // so a panel's size is the shell's state and the tree is derived from it
    // rather than being a second copy of it.
    pub fn solve(
        self: *Frame,
        extent: [2]f32,
        scale: imui.ScaleFactor,
        shell: Shell,
    ) !void {
        self.scale = scale;
        self.root = .{
            .x = 0,
            .y = 0,
            .width = @max(extent[0], 0) / scale.x,
            .height = @max(extent[1], 0) / scale.y,
        };

        // The two docks divide the width with the centre, so neither may take
        // more than what is left after the other and the two edges.
        const for_docks = @max(self.root.width - 2 * metrics.splitter, 0);
        const left = clampDock(shell.left_dock.value, for_docks);
        self.tree.node(.left_dock).width = .{ .fixed = left };
        self.tree.node(.right_dock).width = .{
            .fixed = clampDock(shell.right_dock.value, @max(for_docks - left, 0)),
        };

        // The dock's own height is not known until the solve, so the clamp is
        // against what the two bars leave: the same number the solver will hand
        // the body.
        const body_height = @max(self.root.height - metrics.menu_bar - metrics.status_bar, 0);
        self.tree.node(.scene).height = .{ .fixed = clampSplit(
            shell.scene.value,
            body_height,
            metrics.scene_minimum,
            metrics.files_minimum,
        ) };
        self.tree.node(.output).height = .{ .fixed = clampSplit(
            shell.output.value,
            body_height,
            metrics.output_minimum,
            metrics.centre_minimum,
        ) };

        return self.tree.solveLayout(self.root, .{
            .parents = &self.parents,
            .order = &self.order,
            .measured = &self.measured,
            .rects = &self.scratch,
        });
    }

    // Where a node is, in logical units.
    pub fn logical(self: *const Frame, name: Name) imui.LogicalRect {
        return self.tree.rect(name);
    }

    // And in framebuffer pixels, which is the space a region is tested in and
    // the space a quad is drawn in.
    //
    // The one conversion, so that a rounding happens once per rectangle rather
    // than once per nesting level.
    pub fn rect(self: *const Frame, name: Name) res.Rect {
        return self.tree.rect(name).toFramebufferFilled(self.scale) catch .{
            // A scale is validated when it is built and the tree's rectangles
            // are finite or the solve failed, so what is left is a rectangle so
            // large the conversion leaves the finite range. An empty one
            // registers and draws nothing rather than placing a widget at a
            // coordinate nothing can hit.
            .x = 0,
            .y = 0,
            .width = 0,
            .height = 0,
        };
    }

    // The row height, in pixels, which every list in the editor shares.
    pub fn rowHeight(self: *const Frame) f32 {
        return metrics.row * self.scale.y;
    }
};

const testing = std.testing;

const window_width: f32 = 1600;
const window_height: f32 = 900;

fn solved(shell: Shell) !Frame {
    var frame: Frame = .{};
    try frame.solve(.{ window_width, window_height }, .identity, shell);
    return frame;
}

test "the frame divides the window and leaves nothing between the parts" {
    var frame = try solved(.{});

    try testing.expectEqual(@as(f32, 30), frame.logical(.menu_bar).height);
    try testing.expectEqual(window_height - 22, frame.logical(.status_bar).y);

    // The row across the body tiles it: dock, edge, centre, edge, dock. A
    // pointer anywhere along it falls on exactly one of the five.
    const parts = [_]Name{ .left_dock, .left_edge, .centre, .right_edge, .right_dock };
    var edge: f32 = 0;
    for (parts) |name| {
        const part = frame.logical(name);
        try testing.expectEqual(edge, part.x);
        try testing.expect(part.width > 0);
        edge += part.width;
    }
    try testing.expectEqual(window_width, edge);
}

test "the left dock stacks the scene over the files" {
    var frame = try solved(.{});

    const dock = frame.logical(.left_dock);
    const stacked = [_]Name{ .scene, .left_split, .files };
    var edge = dock.y;
    for (stacked) |name| {
        const part = frame.logical(name);
        try testing.expectEqual(edge, part.y);
        try testing.expectEqual(dock.x, part.x);
        try testing.expectEqual(dock.width, part.width);
        edge += part.height;
    }
    try testing.expectApproxEqAbs(dock.y + dock.height, edge, 1e-3);
}

test "the centre stacks its tabs, its tools, the viewport and the output" {
    var frame = try solved(.{});

    const centre = frame.logical(.centre);
    const stacked = [_]Name{
        .document_tabs,
        .viewport_toolbar,
        .viewport,
        .centre_split,
        .output,
    };
    var edge = centre.y;
    for (stacked) |name| {
        const part = frame.logical(name);
        try testing.expectEqual(edge, part.y);
        edge += part.height;
    }
    try testing.expectApproxEqAbs(centre.y + centre.height, edge, 1e-3);

    try testing.expectEqual(@as(f32, 26), frame.logical(.document_tabs).height);
    try testing.expectEqual(@as(f32, 28), frame.logical(.viewport_toolbar).height);
    // The viewport is what is left, which is what makes it the thing the
    // window is for.
    try testing.expect(frame.logical(.viewport).height > centre.height * 0.5);
}

test "the right dock stacks its tabs over its body" {
    var frame = try solved(.{});

    const dock = frame.logical(.right_dock);
    const stacked = [_]Name{ .inspector_tabs, .inspector_body };
    var edge = dock.y;
    for (stacked) |name| {
        const part = frame.logical(name);
        try testing.expectEqual(edge, part.y);
        edge += part.height;
    }
    try testing.expectApproxEqAbs(dock.y + dock.height, edge, 1e-3);
}

test "the scene's filter sits inside its toolbar" {
    var frame = try solved(.{});

    const toolbar = frame.logical(.scene_toolbar);
    const filter = frame.logical(.scene_filter);
    try testing.expectEqual(toolbar.x + 6, filter.x);
    try testing.expectEqual(toolbar.width - 12, filter.width);
    try testing.expectEqual(@as(f32, 24), filter.height);
    // Centred across the row, so the field has the same air above and below.
    try testing.expectApproxEqAbs(
        toolbar.y + (toolbar.height - filter.height) * 0.5,
        filter.y,
        1e-3,
    );
}

test "a panel's content sits inside its body's padding" {
    var frame = try solved(.{});

    // The body fills the panel and the padding insets what is inside it. On the
    // body itself the padding would inflate what it measures to and move
    // nothing, which is why the content is a child.
    const body = frame.logical(.files_body);
    const content = frame.logical(.files_content);
    try testing.expectEqual(body.x + 6, content.x);
    try testing.expectEqual(body.width - 12, content.width);
    try testing.expectEqual(body.y + 6, content.y);
}

// Written out rather than taken from `metrics`, because what is being pinned is
// what the tree measures to: a test phrased in the name for it would move with
// that name and pass however far the two had drifted apart.
//
// The scene panel is a strip of tabs at 26, a toolbar at 28 and a body padded 6
// on four sides. The files and output panels are their tabs and a padded body.
// The centre keeps its tabs and its tools and nothing else.
test "every panel's floor is what the solver measures it to" {
    var squeezed = try solved(.{ .scene = .{ .value = 0 }, .output = .{ .value = 0 } });
    try testing.expectApproxEqAbs(66, squeezed.logical(.scene).height, 1e-3);
    try testing.expectApproxEqAbs(38, squeezed.logical(.output).height, 1e-3);

    var stretched = try solved(.{ .scene = .{ .value = 5000 }, .output = .{ .value = 5000 } });
    try testing.expectApproxEqAbs(38, stretched.logical(.files).height, 1e-3);
    try testing.expectApproxEqAbs(54, stretched.logical(.document_tabs).height +
        stretched.logical(.viewport_toolbar).height +
        stretched.logical(.viewport).height, 1e-3);

    try testing.expectEqual(@as(f32, 66), metrics.scene_minimum);
    try testing.expectEqual(@as(f32, 38), metrics.files_minimum);
    try testing.expectEqual(@as(f32, 38), metrics.output_minimum);
    try testing.expectEqual(@as(f32, 54), metrics.centre_minimum);
}

test "the frame stays the size of the window however far a splitter is dragged" {
    const cases = [_]Shell{
        .{},
        .{ .scene = .{ .value = 0 }, .output = .{ .value = 0 } },
        .{ .scene = .{ .value = 5000 }, .output = .{ .value = 5000 } },
        .{ .left_dock = .{ .value = 5000 }, .right_dock = .{ .value = 5000 } },
        .{ .left_dock = .{ .value = -50 }, .right_dock = .{ .value = -50 } },
    };
    // Against the window and not against a parent: an overflowing panel grows
    // its parent and its parent's parent, so a panel compared with the one it
    // sits in agrees with itself however far it overflowed.
    for (cases) |shell| {
        var frame = try solved(shell);
        try testing.expectApproxEqAbs(
            window_height - metrics.menu_bar - metrics.status_bar,
            frame.logical(.body).height,
            1e-3,
        );
        try testing.expectApproxEqAbs(window_width, frame.logical(.body).width, 1e-3);
    }
}

test "the two docks together leave the centre a width" {
    var frame = try solved(.{ .left_dock = .{ .value = 5000 }, .right_dock = .{ .value = 5000 } });
    try testing.expect(frame.logical(.centre).width >= 0);
    // Both are clamped to the dock's own maximum, so the centre keeps the rest.
    try testing.expectApproxEqAbs(metrics.dock_maximum, frame.logical(.left_dock).width, 1e-3);
    try testing.expectApproxEqAbs(metrics.dock_maximum, frame.logical(.right_dock).width, 1e-3);
}

test "a window smaller than its own furniture solves rather than failing" {
    var frame: Frame = .{};
    try frame.solve(.{ 120, 20 }, .identity, .{});

    // Every rectangle is well formed, which is what keeps a region from being
    // registered with an extent nothing can hit.
    for ([_]Name{ .menu_bar, .body, .left_dock, .centre, .viewport, .right_dock, .status_bar }) |name| {
        try testing.expect(frame.rect(name).isValid());
    }
}

test "a scaled output moves every edge and no measurement" {
    const scaled: imui.ScaleFactor = try .init(1.5, 1.5);
    var frame: Frame = .{};
    try frame.solve(.{ window_width, window_height }, scaled, .{});

    // The tree is solved in logical units, so the window is smaller in them.
    try testing.expectApproxEqAbs(600, frame.root.height, 1e-3);
    try testing.expectEqual(@as(f32, 30), frame.logical(.menu_bar).height);
    // And the raster answer carries the scale, once.
    try testing.expectApproxEqAbs(45, frame.rect(.menu_bar).height, 1e-3);
    try testing.expectApproxEqAbs(33, frame.rowHeight(), 1e-3);
}

test "a dock edge grabbed off centre does not jump" {
    var shell: Shell = .{};
    var frame = try solved(shell);

    // The left dock's edge is at 260 and the splitter is four wide, so this
    // press lands one unit inside it. Without the anchor the edge would move to
    // 261 on this frame, one unit the user did not ask for.
    const grab: imui.Interaction = .{
        .capture_began = .primary,
        .capture = .primary,
        .pointer = .{ .x = 261, .y = 400 },
    };
    shell.dragLeftEdge(frame, grab);
    try testing.expectApproxEqAbs(260, shell.left_dock.value, 1e-3);

    // And from there it moves by what the pointer moved, not to where it is.
    frame = try solved(shell);
    shell.dragLeftEdge(frame, .{ .capture = .primary, .pointer = .{ .x = 361, .y = 400 } });
    try testing.expectApproxEqAbs(360, shell.left_dock.value, 1e-3);
}

test "the right edge is measured from the right" {
    var shell: Shell = .{};
    const frame = try solved(shell);

    shell.dragRightEdge(frame, .{ .capture = .primary, .pointer = .{ .x = 1200, .y = 400 } });
    try testing.expectApproxEqAbs(400, shell.right_dock.value, 1e-3);
}

test "the output panel is measured from the bottom of the centre" {
    var shell: Shell = .{};
    const frame = try solved(shell);

    // The centre runs to 878, so a pointer 200 above its bottom edge is an
    // output panel 200 tall.
    shell.dragCentreSplit(frame, .{ .capture = .primary, .pointer = .{ .x = 700, .y = 678 } });
    try testing.expectApproxEqAbs(200, shell.output.value, 1e-3);
}

test "a splitter nobody is holding moves nothing" {
    var shell: Shell = .{};
    const frame = try solved(shell);
    const hovering: imui.Interaction = .{ .hovered = true, .pointer = .{ .x = 700, .y = 400 } };

    shell.dragLeftEdge(frame, hovering);
    shell.dragRightEdge(frame, hovering);
    shell.dragLeftSplit(frame, hovering);
    shell.dragCentreSplit(frame, hovering);

    try testing.expectEqual(@as(f32, 260), shell.left_dock.value);
    try testing.expectEqual(@as(f32, 300), shell.right_dock.value);
    try testing.expectEqual(@as(f32, 300), shell.scene.value);
    try testing.expectEqual(@as(f32, 180), shell.output.value);
}

test "a dock edge stops at its limits" {
    var shell: Shell = .{};
    const frame = try solved(shell);

    shell.dragLeftEdge(frame, .{ .capture = .primary, .pointer = .{ .x = 10, .y = 400 } });
    try testing.expectApproxEqAbs(metrics.dock_minimum, shell.left_dock.value, 1e-3);

    shell.dragLeftEdge(frame, .{ .capture = .primary, .pointer = .{ .x = 1500, .y = 400 } });
    try testing.expectApproxEqAbs(metrics.dock_maximum, shell.left_dock.value, 1e-3);
}
