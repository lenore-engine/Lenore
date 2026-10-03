const imui = @import("lenore-imui");

// The editor's colours and the widget styles built from them.
//
// One file because a theme is data that every panel reads and none owns, and
// because the alternative is a colour written twice and changed once.
//
// Colours are authored in sRGB hex, the way a designer names them, and
// premultiplied where they are used. The conversion is not free and a theme
// never changes, so the styles below are built once per frame from the scale
// rather than per widget: what depends on the scale is a border width and a
// radius, and those are the only reason a style is a function at all.

pub const Colour = imui.SrgbColor;

// The surfaces, from the deepest to the nearest.
pub const window = Colour.fromHex("0b0b0e");
pub const panel = Colour.fromHex("131318");
pub const raised = Colour.fromHex("1a1a21");
pub const sunken = Colour.fromHex("0e0e12");
pub const border = Colour.fromHex("26262f");

// One accent, used for what is active and for nothing else. A second accent is
// a second meaning, and an interface with two of them says neither.
pub const accent = Colour.fromHex("ff2d55");
pub const accent_dim = Colour.fromHex("7a1628");

pub const text = Colour.fromHex("e4e4ec");
pub const text_dim = Colour.fromHex("7c7c8a");
pub const text_faint = Colour.fromHex("4e4e5a");

// The row under the pointer and the row that is chosen. The hover is carried at
// partial coverage so that it reads over both the panel and a selected row
// behind it.
pub const row_hovered = Colour.fromHex("ffffff10");
pub const row_selected = Colour.fromHex("2a1620");
pub const selection = Colour.fromHex("47203a");

// The axis colours, which are a convention rather than a choice: X red, Y
// green, Z blue is what every editor and every gizmo uses, and an editor that
// picked its own would be the only one.
pub const axis_x = Colour.fromHex("e05263");
pub const axis_y = Colour.fromHex("6ec46e");
pub const axis_z = Colour.fromHex("4e8fd6");

// Widget styles.
//
// Each takes the scale because a border and a radius are measured in pixels and
// a theme is written in logical units. Everything else in them is a colour,
// which the scale does not touch.

pub fn button(scale: imui.ScaleFactor) imui.ButtonStyle {
    return .{
        .normal_fill = raised.premultiplied(),
        .hovered_fill = Colour.fromHex("22222b").premultiplied(),
        .held_fill = Colour.fromHex("2b2b36").premultiplied(),
        .disabled_fill = panel.premultiplied(),
        .border = border.premultiplied(),
        .focused_border = accent.premultiplied(),
        .border_width = scale.y,
        .radius = 4 * scale.y,
    };
}

// A button that is currently the chosen one of a set: the mode tabs, and the
// selected tab of the inspector.
//
// The accent is the fill rather than an underline, because a tab drawn with a
// rule under it needs the rule to survive the rounding, and at a scale of 1.5
// a one-unit rule is one and a half pixels.
pub fn buttonActive(scale: imui.ScaleFactor) imui.ButtonStyle {
    var style = button(scale);
    style.normal_fill = accent.premultiplied();
    style.hovered_fill = accent.premultiplied();
    style.held_fill = accent_dim.premultiplied();
    style.border = accent.premultiplied();
    return style;
}

// A button with no chrome until the pointer is on it, which is what a menu
// title and a toolbar entry are.
pub fn buttonFlat(scale: imui.ScaleFactor) imui.ButtonStyle {
    var style = button(scale);
    style.normal_fill = Colour.fromHex("00000000").premultiplied();
    style.border = Colour.fromHex("00000000").premultiplied();
    style.border_width = 0;
    return style;
}

pub fn field(scale: imui.ScaleFactor) imui.TextFieldStyle {
    return .{
        .box = .{
            .normal_fill = sunken.premultiplied(),
            .hovered_fill = Colour.fromHex("121218").premultiplied(),
            .held_fill = Colour.fromHex("121218").premultiplied(),
            .disabled_fill = panel.premultiplied(),
            .border = border.premultiplied(),
            .focused_border = accent.premultiplied(),
            .border_width = scale.y,
            .radius = 4 * scale.y,
        },
        .normal_text = text.premultiplied(),
        .disabled_text = text_faint.premultiplied(),
        .selection = selection.premultiplied(),
        .caret = text.premultiplied(),
        // Whole pixels, because a caret is a hairline and a fractional one is
        // drawn at partial coverage on both sides instead of being a line.
        .caret_width = @max(@round(scale.x), 1),
        .padding = 6 * scale.x,
    };
}

pub fn label() imui.LabelStyle {
    return .{
        .normal_text = text.premultiplied(),
        .disabled_text = text_faint.premultiplied(),
    };
}

pub fn labelDim() imui.LabelStyle {
    return .{
        .normal_text = text_dim.premultiplied(),
        .disabled_text = text_faint.premultiplied(),
    };
}

pub fn labelFaint() imui.LabelStyle {
    return .{
        .normal_text = text_faint.premultiplied(),
        .disabled_text = text_faint.premultiplied(),
    };
}

pub fn splitter() imui.SplitterStyle {
    return .{
        .normal_fill = border.premultiplied(),
        .hovered_fill = Colour.fromHex("3a3a48").premultiplied(),
        .held_fill = accent.premultiplied(),
    };
}
