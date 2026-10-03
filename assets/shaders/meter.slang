// The photocell: a fixed grid of taps over the scene target, reduced to one
// mean luminance per cell.
//
// It reads the scene target and not the presented image, and that is the whole
// of why this is a pass of its own rather than arithmetic inside the post
// operator. The scene target holds radiance before the exposure the operator
// applies, so what a driver adapts from does not depend on the exposure it
// chose last frame. Metering the presented picture instead would close the loop
// through the driver's own output, and an adapting exposure would then settle on
// whatever brightness its own lag happened to produce.
//
// One workgroup a cell and one thread a tap. The grid is fixed rather than
// proportional to the target, so what this costs does not grow with the window:
// eight by eight cells of sixteen by sixteen taps is 16 384 bilinear samples
// whatever the frame's extent is. `pass/meter.zig` mirrors both numbers and
// dispatches one group per cell.
//
// The cells are handed back rather than reduced to one number here, because
// which part of a frame a camera meters is the application's answer and not this
// module's: a centre-weighted meter, a spot meter and a plain average are three
// weightings of one grid, and only the application knows which camera it is
// pretending to be.

// The frame, as the scene pass left it.
[[vk::binding(0, 0)]]
Sampler2D scene;

// One mean luminance per cell, in row-major cell order.
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> cells;

// Taps across a cell, and cells across the frame. Both are mirrored by
// `pass/meter.zig`: the first is this shader's local size and the second is the
// dispatch it is given.
static const uint tap_side = 16;
static const uint cell_side = 8;
static const uint taps_per_cell = tap_side * tap_side;

// Linear luminance from linear BT.709 primaries. Khronos Data Format
// Specification 1.3, section 14.1: the middle row of the RGB to XYZ matrix for
// those primaries is the Y row, and Y is luminance.
//
// Not the more familiar 0.2126, 0.7152, 0.0722. Those are section 15.1.1's
// K values, defined for components carrying the ITU transfer function, and the
// scene target holds linear radiance. The two agree to four decimal places and
// only one of them describes this quantity.
static const float3 luminance_709 = float3(0.212639, 0.715169, 0.072192);

groupshared float partial[taps_per_cell];

[shader("compute")]
[numthreads(tap_side, tap_side, 1)]
void meterMain(
    uint3 group: SV_GroupID,
    uint3 local: SV_GroupThreadID,
    uint index: SV_GroupIndex)
{
    // The cell's own square of the frame, then the tap's place inside it, each
    // at the centre of what it stands for.
    const float2 within = (float2(local.xy) + 0.5) / float(tap_side);
    const float2 uv = (float2(group.xy) + within) / float(cell_side);

    // `SampleLevel` and not `Sample`: a compute shader has no derivatives to
    // choose a level from, and the target has one level in any case.
    //
    // Linear luminance rather than its logarithm. A camera's cell integrates the
    // light that falls on it, which is what makes a bright source in the frame
    // close the iris; a geometric mean would report a lamp and a wall alike and
    // is a photographer's construct rather than a photocell's.
    partial[index] = dot(scene.SampleLevel(uv, 0.0).rgb, luminance_709);
    GroupMemoryBarrierWithGroupSync();

    // The ordinary tree, halving the live half each round. Every thread reaches
    // the barrier, including the ones with nothing left to add, because a
    // barrier inside a divergent branch is undefined.
    for (uint stride = taps_per_cell / 2u; stride > 0u; stride >>= 1u)
    {
        if (index < stride) partial[index] += partial[index + stride];
        GroupMemoryBarrierWithGroupSync();
    }

    if (index == 0u)
        cells[group.y * cell_side + group.x] = partial[0] / float(taps_per_cell);
}
