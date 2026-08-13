const lenore = @import("lenore");

// What the compiler would otherwise never look at.
//
// A test reaches only what it calls, and the engine's surface begins at a window
// and a device, which a test cannot open. Referencing a function compiles its
// body, so this is what makes `zig build test` a check on the composition rather
// than on the arithmetic that happens to be host-side.
//
// `run` is the whole frame loop, so what it calls is compiled through it and
// needs no line of its own. Anything here that gains a test which really calls
// it should lose its line.

test "the device-facing surface is compiled" {
    _ = &lenore.Engine.init;
    _ = &lenore.Engine.deinit;
    _ = &lenore.Engine.install;
    _ = &lenore.Engine.frameCamera;
    _ = &lenore.Engine.unload;
    // Its fallback drains the device, so nothing host-side reaches the body.
    _ = &lenore.Engine.retire;
    _ = &lenore.Engine.refitSun;
    // The loop is generic over its driver, and `_ = &fn` does not instantiate a
    // generic function. `tests/engine.zig` calls it with a driver instead, which
    // is what compiles the body.

    // Reaches the morph prepass, so it needs a device. What it orders is tested
    // through `DrawPlan.rebuild`, `World.packJoints` and the animators
    // separately; this is what compiles the body that puts them in sequence.
    _ = &lenore.World.update;
    _ = &lenore.fillVertexSources;

    // The upload half of the image pass. Everything it decides is reachable
    // without a device and tested separately; what needs one is the transfer it
    // hands each decoded image to.
    _ = &lenore.ImageLoader.load;
    _ = &lenore.ImageLoader.textureSet;

    // Reading an environment is three transfers and a cache. What decides
    // whether the table it read is the right one is `lutChannels`, which is
    // tested on its own.
    _ = &lenore.loadEnvironment;
    _ = &lenore.LoadedEnvironment.release;

    // Opening a model is one upload batch and one prepass registration, so all
    // of it needs a device. What it decides that does not is in `World`, which
    // has its own tests.
    _ = &lenore.Level.init;
    _ = &lenore.Level.deinit;
    _ = &lenore.Level.openEnvironment;
    _ = &lenore.Level.install;
}
