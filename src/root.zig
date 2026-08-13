// The engine: the one place that consumes every lenore-* module at once.
//
// That is a structural position and not a convenience. A module names no
// sibling, so nothing else may hold a value from two of them, and every
// translation between them is therefore written here or written again by
// whoever declines to use this. The same rule places composition and policy:
// the frame loop, lifetimes, the world, and time.
//
// The modules remain usable without this. Each builds and tests from its own
// directory, and an application that wants its own composition imports them
// directly. What it inherits in exchange is the translation layer above.

const time = @import("time.zig");
const engine = @import("engine.zig");
const environment = @import("environment.zig");
const images = @import("images.zig");
const level = @import("level.zig");
const shaders = @import("shaders.zig");
const translate = @import("translate.zig");
const world = @import("world.zig");

pub const Engine = engine.Engine;
pub const EngineOptions = engine.Options;
pub const Look = engine.Look;
pub const NoDriver = engine.NoDriver;
pub const Sun = engine.Sun;
pub const FramePhases = engine.FramePhases;
pub const ShadowBakeRequest = engine.ShadowBakeRequest;

pub const FrameClock = time.FrameClock;
pub const FrameTime = time.FrameTime;
pub const FpsCounter = time.FpsCounter;
pub const PhaseTimer = time.PhaseTimer;
pub const max_frame_delta_ns = time.max_frame_delta_ns;
pub const seconds = time.seconds;

pub const RecordPlan = translate.RecordPlan;
pub const BatchError = translate.BatchError;
pub const VertexSourceError = translate.VertexSourceError;
pub const recordBatches = translate.recordBatches;
pub const reduceCube = environment.reduceCube;
pub const lambertianExtent = environment.lambertian_extent;
pub const jointBase = translate.jointBase;
pub const fillVertexSources = translate.fillVertexSources;
pub const sceneLight = translate.sceneLight;
pub const packLight = translate.packLight;
pub const packDocumentLights = translate.packDocumentLights;

pub const World = world.World;
pub const WorldError = world.WorldError;
pub const FrameState = world.FrameState;
pub const DrawPlan = world.DrawPlan;
pub const Skin = world.Skin;
pub const instanceMatrix = world.instanceMatrix;
pub const placePoint = world.placePoint;
pub const skinForIndex = world.skinForIndex;
pub const WorldOptions = world.World.Options;

pub const ImageLoader = images.ImageLoader;
pub const ImageError = images.ImageError;
pub const ImageRecord = images.ImageRecord;
pub const ImageReport = images.Report;
pub const decodedByteSize = images.decodedByteSize;
pub const decode_window_bytes = images.decode_window_bytes;
pub const windowWouldExceed = images.windowWouldExceed;

pub const loadEnvironment = environment.load;
pub const LoadedEnvironment = environment.Loaded;
pub const EnvironmentError = environment.EnvironmentError;
pub const LutChannels = environment.LutChannels;
pub const lutChannels = environment.lutChannels;
pub const environment_keys = environment.environment_keys;
pub const max_environment_bytes = environment.max_environment_bytes;

// The shading the engine ships. `Shaders` is the whole table for the checks
// that walk it; the named values are what a pass is built from.
pub const Shaders = shaders;
pub const sceneShader = shaders.scene;
pub const skyShader = shaders.sky;
pub const postShader = shaders.post;
pub const bloomShader = shaders.bloom;
pub const shadowShader = shaders.shadow;
pub const morphShader = shaders.morph;
pub const rendererShaders = shaders.renderer;

pub const Level = level.Level;
pub const LevelDeps = level.Deps;
pub const LevelOptions = level.Options;
pub const LevelError = level.LevelError;
pub const LevelTimings = level.Timings;
pub const decoded_slots = images.decoded_slots;
pub const imageIndex = images.imageIndex;
