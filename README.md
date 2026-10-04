# oxideav-render-vulkan

GPU-accelerated 3D-scene renderer for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework. It
implements [`oxideav-render`](https://github.com/OxideAV/oxideav-render)'s
`Renderer` trait on top of [wgpu](https://wgpu.rs), so the same
`oxideav_mesh3d::Scene3D` + `RenderOptions` that drive the CPU backends
render on the GPU.

wgpu loads the platform graphics driver (Vulkan, Metal, DX12 or
OpenGL/GLES) **at runtime**, so no graphics SDK is needed to build. On a
machine without a usable adapter, `GpuRenderer::new()` returns
`Error::Backend` and callers fall back to a CPU backend. Shaders are
WGSL, compiled by wgpu's bundled naga when the pipelines are created.

This crate is deliberately **not** in the framework's default
dependency tree because wgpu is large. Link it on demand, as
oxideplay's 3D viewer does.

## Status

| Feature | State |
|---|---|
| Headless offscreen render + readback to `RgbaImage` | done |
| Built on `oxideav-render`'s `PreparedScene` + `Camera` (same posing, framing, materials, lights as the CPU backends) | done |
| Legacy modes with scanline parity: Flat / Gouraud / Phong / Wireframe / NormalDebug / DepthDebug | done |
| `Pbr`: glTF 2.0 metallic-roughness (Appendix B BRDF), base colour / metallic-roughness / normal / occlusion / emissive textures with mips, `KHR_texture_transform`, vertex colours, unlit | done |
| Alpha OPAQUE / MASK / BLEND (per-triangle back-to-front sort, linear-space compositing), double-sided culling | done |
| Up to 16 `KHR_lights_punctual` lights (or the options light), ambient, exposure, tone mapping (Clamp / Reinhard / ACES fitted) | done |
| Scene-linear `Rgba16Float` pass, GPU resolve (background composite, supersample average, sRGB encode) | done |
| Supersampled AA (`aa` 1..=8, shrunk to fit the device's texture limit) | done |
| Upload once / draw many (`upload` + `draw` → GPU texture, no readback) for interactive viewers; shared device via `from_device` | done |
| Shadow maps for directional / spot lights (`opts.shadows`), mirroring the scanline maps: linear light depth, same light-space fit, normal offset, 3×3 bilinear PCF, MASK casters | done |
| Native `render_hdr`: scene-linear `Rgba32Float` resolve + readback (matches the scanline float path, MAE < 1e-4) | done |
| GPU path tracer (`GpuPathTracer`, `GpuMode::PathTrace`, registry name `"gpu-pathtrace"`): WGSL compute port of oxideav-render's path tracer — same Owen-scrambled Sobol' sequences, NEE + MIS, roulette, stochastic BLEND, layered glTF BSDF (transmission, volume, clearcoat, sheen), emissive-triangle and environment-map lights | done |
| GPU path tracer: progressive `sync` / `refine` / `image` / `hdr` / `draw_texture` (no readback) with the CPU tracer's reset rules; statistical (in practice ~65 dB PSNR) parity with the CPU tracer | done |
| Hardware ray queries (`EXPERIMENTAL_RAY_QUERY`) | out of scope — wgpu enables them only through an `unsafe` call and this crate forbids `unsafe` |

## Usage

```rust,no_run
use oxideav_render::{RenderOptions, Renderer};
use oxideav_render_vulkan::GpuRenderer;

fn render(scene: &oxideav_mesh3d::Scene3D) -> oxideav_render::Result<()> {
    let mut gpu = GpuRenderer::new()?; // or with_backend(GpuBackend::Vulkan)
    println!("{}", gpu.adapter_summary());
    let _image = gpu.render(scene, &RenderOptions::default())?;
    Ok(())
}
```

`register_into(&mut RenderRegistry)` adds the rasteriser under the name
`"gpu"` and the path tracer under `"gpu-pathtrace"`. The factories open
the device lazily, on `make`.

### Path tracing

`GpuPathTracer` mirrors `oxideav_render::PathTracer` (whose module docs
are the estimator specification): sample `s` of a pixel draws the same
random numbers on both, so they converge to the same image. Use it
directly, through `GpuRenderer::set_mode(GpuMode::PathTrace)` (then
`render` / `render_hdr` take `opts.path_trace.samples_per_pixel`
samples), or progressively on a renderer's device:

```rust,no_run
use oxideav_render::RenderOptions;
use oxideav_render_vulkan::GpuRenderer;

fn frame(gpu: &mut GpuRenderer, scene: &oxideav_mesh3d::Scene3D, opts: &RenderOptions)
    -> oxideav_render::Result<()> {
    let pt = gpu.path_tracer()?;
    pt.sync(scene, opts)?;      // re-uploads / resets only when needed
    if !pt.is_converged() {
        pt.refine(4)?;          // queued compute dispatches
    }
    let _tex = pt.draw_texture()?; // COLOR_FORMAT texture, no readback
    Ok(())
}
```

`sync` resets the accumulation exactly when the CPU tracer's would
(camera, size, lights, path-trace options, `invalidate_scene`,
`set_environment`, a new texture resolver); display options (background,
tone map, exposure, sample target) never reset it. A changed `Scene3D`
must be signalled with `invalidate_scene()`.

Design:

- **Scene** — `oxideav_render::trace::TraceScene` (the CPU tracer's own
  triangle soup and binned-SAH BVH from `oxideav-mesh3d`) is uploaded as
  four storage buffers, the WebGPU minimum: the 32-byte BVH nodes followed
  by the triangles in leaf order, per-triangle attributes, one table
  buffer (the CPU's Sobol' direction numbers, sheen albedo LUT and
  emissive-light CDF, plus materials, punctual lights, texture
  descriptors and environment CDFs), and the accumulator.
- **Textures** — WGSL cannot index texture arrays without native-only
  features, so every mip chain is shelf-packed into one `Rgba16Float`
  2-D array-texture atlas (layers up to 4096²) and filtered in the
  shader with `textureLoad`, reproducing the CPU sampler exactly: wrap
  modes, nearest / bilinear taps, mip selection, ray-cone LOD on camera
  hits and level 0 on secondary hits.
- **Kernel** — a megakernel, one invocation per pixel running several
  consecutive samples per dispatch. Each path is a small state machine
  that traces one ray per iteration (the continuation ray or one
  next-event shadow ray), so the shader contains one BVH traversal
  (stack-based, ordered near-first; Aila & Laine 2009), one material
  evaluation and two BSDF evaluations. This keeps cold pipeline
  creation around 1.5 s on NVIDIA. Large frames are split into row
  tiles of at most 2²⁰ paths per dispatch, to keep driver watchdogs
  happy.
- **Limits** — needs compute shaders, 4 storage buffers per stage and
  8×8 workgroups; anything less (e.g. GL ES without compute) gives
  `Error::Backend`, as does a scene larger than the device's
  storage-buffer binding limit. `GpuRenderer::new` asks for the
  adapter's own buffer limits.
- **Differences from the CPU** — f32 throughout (the CPU evaluates
  solid angles and Arvo sampling in f64), no f64 fallback in the
  watertight triangle test, `Rgba16Float` texels, UV sets 0 and 1 only.

Timings are in [BENCHMARKS.md](BENCHMARKS.md). On an RTX 5080, the
Cornell box at 256², 64 spp takes 32 ms, against 281 ms for the CPU
tracer on a 64-thread Threadripper.

Textures decode through a `TextureResolver`: install one with
`set_texture_resolver` (e.g. `oxideav_render::RegistryTextureResolver`
over the framework codec registry). Without one, only the raw
`image/x-oxideav-rgba8` container decodes.

`cargo run --example gpu_vs_cpu -- <dir>` renders the same scene with
the GPU and scanline backends (`gpu.ppm` / `cpu.ppm`), plus a `Pbr`
metallic × roughness sphere grid (`gpu_pbr.ppm`).

## Tests

The integration tests compare GPU output against the scanline backend
(coverage, exact flat colour, mean channel error for Phong) and check
the `Pbr` path: exposure, unlit, MASK discard, BLEND compositing over
opaque geometry, and nearest-filtered texture sampling. `tests/parity.rs`
renders `oxideav-render`'s shared test scenes (Cornell box, sphere grid,
checker floor, textured quad, alpha planes, shadow box, skinned/morphed
beam, normal-mapped quad) with both backends in `Pbr` and requires PSNR
between 32 and 60 dB depending on the scene. Two scenes are bit-identical.
`tests/pathtrace.rs` checks the GPU path tracer against
`oxideav_render::PathTraceRenderer`:

- the white furnace (MIS, light-only and BSDF-only);
- image and 4×4 region means within noise, plus a PSNR floor, on the
  Cornell box, sphere grid, shadow box, alpha planes, textured quad,
  checker floor (mip LOD), an extension-material scene (volume glass,
  thin transmission, clearcoat, sheen, metal) and an environment map;
- direct-lighting parity (`max_bounces = 1`);
- determinism for a given seed;
- `refine(a)` + `refine(b)` equal to `refine(a + b)` bit for bit;
- the reset rules, and background bytes on uncovered pixels.

Measured agreement is 63–85 dB PSNR at 64 spp. Set
`OXIDEAV_PARITY_DUMP=<dir>` to write the frame pairs as PPM. Each test
**skips** when no adapter is available, so GPU-less CI stays green.

## License

MIT — see `LICENSE`.
