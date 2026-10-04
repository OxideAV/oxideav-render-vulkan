# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.0.1](https://github.com/OxideAV/oxideav-render-vulkan/compare/v0.0.0...v0.0.1) - 2026-10-04

### Added

- upload-once / draw-many GPU API for interactive viewers
- wgpu headless GPU renderer — scanline-parity shading modes, framing, SSAA

### Added

- `GpuPathTracer`: a wgpu compute-shader port of oxideav-render's
  unbiased path tracer. It follows the CPU estimator spec
  (`oxideav_render::pathtrace` §1–5): the same PCG / Owen-scrambled
  Sobol' sequences and dimension layout, NEE to punctual lights plus
  one emissive-triangle (Arvo spherical-triangle sampling) or
  environment sample, power-heuristic MIS, Russian roulette,
  stochastic BLEND coins, and the layered glTF BSDF (transmission,
  volume, clearcoat, sheen). Its images match the CPU tracer to float
  rounding (63–85 dB PSNR at 64 spp).
- Software BVH traversal in WGSL over `TraceScene`'s SAH BVH, using
  four storage buffers. Textures go into an `Rgba16Float` array-texture
  atlas, filtered manually so the result matches the CPU sampler.
- Progressive API: `sync` / `refine` / `samples` / `is_converged` /
  `image` / `hdr` / `draw_texture` (no readback) / `finish` /
  `set_environment` / `invalidate_scene` / `reset`, with the CPU
  `PathTracer`'s reset rules. It implements `Renderer` (`render` and
  `render_hdr` take `samples_per_pixel` samples).
- `GpuMode` and `GpuRenderer::set_mode` / `mode` / `path_tracer`. The
  path tracer is registered as `"gpu-pathtrace"`
  (`PATHTRACE_BACKEND_NAME`) next to `"gpu"`.
- `tests/pathtrace.rs` (GPU-vs-CPU parity suite), the `pt_bench`
  example and `BENCHMARKS.md`. The Cornell box at 256², 64 spp renders
  in 32 ms on an RTX 5080, against 281 ms for the CPU tracer.

- `GpuRenderer`: wgpu-backed headless implementation of
  `oxideav_render::Renderer` (runtime-loaded Vulkan / Metal / DX12 / GL).
  Covers the scanline backend's shading modes (Flat, Gouraud, Phong,
  Wireframe, NormalDebug, DepthDebug), camera framing, and supersampled
  AA, plus hardware clipping and depth testing.
- `GpuBackend` API selector, `GpuRenderer::from_device` for sharing an
  application's wgpu device, `adapter_summary()`.
- `register_into` / `BACKEND_NAME` (`"gpu"`) for `RenderRegistry`.
- GPU-vs-scanline parity tests (skip without an adapter) and the
  `gpu_vs_cpu` example.
- `GpuRenderer::upload` / `draw`: keep a scene resident on the GPU and
  redraw it into an offscreen texture (`COLOR_FORMAT`, sRGB-encoded,
  `COPY_SRC | TEXTURE_BINDING`) without readback, for interactive
  viewers; `device()` / `queue()` accessors.
- Rebuilt on `oxideav-render`'s scene-preparation layer: `PreparedScene`
  upload (posing at `time`, materials, textures, lights, cameras) and
  `Camera::resolve` framing.
- `ShadingMode::Pbr`: glTF 2.0 Appendix B metallic-roughness BRDF with
  all five core texture slots (uploaded as `Rgba16Float` mip chains),
  `KHR_texture_transform`, TEXCOORD_1, vertex colours, unlit, MASK /
  BLEND (sorted, linear compositing), double-sided culling, up to 16
  punctual lights, ambient, exposure and tone mapping.
- Two-pass frame: scene-linear `Rgba16Float` + coverage target, then a
  GPU resolve pass (background composite, supersample average, sRGB
  encode); `draw` now honours `aa`.
- `set_texture_resolver`.
- Shadow maps (`opts.shadows`) for up to 4 directional / spot lights:
  `R32Float` linear-depth array, same light-space fit and normal-offset
  3×3 bilinear PCF as the scanline backend, MASK casters honour their
  cutoff, BLEND surfaces cast nothing.
- GPU-vs-scanline `Pbr` parity suite over the shared test scenes.
- Native `Renderer::render_hdr`: scene-linear float resolve on the GPU,
  same averaging contract as the scanline backend.
- Resolve averages fall back to the straight mean when every sample is
  fully transparent (scanline parity).

### Changed

- `GpuRenderer::new` / `with_backend` now request the adapter's
  storage-buffer binding size, buffer size and storage-buffer count
  (up to 8), on top of the downlevel defaults. The `upload` / `draw`
  signatures are unchanged.
- `GpuRenderer::upload` takes `&mut self` and `&RenderOptions` (for
  time / animation / material variant).
