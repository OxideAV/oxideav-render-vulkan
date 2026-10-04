# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

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

- `GpuRenderer::upload` takes `&mut self` and `&RenderOptions` (for
  time / animation / material variant).
