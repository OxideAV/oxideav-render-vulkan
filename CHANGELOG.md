# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.0.1](https://github.com/OxideAV/oxideav-render-vulkan/compare/v0.0.0...v0.0.1) - 2026-10-04

### Added

- upload-once / draw-many GPU API for interactive viewers
- wgpu headless GPU renderer — scanline-parity shading modes, framing, SSAA

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
