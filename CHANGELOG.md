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
