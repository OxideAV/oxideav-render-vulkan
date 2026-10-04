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
| Native `render_hdr` (float readback) | planned |
| Hardware ray tracing / GPU path tracer (ray queries) | planned |

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

`register_into(&mut RenderRegistry)` adds the backend under the name
`"gpu"`. The factory opens the device lazily, on `make`.

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
Set `OXIDEAV_PARITY_DUMP=<dir>` to write the frame pairs as PPM. Each
test **skips** when no adapter is available, so GPU-less CI stays green.

## License

MIT — see `LICENSE`.
