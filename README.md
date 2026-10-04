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
| Shading modes: Flat / Gouraud / Phong / Wireframe / NormalDebug / DepthDebug (parity with the scanline backend) | done |
| Perspective and orthographic framing (auto-frame and orbit `CameraSpec`) | done |
| Supersampled AA (`aa` 1..=8, shrunk to fit the device's texture limit) | done |
| Hardware clipping and depth test | done |
| Sharing a device with a windowed application (`GpuRenderer::from_device`) | done |
| PBR, textures, scene lights/cameras, alpha modes (via `oxideav-render`'s scene-preparation layer) | next |
| Upload once / draw many (`upload` + `draw` → GPU texture, no readback) for interactive viewers | done |
| Shadow maps, MSAA | planned |
| Hardware ray tracing / GPU path tracer (ray queries) | planned |

## Usage

```rust,no_run
use oxideav_render::{RenderOptions, Renderer};
use oxideav_render_vulkan::GpuRenderer;

fn render(scene: &oxideav_mesh3d::Scene3D) -> oxideav_render::Result<()> {
    let mut gpu = GpuRenderer::new()?; // or with_backend(GpuBackend::Vulkan)
    println!("{}", gpu.adapter_summary());
    let image = gpu.render(scene, &RenderOptions::default())?;
    # let _ = image;
    Ok(())
}
```

`register_into(&mut RenderRegistry)` adds the backend under the name
`"gpu"`. The factory opens the device lazily, on `make`.

`cargo run --example gpu_vs_cpu -- <dir>` renders the same scene with
the GPU and scanline backends and writes `gpu.ppm` / `cpu.ppm`.

## Tests

The integration tests compare GPU output against the scanline backend:
coverage, exact flat colour, and mean channel error for Phong. Each
test **skips** when no adapter is available, so GPU-less CI stays green.

## License

MIT — see `LICENSE`.
