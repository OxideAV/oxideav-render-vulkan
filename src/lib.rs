//! # oxideav-render-vulkan
//!
//! GPU-accelerated 3D-scene renderer for the
//! [oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
//! Implements [`oxideav_render::Renderer`] on top of
//! [wgpu](https://wgpu.rs), which loads the platform's Vulkan, Metal,
//! DX12 or OpenGL driver **at runtime** — no graphics SDK is needed to
//! build, and a machine without a usable GPU simply gets an
//! [`Error::Backend`] from [`GpuRenderer::new`]. Shaders are WGSL,
//! compiled at pipeline-creation time by wgpu's bundled naga.
//!
//! This crate is deliberately **not** part of the framework's default
//! dependency tree (wgpu is large). Link it on demand — e.g. from
//! oxideplay's 3D viewer — and add it to a
//! [`RenderRegistry`] with [`register_into`].
//!
//! ## Status
//!
//! Phase 1: headless forward rasteriser reproducing the scanline
//! backend's contract — Flat / Gouraud / Phong / Wireframe /
//! NormalDebug / DepthDebug shading, perspective + orthographic
//! framing, one directional light + ambient, supersampled AA — with
//! hardware clipping and depth testing. PBR, textures, scene lights
//! and cameras arrive with `oxideav-render`'s shared scene-preparation
//! layer; hardware ray tracing is a later phase.
//!
//! ```no_run
//! use oxideav_render::{RenderOptions, Renderer};
//! use oxideav_render_vulkan::GpuRenderer;
//!
//! # fn demo(scene: &oxideav_mesh3d::Scene3D) -> oxideav_render::Result<()> {
//! let mut gpu = GpuRenderer::new()?;
//! let image = gpu.render(scene, &RenderOptions::default())?;
//! # let _ = image; Ok(()) }
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod camera;
mod flatten;
mod gpu;
mod math;

use oxideav_render::{RenderOptions, RenderRegistry, Renderer, Result, RgbaImage};

pub use oxideav_render::Error;

/// Crate identifier.
pub const CRATE_NAME: &str = "oxideav-render-vulkan";

/// Name under which [`register_into`] registers the GPU backend.
pub const BACKEND_NAME: &str = "gpu";

/// Which wgpu backend to open.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    /// Let wgpu pick the best available API (Vulkan / Metal / DX12,
    /// then OpenGL).
    #[default]
    Auto,
    /// Vulkan only.
    Vulkan,
    /// OpenGL / OpenGL ES (via EGL / WGL) only.
    Gl,
    /// Metal only (Apple platforms).
    Metal,
    /// Direct3D 12 only (Windows).
    Dx12,
}

/// GPU renderer. Owns a wgpu device plus compiled pipelines; reuse one
/// instance across frames to avoid re-initialising the device.
pub struct GpuRenderer {
    ctx: gpu::GpuContext,
}

impl std::fmt::Debug for GpuRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuRenderer")
            .field("adapter", &self.adapter_summary())
            .finish()
    }
}

impl GpuRenderer {
    /// Open the best available GPU headlessly.
    pub fn new() -> Result<Self> {
        Self::with_backend(GpuBackend::Auto)
    }

    /// Open a GPU through a specific API.
    pub fn with_backend(backend: GpuBackend) -> Result<Self> {
        Ok(Self {
            ctx: gpu::GpuContext::new(backend)?,
        })
    }

    /// Build the renderer on an existing wgpu device — lets a windowed
    /// application (e.g. oxideplay) share its device with the renderer.
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue, info: wgpu::AdapterInfo) -> Self {
        Self {
            ctx: gpu::GpuContext::from_device(device, queue, info),
        }
    }

    /// Human-readable adapter description: `"<name> (<device type>,
    /// <api>)"`.
    pub fn adapter_summary(&self) -> String {
        let info = self.ctx.adapter_info();
        format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend)
    }
}

impl Renderer for GpuRenderer {
    fn render(
        &mut self,
        scene: &oxideav_mesh3d::Scene3D,
        opts: &RenderOptions,
    ) -> Result<RgbaImage> {
        self.ctx.render(scene, opts)
    }
}

/// Register the GPU backend into `registry` under [`BACKEND_NAME`].
/// The factory opens the device lazily, on each `make` call, so
/// registration itself never touches the GPU.
pub fn register_into(registry: &mut RenderRegistry) {
    registry.register(
        BACKEND_NAME,
        Box::new(|| Ok(Box::new(GpuRenderer::new()?) as Box<dyn Renderer>)),
    );
}
