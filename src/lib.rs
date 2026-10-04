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
//! Built on `oxideav-render`'s scene-preparation layer
//! ([`oxideav_render::PreparedScene`]) and [`oxideav_render::Camera`],
//! so framing, posing, materials and lights match the CPU backends:
//!
//! * the scanline backend's legacy modes (Flat / Gouraud / Phong /
//!   Wireframe / NormalDebug / DepthDebug);
//! * `Pbr`: glTF 2.0 metallic-roughness (Appendix B BRDF) with base
//!   colour / metallic-roughness / normal / occlusion / emissive
//!   textures, `KHR_texture_transform`, vertex colours, unlit,
//!   OPAQUE / MASK / BLEND (per-triangle back-to-front sorting,
//!   linear-space compositing), double-sided culling, up to 16
//!   punctual lights (`KHR_lights_punctual` falloff), ambient, exposure
//!   and tone mapping;
//! * a scene-linear `Rgba16Float` pass resolved (tone map, background
//!   composite, supersample average, sRGB encode) on the GPU.
//!
//! Shadow maps and hardware ray tracing are later phases.
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

mod gpu;
mod scene;
mod shadow;

use oxideav_render::{
    HdrImage, RenderOptions, RenderRegistry, Renderer, Result, RgbaImage, TextureResolver,
};

pub use oxideav_render::Error;

/// Crate identifier.
pub const CRATE_NAME: &str = "oxideav-render-vulkan";

/// Format of the texture returned by [`GpuRenderer::draw`]. Holds
/// sRGB-encoded values despite the UNORM format.
pub const COLOR_FORMAT: wgpu::TextureFormat = gpu::COLOR_FORMAT;

/// A scene resident in GPU memory, created by [`GpuRenderer::upload`].
pub struct GpuScene(scene::GpuScene);

impl GpuScene {
    /// Number of triangles uploaded.
    pub fn triangle_count(&self) -> usize {
        self.0.triangle_count
    }
}

impl std::fmt::Debug for GpuScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuScene")
            .field("triangles", &self.triangle_count())
            .finish()
    }
}

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

    /// Upload `scene` to GPU memory once so it can be redrawn cheaply
    /// with different options (interactive viewers: orbit, zoom,
    /// shading-mode switches). Re-upload when the scene changes.
    ///
    /// Upload-time options: `time` / `animation` (pose), and
    /// `material_variant`. Everything else (camera, shading, lights,
    /// tone mapping, size, AA) is read at [`GpuRenderer::draw`] time.
    pub fn upload(&mut self, scene: &oxideav_mesh3d::Scene3D, opts: &RenderOptions) -> GpuScene {
        GpuScene(self.ctx.upload(scene, opts))
    }

    /// Install the resolver used to decode texture images (e.g.
    /// `oxideav_render::RegistryTextureResolver` over the framework's
    /// codec registry). Without one, only raw `RAW_RGBA8_MIME`
    /// payloads decode and other textured materials render untextured.
    pub fn set_texture_resolver(&mut self, resolver: std::sync::Arc<dyn TextureResolver>) {
        self.ctx.texture_cache_mut().set_resolver(resolver);
    }

    /// Draw an uploaded scene at `opts.width × opts.height`
    /// (supersampled by `opts.aa`) into the renderer's offscreen colour
    /// texture and return it — no CPU readback. The texture has format
    /// [`COLOR_FORMAT`] and holds **sRGB-encoded** values (the shader
    /// encodes); it supports `COPY_SRC` and `TEXTURE_BINDING`, so a
    /// viewer can copy or sample it onto its surface. The work is
    /// submitted to [`GpuRenderer::queue`] before returning.
    pub fn draw(&mut self, scene: &mut GpuScene, opts: &RenderOptions) -> Result<&wgpu::Texture> {
        self.ctx.draw(&mut scene.0, opts)
    }

    /// The wgpu device the renderer draws with.
    pub fn device(&self) -> &wgpu::Device {
        self.ctx.device()
    }

    /// The wgpu queue the renderer submits to.
    pub fn queue(&self) -> &wgpu::Queue {
        self.ctx.queue()
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

    /// Native float path: scene-linear radiance resolved on the GPU
    /// (no exposure, tone map or encode), uncovered pixels holding the
    /// linearised background — the same contract as the scanline
    /// backend's `render_hdr`.
    fn render_hdr(
        &mut self,
        scene: &oxideav_mesh3d::Scene3D,
        opts: &RenderOptions,
    ) -> Result<HdrImage> {
        self.ctx.render_hdr(scene, opts)
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
