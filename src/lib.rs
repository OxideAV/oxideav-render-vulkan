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
//!   composite, supersample average, sRGB encode) on the GPU;
//! * shadow maps for directional / spot lights;
//! * [`GpuPathTracer`]: a compute-shader port of oxideav-render's
//!   unbiased path tracer ([`oxideav_render::PathTracer`]) — same
//!   estimator, same random sequences, same progressive API and reset
//!   rules — with a software BVH traversal in WGSL. Select it with
//!   [`GpuRenderer::set_mode`]`(`[`GpuMode::PathTrace`]`)`, use it
//!   directly, or make it from a registry as
//!   [`PATHTRACE_BACKEND_NAME`]. Hardware ray queries are out of scope:
//!   wgpu only enables them through an `unsafe` call, and this crate
//!   forbids `unsafe`.
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
//!
//! Progressive path tracing (e.g. one `refine` per UI frame):
//!
//! ```no_run
//! use oxideav_render::RenderOptions;
//! use oxideav_render_vulkan::GpuRenderer;
//!
//! # fn demo(scene: &oxideav_mesh3d::Scene3D) -> oxideav_render::Result<()> {
//! let mut gpu = GpuRenderer::new()?;
//! let opts = RenderOptions::default();
//! let pt = gpu.path_tracer()?;
//! pt.sync(scene, &opts)?; // re-uploads / resets only when needed
//! pt.refine(4)?; // queued compute dispatches
//! let texture = pt.draw_texture()?; // COLOR_FORMAT, no readback
//! # let _ = texture; Ok(()) }
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod gpu;
mod pathtrace;
mod scene;
mod shadow;

pub use pathtrace::{GpuMode, GpuPathTracer};

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

/// Name under which [`register_into`] registers the GPU rasteriser.
pub const BACKEND_NAME: &str = "gpu";

/// Name under which [`register_into`] registers the GPU path tracer
/// ([`GpuPathTracer`]).
pub const PATHTRACE_BACKEND_NAME: &str = "gpu-pathtrace";

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
///
/// [`GpuRenderer::set_mode`] switches `render` / `render_hdr` between
/// the rasteriser ([`GpuMode::Raster`], the default) and the compute
/// path tracer ([`GpuMode::PathTrace`]) sharing the same device.
/// [`GpuRenderer::upload`] / [`GpuRenderer::draw`] always rasterise;
/// interactive path tracing goes through
/// [`GpuRenderer::path_tracer`].
pub struct GpuRenderer {
    ctx: gpu::GpuContext,
    mode: GpuMode,
    pt: Option<GpuPathTracer>,
    resolver: Option<std::sync::Arc<dyn TextureResolver>>,
}

impl std::fmt::Debug for GpuRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuRenderer")
            .field("adapter", &self.adapter_summary())
            .field("mode", &self.mode)
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
        Ok(Self::wrap(gpu::GpuContext::new(backend)?))
    }

    /// Build the renderer on an existing wgpu device — lets a windowed
    /// application (e.g. oxideplay) share its device with the renderer.
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue, info: wgpu::AdapterInfo) -> Self {
        Self::wrap(gpu::GpuContext::from_device(device, queue, info))
    }

    fn wrap(ctx: gpu::GpuContext) -> Self {
        Self {
            ctx,
            mode: GpuMode::Raster,
            pt: None,
            resolver: None,
        }
    }

    /// Select what [`Renderer::render`] / [`Renderer::render_hdr`]
    /// run: the rasteriser or the path tracer (which takes
    /// `opts.path_trace.samples_per_pixel` samples per call).
    pub fn set_mode(&mut self, mode: GpuMode) {
        self.mode = mode;
    }

    /// The current [`GpuMode`].
    pub fn mode(&self) -> GpuMode {
        self.mode
    }

    /// The path tracer on this renderer's device, created on first use
    /// (for progressive `sync` / `refine` / `draw_texture` in viewers).
    /// Fails with [`Error::Backend`] when the device cannot run
    /// compute shaders.
    pub fn path_tracer(&mut self) -> Result<&mut GpuPathTracer> {
        if self.pt.is_none() {
            let mut pt = GpuPathTracer::from_device(
                self.ctx.device().clone(),
                self.ctx.queue().clone(),
                self.ctx.adapter_info().clone(),
            )?;
            if let Some(r) = &self.resolver {
                pt.set_texture_resolver(r.clone());
            }
            self.pt = Some(pt);
        }
        Ok(self.pt.as_mut().expect("created above"))
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
        if let Some(pt) = &mut self.pt {
            pt.set_texture_resolver(resolver.clone());
        }
        self.ctx.texture_cache_mut().set_resolver(resolver.clone());
        self.resolver = Some(resolver);
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
        match self.mode {
            GpuMode::PathTrace => self.path_tracer()?.render(scene, opts),
            GpuMode::Raster => self.ctx.render(scene, opts),
        }
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
        match self.mode {
            GpuMode::PathTrace => self.path_tracer()?.render_hdr(scene, opts),
            GpuMode::Raster => self.ctx.render_hdr(scene, opts),
        }
    }

    fn set_texture_resolver(&mut self, resolver: std::sync::Arc<dyn TextureResolver>) {
        GpuRenderer::set_texture_resolver(self, resolver);
    }
}

/// Describe the adapter [`GpuRenderer::with_backend`] would open for
/// `backend`, without creating a device. `None` when no adapter is
/// available. Lets callers (and test suites) inspect the device type —
/// e.g. tell a hardware GPU from a software rasteriser such as WARP or
/// llvmpipe (`wgpu::DeviceType::Cpu`) — before committing to it.
pub fn probe_adapter(backend: GpuBackend) -> Option<wgpu::AdapterInfo> {
    gpu::probe_blocking(backend)
}

/// Register the GPU backends into `registry`: the rasteriser under
/// [`BACKEND_NAME`] and the path tracer under
/// [`PATHTRACE_BACKEND_NAME`]. The factories open the device lazily,
/// on each `make` call, so registration itself never touches the GPU.
pub fn register_into(registry: &mut RenderRegistry) {
    registry.register(
        BACKEND_NAME,
        Box::new(|| Ok(Box::new(GpuRenderer::new()?) as Box<dyn Renderer>)),
    );
    registry.register(
        PATHTRACE_BACKEND_NAME,
        Box::new(|| Ok(Box::new(GpuPathTracer::new()?) as Box<dyn Renderer>)),
    );
}
