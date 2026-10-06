//! GPU path tracer: a wgpu compute-shader port of oxideav-render's
//! [`oxideav_render::pathtrace`] estimator (its module docs §1–5 are
//! the specification this mirrors — same Sobol' / Owen sequences and
//! dimension allocation, same lobe and light probabilities, same MIS,
//! roulette and stochastic-transparency coins).
//!
//! # Scene upload
//!
//! The prepared scene goes through [`TraceScene`] — the CPU tracer's
//! own triangle soup and binned-SAH [`oxideav_mesh3d::Bvh`] — so global
//! triangle ids (which seed the BLEND coins) and the emissive-light
//! table ([`EmissiveLights`]) are identical. Four storage buffers stay
//! within the WebGPU / downlevel minimum of four per stage:
//!
//! * `geom` — the BVH's 32-byte nodes ([`oxideav_mesh3d::Bvh::node_words`]
//!   layout) followed by the triangles **in leaf-slot order** (three
//!   `vec4`s each: positions plus global id / light index / material);
//! * `attrs` — per-slot vertex attributes (normals, tangents, two UV
//!   sets, colours; 192 bytes);
//! * `tables` — one flat word array: Sobol' direction numbers
//!   ([`sobol_direction_numbers`]), the sheen albedo LUT
//!   ([`sheen_albedo_table`]), texture descriptors, materials,
//!   punctual lights, the emissive CDF and the environment CDFs;
//! * `accum` — per pixel `Σ rgb` over covered samples + covered count.
//!
//! # Textures
//!
//! WGSL cannot index a texture array without native-only features, so
//! every (image, colour space) mip chain is shelf-packed into one
//! `Rgba16Float` 2-D **array texture atlas** (square layers, side = the
//! smallest power of two holding the largest level 0, capped at 4096).
//! The kernel filters manually with `textureLoad` — nearest / bilinear
//! taps, wrap modes, and mip selection exactly as
//! `PreparedTexture::sample_lod` — so no atlas bleeding and no sampler
//! objects. Primary hits use the CPU's ray-cone LOD, secondary hits
//! and any-hit alpha level 0. Levels larger than a layer are dropped
//! (a level index below the first kept level reads the first kept).
//!
//! # Kernel
//!
//! A megakernel: one invocation traces whole paths for one pixel (8×8
//! workgroups), several consecutive samples per dispatch. Each path is
//! a small per-thread state machine — every loop iteration traces
//! exactly one ray, either the path's continuation ray or one of the
//! vertex's NEE shadow rays (punctual lights in order, then the area /
//! environment sample) — so the kernel has a single BVH traversal, a
//! single material evaluation and two BSDF evaluations. That keeps the
//! driver's fully-inlined shader small (pipeline creation ~1.5 s cold
//! instead of ~16 s for the naive structure). Traversal is a per-thread
//! stack walk, ordered near-first with entry-distance culling on pop
//! (Aila & Laine 2009), shared by closest-hit and any-hit queries.
//! Large frames are dispatched in row tiles with a bounded number of
//! paths per dispatch to stay clear of driver watchdogs.
//!
//! Sample `s` of every pixel uses the CPU tracer's random numbers, and
//! contributions are added in the CPU's order, so images agree with
//! [`oxideav_render::PathTracer`] to float rounding (the tests measure
//! 63–85 dB PSNR at 64 spp); only paths whose rounding flips a
//! discrete choice diverge, which is statistically neutral.

use std::collections::HashMap;
use std::sync::Arc;

use oxideav_mesh3d::{AlphaMode, MagFilter, MinFilter, Scene3D, WrapMode};
use oxideav_render::pathtrace::{
    radiance_key, sheen_albedo_table, sobol_direction_numbers, EmissiveLights, EnvironmentMap,
    SHEEN_LUT_COS, SHEEN_LUT_ROUGH,
};
use oxideav_render::prepare::{DrawItem, LightKind, PrepareOptions, TextureBinding};
use oxideav_render::texture::{ColorSpace, TextureData};
use oxideav_render::trace::TraceScene;
use oxideav_render::{
    Camera, Error, HdrImage, LightStrategy, PreparedScene, Projection, RenderOptions, Result,
    RgbaImage, TextureCache, TextureResolver, ToneMap,
};
use wgpu::util::DeviceExt;

use crate::gpu::{backend_err, readback_texture, COLOR_FORMAT};
use crate::scene::f32_to_f16;

const _: () = assert!(SHEEN_LUT_COS == 32 && SHEEN_LUT_ROUGH == 16);

const HDR_OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Largest atlas layer side.
const MAX_LAYER: u32 = 4096;
/// Paths (pixel-samples) per dispatch — keeps a single dispatch well
/// under a second on any GPU that can run the kernel at all.
const PATHS_PER_DISPATCH: u64 = 1 << 20;
/// Paths per queue submission.
const PATHS_PER_SUBMIT: u64 = 1 << 22;
/// Dynamic-offset stride of the per-dispatch uniform.
const FRAME_SLOT: u64 = 256;
/// Dispatches per per-dispatch-uniform buffer.
const FRAMES_PER_BUFFER: usize = 1024;
const MAT_STRIDE: usize = 136;
const MAT_SLOTS: usize = 32;
const NONE: u32 = u32::MAX;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    eye: [f32; 4],
    forward: [f32; 4],
    side: [f32; 4],
    up: [f32; 4],
    dims: [u32; 4],
    cfg: [u32; 4],
    env: [u32; 4],
    offs0: [u32; 4],
    offs1: [u32; 4],
    offs2: [u32; 4],
    fparams: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameParams {
    first: u32,
    count: u32,
    y0: u32,
    rows: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolveParams {
    cfg: [u32; 4],
    exposure: [f32; 4],
    background: [f32; 4],
}

/// Which pipeline [`crate::GpuRenderer`] renders with.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuMode {
    /// The forward rasteriser (all shading modes, shadow maps).
    #[default]
    Raster,
    /// The progressive compute path tracer ([`GpuPathTracer`]); takes
    /// `opts.path_trace.samples_per_pixel` samples per `render`.
    PathTrace,
}

/// Scene-dependent GPU state (rebuilt on re-prepare).
struct SceneGpu {
    /// Lightweight copy (per-vertex arrays dropped) for camera framing.
    prepared: PreparedScene,
    geom: wgpu::Buffer,
    attrs: wgpu::Buffer,
    atlas: wgpu::TextureView,
    /// Scene part of the `tables` words (everything but the
    /// environment).
    tables: Vec<u32>,
    node_count: u32,
    tri_base: u32,
    n_punctual: u32,
    n_emissive: u32,
    off_sobol: u32,
    off_sheen: u32,
    off_mat: u32,
    off_plights: u32,
    off_elights: u32,
}

/// Environment section of the tables + its offsets (relative).
struct EnvWords {
    words: Vec<u32>,
    w: u32,
    h: u32,
    marg: u32,
    cond: u32,
    mass: u32,
    rad: u32,
}

/// Size-dependent output targets.
struct Outputs {
    size: (u32, u32),
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    hdr: wgpu::Texture,
    hdr_view: wgpu::TextureView,
}

/// Progressive GPU path tracer — the compute-shader twin of
/// [`oxideav_render::PathTracer`] with the same progressive API and
/// reset rules:
///
/// ```no_run
/// use oxideav_render::RenderOptions;
/// use oxideav_render_vulkan::GpuPathTracer;
/// # fn demo(scene: &oxideav_mesh3d::Scene3D) -> oxideav_render::Result<()> {
/// let opts = RenderOptions::default();
/// let mut pt = GpuPathTracer::new()?;
/// pt.sync(scene, &opts)?; // (re)uploads only when needed
/// while pt.samples() < 256 {
///     pt.refine(16)?; // queues GPU work, returns immediately
///     let _frame = pt.image()?; // or draw_texture() for viewers
/// }
/// # Ok(()) }
/// ```
///
/// The accumulation resets whenever the radiance-relevant inputs
/// change: [`Self::sync`] with different camera / resolution /
/// lighting / path-trace options, [`Self::invalidate_scene`],
/// [`Self::set_environment`] or a new texture resolver. Display-only
/// options (background, tone map, exposure, sample target) never reset
/// it. Sample `s` of pixel `(x, y)` is the CPU tracer's sample `s` —
/// same random numbers — so both converge to the same image.
pub struct GpuPathTracer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    trace_pipeline: wgpu::ComputePipeline,
    resolve_color: wgpu::RenderPipeline,
    resolve_hdr: wgpu::RenderPipeline,
    scene_layout: wgpu::BindGroupLayout,
    frame_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
    accum_layout: wgpu::BindGroupLayout,
    cache: TextureCache,
    opts: RenderOptions,
    prep: Option<PrepareOptions>,
    env: Option<Arc<EnvironmentMap>>,
    dirty: bool,
    scene: Option<SceneGpu>,
    camera: Option<Camera>,
    size: (u32, u32),
    /// Scene bind group (params, buffers, atlas — the group keeps its
    /// resources alive); rebuilt when the environment, options or size
    /// change.
    binding: Option<wgpu::BindGroup>,
    accum: Option<wgpu::Buffer>,
    outputs: Option<Outputs>,
    samples: u32,
}

impl std::fmt::Debug for GpuPathTracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPathTracer")
            .field("adapter", &self.info.name)
            .field("size", &self.size)
            .field("samples", &self.samples)
            .field("prepared", &self.scene.is_some())
            .finish()
    }
}

/// Fail with [`Error::Backend`] when the device cannot run the kernel
/// (e.g. GL ES without compute shaders).
fn check_limits(device: &wgpu::Device) -> Result<()> {
    let l = device.limits();
    let missing = if l.max_compute_workgroups_per_dimension == 0
        || l.max_compute_invocations_per_workgroup < 64
        || l.max_compute_workgroup_size_x < 8
        || l.max_compute_workgroup_size_y < 8
    {
        Some("compute shaders (8×8 workgroups)")
    } else if l.max_storage_buffers_per_shader_stage < 4 {
        Some("4 storage buffers per shader stage")
    } else if l.max_compute_workgroup_storage_size < 4096 {
        Some("4 KiB workgroup storage")
    } else {
        None
    };
    match missing {
        Some(what) => Err(Error::Backend(format!(
            "wgpu: the GPU path tracer needs {what}, which this device does not offer"
        ))),
        None => Ok(()),
    }
}

fn storage_entry(
    binding: u32,
    read_only: bool,
    vis: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(
    binding: u32,
    vis: wgpu::ShaderStages,
    dynamic: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GpuPathTracer {
    /// Open the best available GPU headlessly.
    pub fn new() -> Result<Self> {
        Self::with_backend(crate::GpuBackend::Auto)
    }

    /// Open a GPU through a specific API.
    pub fn with_backend(backend: crate::GpuBackend) -> Result<Self> {
        let (device, queue, info) = crate::gpu::open_device_blocking(backend)?;
        Self::from_device(device, queue, info)
    }

    /// Build the tracer on an existing wgpu device (shared with a
    /// viewer or a [`crate::GpuRenderer`]). Fails with
    /// [`Error::Backend`] when the device lacks compute / storage
    /// support or the kernel does not compile.
    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        info: wgpu::AdapterInfo,
    ) -> Result<Self> {
        check_limits(&device)?;
        // Capture both validation and internal errors: a backend shader
        // compiler rejecting the kernel (e.g. D3D12's FXC failing to
        // unroll a loop) is reported as an *internal* error, which would
        // otherwise reach wgpu's default handler and panic.
        let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let cs = wgpu::ShaderStages::COMPUTE;
        let fs = wgpu::ShaderStages::FRAGMENT;
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pt scene"),
            entries: &[
                uniform_entry(0, cs, false),
                storage_entry(1, true, cs),
                storage_entry(2, true, cs),
                storage_entry(3, true, cs),
                storage_entry(4, false, cs),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: cs,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pt frame"),
            entries: &[uniform_entry(0, cs, true)],
        });
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pt resolve"),
            entries: &[uniform_entry(0, fs, false)],
        });
        let accum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pt accum"),
            entries: &[storage_entry(0, true, fs)],
        });
        let trace_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pathtrace.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pathtrace.wgsl").into()),
        });
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resolve.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/resolve.wgsl").into()),
        });
        let trace_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pt trace"),
            bind_group_layouts: &[Some(&scene_layout), Some(&frame_layout)],
            immediate_size: 0,
        });
        let trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pt trace"),
            layout: Some(&trace_layout),
            module: &trace_shader,
            entry_point: Some("trace_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let rlayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pt resolve"),
            bind_group_layouts: &[Some(&resolve_layout), Some(&accum_layout)],
            immediate_size: 0,
        });
        let resolve_for = |format| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("pt resolve"),
                layout: Some(&rlayout),
                vertex: wgpu::VertexState {
                    module: &resolve_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &resolve_shader,
                    entry_point: Some("fs_resolve_pt"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let resolve_color = resolve_for(COLOR_FORMAT);
        let resolve_hdr = resolve_for(HDR_OUT_FORMAT);
        let validation = pollster::block_on(scope.pop());
        let internal = pollster::block_on(internal_scope.pop());
        if let Some(e) = validation.or(internal) {
            return Err(backend_err("path tracer pipelines", e));
        }
        Ok(Self {
            device,
            queue,
            info,
            trace_pipeline,
            resolve_color,
            resolve_hdr,
            scene_layout,
            frame_layout,
            resolve_layout,
            accum_layout,
            cache: TextureCache::default(),
            opts: RenderOptions::default(),
            prep: None,
            env: None,
            dirty: true,
            scene: None,
            camera: None,
            size: (1, 1),
            binding: None,
            accum: None,
            outputs: None,
            samples: 0,
        })
    }

    /// The wgpu device the tracer runs on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Human-readable adapter description: `"<name> (<device type>,
    /// <api>)"`.
    pub fn adapter_summary(&self) -> String {
        let i = &self.info;
        format!("{} ({:?}, {:?})", i.name, i.device_type, i.backend)
    }

    /// The wgpu queue the tracer submits to.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Replace the texture decoder (re-prepares on the next sync).
    pub fn set_texture_resolver(&mut self, resolver: Arc<dyn TextureResolver>) {
        self.cache.set_resolver(resolver);
        self.dirty = true;
    }

    /// Install (or remove) an HDR environment map; replaces the
    /// constant [`RenderOptions::ambient`] sky. Resets accumulation.
    pub fn set_environment(&mut self, env: Option<Arc<EnvironmentMap>>) {
        self.env = env;
        self.binding = None;
        self.reset();
    }

    /// Mark the scene content as changed: the next [`Self::sync`]
    /// re-prepares and re-uploads it.
    pub fn invalidate_scene(&mut self) {
        self.dirty = true;
    }

    /// Discard accumulated samples.
    pub fn reset(&mut self) {
        self.samples = 0;
        if let Some(acc) = &self.accum {
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.clear_buffer(acc, 0, None);
            self.queue.submit([enc.finish()]);
        }
    }

    /// Samples accumulated per pixel so far.
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// `true` once [`oxideav_render::PathTraceOptions::samples_per_pixel`]
    /// samples are in.
    pub fn is_converged(&self) -> bool {
        self.samples >= self.opts.path_trace.samples_per_pixel
    }

    /// Output size of the current accumulation.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Bring the tracer in line with `scene` + `opts` — the same rules
    /// as [`oxideav_render::PathTracer::sync`]: re-prepares (and
    /// re-uploads) when the scene was invalidated or a preparation
    /// input changed, resets the accumulation when anything affecting
    /// radiance changed. Returns `Ok(true)` when it reset. Cheap when
    /// nothing changed.
    pub fn sync(&mut self, scene: &Scene3D, opts: &RenderOptions) -> Result<bool> {
        let prep = PrepareOptions::from_render_options(opts);
        let need_prepare = self.dirty || self.scene.is_none() || self.prep.as_ref() != Some(&prep);
        let changed = need_prepare || radiance_key(opts) != radiance_key(&self.opts);
        self.opts = opts.clone();
        if !changed {
            return Ok(false);
        }
        let width = opts.width.max(1);
        let height = opts.height.max(1);
        let max_dim = self.device.limits().max_texture_dimension_2d;
        if width > max_dim || height > max_dim {
            return Err(Error::InvalidOptions(format!(
                "{width}x{height} exceeds the GPU's {max_dim} max texture dimension"
            )));
        }
        if need_prepare {
            let prepared = PreparedScene::build(scene, &prep, &mut self.cache);
            let ts = TraceScene::new(prepared);
            self.scene = Some(self.upload(ts)?);
            self.prep = Some(prep);
            self.dirty = false;
        }
        let st = self.scene.as_ref().expect("uploaded above");
        self.camera = Some(Camera::resolve(&st.prepared, opts, width, height));
        if self.size != (width, height) || self.accum.is_none() {
            self.size = (width, height);
            let bytes = width as u64 * height as u64 * 16;
            self.check_size(bytes, "accumulation buffer")?;
            self.accum = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pt accum"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        self.binding = None;
        self.reset();
        Ok(true)
    }

    fn check_size(&self, bytes: u64, what: &str) -> Result<()> {
        let l = self.device.limits();
        let cap = l.max_storage_buffer_binding_size.min(l.max_buffer_size);
        if bytes > cap {
            return Err(Error::Backend(format!(
                "wgpu: {what} needs {bytes} bytes; the device binds at most {cap}"
            )));
        }
        Ok(())
    }

    /// Queue `samples` more samples per pixel (compute dispatches; the
    /// call returns once the work is submitted — readbacks wait for
    /// it, [`Self::finish`] blocks explicitly). Sample indices continue
    /// from [`Self::samples`], so `refine(a); refine(b)` takes exactly
    /// the samples of `refine(a + b)`. No-op before the first
    /// [`Self::sync`].
    pub fn refine(&mut self, samples: u32) -> Result<()> {
        if self.scene.is_none() || samples == 0 {
            return Ok(());
        }
        let samples = samples.min(u32::MAX - self.samples);
        self.ensure_binding()?;
        let (w, h) = self.size;
        let pixels = w as u64 * h as u64;
        // Per dispatch: a row tile and a run of consecutive samples.
        let (rows, per) = if pixels >= PATHS_PER_DISPATCH {
            let rows = ((PATHS_PER_DISPATCH / w as u64).max(8) as u32).min(h);
            (rows, 1)
        } else {
            (h, (PATHS_PER_DISPATCH / pixels).clamp(1, 1 << 16) as u32)
        };
        let mut frames = Vec::new();
        let mut first = self.samples;
        let end = self.samples + samples;
        while first < end {
            let count = per.min(end - first);
            let mut y0 = 0;
            while y0 < h {
                frames.push(FrameParams {
                    first,
                    count,
                    y0,
                    rows: rows.min(h - y0),
                });
                y0 += rows;
            }
            first += count;
        }
        let scene_group = self.binding.as_ref().expect("ensured");
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut pending = 0u64;
        // One small uniform buffer of per-dispatch parameters per
        // chunk keeps memory bounded for huge `samples`.
        for chunk in frames.chunks(FRAMES_PER_BUFFER) {
            let mut bytes = vec![0u8; chunk.len() * FRAME_SLOT as usize];
            for (i, f) in chunk.iter().enumerate() {
                let at = i * FRAME_SLOT as usize;
                bytes[at..at + 16].copy_from_slice(bytemuck::bytes_of(f));
            }
            let fbuf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pt frames"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let fgroup = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pt frames"),
                layout: &self.frame_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &fbuf,
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                }],
            });
            for (i, f) in chunk.iter().enumerate() {
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("pt trace"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.trace_pipeline);
                    pass.set_bind_group(0, scene_group, &[]);
                    pass.set_bind_group(1, &fgroup, &[(i as u64 * FRAME_SLOT) as u32]);
                    pass.dispatch_workgroups(w.div_ceil(8), f.rows.div_ceil(8), 1);
                }
                pending += w as u64 * f.rows as u64 * f.count as u64;
                if pending >= PATHS_PER_SUBMIT {
                    let done = std::mem::replace(
                        &mut encoder,
                        self.device.create_command_encoder(&Default::default()),
                    );
                    self.queue.submit([done.finish()]);
                    pending = 0;
                }
            }
        }
        self.queue.submit([encoder.finish()]);
        self.samples += samples;
        Ok(())
    }

    /// Block until every queued [`Self::refine`] dispatch finished.
    pub fn finish(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| backend_err("poll", e))
    }

    /// Resolve the current estimate into the tracer's
    /// [`crate::COLOR_FORMAT`] output texture (exposure, tone map,
    /// background composite, sRGB encode — the CPU tracer's
    /// [`oxideav_render::PathTracer::image`] contract) without
    /// readback, and return it. The texture supports `COPY_SRC` and
    /// `TEXTURE_BINDING`; the work is submitted before returning.
    pub fn draw_texture(&mut self) -> Result<&wgpu::Texture> {
        let encoder = self.encode_resolve(false)?;
        self.queue.submit([encoder.finish()]);
        Ok(&self.outputs.as_ref().expect("ensured").color)
    }

    /// Current estimate as display bytes (uncovered pixels keep the
    /// background bytes exactly).
    pub fn image(&mut self) -> Result<RgbaImage> {
        let encoder = self.encode_resolve(false)?;
        let (w, h) = self.size;
        let out = self.outputs.as_ref().expect("ensured");
        let pixels = readback_texture(&self.device, &self.queue, encoder, &out.color, w, h, 4)?;
        Ok(RgbaImage {
            width: w,
            height: h,
            stride: w as usize * 4,
            pixels,
        })
    }

    /// Current estimate in scene-linear floats (no exposure / tone
    /// map; uncovered pixels hold the decoded background).
    pub fn hdr(&mut self) -> Result<HdrImage> {
        let encoder = self.encode_resolve(true)?;
        let (w, h) = self.size;
        let out = self.outputs.as_ref().expect("ensured");
        let bytes = readback_texture(&self.device, &self.queue, encoder, &out.hdr, w, h, 16)?;
        Ok(HdrImage {
            width: w,
            height: h,
            pixels: bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        })
    }

    fn ensure_outputs(&mut self) {
        if self.outputs.as_ref().is_some_and(|o| o.size == self.size) {
            return;
        }
        let (w, h) = self.size;
        let tex = |format, usage, label| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = tex(
            COLOR_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            "pt output",
        );
        let hdr = tex(
            HDR_OUT_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            "pt hdr output",
        );
        self.outputs = Some(Outputs {
            size: self.size,
            color_view: color.create_view(&Default::default()),
            hdr_view: hdr.create_view(&Default::default()),
            color,
            hdr,
        });
    }

    fn encode_resolve(&mut self, hdr: bool) -> Result<wgpu::CommandEncoder> {
        let (w, h) = self.size;
        if self.accum.is_none() {
            // Never synced: an all-background frame.
            self.accum = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pt accum"),
                size: w as u64 * h as u64 * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        self.ensure_outputs();
        let bg = self.opts.background.0.map(|c| c as f32 / 255.0);
        let params = ResolveParams {
            cfg: [1, 3, tone_map_code(self.opts.tone_map), u32::from(hdr)],
            exposure: [
                self.opts.exposure.max(0.0),
                self.samples as f32,
                w as f32,
                0.0,
            ],
            background: bg,
        };
        let pbuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pt resolve params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let g0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt resolve"),
            layout: &self.resolve_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: pbuf.as_entire_binding(),
            }],
        });
        let g1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt accum"),
            layout: &self.accum_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.accum.as_ref().expect("ensured").as_entire_binding(),
            }],
        });
        let out = self.outputs.as_ref().expect("ensured");
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pt resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: if hdr { &out.hdr_view } else { &out.color_view },
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(if hdr {
                &self.resolve_hdr
            } else {
                &self.resolve_color
            });
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.draw(0..3, 0..1);
        }
        Ok(encoder)
    }

    /// (Re)build the tables buffer, params and the scene bind group.
    fn ensure_binding(&mut self) -> Result<()> {
        if self.binding.is_some() {
            return Ok(());
        }
        let st = self.scene.as_ref().expect("synced");
        let cam = self.camera.as_ref().expect("synced");
        let env = self.env.as_deref().map(env_words);
        let mut tables = st.tables.clone();
        let env_base = tables.len() as u32;
        if let Some(e) = &env {
            tables.extend_from_slice(&e.words);
        }
        if tables.is_empty() {
            tables.push(0);
        }
        self.check_size(tables.len() as u64 * 4, "scene tables")?;
        let tables_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pt tables"),
                contents: bytemuck::cast_slice(&tables),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let (w, h) = self.size;
        let pt = &self.opts.path_trace;
        let has_env = env.is_some();
        let p_env = match (has_env, st.n_emissive > 0) {
            (true, true) => 0.5,
            (true, false) => 1.0,
            _ => 0.0,
        };
        let strategy = match pt.strategy {
            LightStrategy::LightOnly => 1,
            LightStrategy::BsdfOnly => 2,
            _ => 0,
        };
        let ortho = cam.projection == Projection::Orthographic;
        let e = env.as_ref();
        let params = Params {
            eye: [
                cam.eye[0],
                cam.eye[1],
                cam.eye[2],
                f32::from(u8::from(ortho)),
            ],
            forward: [cam.forward[0], cam.forward[1], cam.forward[2], cam.half_w],
            side: [cam.side[0], cam.side[1], cam.side[2], cam.half_h],
            up: [cam.up[0], cam.up[1], cam.up[2], 2.0 * cam.half_h / h as f32],
            dims: [w, h, pt.seed, pt.max_bounces],
            cfg: [pt.rr_start, strategy, st.n_punctual, st.n_emissive],
            env: [
                e.map_or(0, |e| e.w),
                e.map_or(0, |e| e.h),
                u32::from(has_env),
                st.node_count,
            ],
            offs0: [st.off_sobol, st.off_sheen, st.off_mat, st.off_plights],
            offs1: [
                st.off_elights,
                0,
                e.map_or(0, |e| env_base + e.marg),
                e.map_or(0, |e| env_base + e.cond),
            ],
            offs2: [
                e.map_or(0, |e| env_base + e.mass),
                e.map_or(0, |e| env_base + e.rad),
                st.tri_base,
                0,
            ],
            fparams: [
                if self.opts.ambient.is_finite() {
                    self.opts.ambient.max(0.0)
                } else {
                    0.0
                },
                pt.clamp,
                p_env,
                0.0,
            ],
        };
        let pbuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pt params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt scene"),
            layout: &self.scene_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pbuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: st.geom.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: st.attrs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tables_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.accum.as_ref().expect("synced").as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&st.atlas),
                },
            ],
        });
        self.binding = Some(group);
        Ok(())
    }

    /// Upload a trace scene: BVH + triangles, attributes, atlas,
    /// tables.
    fn upload(&mut self, ts: TraceScene) -> Result<SceneGpu> {
        let prepared = &ts.prepared;
        let lights = EmissiveLights::build(&ts);
        // ---- BVH + triangles in leaf-slot order.
        let (node_words, slots): (Vec<u32>, Vec<u32>) = match ts.bvh() {
            Some(b) => (b.node_words(), b.triangles.clone()),
            None => (Vec::new(), Vec::new()),
        };
        let node_count = (node_words.len() / 8) as u32;
        let tri_base = node_count * 2;
        let mut slot_of = vec![NONE; ts.triangle_count()];
        for (s, &g) in slots.iter().enumerate() {
            slot_of[g as usize] = s as u32;
        }
        let mut geom: Vec<u32> = node_words;
        geom.reserve(slots.len() * 12);
        let mut attrs: Vec<f32> = Vec::with_capacity(slots.len() * 48);
        let refs = ts.tri_refs();
        for &g in &slots {
            let r = refs[g as usize];
            let item = &prepared.items[r.item as usize];
            let pos = ts.triangle_positions(g);
            let light = lights.lookup.get(g as usize).copied().unwrap_or(NONE);
            let ws = [g, light, item.material as u32];
            for (p, wd) in pos.iter().zip(ws) {
                geom.extend_from_slice(&[p[0].to_bits(), p[1].to_bits(), p[2].to_bits(), wd]);
            }
            push_attrs(&mut attrs, item, 3 * r.tri as usize);
        }
        if geom.is_empty() {
            geom.resize(8, 0);
        }
        if attrs.is_empty() {
            attrs.resize(48, 0.0);
        }
        self.check_size(geom.len() as u64 * 4, "BVH + triangle buffer")?;
        self.check_size(attrs.len() as u64 * 4, "vertex attribute buffer")?;
        let geom = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pt geometry"),
                contents: bytemuck::cast_slice(&geom),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let attrs = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pt attributes"),
                contents: bytemuck::cast_slice(&attrs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // ---- Tables.
        let mut t: Vec<u32> = Vec::new();
        let off_sobol = t.len() as u32;
        for d in sobol_direction_numbers() {
            t.extend_from_slice(d);
        }
        let off_sheen = t.len() as u32;
        t.extend(sheen_albedo_table().iter().map(|v| v.to_bits()));
        // Textures: atlas + descriptors, keyed by (texture, space).
        let mut atlas = AtlasBuilder::new(&self.device);
        let mut descs: HashMap<(usize, bool), u32> = HashMap::new();
        let mut want = |b: &Option<TextureBinding>, space: ColorSpace| {
            if let Some(b) = b {
                if prepared.texture(b).is_some() {
                    descs
                        .entry((b.texture, space == ColorSpace::Srgb))
                        .or_insert(0);
                }
            }
        };
        let ext_bindings = |m: &oxideav_render::prepare::PreparedMaterial| {
            let e = &m.ext;
            [
                (
                    e.specular.as_ref().map(|s| s.factor_texture),
                    ColorSpace::Linear,
                ),
                (
                    e.specular.as_ref().map(|s| s.color_texture),
                    ColorSpace::Srgb,
                ),
                (
                    e.transmission.as_ref().map(|s| s.factor_texture),
                    ColorSpace::Linear,
                ),
                (
                    e.volume.as_ref().map(|s| s.thickness_texture),
                    ColorSpace::Linear,
                ),
                (
                    e.clearcoat.as_ref().map(|s| s.factor_texture),
                    ColorSpace::Linear,
                ),
                (
                    e.clearcoat.as_ref().map(|s| s.roughness_texture),
                    ColorSpace::Linear,
                ),
                (
                    e.clearcoat.as_ref().map(|s| s.normal_texture),
                    ColorSpace::Linear,
                ),
                (e.sheen.as_ref().map(|s| s.color_texture), ColorSpace::Srgb),
                (
                    e.sheen.as_ref().map(|s| s.roughness_texture),
                    ColorSpace::Linear,
                ),
            ]
            .map(|(r, sp)| (r.flatten().and_then(|r| ts.bind_ref(&Some(r))), sp))
        };
        let mut slot_bindings: Vec<[(Option<TextureBinding>, ColorSpace); 13]> = Vec::new();
        for m in &prepared.materials {
            let ext = ext_bindings(m);
            let core = [
                (m.base_color_texture, ColorSpace::Srgb),
                (m.metallic_roughness_texture, ColorSpace::Linear),
                (m.normal_texture, ColorSpace::Linear),
                (m.emissive_texture, ColorSpace::Srgb),
            ];
            let all: [(Option<TextureBinding>, ColorSpace); 13] =
                std::array::from_fn(|i| if i < 4 { core[i] } else { ext[i - 4] });
            for (b, sp) in &all {
                want(b, *sp);
            }
            slot_bindings.push(all);
        }
        let mut keys: Vec<(usize, bool)> = descs.keys().copied().collect();
        keys.sort_unstable();
        for &(ti, srgb) in &keys {
            let tex = prepared.textures[ti].as_ref().expect("checked");
            let space = if srgb {
                ColorSpace::Srgb
            } else {
                ColorSpace::Linear
            };
            atlas.add(&tex.data, space);
        }
        let placed = atlas.pack()?;
        for &(ti, srgb) in &keys {
            let tex = prepared.textures[ti].as_ref().expect("checked");
            let space = if srgb {
                ColorSpace::Srgb
            } else {
                ColorSpace::Linear
            };
            let entry = &placed[&(Arc::as_ptr(&tex.data) as usize, srgb)];
            descs.insert((ti, srgb), t.len() as u32);
            let img = tex.data.image();
            let full = tex.data.mips(space).len() as u32;
            t.extend_from_slice(&[
                full,
                sampler_flags(&tex.sampler),
                img.width,
                img.height,
                entry.skip,
                entry.rects.len() as u32,
                0,
                0,
            ]);
            for r in &entry.rects {
                t.extend_from_slice(&[r.layer, r.x, r.y, r.w | (r.h << 16)]);
            }
        }
        let atlas_view = atlas.upload(&self.queue)?;
        // Materials.
        let off_mat = t.len() as u32;
        for (m, bindings) in prepared.materials.iter().zip(&slot_bindings) {
            let mut w = material_words(m);
            for (s, (b, sp)) in bindings.iter().enumerate() {
                let at = MAT_SLOTS + 8 * s;
                let desc = b
                    .as_ref()
                    .and_then(|b| descs.get(&(b.texture, *sp == ColorSpace::Srgb)).copied());
                let (Some(b), Some(desc)) = (b, desc) else {
                    w[at] = NONE;
                    continue;
                };
                let mx = b.transform.map(|x| x.to_matrix()).unwrap_or([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]);
                w[at] = desc;
                w[at + 1] = b.uv_set;
                for (k, v) in [mx[0][0], mx[0][1], mx[0][2], mx[1][0], mx[1][1], mx[1][2]]
                    .iter()
                    .enumerate()
                {
                    w[at + 2 + k] = v.to_bits();
                }
            }
            t.extend_from_slice(&w);
        }
        // Punctual lights.
        let off_plights = t.len() as u32;
        for l in &prepared.lights {
            let kind = match l.kind {
                LightKind::Directional => 0u32,
                LightKind::Point => 1,
                LightKind::Spot => 2,
            };
            let base = [
                l.color[0] * l.intensity,
                l.color[1] * l.intensity,
                l.color[2] * l.intensity,
            ];
            let cos_outer = l.outer_cone_angle.cos();
            let cos_inner = l.inner_cone_angle.cos();
            let scale = 1.0 / (cos_inner - cos_outer).max(0.001);
            let offset = -cos_outer * scale;
            let f = |v: f32| v.to_bits();
            t.extend_from_slice(&[
                f(l.position[0]),
                f(l.position[1]),
                f(l.position[2]),
                kind,
                f(l.direction[0]),
                f(l.direction[1]),
                f(l.direction[2]),
                f(l.range.unwrap_or(0.0)),
                f(base[0]),
                f(base[1]),
                f(base[2]),
                0,
                f(scale),
                f(offset),
                0,
                0,
            ]);
        }
        // Emissive triangles.
        let off_elights = t.len() as u32;
        for (li, &g) in lights.tris.iter().enumerate() {
            t.extend_from_slice(&[
                lights.cdf[li].to_bits(),
                lights.pmf[li].to_bits(),
                g,
                slot_of[g as usize],
            ]);
        }
        let n_punctual = prepared.lights.len() as u32;
        // Camera framing only needs bounds + cameras: drop the bulky
        // per-vertex arrays.
        let mut light = ts.prepared;
        for it in &mut light.items {
            it.positions = Vec::new();
            it.normals = Vec::new();
            it.tangents = Vec::new();
            it.uvs = Vec::new();
            it.colors = Vec::new();
        }
        light.textures = Vec::new();
        Ok(SceneGpu {
            prepared: light,
            geom,
            attrs,
            atlas: atlas_view,
            tables: t,
            node_count,
            tri_base,
            n_punctual,
            n_emissive: lights.tris.len() as u32,
            off_sobol,
            off_sheen,
            off_mat,
            off_plights,
            off_elights,
        })
    }

    /// Full render: re-prepare, take
    /// `opts.path_trace.samples_per_pixel` samples, resolve.
    fn run(&mut self, scene: &Scene3D, opts: &RenderOptions) -> Result<()> {
        self.invalidate_scene();
        self.sync(scene, opts)?;
        self.refine(opts.path_trace.samples_per_pixel.max(1))
    }
}

impl oxideav_render::Renderer for GpuPathTracer {
    fn render(&mut self, scene: &Scene3D, opts: &RenderOptions) -> Result<RgbaImage> {
        self.run(scene, opts)?;
        self.image()
    }

    fn render_hdr(&mut self, scene: &Scene3D, opts: &RenderOptions) -> Result<HdrImage> {
        self.run(scene, opts)?;
        self.hdr()
    }

    fn set_texture_resolver(&mut self, resolver: Arc<dyn TextureResolver>) {
        GpuPathTracer::set_texture_resolver(self, resolver);
    }
}

pub(crate) fn tone_map_code(t: ToneMap) -> u32 {
    match t {
        ToneMap::Reinhard => 1,
        ToneMap::AcesFitted => 2,
        _ => 0,
    }
}

/// 12 `vec4`s of per-slot attributes (see pathtrace.wgsl).
fn push_attrs(out: &mut Vec<f32>, item: &DrawItem, base: usize) {
    let n = item.positions.len();
    let normals = item.has_vertex_normals && item.normals.len() == n;
    let tangents = item.tangents.len() == n;
    let colors = item.colors.len() == n;
    let uv = |s: u32| item.uv_set(s).filter(|v| v.len() == n);
    let (uv0, uv1) = (uv(0), uv(1));
    let flags = u32::from(normals)
        | u32::from(tangents) << 1
        | u32::from(colors) << 2
        | u32::from(uv0.is_some()) << 3
        | u32::from(uv1.is_some()) << 4;
    for v in 0..3 {
        let nv = if normals {
            item.normals[base + v]
        } else {
            [0.0; 3]
        };
        let w = if v == 0 { f32::from_bits(flags) } else { 0.0 };
        out.extend_from_slice(&[nv[0], nv[1], nv[2], w]);
    }
    for v in 0..3 {
        out.extend_from_slice(&if tangents {
            item.tangents[base + v]
        } else {
            [0.0; 4]
        });
    }
    for v in 0..3 {
        let a = uv0.map_or([0.0; 2], |u| u[base + v]);
        let b = uv1.map_or([0.0; 2], |u| u[base + v]);
        out.extend_from_slice(&[a[0], a[1], b[0], b[1]]);
    }
    for v in 0..3 {
        out.extend_from_slice(&if colors {
            item.colors[base + v]
        } else {
            [1.0; 4]
        });
    }
}

fn sampler_flags(s: &oxideav_mesh3d::Sampler) -> u32 {
    let wrap = |w: WrapMode| match w {
        WrapMode::ClampToEdge => 0u32,
        WrapMode::MirroredRepeat => 1,
        WrapMode::Repeat => 2,
    };
    let mag = u32::from(s.effective_mag_filter() == MagFilter::Linear);
    let min = match s.effective_min_filter() {
        MinFilter::Nearest => 0u32,
        MinFilter::Linear => 1,
        MinFilter::NearestMipNearest => 2,
        MinFilter::LinearMipNearest => 3,
        MinFilter::NearestMipLinear => 4,
        MinFilter::LinearMipLinear => 5,
    };
    wrap(s.wrap_s) | wrap(s.wrap_t) << 2 | mag << 4 | min << 5
}

/// Scalar part of a material record (`MAT_STRIDE` words; texture
/// slots filled by the caller). Mirrors `TraceScene::material_oriented`
/// factor handling.
fn material_words(m: &oxideav_render::prepare::PreparedMaterial) -> Vec<u32> {
    let mut w = vec![0u32; MAT_STRIDE];
    let mut put = |i: usize, v: f32| w[i] = v.to_bits();
    for k in 0..4 {
        put(k, m.base_color[k]);
    }
    for k in 0..3 {
        put(4 + k, m.emissive[k]);
    }
    let (mode, cutoff) = match m.alpha_mode {
        AlphaMode::Mask { cutoff } => (1u32, cutoff),
        AlphaMode::Blend => (2, 0.0),
        _ => (0, 0.0),
    };
    put(8, m.metallic);
    put(9, m.roughness);
    put(10, m.normal_scale);
    put(11, cutoff);
    put(12, m.ior);
    let fin = |v: f32, d: f32| if v.is_finite() { v } else { d };
    let ext = &m.ext;
    let (mut specular, mut specular_color) = (1.0, [1.0f32; 3]);
    if let Some(s) = &ext.specular {
        specular = fin(s.factor, 1.0).clamp(0.0, 1.0);
        specular_color = [
            fin(s.color_factor[0], 1.0).max(0.0),
            fin(s.color_factor[1], 1.0).max(0.0),
            fin(s.color_factor[2], 1.0).max(0.0),
        ];
    }
    put(13, specular);
    for (k, v) in specular_color.into_iter().enumerate() {
        put(14 + k, v);
    }
    let transmission = ext
        .transmission
        .as_ref()
        .map_or(0.0, |t| fin(t.factor, 0.0).clamp(0.0, 1.0));
    put(17, transmission);
    let (mut thickness, mut ac, mut ad) = (0.0, [1.0f32; 3], -1.0);
    if let Some(v) = &ext.volume {
        thickness = fin(v.thickness, 0.0).max(0.0);
        ac = [
            fin(v.attenuation_color[0], 1.0).clamp(0.0, 1.0),
            fin(v.attenuation_color[1], 1.0).clamp(0.0, 1.0),
            fin(v.attenuation_color[2], 1.0).clamp(0.0, 1.0),
        ];
        let d = v.effective_attenuation_distance();
        if d.is_finite() && d > 0.0 {
            ad = d;
        }
    }
    put(18, thickness);
    for (k, v) in ac.into_iter().enumerate() {
        put(19 + k, v);
    }
    put(22, ad);
    let (mut cc, mut ccr, mut ccs) = (0.0, 0.0, 1.0);
    if let Some(c) = &ext.clearcoat {
        cc = fin(c.factor, 0.0).clamp(0.0, 1.0);
        ccr = fin(c.roughness, 0.0).clamp(0.0, 1.0);
        ccs = fin(c.normal_scale, 1.0);
    }
    put(23, cc);
    put(24, ccr);
    put(25, ccs);
    let (mut sc, mut sr) = ([0.0f32; 3], 0.0);
    if let Some(s) = &ext.sheen {
        sc = [
            fin(s.color_factor[0], 0.0).clamp(0.0, 1.0),
            fin(s.color_factor[1], 0.0).clamp(0.0, 1.0),
            fin(s.color_factor[2], 0.0).clamp(0.0, 1.0),
        ];
        sr = fin(s.roughness, 0.0).clamp(0.0, 1.0);
    }
    for (k, v) in sc.into_iter().enumerate() {
        put(26 + k, v);
    }
    put(29, sr);
    let emissive_lit = m.emissive.iter().any(|&c| c > 0.0);
    w[7] = mode
        | u32::from(m.double_sided) << 2
        | u32::from(m.unlit) << 3
        | u32::from(emissive_lit) << 4;
    w
}

/// Environment words: marginal, conditional, mass, radiance (rgb per
/// texel, already clamped / scaled / zero when uniform —
/// `EnvironmentMap::radiance`).
fn env_words(env: &EnvironmentMap) -> EnvWords {
    let img = env.image();
    let (w, h) = (img.width.max(1), img.height.max(1));
    let mut words = Vec::new();
    let marg = 0u32;
    words.extend(env.marginal_cdf().iter().map(|v| v.to_bits()));
    let cond = words.len() as u32;
    words.extend(env.conditional_cdf().iter().map(|v| v.to_bits()));
    let mass = words.len() as u32;
    words.extend(env.texel_mass().iter().map(|v| v.to_bits()));
    let rad = words.len() as u32;
    let k = env.intensity();
    for i in 0..(w * h) as usize {
        let c = img
            .pixels
            .get(i * 4..i * 4 + 3)
            .filter(|_| !env.is_uniform())
            .map(|p| [p[0], p[1], p[2]])
            .filter(|c| c.iter().all(|v| v.is_finite()))
            .map_or([0.0; 3], |c| {
                [c[0].max(0.0) * k, c[1].max(0.0) * k, c[2].max(0.0) * k]
            });
        words.extend(c.iter().map(|v| v.to_bits()));
    }
    EnvWords {
        words,
        w,
        h,
        marg,
        cond,
        mass,
        rad,
    }
}

// =====================================================================
// Texture atlas.
// =====================================================================

#[derive(Debug, Clone, Copy)]
struct Rect {
    layer: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// Placement of one mip chain: dropped leading levels + kept rects.
struct Placed {
    skip: u32,
    rects: Vec<Rect>,
}

struct AtlasBuilder<'a> {
    device: &'a wgpu::Device,
    chains: Vec<(Arc<TextureData>, ColorSpace)>,
    layer: u32,
    layers: u32,
    texels: Vec<Vec<u16>>,
}

impl<'a> AtlasBuilder<'a> {
    fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            chains: Vec::new(),
            layer: 1,
            layers: 1,
            texels: Vec::new(),
        }
    }

    fn add(&mut self, data: &Arc<TextureData>, space: ColorSpace) {
        let key = (Arc::as_ptr(data) as usize, space == ColorSpace::Srgb);
        if !self
            .chains
            .iter()
            .any(|(d, s)| (Arc::as_ptr(d) as usize, *s == ColorSpace::Srgb) == key)
        {
            self.chains.push((data.clone(), space));
        }
    }

    /// Shelf-pack every kept level (tallest first) into square layers.
    fn pack(&mut self) -> Result<HashMap<(usize, bool), Placed>> {
        let limits = self.device.limits();
        let cap = limits.max_texture_dimension_2d.min(MAX_LAYER);
        let biggest = self
            .chains
            .iter()
            .filter_map(|(d, s)| d.mips(*s).first().map(|m| m.width.max(m.height)))
            .max()
            .unwrap_or(1);
        self.layer = biggest.next_power_of_two().clamp(1, cap);
        let side = self.layer;
        let mut items: Vec<(usize, usize, u32, u32)> = Vec::new();
        let mut placed: HashMap<(usize, bool), Placed> = HashMap::new();
        for (ci, (d, s)) in self.chains.iter().enumerate() {
            let mips = d.mips(*s);
            let skip = mips
                .iter()
                .take_while(|m| m.width > side || m.height > side)
                .count();
            placed.insert(
                (Arc::as_ptr(d) as usize, *s == ColorSpace::Srgb),
                Placed {
                    skip: skip as u32,
                    rects: Vec::new(),
                },
            );
            for (li, m) in mips.iter().enumerate().skip(skip) {
                items.push((ci, li, m.width, m.height));
            }
        }
        items.sort_by(|a, b| b.3.cmp(&a.3).then(b.2.cmp(&a.2)));
        let (mut layer, mut x, mut y, mut shelf) = (0u32, 0u32, 0u32, 0u32);
        let mut rects: Vec<(usize, usize, Rect)> = Vec::new();
        for &(ci, li, w, h) in &items {
            if x + w > side {
                x = 0;
                y += shelf;
                shelf = 0;
            }
            if y + h > side {
                layer += 1;
                x = 0;
                y = 0;
                shelf = 0;
            }
            rects.push((ci, li, Rect { layer, x, y, w, h }));
            x += w;
            shelf = shelf.max(h);
        }
        self.layers = layer + 1;
        if self.layers > limits.max_texture_array_layers {
            return Err(Error::Backend(format!(
                "wgpu: path-tracer texture atlas needs {} layers of {side}²; the device allows {}",
                self.layers, limits.max_texture_array_layers
            )));
        }
        rects.sort_by_key(|(ci, li, _)| (*ci, *li));
        let texels_per_layer = (side * side * 4) as usize;
        self.texels = vec![vec![0u16; texels_per_layer]; self.layers as usize];
        for (ci, li, r) in rects {
            let (d, s) = &self.chains[ci];
            let m = &d.mips(*s)[li];
            let dst = &mut self.texels[r.layer as usize];
            for row in 0..r.h {
                for col in 0..r.w {
                    let t = m.texels[(row * r.w + col) as usize];
                    let at = (((r.y + row) * side + r.x + col) * 4) as usize;
                    for c in 0..4 {
                        dst[at + c] = f32_to_f16(t[c]);
                    }
                }
            }
            placed
                .get_mut(&(Arc::as_ptr(d) as usize, *s == ColorSpace::Srgb))
                .expect("inserted")
                .rects
                .push(r);
        }
        Ok(placed)
    }

    fn upload(self, queue: &wgpu::Queue) -> Result<wgpu::TextureView> {
        let side = self.layer;
        let mut bytes: Vec<u8> = Vec::with_capacity(self.texels.iter().map(|l| l.len() * 2).sum());
        if self.texels.is_empty() {
            bytes.resize((side * side * 8) as usize, 0);
        }
        for l in &self.texels {
            bytes.extend_from_slice(bytemuck::cast_slice(l));
        }
        let tex = self.device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("pt texture atlas"),
                size: wgpu::Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: self.layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: ATLAS_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &bytes,
        );
        Ok(tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        }))
    }
}
