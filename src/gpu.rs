//! wgpu device, pipelines, and the two-pass frame:
//!
//! 1. **scene pass** at `aa ×` the output size into an `Rgba16Float`
//!    scene-linear target plus an `R8Unorm` coverage mask, with depth;
//! 2. **resolve pass** at output size: tone map, background composite,
//!    supersample average, sRGB encode into `Rgba8Unorm`.

use bytemuck::{Pod, Zeroable};
use oxideav_render::prepare::{DrawTopology, LightKind, PrepareOptions, PreparedLight};
use oxideav_render::{
    Camera, DepthRange, Error, PreparedScene, RenderOptions, Result, RgbaImage, ShadingMode,
    TextureCache, ToneMap,
};
use wgpu::util::DeviceExt;

use crate::scene::{
    material_layout, GpuScene, Pass, ResourceCache, Uploader, Vertex, VERTEX_ATTRIBUTES,
};
use crate::GpuBackend;

/// Output format of the resolve pass (holds sRGB-encoded values).
pub(crate) const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const COVERAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Legacy-mode ambient term — identical to the scanline backend's.
const LEGACY_AMBIENT: f32 = 0.2;
const MAX_LIGHTS: usize = 16;
const MAX_SHADOWS: usize = 4;
const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Dynamic-offset stride of the per-light shadow-pass uniform.
const SHADOW_SLOT: u64 = 256;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
struct ShadowUniform {
    view_proj: [[f32; 4]; 4],
    eye_texel: [f32; 4],
    dir_persp: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ShadowPassUniform {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    dir: [f32; 4],
}

/// Shadow-map array (one `R32Float` linear-depth layer per shadowed
/// light) plus its depth buffer, cached per (size, layers).
struct ShadowTargets {
    key: (u32, u32),
    array_view: wgpu::TextureView,
    layer_views: Vec<wgpu::TextureView>,
    depth_view: wgpu::TextureView,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
struct LightUniform {
    position: [f32; 4],
    direction: [f32; 4],
    radiance: [f32; 4],
    cone: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    legacy_light: [f32; 4],
    params: [f32; 4],
    mode: [u32; 4],
    lights: [LightUniform; MAX_LIGHTS],
    shadows: [ShadowUniform; MAX_SHADOWS],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ResolveParams {
    cfg: [u32; 4],
    exposure: [f32; 4],
    background: [f32; 4],
}

/// Frame targets, cached across frames of the same geometry.
struct Targets {
    key: (u32, u32, u32, u32),
    hdr_view: wgpu::TextureView,
    coverage_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
}

struct Pipelines {
    /// `[cull none, cull back]`.
    opaque: [wgpu::RenderPipeline; 2],
    /// `[cull none, cull back]`, premultiplied over, no depth write.
    blend: [wgpu::RenderPipeline; 2],
    lines: wgpu::RenderPipeline,
    points: wgpu::RenderPipeline,
    resolve: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
}

/// A live GPU device with the scene pipelines compiled.
pub(crate) struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    max_dim: u32,
    globals_layout: wgpu::BindGroupLayout,
    material_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
    shadow_pass_layout: wgpu::BindGroupLayout,
    empty_group: wgpu::BindGroup,
    pipelines: Pipelines,
    cache: ResourceCache,
    targets: Option<Targets>,
    shadow_targets: Option<ShadowTargets>,
    texture_cache: TextureCache,
}

fn backend_err(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("wgpu: {what}: {e}"))
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GpuContext {
    /// Open an adapter + device headlessly (no surface).
    pub(crate) fn new(backend: GpuBackend) -> Result<Self> {
        pollster::block_on(Self::new_async(backend))
    }

    async fn new_async(backend: GpuBackend) -> Result<Self> {
        let backends = match backend {
            GpuBackend::Auto => wgpu::Backends::all(),
            GpuBackend::Vulkan => wgpu::Backends::VULKAN,
            GpuBackend::Gl => wgpu::Backends::GL,
            GpuBackend::Metal => wgpu::Backends::METAL,
            GpuBackend::Dx12 => wgpu::Backends::DX12,
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| backend_err("no suitable adapter", e))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("oxideav-render-vulkan"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| backend_err("request_device", e))?;
        Ok(Self::from_device(device, queue, adapter.get_info()))
    }

    /// Build pipelines on an existing device (e.g. one shared with a
    /// windowed viewer).
    pub(crate) fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
    ) -> Self {
        let max_dim = device.limits().max_texture_dimension_2d;
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let material_layout = material_layout(&device);
        let shadow_pass_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow pass"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<ShadowPassUniform>() as u64,
                        ),
                    },
                    count: None,
                }],
            });
        // The shadow pass renders into the array `@group(0)` samples,
        // so it binds an empty group 0 instead of the globals.
        let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("empty"),
            entries: &[],
        });
        let empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("empty"),
            layout: &empty_layout,
            entries: &[],
        });
        let scene_texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resolve"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                scene_texture(1),
                scene_texture(2),
            ],
        });
        let pipelines = build_pipelines(
            &device,
            [&globals_layout, &material_layout, &resolve_layout],
            [&empty_layout, &shadow_pass_layout],
        );
        let cache = ResourceCache::new(&device, &queue);
        Self {
            device,
            queue,
            adapter_info,
            max_dim,
            globals_layout,
            material_layout,
            resolve_layout,
            shadow_pass_layout,
            empty_group,
            pipelines,
            cache,
            targets: None,
            shadow_targets: None,
            texture_cache: TextureCache::default(),
        }
    }

    pub(crate) fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn texture_cache_mut(&mut self) -> &mut TextureCache {
        &mut self.texture_cache
    }

    /// Prepare `scene` (pose at `opts.time`, textures, lights,
    /// cameras) and upload it.
    pub(crate) fn upload(
        &mut self,
        scene: &oxideav_mesh3d::Scene3D,
        opts: &RenderOptions,
    ) -> GpuScene {
        let mut popts = PrepareOptions::from_render_options(opts);
        // Scene lights are always kept; whether they are used is a
        // draw-time choice (`use_scene_lights`).
        popts.use_scene_lights = true;
        popts.generate_tangents = true;
        let prepared = PreparedScene::build(scene, &popts, &mut self.texture_cache);
        Uploader {
            device: &self.device,
            queue: &self.queue,
            material_layout: &self.material_layout,
            cache: &mut self.cache,
        }
        .upload(prepared)
    }

    fn ensure_targets(&mut self, rw: u32, rh: u32, w: u32, h: u32) {
        let key = (rw, rh, w, h);
        if self.targets.as_ref().is_some_and(|t| t.key == key) {
            return;
        }
        let tex = |width, height, format, usage, label| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
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
        let attach = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let hdr = tex(rw, rh, HDR_FORMAT, attach, "scene hdr");
        let coverage = tex(rw, rh, COVERAGE_FORMAT, attach, "scene coverage");
        let depth = tex(
            rw,
            rh,
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            "scene depth",
        );
        let output = tex(
            w,
            h,
            COLOR_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            "output",
        );
        self.targets = Some(Targets {
            key,
            hdr_view: hdr.create_view(&Default::default()),
            coverage_view: coverage.create_view(&Default::default()),
            depth_view: depth.create_view(&Default::default()),
            output_view: output.create_view(&Default::default()),
            output,
        });
    }

    /// Supersampling factor that fits the device limits.
    fn effective_aa(&self, opts: &RenderOptions, w: u32, h: u32) -> u32 {
        let mut aa = opts.aa.clamp(1, 8);
        while aa > 1 && (w * aa > self.max_dim || h * aa > self.max_dim) {
            aa -= 1;
        }
        aa
    }

    /// Draw `gs` at `opts.width × opts.height` (supersampled by
    /// `opts.aa`) into the output texture and submit.
    pub(crate) fn draw(
        &mut self,
        gs: &mut GpuScene,
        opts: &RenderOptions,
    ) -> Result<&wgpu::Texture> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode_frame(&mut encoder, gs, opts)?;
        self.queue.submit([encoder.finish()]);
        Ok(&self.targets.as_ref().expect("targets ensured").output)
    }

    fn encode_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gs: &mut GpuScene,
        opts: &RenderOptions,
    ) -> Result<()> {
        let (w, h) = (opts.width.max(1), opts.height.max(1));
        if w > self.max_dim || h > self.max_dim {
            return Err(Error::InvalidOptions(format!(
                "{w}x{h} exceeds the GPU's {} max texture dimension",
                self.max_dim
            )));
        }
        let aa = self.effective_aa(opts, w, h);
        let (rw, rh) = (w * aa, h * aa);
        self.ensure_targets(rw, rh, w, h);

        let mode = shading_code(opts.shading);
        let pbr = mode == 6;
        let wireframe = opts.shading == ShadingMode::Wireframe;
        let camera = Camera::resolve(&gs.prepared, opts, rw, rh);

        // Lights.
        let fallback;
        let lights: &[PreparedLight] = if opts.use_scene_lights && !gs.scene_lights.is_empty() {
            &gs.scene_lights
        } else {
            fallback = [PreparedLight::from_light_spec(opts.light)];
            &fallback
        };
        let mut light_uniforms = [LightUniform::default(); MAX_LIGHTS];
        for (u, l) in light_uniforms.iter_mut().zip(lights) {
            let kind = match l.kind {
                LightKind::Directional => 0.0,
                LightKind::Point => 1.0,
                _ => 2.0,
            };
            *u = LightUniform {
                position: [l.position[0], l.position[1], l.position[2], kind],
                direction: [
                    l.direction[0],
                    l.direction[1],
                    l.direction[2],
                    l.range.unwrap_or(0.0),
                ],
                radiance: [
                    l.color[0] * l.intensity,
                    l.color[1] * l.intensity,
                    l.color[2] * l.intensity,
                    0.0,
                ],
                cone: [
                    l.inner_cone_angle.cos(),
                    l.outer_cone_angle.cos(),
                    -1.0,
                    0.0,
                ],
            };
        }
        // Shadow maps (Pbr + `opts.shadows`): directional / spot lights,
        // first MAX_SHADOWS of them.
        let mut shadow_uniforms = [ShadowUniform::default(); MAX_SHADOWS];
        let mut shadow_setups = Vec::new();
        let shadow_size = opts.shadow_map_size.clamp(16, 8192).min(self.max_dim);
        if pbr && opts.shadows {
            if let Some(bounds) = gs.prepared.bounds() {
                for (li, l) in lights.iter().enumerate().take(MAX_LIGHTS) {
                    if shadow_setups.len() == MAX_SHADOWS {
                        break;
                    }
                    if let Some(sh) = crate::shadow::setup(l, bounds, shadow_size) {
                        let layer = shadow_setups.len();
                        light_uniforms[li].cone[2] = layer as f32;
                        shadow_uniforms[layer] = ShadowUniform {
                            view_proj: crate::shadow::column_major(&sh.view_proj),
                            eye_texel: [sh.eye[0], sh.eye[1], sh.eye[2], sh.texel],
                            dir_persp: [
                                sh.dir[0],
                                sh.dir[1],
                                sh.dir[2],
                                sh.perspective as u32 as f32,
                            ],
                        };
                        shadow_setups.push(sh);
                    }
                }
            }
        }
        let shadow_key = if shadow_setups.is_empty() {
            (1, 1)
        } else {
            (shadow_size, shadow_setups.len() as u32)
        };
        self.ensure_shadow_targets(shadow_key);
        if !shadow_setups.is_empty() {
            self.encode_shadow_maps(encoder, gs, &shadow_setups);
        }

        let legacy = {
            let az = opts.light.azimuth_deg.to_radians();
            let el = opts.light.elevation_deg.to_radians();
            let d = [el.cos() * az.sin(), el.sin(), el.cos() * az.cos()];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12);
            [
                d[0] / len,
                d[1] / len,
                d[2] / len,
                opts.light.intensity.max(0.0),
            ]
        };
        let vp = camera.view_projection(DepthRange::ZeroToOne);
        let globals = Globals {
            view_proj: std::array::from_fn(|c| std::array::from_fn(|r| vp[r][c])),
            eye: [camera.eye[0], camera.eye[1], camera.eye[2], 1.0],
            legacy_light: legacy,
            params: [LEGACY_AMBIENT, opts.ambient.max(0.0), 0.0, 0.0],
            mode: [mode, lights.len().min(MAX_LIGHTS) as u32, 0, 0],
            lights: light_uniforms,
            shadows: shadow_uniforms,
        };
        let globals_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("globals"),
                contents: bytemuck::bytes_of(&globals),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let globals_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &self.globals_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &self.shadow_targets.as_ref().expect("ensured").array_view,
                    ),
                },
            ],
        });

        // Wireframe edges (legacy) and sorted BLEND triangles (Pbr).
        if wireframe && gs.edges.is_none() {
            let mut idx = Vec::new();
            for it in gs
                .items
                .iter()
                .filter(|i| i.topology == DrawTopology::Triangles)
            {
                for t in (it.first..it.first + it.count).step_by(3) {
                    idx.extend_from_slice(&[t, t + 1, t + 1, t + 2, t + 2, t]);
                }
            }
            gs.edges = Some(self.index_buffer(&idx, "edges"));
        }
        let mut blend_runs: Vec<(usize, std::ops::Range<u32>)> = Vec::new();
        let blend_buf = if pbr && !gs.blend_tris.is_empty() {
            let mut order: Vec<(f32, usize)> = gs
                .blend_tris
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let d: f32 = (0..3)
                        .map(|k| (t.centroid[k] - camera.eye[k]) * camera.forward[k])
                        .sum();
                    (d, i)
                })
                .collect();
            // Back to front: farthest view depth first.
            order.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut idx = Vec::with_capacity(order.len() * 3);
            for (_, i) in &order {
                let t = gs.blend_tris[*i];
                let start = idx.len() as u32;
                idx.extend_from_slice(&[t.first, t.first + 1, t.first + 2]);
                match blend_runs.last_mut() {
                    Some((m, r)) if *m == t.material => r.end = start + 3,
                    _ => blend_runs.push((t.material, start..start + 3)),
                }
            }
            self.index_buffer(&idx, "blend order")
        } else {
            None
        };

        let targets = self.targets.as_ref().expect("targets ensured above");
        let bg = opts.background.0.map(|c| c as f32 / 255.0);
        let bg_lin = bg.map(|c| oxideav_render::hdr::srgb_to_linear(c) as f64);
        let clear = if pbr {
            let a = bg[3] as f64;
            wgpu::Color {
                r: bg_lin[0] * a,
                g: bg_lin[1] * a,
                b: bg_lin[2] * a,
                a,
            }
        } else {
            wgpu::Color::TRANSPARENT
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.hdr_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(clear),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.coverage_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(vbuf) = &gs.vertices {
                pass.set_bind_group(0, &globals_bg, &[]);
                pass.set_vertex_buffer(0, vbuf.slice(..));
                let p = &self.pipelines;
                if wireframe {
                    if let Some(Some((ibuf, _))) = &gs.edges {
                        // Each item keeps its own material colour, so
                        // edges are drawn per item.
                        pass.set_pipeline(&p.lines);
                        pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        let mut at = 0u32;
                        for it in gs
                            .items
                            .iter()
                            .filter(|i| i.topology == DrawTopology::Triangles)
                        {
                            let k = it.count * 2;
                            pass.set_bind_group(1, &gs.materials[it.material].bind_group, &[]);
                            pass.draw_indexed(at..at + k, 0, 0..1);
                            at += k;
                        }
                    }
                } else {
                    for it in gs
                        .items
                        .iter()
                        .filter(|i| i.topology == DrawTopology::Triangles)
                    {
                        let m = &gs.materials[it.material];
                        if pbr && m.pass == Pass::Blend {
                            continue;
                        }
                        let cull = usize::from(pbr && !m.double_sided);
                        pass.set_pipeline(&p.opaque[cull]);
                        pass.set_bind_group(1, &m.bind_group, &[]);
                        pass.draw(it.first..it.first + it.count, 0..1);
                    }
                }
                for it in gs
                    .items
                    .iter()
                    .filter(|i| i.topology != DrawTopology::Triangles)
                {
                    pass.set_pipeline(if it.topology == DrawTopology::Lines {
                        &p.lines
                    } else {
                        &p.points
                    });
                    pass.set_bind_group(1, &gs.materials[it.material].bind_group, &[]);
                    pass.draw(it.first..it.first + it.count, 0..1);
                }
                if let Some((ibuf, _)) = &blend_buf {
                    pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    for (mat, range) in &blend_runs {
                        let m = &gs.materials[*mat];
                        pass.set_pipeline(&p.blend[usize::from(!m.double_sided)]);
                        pass.set_bind_group(1, &m.bind_group, &[]);
                        pass.draw_indexed(range.clone(), 0, 0..1);
                    }
                }
            }
        }

        // Resolve.
        let class = match opts.shading {
            ShadingMode::NormalDebug | ShadingMode::DepthDebug => 2,
            _ if pbr => 0,
            _ => 1,
        };
        let (tone, exposure) = if pbr {
            (tone_map_code(opts.tone_map), opts.exposure.max(0.0))
        } else {
            (0, 1.0)
        };
        let params = ResolveParams {
            cfg: [aa, class, tone, 0],
            exposure: [exposure, 0.0, 0.0, 0.0],
            background: bg,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("resolve params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let resolve_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resolve"),
            layout: &self.resolve_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&targets.hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&targets.coverage_view),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("resolve"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.output_view,
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
        pass.set_pipeline(&self.pipelines.resolve);
        pass.set_bind_group(0, &resolve_bg, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    fn ensure_shadow_targets(&mut self, key: (u32, u32)) {
        if self.shadow_targets.as_ref().is_some_and(|t| t.key == key) {
            return;
        }
        let (size, layers) = key;
        let color = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow maps"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow depth"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let layer_views = (0..layers)
            .map(|l| {
                color.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: l,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        self.shadow_targets = Some(ShadowTargets {
            key,
            array_view: color.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            }),
            layer_views,
            depth_view: depth.create_view(&Default::default()),
        });
    }

    /// Render one linear-depth map per shadowed light. Casters: OPAQUE
    /// and MASK triangles (BLEND surfaces cast no shadow), both faces.
    fn encode_shadow_maps(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gs: &GpuScene,
        setups: &[crate::shadow::ShadowSetup],
    ) {
        let Some(vbuf) = &gs.vertices else { return };
        let mut bytes = vec![0u8; SHADOW_SLOT as usize * setups.len()];
        for (k, sh) in setups.iter().enumerate() {
            let u = ShadowPassUniform {
                view_proj: crate::shadow::column_major(&sh.view_proj),
                eye: [sh.eye[0], sh.eye[1], sh.eye[2], 1.0],
                dir: [sh.dir[0], sh.dir[1], sh.dir[2], 0.0],
            };
            let at = k * SHADOW_SLOT as usize;
            bytes[at..at + std::mem::size_of::<ShadowPassUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&u));
        }
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shadow pass"),
                contents: &bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow pass"),
            layout: &self.shadow_pass_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<ShadowPassUniform>() as u64),
                }),
            }],
        });
        let targets = self.shadow_targets.as_ref().expect("ensured");
        for k in 0..setups.len() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow map"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.layer_views[k],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f32::MAX as f64,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipelines.shadow);
            pass.set_bind_group(0, &self.empty_group, &[]);
            pass.set_bind_group(2, &group, &[(k as u64 * SHADOW_SLOT) as u32]);
            pass.set_vertex_buffer(0, vbuf.slice(..));
            for it in gs
                .items
                .iter()
                .filter(|i| i.topology == DrawTopology::Triangles)
            {
                let m = &gs.materials[it.material];
                if m.pass == Pass::Blend {
                    continue;
                }
                pass.set_bind_group(1, &m.bind_group, &[]);
                pass.draw(it.first..it.first + it.count, 0..1);
            }
        }
    }

    fn index_buffer(&self, idx: &[u32], label: &str) -> Option<(wgpu::Buffer, u32)> {
        (!idx.is_empty()).then(|| {
            let buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(idx),
                    usage: wgpu::BufferUsages::INDEX,
                });
            (buf, idx.len() as u32)
        })
    }

    /// Full offscreen render: prepare + upload + draw + readback.
    pub(crate) fn render(
        &mut self,
        scene: &oxideav_mesh3d::Scene3D,
        opts: &RenderOptions,
    ) -> Result<RgbaImage> {
        let mut gs = self.upload(scene, opts);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode_frame(&mut encoder, &mut gs, opts)?;
        let (w, h) = (opts.width.max(1), opts.height.max(1));

        // Readback buffer rows must be 256-byte aligned.
        let unpadded = w as usize * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded = unpadded.div_ceil(align) * align;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * h as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let targets = self.targets.as_ref().expect("targets ensured");
        encoder.copy_texture_to_buffer(
            targets.output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| backend_err("poll", e))?;
        rx.recv()
            .map_err(|e| backend_err("map_async", e))?
            .map_err(|e| backend_err("map_async", e))?;
        let mut pixels = Vec::with_capacity(unpadded * h as usize);
        {
            let data = slice.get_mapped_range();
            for row in data.chunks_exact(padded) {
                pixels.extend_from_slice(&row[..unpadded]);
            }
        }
        readback.unmap();
        Ok(RgbaImage {
            width: w,
            height: h,
            pixels,
            stride: unpadded,
        })
    }
}

fn shading_code(mode: ShadingMode) -> u32 {
    match mode {
        ShadingMode::Flat => 0,
        ShadingMode::Gouraud => 1,
        ShadingMode::Phong => 2,
        ShadingMode::Wireframe => 3,
        ShadingMode::NormalDebug => 4,
        ShadingMode::DepthDebug => 5,
        ShadingMode::Pbr => 6,
        // Modes added later render with the physically based path.
        _ => 6,
    }
}

fn tone_map_code(t: ToneMap) -> u32 {
    match t {
        ToneMap::Reinhard => 1,
        ToneMap::AcesFitted => 2,
        _ => 0,
    }
}

fn build_pipelines(
    device: &wgpu::Device,
    [globals, material, resolve]: [&wgpu::BindGroupLayout; 3],
    [empty, shadow_pass]: [&wgpu::BindGroupLayout; 2],
) -> Pipelines {
    let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scene.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scene.wgsl").into()),
    });
    let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resolve.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/resolve.wgsl").into()),
    });
    let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("scene"),
        bind_group_layouts: &[Some(globals), Some(material)],
        immediate_size: 0,
    });
    let over = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    let premultiplied_over = wgpu::BlendState {
        color: over,
        alpha: over,
    };
    let make = |label: &str,
                topology: wgpu::PrimitiveTopology,
                cull: Option<wgpu::Face>,
                blend: Option<wgpu::BlendState>,
                fs: &str| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &VERTEX_ATTRIBUTES,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: Some(fs),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: COVERAGE_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                topology,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: cull,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(blend.is_none()),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };
    let tri = wgpu::PrimitiveTopology::TriangleList;
    let back = Some(wgpu::Face::Back);
    let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("resolve"),
        bind_group_layouts: &[Some(resolve)],
        immediate_size: 0,
    });
    let resolve = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("resolve"),
        layout: Some(&resolve_layout),
        vertex: wgpu::VertexState {
            module: &resolve_shader,
            entry_point: Some("vs_fullscreen"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &resolve_shader,
            entry_point: Some("fs_resolve"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: COLOR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    Pipelines {
        opaque: [
            make("opaque", tri, None, None, "fs_main"),
            make("opaque-cull", tri, back, None, "fs_main"),
        ],
        blend: [
            make("blend", tri, None, Some(premultiplied_over), "fs_main"),
            make("blend-cull", tri, back, Some(premultiplied_over), "fs_main"),
        ],
        lines: make(
            "lines",
            wgpu::PrimitiveTopology::LineList,
            None,
            None,
            "fs_unlit",
        ),
        points: make(
            "points",
            wgpu::PrimitiveTopology::PointList,
            None,
            None,
            "fs_unlit",
        ),
        resolve,
        shadow: shadow_pipeline(device, &scene_shader, [empty, material, shadow_pass]),
    }
}

fn shadow_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    groups: [&wgpu::BindGroupLayout; 3],
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("shadow"),
        bind_group_layouts: &groups.map(Some),
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_shadow"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &VERTEX_ATTRIBUTES,
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_shadow"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SHADOW_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}
