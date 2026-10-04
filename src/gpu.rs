//! wgpu device, pipelines and the offscreen draw + readback path.

use bytemuck::{Pod, Zeroable};
use oxideav_render::{Error, RenderOptions, Result, RgbaImage, ShadingMode};
use wgpu::util::DeviceExt;

use crate::camera::{light_direction, view_proj};
use crate::flatten::{flatten, Vertex};
use crate::math::to_column_major;
use crate::GpuBackend;

/// Colour target format. Plain UNORM: the fragment shader performs
/// the sRGB encode itself (see `shaders/scene.wgsl`).
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Ambient term — identical to the scanline backend's.
const AMBIENT: f32 = 0.2;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    light: [f32; 4],
    params: [f32; 4],
    mode: [u32; 4],
}

/// Offscreen colour + depth targets, cached across frames of the same
/// size.
struct Targets {
    width: u32,
    height: u32,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

/// A live GPU device with the scene pipelines compiled.
pub(crate) struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    max_dim: u32,
    bind_group_layout: wgpu::BindGroupLayout,
    tri_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    targets: Option<Targets>,
}

fn backend_err(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("wgpu: {what}: {e}"))
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
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scene.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let make = |topology: wgpu::PrimitiveTopology, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x3,
                            1 => Float32x3,
                            2 => Float32x4
                        ],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    // The scanline backend draws both faces; match it.
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
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
        };
        let tri_pipeline = make(wgpu::PrimitiveTopology::TriangleList, "scene-triangles");
        let line_pipeline = make(wgpu::PrimitiveTopology::LineList, "scene-lines");
        Self {
            device,
            queue,
            adapter_info,
            max_dim,
            bind_group_layout,
            tri_pipeline,
            line_pipeline,
            targets: None,
        }
    }

    pub(crate) fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    fn ensure_targets(&mut self, width: u32, height: u32) {
        let stale = self
            .targets
            .as_ref()
            .is_none_or(|t| t.width != width || t.height != height);
        if stale {
            let size = wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };
            let tex = |format, usage, label| {
                self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
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
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                "color",
            );
            let depth = tex(
                DEPTH_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
                "depth",
            );
            self.targets = Some(Targets {
                width,
                height,
                color_view: color.create_view(&Default::default()),
                depth_view: depth.create_view(&Default::default()),
                color,
            });
        }
    }

    pub(crate) fn render(
        &mut self,
        scene: &oxideav_mesh3d::Scene3D,
        opts: &RenderOptions,
    ) -> Result<RgbaImage> {
        let width = opts.width.max(1);
        let height = opts.height.max(1);
        if width > self.max_dim || height > self.max_dim {
            return Err(Error::InvalidOptions(format!(
                "{width}x{height} exceeds the GPU's {} max texture dimension",
                self.max_dim
            )));
        }
        // Supersample like the scanline backend, shrinking the factor
        // when the enlarged target would exceed the device limit.
        let mut aa = opts.aa.clamp(1, 8);
        while aa > 1 && (width * aa > self.max_dim || height * aa > self.max_dim) {
            aa -= 1;
        }
        let (rw, rh) = (width * aa, height * aa);

        let wireframe = opts.shading == ShadingMode::Wireframe;
        let flat = flatten(scene, wireframe);
        let mode = match opts.shading {
            ShadingMode::Flat => 0,
            ShadingMode::Gouraud => 1,
            ShadingMode::Phong => 2,
            ShadingMode::Wireframe => 3,
            ShadingMode::NormalDebug => 4,
            ShadingMode::DepthDebug => 5,
            // Future modes fall back to per-fragment lighting.
            #[allow(unreachable_patterns)]
            _ => 2,
        };
        let light = light_direction(opts);
        let globals = Globals {
            view_proj: to_column_major(&view_proj(rw, rh, flat.bounds_or_unit(), opts)),
            light: [light[0], light[1], light[2], opts.light.intensity.max(0.0)],
            params: [AMBIENT, 0.0, 0.0, 0.0],
            mode: [mode, 0, 0, 0],
        };

        let device = self.device.clone();
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globals"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let vbuf = |data: &[Vertex], label| {
            (!data.is_empty()).then(|| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            })
        };
        let tri_buf = vbuf(&flat.triangles, "triangles");
        let line_buf = vbuf(&flat.lines, "lines");

        // Readback buffer rows must be 256-byte aligned.
        let unpadded = rw as usize * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded = unpadded.div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * rh as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = opts.background.0.map(|c| c as f64 / 255.0);
        let mut encoder = device.create_command_encoder(&Default::default());
        self.ensure_targets(rw, rh);
        let targets = self.targets.as_ref().expect("targets ensured above");
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg[0],
                            g: bg[1],
                            b: bg[2],
                            a: bg[3],
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
            pass.set_bind_group(0, &bind_group, &[]);
            if let Some(buf) = &tri_buf {
                pass.set_pipeline(&self.tri_pipeline);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..flat.triangles.len() as u32, 0..1);
            }
            if let Some(buf) = &line_buf {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..flat.lines.len() as u32, 0..1);
            }
        }
        encoder.copy_texture_to_buffer(
            targets.color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(rh),
                },
            },
            wgpu::Extent3d {
                width: rw,
                height: rh,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| backend_err("poll", e))?;
        rx.recv()
            .map_err(|e| backend_err("map_async", e))?
            .map_err(|e| backend_err("map_async", e))?;
        let mut pixels = Vec::with_capacity(unpadded * rh as usize);
        {
            let data = slice.get_mapped_range();
            for row in data.chunks_exact(padded) {
                pixels.extend_from_slice(&row[..unpadded]);
            }
        }
        readback.unmap();

        let full = RgbaImage {
            width: rw,
            height: rh,
            pixels,
            stride: unpadded,
        };
        Ok(if aa == 1 {
            full
        } else {
            downsample_box(&full, width, height, aa)
        })
    }
}

/// `aa × aa` box filter, the same reduction the scanline backend uses.
fn downsample_box(src: &RgbaImage, dst_w: u32, dst_h: u32, aa: u32) -> RgbaImage {
    let aa = aa as usize;
    let div = (aa * aa) as u32;
    let mut pixels = Vec::with_capacity(dst_w as usize * dst_h as usize * 4);
    for dy in 0..dst_h as usize {
        for dx in 0..dst_w as usize {
            let mut acc = [0u32; 4];
            for j in 0..aa {
                let row = (dy * aa + j) * src.stride + dx * aa * 4;
                for i in 0..aa {
                    let p = row + i * 4;
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a += src.pixels[p + c] as u32;
                    }
                }
            }
            pixels.extend(acc.map(|a| (a / div) as u8));
        }
    }
    RgbaImage {
        width: dst_w,
        height: dst_h,
        pixels,
        stride: dst_w as usize * 4,
    }
}
