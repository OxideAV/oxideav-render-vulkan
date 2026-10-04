//! Upload of an `oxideav_render::PreparedScene` into GPU resources:
//! one interleaved world-space vertex buffer, per-item draw ranges,
//! material uniform buffers + bind groups, and textures (deduplicated
//! per decoded image and colour space).

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use oxideav_mesh3d::{AlphaMode, MagFilter, MinFilter, Sampler, WrapMode};
use oxideav_render::prepare::{DrawTopology, PreparedLight, PreparedMaterial, TextureBinding};
use oxideav_render::texture::{ColorSpace, TextureData};
use oxideav_render::PreparedScene;
use wgpu::util::DeviceExt;

/// One GPU vertex (72 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct Vertex {
    pub(crate) position: [f32; 3],
    pub(crate) normal: [f32; 3],
    pub(crate) tangent: [f32; 4],
    pub(crate) uv0: [f32; 2],
    pub(crate) uv1: [f32; 2],
    pub(crate) color: [f32; 4],
}

pub(crate) const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
    0 => Float32x3,
    1 => Float32x3,
    2 => Float32x4,
    3 => Float32x2,
    4 => Float32x2,
    5 => Float32x4
];

/// Material uniform block — layout matches `Material` in scene.wgsl.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MaterialUniform {
    base_color: [f32; 4],
    emissive: [f32; 4],
    factors: [f32; 4],
    extra: [f32; 4],
    flags: [u32; 4],
    uv_xform: [[f32; 4]; 10],
}

/// How an item's triangles composite in `Pbr` mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pass {
    Opaque,
    Blend,
}

/// A contiguous vertex range drawn with one material.
#[derive(Debug, Clone)]
pub(crate) struct ItemDraw {
    pub(crate) topology: DrawTopology,
    pub(crate) first: u32,
    pub(crate) count: u32,
    pub(crate) material: usize,
}

/// A BLEND triangle, kept CPU-side for per-frame back-to-front sorting.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlendTri {
    pub(crate) centroid: [f32; 3],
    pub(crate) first: u32,
    pub(crate) material: usize,
}

/// Per-material GPU state.
pub(crate) struct GpuMaterial {
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) pass: Pass,
    pub(crate) double_sided: bool,
}

/// A prepared scene resident on the GPU.
pub(crate) struct GpuScene {
    pub(crate) vertices: Option<wgpu::Buffer>,
    pub(crate) items: Vec<ItemDraw>,
    pub(crate) blend_tris: Vec<BlendTri>,
    pub(crate) materials: Vec<GpuMaterial>,
    /// Scene lights (world space); empty when the scene has none.
    pub(crate) scene_lights: Vec<PreparedLight>,
    /// Kept for camera resolution (bounds + scene cameras); the bulky
    /// per-item vertex arrays are dropped after upload.
    pub(crate) prepared: PreparedScene,
    /// Lazily built index buffer of triangle edges (legacy Wireframe).
    pub(crate) edges: Option<Option<(wgpu::Buffer, u32)>>,
    pub(crate) triangle_count: usize,
}

/// Shared texture / sampler caches and fallbacks, owned by the context
/// so repeated uploads of the same assets reuse GPU memory.
pub(crate) struct ResourceCache {
    textures: HashMap<(usize, bool), (Arc<TextureData>, wgpu::TextureView)>,
    samplers: HashMap<[u8; 4], wgpu::Sampler>,
    white: wgpu::TextureView,
}

impl ResourceCache {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let white = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("white"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255, 255, 255, 255],
        );
        Self {
            textures: HashMap::new(),
            samplers: HashMap::new(),
            white: white.create_view(&Default::default()),
        }
    }

    /// GPU view of `data` decoded in `space`, uploaded once as an
    /// `Rgba16Float` texture with the full mip chain.
    fn texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &Arc<TextureData>,
        space: ColorSpace,
    ) -> wgpu::TextureView {
        let key = (Arc::as_ptr(data) as usize, space == ColorSpace::Srgb);
        if let Some((_, view)) = self.textures.get(&key) {
            return view.clone();
        }
        let max_dim = device.limits().max_texture_dimension_2d;
        let mips: Vec<_> = data
            .mips(space)
            .iter()
            .skip_while(|m| m.width > max_dim || m.height > max_dim)
            .collect();
        let Some(level0) = mips.first() else {
            return self.white.clone();
        };
        let mut bytes = Vec::new();
        for level in &mips {
            for t in &level.texels {
                for c in t {
                    bytes.extend_from_slice(&f32_to_f16(*c).to_le_bytes());
                }
            }
        }
        let tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("material texture"),
                size: wgpu::Extent3d {
                    width: level0.width,
                    height: level0.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mips.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::MipMajor,
            &bytes,
        );
        let view = tex.create_view(&Default::default());
        // Holding the Arc keeps the pointer key from being recycled.
        self.textures.insert(key, (data.clone(), view.clone()));
        view
    }

    fn sampler(&mut self, device: &wgpu::Device, s: &Sampler) -> wgpu::Sampler {
        let wrap = |w: WrapMode| match w {
            WrapMode::ClampToEdge => 0u8,
            WrapMode::MirroredRepeat => 1,
            _ => 2,
        };
        let mag = match s.mag_filter {
            Some(MagFilter::Nearest) => 0u8,
            _ => 1,
        };
        // (min filter, mip filter); undefined → trilinear.
        let min = match s.min_filter {
            Some(MinFilter::Nearest) | Some(MinFilter::NearestMipNearest) => 0u8,
            Some(MinFilter::Linear) | Some(MinFilter::LinearMipNearest) => 1,
            Some(MinFilter::NearestMipLinear) => 2,
            _ => 3,
        };
        let key = [wrap(s.wrap_s), wrap(s.wrap_t), mag, min];
        self.samplers
            .entry(key)
            .or_insert_with(|| {
                let address = |w: u8| match w {
                    0 => wgpu::AddressMode::ClampToEdge,
                    1 => wgpu::AddressMode::MirrorRepeat,
                    _ => wgpu::AddressMode::Repeat,
                };
                let filter = |linear: bool| {
                    if linear {
                        wgpu::FilterMode::Linear
                    } else {
                        wgpu::FilterMode::Nearest
                    }
                };
                let mip = |linear: bool| {
                    if linear {
                        wgpu::MipmapFilterMode::Linear
                    } else {
                        wgpu::MipmapFilterMode::Nearest
                    }
                };
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("material sampler"),
                    address_mode_u: address(key[0]),
                    address_mode_v: address(key[1]),
                    address_mode_w: wgpu::AddressMode::Repeat,
                    mag_filter: filter(key[2] == 1),
                    min_filter: filter(key[3] == 1 || key[3] == 3),
                    mipmap_filter: mip(key[3] >= 2),
                    ..Default::default()
                })
            })
            .clone()
    }
}

/// IEEE 754 binary32 → binary16 bits, round-to-nearest-even, with
/// overflow to infinity, subnormals, and NaN preserved.
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let round = rem > mid || (rem == mid && (half & 1) == 1);
        return sign | (half + round as u32) as u16;
    }
    let half = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    let round = rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1);
    sign | (half + round as u32) as u16
}

pub(crate) struct Uploader<'a> {
    pub(crate) device: &'a wgpu::Device,
    pub(crate) queue: &'a wgpu::Queue,
    pub(crate) material_layout: &'a wgpu::BindGroupLayout,
    pub(crate) cache: &'a mut ResourceCache,
}

impl Uploader<'_> {
    pub(crate) fn upload(&mut self, mut prepared: PreparedScene) -> GpuScene {
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut items = Vec::new();
        let mut blend_tris = Vec::new();
        let materials: Vec<GpuMaterial> = prepared
            .materials
            .iter()
            .map(|m| self.material(m, &prepared))
            .collect();
        let mut triangle_count = 0;

        for item in &mut prepared.items {
            let n = item.positions.len();
            let first = vertices.len() as u32;
            let get = |v: &[[f32; 2]], i: usize| v.get(i).copied().unwrap_or([0.0; 2]);
            let uv0 = item.uvs.first().map(Vec::as_slice).unwrap_or(&[]);
            let uv1 = item.uvs.get(1).map(Vec::as_slice).unwrap_or(&[]);
            for i in 0..n {
                vertices.push(Vertex {
                    position: item.positions[i],
                    normal: item.normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]),
                    tangent: item.tangents.get(i).copied().unwrap_or([0.0; 4]),
                    uv0: get(uv0, i),
                    uv1: get(uv1, i),
                    color: item.colors.get(i).copied().unwrap_or([1.0; 4]),
                });
            }
            let material = item.material.min(materials.len().saturating_sub(1));
            let mut count = n;
            if item.topology == DrawTopology::Triangles {
                count -= n % 3;
                triangle_count += count / 3;
                if materials.get(material).map(|m| m.pass) == Some(Pass::Blend) {
                    for t in 0..count / 3 {
                        let p = &item.positions[t * 3..t * 3 + 3];
                        blend_tris.push(BlendTri {
                            centroid: std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0),
                            first: first + (t * 3) as u32,
                            material,
                        });
                    }
                }
            }
            items.push(ItemDraw {
                topology: item.topology,
                first,
                count: count as u32,
                material,
            });
            // Drop the bulky arrays; bounds / cameras stay for framing.
            item.positions = Vec::new();
            item.normals = Vec::new();
            item.tangents = Vec::new();
            item.uvs = Vec::new();
            item.colors = Vec::new();
        }

        let vertices = (!vertices.is_empty()).then(|| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene vertices"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
        });
        let scene_lights = if prepared.scene_lights {
            prepared.lights.clone()
        } else {
            Vec::new()
        };
        GpuScene {
            vertices,
            items,
            blend_tris,
            materials,
            scene_lights,
            prepared,
            edges: None,
            triangle_count,
        }
    }

    fn material(&mut self, m: &PreparedMaterial, prepared: &PreparedScene) -> GpuMaterial {
        let slots: [(Option<TextureBinding>, ColorSpace); 5] = [
            (m.base_color_texture, ColorSpace::Srgb),
            (m.metallic_roughness_texture, ColorSpace::Linear),
            (m.normal_texture, ColorSpace::Linear),
            (m.occlusion_texture, ColorSpace::Linear),
            (m.emissive_texture, ColorSpace::Srgb),
        ];
        let mut present = 0u32;
        let mut uv_bits = 0u32;
        let mut uv_xform = [[0.0f32; 4]; 10];
        let mut views = Vec::with_capacity(5);
        let mut samplers = Vec::with_capacity(5);
        for (slot, (binding, space)) in slots.iter().enumerate() {
            let tex = binding.and_then(|b| {
                let t = prepared.textures.get(b.texture)?.as_ref()?;
                Some((b, t))
            });
            let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
            match tex {
                Some((b, t)) => {
                    present |= 1 << slot;
                    if b.uv_set == 1 {
                        uv_bits |= 1 << slot;
                    }
                    let mx = b.transform.map(|x| x.to_matrix());
                    let rows = mx.map_or(identity, |mx| [mx[0], mx[1]]);
                    uv_xform[slot * 2] = [rows[0][0], rows[0][1], rows[0][2], 0.0];
                    uv_xform[slot * 2 + 1] = [rows[1][0], rows[1][1], rows[1][2], 0.0];
                    views.push(self.cache.texture(self.device, self.queue, &t.data, *space));
                    samplers.push(self.cache.sampler(self.device, &t.sampler));
                }
                None => {
                    uv_xform[slot * 2] = [1.0, 0.0, 0.0, 0.0];
                    uv_xform[slot * 2 + 1] = [0.0, 1.0, 0.0, 0.0];
                    views.push(self.cache.white.clone());
                    samplers.push(self.cache.sampler(self.device, &Sampler::default()));
                }
            }
        }
        let (alpha_mode, cutoff, pass) = match m.alpha_mode {
            AlphaMode::Mask { cutoff } => (1u32, cutoff, Pass::Opaque),
            AlphaMode::Blend => (2, 0.0, Pass::Blend),
            _ => (0, 0.0, Pass::Opaque),
        };
        let uniform = MaterialUniform {
            base_color: m.base_color,
            emissive: [m.emissive[0], m.emissive[1], m.emissive[2], cutoff],
            factors: [
                m.metallic,
                m.roughness,
                m.normal_scale,
                m.occlusion_strength,
            ],
            extra: [m.dielectric_f0(), 0.0, 0.0, 0.0],
            flags: [present, alpha_mode, m.unlit as u32, uv_bits],
            uv_xform,
        };
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }];
        for (i, (view, sampler)) in views.iter().zip(&samplers).enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 1 + 2 * i as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 2 + 2 * i as u32,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material"),
            layout: self.material_layout,
            entries: &entries,
        });
        GpuMaterial {
            bind_group,
            pass,
            double_sided: m.double_sided,
        }
    }
}

/// Bind-group layout for `@group(1)` in scene.wgsl.
pub(crate) fn material_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    for i in 0..5u32 {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 1 + 2 * i,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2 + 2 * i,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("material"),
        entries: &entries,
    })
}

#[cfg(test)]
mod tests {
    use super::f32_to_f16;

    #[test]
    fn f16_conversion_known_values() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(1.0e6), 0x7c00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NAN) & 0x7e00, 0x7e00);
        // Smallest subnormal 2^-24.
        assert_eq!(f32_to_f16(5.960_464_5e-8), 0x0001);
        // 1/3 rounds to nearest.
        assert_eq!(f32_to_f16(1.0 / 3.0), 0x3555);
    }
}
