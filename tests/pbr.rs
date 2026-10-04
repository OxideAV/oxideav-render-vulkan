//! Physically based (`ShadingMode::Pbr`) GPU tests. Each skips when no
//! adapter is available.

mod common;

use std::sync::Arc;

use oxideav_mesh3d::{
    AlphaMode, ImageData, InMemoryAsset, MagFilter, Material, MaterialId, Mesh, MeshId, MinFilter,
    Node, NodeId, Primitive, Sampler, Scene3D, Texture, TextureId, TextureRef, Topology,
};
use oxideav_render::texture::{encode_raw_rgba8, RAW_RGBA8_MIME};
use oxideav_render::{BackgroundColor, RenderOptions, Renderer, RgbaImage, ShadingMode, ToneMap};
use oxideav_render_vulkan::GpuRenderer;

const BG: [u8; 4] = [10, 20, 30, 255];

fn gpu() -> Option<GpuRenderer> {
    if !common::gpu_tests_enabled() {
        return None;
    }
    GpuRenderer::new()
        .map_err(|e| eprintln!("skipping GPU test: {e}"))
        .ok()
}

/// Axis-aligned unit quad in the plane `z`, facing +Z, UVs spanning
/// `[0, 1]` (v down, glTF convention).
fn quad(z: f32, half: f32, material: u32) -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    let c = [
        [-half, -half, z],
        [half, -half, z],
        [half, half, z],
        [-half, half, z],
    ];
    let uv = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let order = [0, 1, 2, 0, 2, 3];
    p.positions = order.iter().map(|&i| c[i]).collect();
    p.uvs = vec![order.iter().map(|&i| uv[i]).collect()];
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 6]);
    p.material = Some(MaterialId(material));
    p
}

fn scene(prims: Vec<Primitive>, materials: Vec<Material>) -> Scene3D {
    let mut s = Scene3D::new();
    s.materials = materials;
    let mut mesh = Mesh::default();
    mesh.primitives = prims;
    s.meshes.push(mesh);
    s.nodes.push(Node {
        mesh: Some(MeshId(0)),
        ..Node::default()
    });
    s.roots.push(NodeId(0));
    s
}

fn opts() -> RenderOptions {
    RenderOptions {
        width: 64,
        height: 64,
        background: BackgroundColor(BG),
        shading: ShadingMode::Pbr,
        ..RenderOptions::default()
    }
}

fn centre(img: &RgbaImage) -> [u8; 4] {
    img.pixel(img.width / 2, img.height / 2).unwrap()
}

fn mean_luma(img: &RgbaImage) -> f64 {
    let px: Vec<_> = img.pixels.chunks_exact(4).filter(|p| p != &BG).collect();
    px.iter()
        .map(|p| p[0] as f64 + p[1] as f64 + p[2] as f64)
        .sum::<f64>()
        / px.len().max(1) as f64
}

#[test]
fn pbr_lit_quad_draws_and_exposure_brightens() {
    let Some(mut r) = gpu() else { return };
    let mat = Material {
        base_color: [0.5, 0.5, 0.5, 1.0],
        metallic: 0.0,
        roughness: 0.8,
        ..Material::default()
    };
    let s = scene(vec![quad(0.0, 0.5, 0)], vec![mat]);
    let dim = r.render(&s, &opts()).unwrap();
    assert_ne!(centre(&dim), BG, "quad must cover the centre");
    let bright = r
        .render(
            &s,
            &RenderOptions {
                exposure: 2.0,
                ..opts()
            },
        )
        .unwrap();
    assert!(
        mean_luma(&bright) > mean_luma(&dim) * 1.2,
        "exposure 2 should brighten: {} vs {}",
        mean_luma(&bright),
        mean_luma(&dim)
    );
    // Uncovered corners keep the background bytes exactly, whatever
    // the tone map.
    let aces = r
        .render(
            &s,
            &RenderOptions {
                tone_map: ToneMap::AcesFitted,
                ..opts()
            },
        )
        .unwrap();
    assert_eq!(aces.pixel(0, 0), Some(BG));
}

#[test]
fn unlit_material_outputs_base_colour() {
    let Some(mut r) = gpu() else { return };
    let mut mat = Material {
        base_color: [1.0, 0.0, 0.0, 1.0],
        ..Material::default()
    };
    mat.ext.unlit = true;
    let s = scene(vec![quad(0.0, 0.5, 0)], vec![mat]);
    assert_eq!(centre(&r.render(&s, &opts()).unwrap()), [255, 0, 0, 255]);
}

#[test]
fn mask_below_cutoff_is_discarded() {
    let Some(mut r) = gpu() else { return };
    let mat = Material {
        base_color: [1.0, 1.0, 1.0, 0.3],
        alpha_mode: AlphaMode::Mask { cutoff: 0.5 },
        ..Material::default()
    };
    let s = scene(vec![quad(0.0, 0.5, 0)], vec![mat]);
    let img = r.render(&s, &opts()).unwrap();
    assert!(img.pixels.chunks_exact(4).all(|p| p == BG));
}

#[test]
fn blend_composites_over_opaque() {
    let Some(mut r) = gpu() else { return };
    let mut red = Material {
        base_color: [1.0, 0.0, 0.0, 0.5],
        alpha_mode: AlphaMode::Blend,
        ..Material::default()
    };
    red.ext.unlit = true;
    let mut blue = Material {
        base_color: [0.0, 0.0, 1.0, 1.0],
        ..Material::default()
    };
    blue.ext.unlit = true;
    // Blend quad in front (+Z, towards the default camera) of the
    // opaque one; listed first so ordering is up to the renderer.
    let s = scene(
        vec![quad(0.25, 0.5, 0), quad(-0.25, 0.5, 1)],
        vec![red, blue],
    );
    let c = centre(&r.render(&s, &opts()).unwrap());
    // 50/50 linear mix of red and blue → both ≈ sRGB(0.5) ≈ 188.
    assert!((180..=196).contains(&c[0]), "red {c:?}");
    assert!((180..=196).contains(&c[2]), "blue {c:?}");
    assert!(c[1] < 10, "green {c:?}");
}

#[test]
fn base_colour_texture_is_sampled() {
    let Some(mut r) = gpu() else { return };
    // 2×1 texture: left red, right green; nearest filtering.
    let bytes = encode_raw_rgba8(2, 1, &[255, 0, 0, 255, 0, 255, 0, 255]);
    let mut mat = Material {
        base_color_texture: Some(TextureRef {
            texture: TextureId(0),
            uv_set: 0,
            transform: None,
        }),
        ..Material::default()
    };
    mat.ext.unlit = true;
    let mut s = scene(vec![quad(0.0, 0.5, 0)], vec![mat]);
    s.textures.push(Texture {
        name: None,
        image: ImageData::Source(Arc::new(InMemoryAsset::new(
            Some(RAW_RGBA8_MIME.to_string()),
            bytes,
        ))),
        sampler: Sampler {
            mag_filter: Some(MagFilter::Nearest),
            min_filter: Some(MinFilter::Nearest),
            ..Sampler::default()
        },
    });
    let img = r.render(&s, &opts()).unwrap();
    let left = img.pixel(img.width / 2 - 8, img.height / 2).unwrap();
    let right = img.pixel(img.width / 2 + 8, img.height / 2).unwrap();
    assert_eq!(left, [255, 0, 0, 255], "left half samples red");
    assert_eq!(right, [0, 255, 0, 255], "right half samples green");
}
