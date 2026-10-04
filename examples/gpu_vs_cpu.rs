//! Render a procedural scene with both the GPU backend and the
//! scanline CPU backend and write the two frames as binary PPM (P6)
//! files for side-by-side inspection, plus a GPU `Pbr` sphere grid
//! (`gpu_pbr.ppm`: metallic rises left→right, roughness top→bottom).
//!
//! `cargo run --example gpu_vs_cpu -- <out-dir>`

use oxideav_mesh3d::{
    Material, MaterialId, Mesh, MeshId, Node, NodeId, Primitive, Scene3D, Topology,
};
use oxideav_render::{
    make_renderer, BackgroundColor, CameraSpec, RenderBackend, RenderOptions, Renderer, RgbaImage,
    ShadingMode,
};
use oxideav_render_vulkan::GpuRenderer;

/// UV sphere with smooth normals.
fn sphere(segments: u32, rings: u32) -> Primitive {
    let mut prim = Primitive::new(Topology::Triangles);
    let mut normals = Vec::new();
    for r in 0..=rings {
        let theta = std::f32::consts::PI * r as f32 / rings as f32;
        for s in 0..=segments {
            let phi = std::f32::consts::TAU * s as f32 / segments as f32;
            let n = [
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ];
            prim.positions.push(n);
            normals.push(n);
        }
    }
    let mut idx = Vec::new();
    let row = segments + 1;
    for r in 0..rings {
        for s in 0..segments {
            let a = r * row + s;
            let b = a + row;
            idx.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    prim.normals = Some(normals);
    prim.indices = Some(oxideav_mesh3d::Indices::U32(idx));
    prim.material = Some(MaterialId(0));
    prim
}

fn write_ppm(path: &std::path::Path, img: &RgbaImage) -> std::io::Result<()> {
    let mut out = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    for p in img.pixels.chunks_exact(4) {
        out.extend_from_slice(&p[..3]);
    }
    std::fs::write(path, out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".into()));
    let mut scene = Scene3D::new();
    scene.materials.push(Material {
        base_color: [0.2, 0.45, 0.9, 1.0],
        ..Material::default()
    });
    let mut mesh = Mesh::default();
    mesh.primitives.push(sphere(48, 24));
    scene.meshes.push(mesh);
    scene.nodes.push(Node {
        mesh: Some(MeshId(0)),
        ..Node::default()
    });
    scene.roots.push(NodeId(0));

    let opts = RenderOptions {
        width: 320,
        height: 240,
        background: BackgroundColor([24, 24, 28, 255]),
        shading: ShadingMode::Phong,
        camera: Some(CameraSpec {
            elevation_deg: 20.0,
            azimuth_deg: 30.0,
            distance: 1.0,
        }),
        aa: 2,
        ..RenderOptions::default()
    };
    let mut gpu = GpuRenderer::new()?;
    eprintln!("GPU: {}", gpu.adapter_summary());
    write_ppm(&dir.join("gpu.ppm"), &gpu.render(&scene, &opts)?)?;
    let mut cpu = make_renderer(RenderBackend::Scanline)?;
    write_ppm(&dir.join("cpu.ppm"), &cpu.render(&scene, &opts)?)?;

    // Pbr grid: 5 metallic steps × 4 roughness steps.
    let mut grid = Scene3D::new();
    let mut mesh = Mesh::default();
    mesh.primitives.push(sphere(48, 24));
    grid.meshes.push(mesh);
    for r in 0..4 {
        for m in 0..5 {
            grid.materials.push(Material {
                base_color: [0.9, 0.55, 0.2, 1.0],
                metallic: m as f32 / 4.0,
                roughness: 0.1 + 0.3 * r as f32,
                ..Material::default()
            });
            let mut mesh = Mesh::default();
            let mut prim = sphere(48, 24);
            prim.material = Some(MaterialId((r * 5 + m) as u32));
            mesh.primitives.push(prim);
            grid.meshes.push(mesh);
            grid.nodes.push(Node {
                mesh: Some(MeshId(grid.meshes.len() as u32 - 1)),
                transform: oxideav_mesh3d::Transform::Trs {
                    translation: [m as f32 * 2.2, -(r as f32) * 2.2, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0; 3],
                },
                ..Node::default()
            });
            grid.roots.push(NodeId(grid.nodes.len() as u32 - 1));
        }
    }
    let pbr = RenderOptions {
        width: 640,
        height: 520,
        shading: ShadingMode::Pbr,
        camera: None,
        tone_map: oxideav_render::ToneMap::AcesFitted,
        exposure: 1.5,
        ambient: 0.15,
        ..opts
    };
    write_ppm(&dir.join("gpu_pbr.ppm"), &gpu.render(&grid, &pbr)?)?;
    Ok(())
}
