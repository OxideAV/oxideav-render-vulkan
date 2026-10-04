//! Render a procedural scene with both the GPU backend and the
//! scanline CPU backend and write the two frames as binary PPM (P6)
//! files for side-by-side inspection.
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
    Ok(())
}
