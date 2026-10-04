//! GPU render tests. Every test skips (passes with a note) when the
//! machine has no usable adapter, so CI without a GPU stays green.

use oxideav_mesh3d::{Material, MaterialId, Mesh, MeshId, Node, Primitive, Scene3D, Topology};
use oxideav_render::{
    make_renderer, BackgroundColor, CameraSpec, RenderBackend, RenderOptions, Renderer, RgbaImage,
    ShadingMode,
};
use oxideav_render_vulkan::GpuRenderer;

fn gpu() -> Option<GpuRenderer> {
    match GpuRenderer::new() {
        Ok(r) => {
            eprintln!("using {}", r.adapter_summary());
            Some(r)
        }
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

/// A unit cube (12 triangles, per-face normals) with one red material.
fn cube_scene() -> Scene3D {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
    ];
    for (n, u, v) in faces {
        let corner = |a: f32, b: f32| -> [f32; 3] {
            std::array::from_fn(|i| 0.5 * n[i] + 0.5 * a * u[i] + 0.5 * b * v[i])
        };
        let quad = [
            corner(-1.0, -1.0),
            corner(1.0, -1.0),
            corner(1.0, 1.0),
            corner(-1.0, 1.0),
        ];
        for idx in [0, 1, 2, 0, 2, 3] {
            positions.push(quad[idx]);
            normals.push(n);
        }
    }
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = positions;
    prim.normals = Some(normals);
    prim.material = Some(MaterialId(0));
    scene_with(prim, [0.8, 0.1, 0.1, 1.0])
}

fn scene_with(prim: Primitive, colour: [f32; 4]) -> Scene3D {
    let mut scene = Scene3D::new();
    let material = Material {
        base_color: colour,
        ..Material::default()
    };
    scene.materials.push(material);
    let mut mesh = Mesh::default();
    mesh.primitives.push(prim);
    scene.meshes.push(mesh);
    let node = Node {
        mesh: Some(MeshId(0)),
        ..Node::default()
    };
    scene.nodes.push(node);
    scene.roots.push(oxideav_mesh3d::NodeId(0));
    scene
}

fn opts(shading: ShadingMode) -> RenderOptions {
    RenderOptions {
        width: 96,
        height: 64,
        background: BackgroundColor([255, 255, 255, 255]),
        shading,
        camera: Some(CameraSpec {
            elevation_deg: 30.0,
            azimuth_deg: 40.0,
            distance: 1.0,
        }),
        ..RenderOptions::default()
    }
}

fn coverage(img: &RgbaImage) -> Vec<bool> {
    img.pixels
        .chunks_exact(4)
        .map(|p| p != [255, 255, 255, 255])
        .collect()
}

#[test]
fn renders_requested_dimensions_and_draws_geometry() {
    let Some(mut r) = gpu() else { return };
    let img = r.render(&cube_scene(), &opts(ShadingMode::Phong)).unwrap();
    assert_eq!((img.width, img.height), (96, 64));
    assert_eq!(img.pixels.len(), 96 * 64 * 4);
    let drawn = coverage(&img).iter().filter(|&&c| c).count();
    assert!(
        drawn > 96 * 64 / 10,
        "cube should cover a good part of the frame, got {drawn}"
    );
    // Corners stay background.
    assert_eq!(img.pixel(0, 0), Some([255, 255, 255, 255]));
}

#[test]
fn coverage_matches_scanline_backend() {
    let Some(mut r) = gpu() else { return };
    let scene = cube_scene();
    for mode in [
        ShadingMode::Flat,
        ShadingMode::Phong,
        ShadingMode::DepthDebug,
    ] {
        let o = opts(mode);
        let gpu_img = r.render(&scene, &o).unwrap();
        let cpu_img = make_renderer(RenderBackend::Scanline)
            .unwrap()
            .render(&scene, &o)
            .unwrap();
        let (a, b) = (coverage(&gpu_img), coverage(&cpu_img));
        let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        // Only silhouette-edge pixels may disagree (rasterisation
        // rules differ slightly at edges).
        assert!(
            differ * 100 < a.len() * 3,
            "{mode:?}: {differ} of {} pixels differ in coverage",
            a.len()
        );
    }
}

#[test]
fn flat_colour_matches_scanline_exactly_in_interior() {
    let Some(mut r) = gpu() else { return };
    let scene = cube_scene();
    let o = opts(ShadingMode::Flat);
    let gpu_img = r.render(&scene, &o).unwrap();
    let cpu_img = make_renderer(RenderBackend::Scanline)
        .unwrap()
        .render(&scene, &o)
        .unwrap();
    let (cx, cy) = (48, 32);
    assert_eq!(gpu_img.pixel(cx, cy), cpu_img.pixel(cx, cy));
}

#[test]
fn phong_shading_is_close_to_scanline() {
    let Some(mut r) = gpu() else { return };
    let scene = cube_scene();
    let o = opts(ShadingMode::Phong);
    let g = r.render(&scene, &o).unwrap();
    let c = make_renderer(RenderBackend::Scanline)
        .unwrap()
        .render(&scene, &o)
        .unwrap();
    let mut worst = 0i32;
    let mut sum = 0u64;
    let mut n = 0u64;
    for (pg, pc) in g.pixels.chunks_exact(4).zip(c.pixels.chunks_exact(4)) {
        let bg = [255, 255, 255, 255];
        if pg == bg || pc == bg {
            continue;
        }
        for k in 0..3 {
            let d = (pg[k] as i32 - pc[k] as i32).abs();
            worst = worst.max(d);
            sum += d as u64;
            n += 1;
        }
    }
    let mean = sum as f64 / n.max(1) as f64;
    assert!(mean < 2.0, "mean channel error {mean} (worst {worst})");
}

#[test]
fn wireframe_draws_lines_only() {
    let Some(mut r) = gpu() else { return };
    let img = r
        .render(&cube_scene(), &opts(ShadingMode::Wireframe))
        .unwrap();
    let drawn = coverage(&img).iter().filter(|&&c| c).count();
    assert!(drawn > 50, "wireframe should draw edges");
    assert!(
        drawn < 96 * 64 / 4,
        "wireframe must not fill faces ({drawn})"
    );
}

#[test]
fn empty_scene_is_background() {
    let Some(mut r) = gpu() else { return };
    let img = r
        .render(&Scene3D::new(), &opts(ShadingMode::Phong))
        .unwrap();
    assert!(img
        .pixels
        .chunks_exact(4)
        .all(|p| p == [255, 255, 255, 255]));
}

#[test]
fn supersampling_keeps_output_size() {
    let Some(mut r) = gpu() else { return };
    let o = RenderOptions {
        aa: 4,
        ..opts(ShadingMode::Phong)
    };
    let img = r.render(&cube_scene(), &o).unwrap();
    assert_eq!((img.width, img.height), (96, 64));
}

#[test]
fn registry_constructs_gpu_backend() {
    let mut reg = oxideav_render::RenderRegistry::new();
    oxideav_render_vulkan::register_into(&mut reg);
    assert!(reg.names().contains(&oxideav_render_vulkan::BACKEND_NAME));
}

#[test]
fn upload_once_draw_many() {
    let Some(mut r) = gpu() else { return };
    let mut gs = r.upload(&cube_scene());
    assert_eq!(gs.triangle_count(), 12);
    for mode in [
        ShadingMode::Phong,
        ShadingMode::Wireframe,
        ShadingMode::Flat,
    ] {
        let tex = r.draw(&mut gs, &opts(mode)).unwrap();
        assert_eq!((tex.width(), tex.height()), (96, 64));
        assert_eq!(tex.format(), oxideav_render_vulkan::COLOR_FORMAT);
    }
}
