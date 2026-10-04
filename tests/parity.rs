//! GPU `Pbr` vs scanline `Pbr` parity on oxideav-render's shared
//! procedural test scenes. Skips without an adapter.

use oxideav_mesh3d::{Sampler, Scene3D};
use oxideav_render::testscenes::{self, mean_abs_error, psnr};
use oxideav_render::{
    make_renderer, BackgroundColor, RenderBackend, RenderOptions, Renderer, ShadingMode,
};
use oxideav_render_vulkan::GpuRenderer;

fn opts() -> RenderOptions {
    RenderOptions {
        width: 160,
        height: 120,
        background: BackgroundColor([16, 16, 20, 255]),
        shading: ShadingMode::Pbr,
        // Every shared test scene carries its own framing camera.
        scene_camera: Some(0),
        ..RenderOptions::default()
    }
}

fn compare(name: &str, scene: &Scene3D, o: &RenderOptions, min_psnr: f64) {
    let Ok(mut gpu) = GpuRenderer::new() else {
        eprintln!("skipping GPU parity test");
        return;
    };
    let g = gpu.render(scene, o).unwrap();
    let c = make_renderer(RenderBackend::Scanline)
        .unwrap()
        .render(scene, o)
        .unwrap();
    let (p, mae) = (psnr(&g, &c), mean_abs_error(&g, &c));
    eprintln!("{name}: PSNR {p:.2} dB, MAE {mae:.3}");
    if let Ok(dir) = std::env::var("OXIDEAV_PARITY_DUMP") {
        for (tag, img) in [("gpu", &g), ("cpu", &c)] {
            let mut out = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
            for px in img.pixels.chunks_exact(4) {
                out.extend_from_slice(&px[..3]);
            }
            std::fs::write(format!("{dir}/{name}-{tag}.ppm"), out).unwrap();
        }
    }
    assert!(
        p >= min_psnr,
        "{name}: PSNR {p:.2} dB < {min_psnr} dB (MAE {mae:.3})"
    );
}

#[test]
fn cornell_box() {
    compare("cornell", &testscenes::cornell_box(), &opts(), 32.0);
}

#[test]
fn sphere_grid() {
    compare("spheres", &testscenes::sphere_grid(4, 3), &opts(), 40.0);
}

#[test]
fn checker_floor() {
    compare(
        "checker",
        &testscenes::checker_floor(Sampler::default()),
        &opts(),
        35.0,
    );
}

#[test]
fn textured_quad() {
    compare(
        "textured",
        &testscenes::textured_quad(Sampler::default()),
        &opts(),
        50.0,
    );
}

#[test]
fn alpha_planes() {
    compare("alpha", &testscenes::alpha_planes(), &opts(), 50.0);
}

#[test]
fn shadow_box() {
    let o = RenderOptions {
        shadows: true,
        ..opts()
    };
    compare("shadow", &testscenes::shadow_box(), &o, 50.0);
}

#[test]
fn skinned_morph_beam() {
    let o = RenderOptions {
        time: Some(0.5),
        ..opts()
    };
    compare("skinned", &testscenes::skinned_morph_beam(), &o, 60.0);
}

#[test]
fn normal_mapped_quad() {
    compare(
        "normalmap",
        &testscenes::normal_mapped_quad(0.4),
        &opts(),
        50.0,
    );
}

#[test]
fn shadows_darken_the_floor() {
    let Ok(mut gpu) = GpuRenderer::new() else {
        return;
    };
    let scene = testscenes::shadow_box();
    let lit = gpu.render(&scene, &opts()).unwrap();
    let shadowed = gpu
        .render(
            &scene,
            &RenderOptions {
                shadows: true,
                ..opts()
            },
        )
        .unwrap();
    let sum =
        |img: &oxideav_render::RgbaImage| -> u64 { img.pixels.iter().map(|&b| b as u64).sum() };
    assert!(
        sum(&shadowed) < sum(&lit),
        "enabling shadows must remove light somewhere"
    );
}
