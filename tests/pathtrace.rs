//! GPU path tracer vs oxideav-render's CPU path tracer. Both follow the
//! same normative estimator (pathtrace §1–5) with the same sequences,
//! so they must agree statistically — in practice to within float
//! rounding. Every test skips without an adapter.

use std::sync::Arc;

use oxideav_mesh3d::{
    Clearcoat, Light, Material, Mesh, Node, Primitive, Sampler, Scene3D, Sheen, Specular,
    Transform, Transmission, Volume,
};
use oxideav_render::pathtrace::EnvironmentMap;
use oxideav_render::testscenes::{self, add_camera, add_light, cuboid, diffuse_material, quad};
use oxideav_render::{
    BackgroundColor, HdrImage, LightSpec, LightStrategy, PathTraceOptions, PathTraceRenderer,
    RenderOptions, Renderer, RgbaImage, ToneMap,
};
use oxideav_render_vulkan::{GpuMode, GpuPathTracer, GpuRenderer};

fn gpu() -> Option<GpuPathTracer> {
    match GpuPathTracer::new() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping GPU path-tracer test: {e}");
            None
        }
    }
}

fn opts(w: u32, h: u32, spp: u32, bounces: u32) -> RenderOptions {
    RenderOptions {
        width: w,
        height: h,
        scene_camera: Some(0),
        background: BackgroundColor([16, 16, 20, 255]),
        path_trace: PathTraceOptions {
            samples_per_pixel: spp,
            max_bounces: bounces,
            ..PathTraceOptions::default()
        },
        ..RenderOptions::default()
    }
}

fn mean_rgb(img: &HdrImage, x0: u32, y0: u32, x1: u32, y1: u32) -> ([f64; 3], f64) {
    let mut s = [0.0f64; 3];
    let mut s2 = 0.0f64;
    let mut n = 0.0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = ((y * img.width + x) * 4) as usize;
            let mut l = 0.0;
            for (sk, &v) in s.iter_mut().zip(&img.pixels[i..i + 3]) {
                *sk += v as f64;
                l += v as f64 / 3.0;
            }
            s2 += l * l;
            n += 1.0;
        }
    }
    let m = s.map(|v| v / n);
    let lm = (m[0] + m[1] + m[2]) / 3.0;
    let sd = (s2 / n - lm * lm).max(0.0).sqrt();
    (m, sd / n.sqrt())
}

fn dump(name: &str, imgs: &[(&str, &RgbaImage)]) {
    if let Ok(dir) = std::env::var("OXIDEAV_PARITY_DUMP") {
        for (tag, img) in imgs {
            let mut out = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
            for px in img.pixels.chunks_exact(4) {
                out.extend_from_slice(&px[..3]);
            }
            std::fs::write(format!("{dir}/pt-{name}-{tag}.ppm"), out).unwrap();
        }
    }
}

/// Statistical parity: image mean and 4×4 region means within the
/// region's noise (4 standard errors of the pixel spread, which also
/// counts real signal variation, so it is conservative) plus 1 %, and
/// an LDR PSNR floor.
fn parity(
    name: &str,
    scene: &Scene3D,
    o: &RenderOptions,
    env: Option<Arc<EnvironmentMap>>,
    min_psnr: f64,
) {
    let Some(mut g) = gpu() else { return };
    let mut cpu = PathTraceRenderer::new();
    if env.is_some() {
        g.set_environment(env.clone());
        cpu.tracer_mut().set_environment(env);
    }
    let gh = g.render_hdr(scene, o).unwrap();
    let gi = g.image().unwrap();
    let ch = cpu.render_hdr(scene, o).unwrap();
    let ci = cpu.tracer_mut().image();
    let (w, h) = (o.width, o.height);
    let (gm, _) = mean_rgb(&gh, 0, 0, w, h);
    let (cm, cse) = mean_rgb(&ch, 0, 0, w, h);
    let p = testscenes::psnr(&gi, &ci);
    eprintln!("{name}: mean gpu {gm:.5?} cpu {cm:.5?}, PSNR {p:.2} dB");
    dump(name, &[("gpu", &gi), ("cpu", &ci)]);
    for k in 0..3 {
        let tol = 4.0 * cse + 0.01 * cm[k].abs() + 1e-4;
        assert!(
            (gm[k] - cm[k]).abs() <= tol,
            "{name} ch{k}: {} vs {} (tol {tol})",
            gm[k],
            cm[k]
        );
    }
    let mut worst = 0.0f64;
    for ry in 0..4 {
        for rx in 0..4 {
            let (x0, x1) = (rx * w / 4, (rx + 1) * w / 4);
            let (y0, y1) = (ry * h / 4, (ry + 1) * h / 4);
            let (a, _) = mean_rgb(&gh, x0, y0, x1, y1);
            let (b, se) = mean_rgb(&ch, x0, y0, x1, y1);
            for k in 0..3 {
                let d = (a[k] - b[k]).abs();
                let tol = 4.0 * se + 0.01 * b[k].abs() + 1e-3;
                worst = worst.max(d / tol);
                assert!(
                    d <= tol,
                    "{name} region ({rx},{ry}) ch{k}: {} vs {} (tol {tol})",
                    a[k],
                    b[k]
                );
            }
        }
    }
    eprintln!("{name}: worst region deviation {:.3} of tolerance", worst);
    assert!(p >= min_psnr, "{name}: PSNR {p:.2} dB < {min_psnr} dB");
}

// ---------------------------------------------------------------------
// Scenes.
// ---------------------------------------------------------------------

fn add_prim(scene: &mut Scene3D, prim: Primitive) {
    let m = scene.add_mesh(Mesh::new(None).with_primitive(prim));
    let n = scene.add_node(
        Node::new()
            .with_mesh(m)
            .with_transform(Transform::identity()),
    );
    scene.add_root(n);
}

fn lambert(rgb: [f32; 3]) -> Material {
    let mut m = diffuse_material(rgb, 1.0);
    m.ext.specular = Some(Specular {
        factor: 0.0,
        ..Specular::default()
    });
    m
}

/// Closed box of double-sided emissive Lambertian walls (albedo ρ,
/// emission Le) seen from inside: L = Le / (1 − ρ) everywhere.
fn furnace_scene(rho: f32, le: f32) -> Scene3D {
    let mut scene = Scene3D::new();
    let mut m = lambert([rho; 3]);
    m.emissive_factor = [le; 3];
    m.double_sided = true;
    let mat = scene.add_material(m);
    let mut b = cuboid([-1.0; 3], [1.0; 3]);
    b.material = Some(mat);
    add_prim(&mut scene, b);
    add_camera(&mut scene, [0.2, 0.1, 0.4], [0.0, -0.3, -1.0], 1.2);
    scene
}

/// Floor + four spheres exercising the extension lobes (volume glass,
/// thin transmission, clearcoat, sheen) under an emissive quad and a
/// point light.
fn materials_scene() -> Scene3D {
    let mut scene = Scene3D::new();
    let floor_mat = scene.add_material(diffuse_material([0.7, 0.7, 0.7], 0.8));
    let mut floor = quad(
        [
            [-3.0, 0.0, 3.0],
            [3.0, 0.0, 3.0],
            [3.0, 0.0, -3.0],
            [-3.0, 0.0, -3.0],
        ],
        [1.0, 1.0],
    );
    floor.material = Some(floor_mat);
    add_prim(&mut scene, floor);
    let mut lm = lambert([0.0; 3]);
    lm.emissive_factor = [6.0, 5.5, 5.0];
    let light_mat = scene.add_material(lm);
    let mut light = quad(
        [
            [-0.8, 2.5, -0.8],
            [0.8, 2.5, -0.8],
            [0.8, 2.5, 0.8],
            [-0.8, 2.5, 0.8],
        ],
        [1.0, 1.0],
    );
    light.material = Some(light_mat);
    add_prim(&mut scene, light);

    let mut glass = diffuse_material([0.95, 0.95, 1.0], 0.1);
    glass.ext.transmission = Some(Transmission {
        factor: 1.0,
        factor_texture: None,
    });
    glass.ext.volume = Some(Volume {
        thickness: 1.0,
        thickness_texture: None,
        attenuation_distance: Some(1.5),
        attenuation_color: [0.6, 0.8, 0.9],
    });
    let mut thin = diffuse_material([0.9, 0.7, 0.5], 0.3);
    thin.ext.transmission = Some(Transmission {
        factor: 0.8,
        factor_texture: None,
    });
    let mut cc = diffuse_material([0.6, 0.1, 0.1], 0.6);
    cc.ext.clearcoat = Some(Clearcoat {
        factor: 1.0,
        roughness: 0.15,
        ..Clearcoat::default()
    });
    let mut sh = diffuse_material([0.1, 0.1, 0.3], 0.9);
    sh.ext.sheen = Some(Sheen {
        color_factor: [0.9, 0.8, 0.6],
        roughness: 0.5,
        ..Sheen::default()
    });
    let mut metal = diffuse_material([0.95, 0.8, 0.5], 0.35);
    metal.metallic = 1.0;
    for (i, m) in [glass, thin, cc, sh, metal].into_iter().enumerate() {
        let mid = scene.add_material(m);
        let mut s = testscenes::uv_sphere(0.45, 24, 12);
        s.material = Some(mid);
        let x = -1.6 + 0.8 * i as f32;
        let mesh = scene.add_mesh(Mesh::new(None).with_primitive(s));
        let n = scene.add_node(Node::new().with_mesh(mesh).with_transform(Transform::Trs {
            translation: [x, 0.45, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }));
        scene.add_root(n);
    }
    add_light(
        &mut scene,
        Light::Point {
            color: [1.0, 0.9, 0.8],
            intensity: 6.0,
            range: None,
        },
        [2.0, 2.0, 2.0],
        [0.0, -1.0, 0.0],
    );
    add_camera(&mut scene, [0.0, 1.6, 4.2], [0.0, 0.4, 0.0], 0.8);
    scene
}

/// Small equirectangular sky: blue gradient with a bright sun.
fn sky() -> Arc<EnvironmentMap> {
    let (w, h) = (64u32, 32u32);
    let mut img = HdrImage::filled(w, h, [0.0, 0.0, 0.0, 1.0]);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let t = y as f32 / h as f32;
            img.pixels[i..i + 3].copy_from_slice(&[0.3 + 0.4 * t, 0.5 + 0.3 * t, 0.9]);
        }
    }
    for y in 6..8 {
        for x in 40..43 {
            let i = ((y * w + x) * 4) as usize;
            img.pixels[i..i + 3].copy_from_slice(&[80.0, 70.0, 55.0]);
        }
    }
    Arc::new(EnvironmentMap::new(Arc::new(img), 1.0))
}

// ---------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------

#[test]
fn white_furnace() {
    let Some(mut g) = gpu() else { return };
    let scene = furnace_scene(0.5, 1.0);
    for strategy in [
        LightStrategy::Mis,
        LightStrategy::LightOnly,
        LightStrategy::BsdfOnly,
    ] {
        let mut o = opts(8, 8, 64, 64);
        o.ambient = 0.0;
        o.light = LightSpec {
            intensity: 0.0,
            ..LightSpec::default()
        };
        o.path_trace.strategy = strategy;
        let img = g.render_hdr(&scene, &o).unwrap();
        let (m, _) = mean_rgb(&img, 0, 0, 8, 8);
        for c in m {
            assert!((c - 2.0).abs() < 0.04, "{strategy:?}: {m:?} (expected 2.0)");
        }
    }
}

#[test]
fn parity_cornell_box() {
    parity(
        "cornell",
        &testscenes::cornell_box(),
        &opts(96, 96, 64, 8),
        None,
        40.0,
    );
}

#[test]
fn parity_sphere_grid() {
    parity(
        "spheres",
        &testscenes::sphere_grid(4, 3),
        &opts(128, 96, 64, 8),
        None,
        40.0,
    );
}

#[test]
fn parity_shadow_box() {
    parity(
        "shadowbox",
        &testscenes::shadow_box(),
        &opts(96, 96, 64, 8),
        None,
        40.0,
    );
}

#[test]
fn parity_alpha_planes() {
    parity(
        "alpha",
        &testscenes::alpha_planes(),
        &opts(96, 96, 64, 8),
        None,
        40.0,
    );
}

#[test]
fn parity_textured_quad() {
    parity(
        "textured",
        &testscenes::textured_quad(Sampler::default()),
        &opts(96, 96, 32, 8),
        None,
        40.0,
    );
}

#[test]
fn parity_checker_floor_mips() {
    // Grazing texture footprint: exercises the ray-cone LOD + trilinear
    // filtering from the atlas.
    parity(
        "checker",
        &testscenes::checker_floor(Sampler::default()),
        &opts(128, 96, 32, 4),
        None,
        40.0,
    );
}

#[test]
fn parity_extension_materials() {
    parity(
        "materials",
        &materials_scene(),
        &opts(128, 96, 64, 8),
        None,
        36.0,
    );
}

#[test]
fn parity_environment_map() {
    parity(
        "env",
        &testscenes::sphere_grid(4, 3),
        &opts(96, 72, 64, 6),
        Some(sky()),
        36.0,
    );
}

#[test]
fn direct_lighting_parity() {
    for (name, scene) in [
        ("cornell-direct", testscenes::cornell_box()),
        ("shadowbox-direct", testscenes::shadow_box()),
        ("materials-direct", materials_scene()),
    ] {
        parity(name, &scene, &opts(96, 96, 64, 1), None, 40.0);
    }
}

#[test]
fn deterministic_for_a_seed() {
    let Some(mut g) = gpu() else { return };
    let scene = testscenes::cornell_box();
    let o = opts(64, 64, 8, 8);
    let a = g.render(&scene, &o).unwrap();
    let b = g.render(&scene, &o).unwrap();
    assert_eq!(a.pixels, b.pixels);
    let mut o2 = o.clone();
    o2.path_trace.seed = 7;
    let c = g.render(&scene, &o2).unwrap();
    assert_ne!(a.pixels, c.pixels);
}

#[test]
fn progressive_refines_add_up() {
    let Some(mut g) = gpu() else { return };
    let scene = testscenes::cornell_box();
    let o = opts(64, 48, 0, 8);
    g.sync(&scene, &o).unwrap();
    g.refine(3).unwrap();
    g.refine(5).unwrap();
    assert_eq!(g.samples(), 8);
    let split = g.hdr().unwrap();
    g.reset();
    assert_eq!(g.samples(), 0);
    g.refine(8).unwrap();
    let whole = g.hdr().unwrap();
    assert_eq!(split.pixels, whole.pixels);
}

#[test]
fn reset_rules_match_the_cpu_tracer() {
    let Some(mut g) = gpu() else { return };
    let scene = testscenes::cornell_box();
    let o = opts(32, 32, 16, 4);
    assert!(g.sync(&scene, &o).unwrap(), "first sync prepares");
    g.refine(2).unwrap();
    assert!(!g.sync(&scene, &o).unwrap(), "no change, no reset");
    assert_eq!(g.samples(), 2);
    // Display-only options keep the accumulation.
    let mut d = o.clone();
    d.background = BackgroundColor([200, 0, 0, 255]);
    d.tone_map = ToneMap::AcesFitted;
    d.exposure = 2.0;
    d.path_trace.samples_per_pixel = 999;
    assert!(!g.sync(&scene, &d).unwrap());
    assert_eq!(g.samples(), 2);
    assert!(!g.is_converged());
    // Radiance-relevant options reset it.
    for change in [
        |o: &mut RenderOptions| o.path_trace.seed = 3,
        |o: &mut RenderOptions| o.path_trace.max_bounces = 2,
        |o: &mut RenderOptions| o.width = 40,
        |o: &mut RenderOptions| o.ambient = 0.5,
    ] {
        g.refine(1).unwrap();
        let mut c = d.clone();
        change(&mut c);
        assert!(g.sync(&scene, &c).unwrap());
        assert_eq!(g.samples(), 0);
        g.sync(&scene, &d).unwrap();
    }
    g.refine(1).unwrap();
    g.invalidate_scene();
    assert!(g.sync(&scene, &d).unwrap());
    assert_eq!(g.samples(), 0);
    g.refine(1).unwrap();
    g.set_environment(Some(sky()));
    assert_eq!(g.samples(), 0);
    g.refine(1).unwrap();
    assert_eq!(g.samples(), 1);
    // Refining before any sync is a no-op.
    let mut fresh = gpu().unwrap();
    fresh.refine(4).unwrap();
    assert_eq!(fresh.samples(), 0);
}

#[test]
fn uncovered_pixels_keep_background_bytes() {
    let Some(mut g) = gpu() else { return };
    let scene = testscenes::sphere_grid(2, 1);
    let mut o = opts(64, 48, 4, 2);
    o.background = BackgroundColor([37, 99, 201, 180]);
    let gi = g.render(&scene, &o).unwrap();
    let ci = PathTraceRenderer::new().render(&scene, &o).unwrap();
    let mut bg = 0;
    for (a, b) in gi.pixels.chunks_exact(4).zip(ci.pixels.chunks_exact(4)) {
        if b == [37, 99, 201, 180] {
            assert_eq!(a, b);
            bg += 1;
        }
    }
    assert!(bg > 100, "{bg} background pixels");
}

#[test]
fn draw_texture_and_mode_selection() {
    let Some(mut g) = gpu() else { return };
    let scene = testscenes::cornell_box();
    let o = opts(40, 30, 2, 2);
    g.sync(&scene, &o).unwrap();
    g.refine(2).unwrap();
    let t = g.draw_texture().unwrap();
    assert_eq!((t.width(), t.height()), (40, 30));
    assert_eq!(t.format(), oxideav_render_vulkan::COLOR_FORMAT);

    let mut r = GpuRenderer::new().unwrap();
    r.set_mode(GpuMode::PathTrace);
    assert_eq!(r.mode(), GpuMode::PathTrace);
    let a = r.render(&scene, &o).unwrap();
    let b = g.render(&scene, &o).unwrap();
    assert_eq!(a.pixels, b.pixels);
    r.set_mode(GpuMode::Raster);
    let c = r.render(&scene, &o).unwrap();
    assert_ne!(a.pixels, c.pixels);

    let mut reg = oxideav_render::RenderRegistry::new();
    oxideav_render_vulkan::register_into(&mut reg);
    let mut viareg = reg
        .make(oxideav_render_vulkan::PATHTRACE_BACKEND_NAME)
        .unwrap();
    assert_eq!(viareg.render(&scene, &o).unwrap().pixels, a.pixels);
}

#[test]
fn empty_scene_is_all_background() {
    let Some(mut g) = gpu() else { return };
    let mut o = opts(16, 12, 4, 4);
    o.scene_camera = None;
    o.background = BackgroundColor([1, 2, 3, 4]);
    let img = g.render(&Scene3D::new(), &o).unwrap();
    assert!(img.pixels.chunks_exact(4).all(|p| p == [1, 2, 3, 4]));
}

#[test]
fn gl_backend_runs_or_fails_cleanly() {
    // GL / GLES adapters may lack compute shaders: the tracer must then
    // report Error::Backend instead of panicking.
    match GpuPathTracer::with_backend(oxideav_render_vulkan::GpuBackend::Gl) {
        Ok(mut g) => {
            let scene = testscenes::cornell_box();
            let o = opts(32, 32, 4, 4);
            let a = g.render(&scene, &o).unwrap();
            let b = PathTraceRenderer::new().render(&scene, &o).unwrap();
            let p = testscenes::psnr(&a, &b);
            eprintln!("GL: {} — PSNR vs CPU {p:.2} dB", g.adapter_summary());
            assert!(p > 35.0, "{p}");
        }
        Err(e) => eprintln!("GL path tracer unavailable: {e}"),
    }
}
