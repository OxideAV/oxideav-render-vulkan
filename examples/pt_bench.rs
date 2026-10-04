//! GPU path-tracer timings, mirroring oxideav-render's
//! `cargo bench --bench render -- pathtrace` scenarios (same scenes and
//! options), plus the CPU tracer on the same machine for comparison.
//!
//! `cargo run --release --example pt_bench [-- --no-cpu]`

use std::time::{Duration, Instant};

use oxideav_mesh3d::{Mesh, MeshId, Node, NodeId, Scene3D};
use oxideav_render::{
    BackgroundColor, PathTraceOptions, PathTraceRenderer, PathTracer, RenderOptions, Renderer,
    ShadingMode,
};
use oxideav_render_vulkan::GpuPathTracer;

fn opts(size: u32, spp: u32, scene_camera: Option<usize>) -> RenderOptions {
    RenderOptions {
        width: size,
        height: size,
        shading: ShadingMode::Pbr,
        background: BackgroundColor([16, 16, 24, 255]),
        scene_camera,
        path_trace: PathTraceOptions {
            samples_per_pixel: spp,
            ..PathTraceOptions::default()
        },
        ..RenderOptions::default()
    }
}

fn sphere_scene() -> Scene3D {
    let mut scene = Scene3D::new();
    scene.meshes.push(
        Mesh::new("sphere".to_string())
            .with_primitive(oxideav_render::testscenes::uv_sphere(1.0, 32, 16)),
    );
    scene.nodes.push(Node {
        mesh: Some(MeshId(0)),
        ..Node::default()
    });
    scene.roots.push(NodeId(0));
    scene
}

/// Median of `n` timed runs (after one warm-up).
fn time(n: usize, mut f: impl FnMut()) -> Duration {
    f();
    let mut v: Vec<Duration> = (0..n)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    v.sort();
    v[n / 2]
}

fn row(name: &str, gpu: Duration, cpu: Option<Duration>) {
    match cpu {
        Some(c) => println!(
            "| `{name}` | {:.2} ms | {:.1} ms | {:.0}× |",
            gpu.as_secs_f64() * 1e3,
            c.as_secs_f64() * 1e3,
            c.as_secs_f64() / gpu.as_secs_f64()
        ),
        None => println!("| `{name}` | {:.2} ms | – | – |", gpu.as_secs_f64() * 1e3),
    }
}

fn main() -> oxideav_render::Result<()> {
    let with_cpu = !std::env::args().any(|a| a == "--no-cpu");
    let t = Instant::now();
    let mut gpu = GpuPathTracer::new()?;
    println!(
        "adapter: {}; tracer created in {:?}",
        gpu.adapter_summary(),
        t.elapsed()
    );
    let cornell = oxideav_render::testscenes::cornell_box();
    let sphere = sphere_scene();
    let mut cpu = PathTraceRenderer::new();
    println!("| Scenario | GPU | CPU | speed-up |");
    println!("| --- | --- | --- | --- |");

    for spp in [64u32, 1024] {
        let o = opts(256, spp, Some(0));
        let g = time(5, || {
            gpu.render(&cornell, &o).unwrap();
        });
        let c = (with_cpu && spp == 64).then(|| {
            time(5, || {
                cpu.render(&cornell, &o).unwrap();
            })
        });
        row(&format!("pathtrace_cornell_{spp}spp_256 (render)"), g, c);
        // Kernel only: refine + wait, scene already resident.
        gpu.sync(&cornell, &o)?;
        let k = time(5, || {
            gpu.reset();
            gpu.refine(spp).unwrap();
            gpu.finish().unwrap();
        });
        row(
            &format!("pathtrace_cornell_{spp}spp_256 (refine only)"),
            k,
            None,
        );
    }

    let o = opts(256, 64, Some(0));
    gpu.invalidate_scene();
    gpu.sync(&cornell, &o)?;
    let g = time(20, || {
        gpu.refine(1).unwrap();
        gpu.image().unwrap();
    });
    let c = with_cpu.then(|| {
        let mut t = PathTracer::new();
        t.sync(&cornell, &o);
        time(20, || {
            t.refine(1);
            t.image();
        })
    });
    row("pathtrace_cornell_refine1_256 (+ image readback)", g, c);
    let g = time(20, || {
        gpu.refine(1).unwrap();
        gpu.draw_texture().unwrap();
        gpu.finish().unwrap();
    });
    row("pathtrace_cornell_refine1_256 (+ draw_texture)", g, None);

    let o = opts(256, 16, None);
    let g = time(5, || {
        gpu.render(&sphere, &o).unwrap();
    });
    let c = with_cpu.then(|| {
        time(5, || {
            cpu.render(&sphere, &o).unwrap();
        })
    });
    row("pathtrace_sphere_960tri_16spp_256", g, c);

    let hd = RenderOptions {
        width: 1920,
        height: 1080,
        ..opts(256, 64, Some(0))
    };
    // A different scene object: tell the tracer (sync alone only
    // tracks options, like the CPU `PathTracer`).
    gpu.invalidate_scene();
    gpu.sync(&cornell, &hd)?;
    let g = time(10, || {
        gpu.refine(1).unwrap();
        gpu.draw_texture().unwrap();
        gpu.finish().unwrap();
    });
    row("pathtrace_cornell_refine1_1080p (+ draw_texture)", g, None);
    gpu.refine(1)?;
    let g = time(10, || {
        gpu.image().unwrap();
    });
    row("pathtrace_resolve_image_1080p", g, None);
    Ok(())
}
