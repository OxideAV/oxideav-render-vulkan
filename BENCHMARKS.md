# Benchmarks

## GPU path tracer (2026-10-04)

`cargo run --release --example pt_bench` runs the scenarios of
oxideav-render's `cargo bench --bench render -- pathtrace` (same scenes
and options: 256², 8 bounces, roulette from bounce 3, default seed) on
`GpuPathTracer`, plus the CPU `PathTraceRenderer` in the same process.
Each figure is the median of 5–20 timed runs after a warm-up.

The machine has an NVIDIA GeForce RTX 5080 (Vulkan, driver 595.99.02)
and an AMD Ryzen Threadripper 9970X (64 hardware threads). rustc is
1.98 with the release profile.

| Scenario | GPU | CPU, same run | CPU, oxideav-render BENCHMARKS.md |
| --- | --- | --- | --- |
| Cornell box 256², 64 spp, full `render` | **32.2 ms** | 418.8 ms | 281 ms |
| Cornell box 256², 64 spp, `refine` only (scene resident) | 31.8 ms | – | – |
| Cornell box 256², 1024 spp, full `render` | **514 ms** | – | – |
| Cornell box 256², 1024 spp, `refine` only | 512 ms | – | – |
| Cornell 256², `refine(1)` + `image()` readback | 0.76 ms | 8.9 ms | 8.5 ms |
| Cornell 256², `refine(1)` + `draw_texture()` | 0.70 ms | – | – |
| 960-triangle sphere 256², 16 spp | 2.7 ms | 50.9 ms | 154 ms |
| Cornell 1920×1080, `refine(1)` + `draw_texture()` | 9.1 ms | – | – |
| `image()` resolve + readback at 1080p | 0.97 ms | – | 7.8 ms |

- **Cornell box, 64 spp.** The GPU takes 32 ms against the CPU's
  281 ms in oxideav-render's quiet-machine benchmark, about **8.7×**
  faster. The same-run CPU figure (419 ms, 13×) is pessimistic: other
  jobs held the load average near 45 of 64 threads during the run.
  The kernel sustains about 130 M paths/s on this scene (4.2 M camera
  paths, each with point-light NEE, a shadow ray per vertex and
  roulette). At 1024 spp it holds that rate: 67 M paths in 0.51 s.
- **Full `render` ≈ `refine` only.** Preparing and uploading the scene
  (BVH, attribute and table buffers, texture atlas) and the final
  resolve plus readback add under 1 ms for these scenes.
- **Interactive use.** One progressive pass at 256² costs 0.7 ms
  including the resolve into the `COLOR_FORMAT` texture. At 1080p it
  costs 9 ms (2 M paths), so a viewer can refine at about 100 fps while
  displaying the running estimate with no readback.
- **Sphere row.** The BVH-heavy scene gains the most (19× same-run):
  traversal is where the GPU's parallelism pays most. Its documented CPU
  figure (154 ms) predates later oxideav-render commits and is far
  from today's same-run timing, so compare against the same-run
  column.
- **Pipeline creation.** The first `GpuPathTracer::new` in a process
  costs about 1.5 s when the driver's shader cache is cold (NVIDIA keys
  its disk cache per executable) and about 0.25 s when warm. Restructuring
  the kernel so the BVH traversal, the material evaluation and the BSDF
  evaluation each have a single call site brought the cold cost down
  from about 16 s, and made the kernel about 1.3× faster.
- **Not yet tried.** Wavefront path tracing (Laine, Karras, Aila 2013)
  would keep ray types coherent and cut the megakernel's register
  pressure. A wider BVH, or a compressed node layout, would cut
  traversal memory traffic.
