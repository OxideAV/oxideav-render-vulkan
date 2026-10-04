//! Shared GPU-test gating.
//!
//! GPU tests run on hardware adapters. Software rasterisers
//! (`wgpu::DeviceType::Cpu`: WARP on Windows CI runners, llvmpipe /
//! lavapipe on Linux) are skipped by default: they are slow, and WARP
//! has been seen to crash (access violation) when several test threads
//! drive it at once, which no test can catch. Set
//! `OXIDEAV_GPU_TESTS=software` to run on them anyway, or
//! `OXIDEAV_GPU_TESTS=off` to skip every GPU test.

#![allow(dead_code)]

use oxideav_render_vulkan::{probe_adapter, GpuBackend};

/// Whether GPU tests should run on this machine. Prints the reason
/// when they are skipped.
pub fn gpu_tests_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(decide)
}

fn decide() -> bool {
    let mode = std::env::var("OXIDEAV_GPU_TESTS").unwrap_or_default();
    if mode == "off" {
        eprintln!("skipping GPU test: OXIDEAV_GPU_TESTS=off");
        return false;
    }
    match probe_adapter(GpuBackend::Auto) {
        None => {
            eprintln!("skipping GPU test: no adapter");
            false
        }
        Some(info) if info.device_type == wgpu::DeviceType::Cpu && mode != "software" => {
            eprintln!(
                "skipping GPU test: software adapter {} ({:?}); set OXIDEAV_GPU_TESTS=software to run",
                info.name, info.backend
            );
            false
        }
        Some(_) => true,
    }
}
