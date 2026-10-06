//! Shared GPU-test gating.
//!
//! GPU tests run on any adapter wgpu finds, software rasterisers
//! included (`wgpu::DeviceType::Cpu`: WARP on the Windows CI runners,
//! llvmpipe / lavapipe on Linux). WARP is how CI exercises the D3D12
//! backend and its FXC shader compiler, so it is kept on by default.
//! Set `OXIDEAV_GPU_TESTS=hardware` to skip software adapters, or
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
        Some(info) if info.device_type == wgpu::DeviceType::Cpu && mode == "hardware" => {
            eprintln!(
                "skipping GPU test: software adapter {} ({:?}) with OXIDEAV_GPU_TESTS=hardware",
                info.name, info.backend
            );
            false
        }
        Some(info) => {
            eprintln!(
                "GPU test on {} ({:?}, {:?})",
                info.name, info.device_type, info.backend
            );
            true
        }
    }
}
