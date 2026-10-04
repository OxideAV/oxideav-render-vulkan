//! Camera framing. Reproduces `oxideav-render`'s scanline camera
//! contract (auto-frame looking down `-Z` at a 1.2× bounding margin,
//! or an orbit around the bounds centre when [`RenderOptions::camera`]
//! is set) so GPU and CPU backends frame a scene identically.
//!
//! The projection is built with OpenGL-style `[-1, 1]` clip depth and
//! then remapped to the `[0, 1]` range wgpu uses (`z' = (z + w) / 2`),
//! which keeps the depth hyperbola — and therefore the DepthDebug
//! grayscale — identical to the scanline backend's.

use oxideav_render::{CameraSpec, Projection, RenderOptions};

use crate::flatten::Bounds;
use crate::math::{mat4_mul, vec3_cross, vec3_dot, vec3_normalise, vec3_sub, Mat4};

/// Resolve the view-projection matrix (row-major, `[0, 1]` depth) for
/// a `width × height` target.
pub(crate) fn view_proj(width: u32, height: u32, bounds: Bounds, opts: &RenderOptions) -> Mat4 {
    let centre: [f32; 3] = std::array::from_fn(|i| (bounds.min[i] + bounds.max[i]) * 0.5);
    let extent = (0..3)
        .map(|i| bounds.max[i] - bounds.min[i])
        .fold(1.0e-3_f32, f32::max);
    let aspect = width.max(1) as f32 / height.max(1) as f32;
    let radius = extent * 0.5 * 1.2;
    let fov_y = opts.fov_deg.to_radians();
    let auto_dist = match opts.projection {
        Projection::Perspective => radius / (fov_y * 0.5).tan(),
        Projection::Orthographic => extent * 1.5,
        _ => radius / (fov_y * 0.5).tan(),
    };

    let (eye, dist) = match opts.camera {
        Some(CameraSpec {
            elevation_deg,
            azimuth_deg,
            distance,
        }) => {
            let (el, az) = (elevation_deg.to_radians(), azimuth_deg.to_radians());
            let dir = [el.cos() * az.sin(), el.sin(), el.cos() * az.cos()];
            let d = auto_dist * distance;
            (std::array::from_fn(|i| centre[i] + dir[i] * d), d)
        }
        None => ([centre[0], centre[1], centre[2] + auto_dist], auto_dist),
    };

    let view = look_at(eye, centre, [0.0, 1.0, 0.0]);
    let proj = match opts.projection {
        Projection::Orthographic => {
            let half = radius;
            let (half_w, half_h) = if aspect >= 1.0 {
                (half * aspect, half)
            } else {
                (half, half / aspect)
            };
            let near = (dist - extent * 2.0).min(-extent);
            let far = dist + extent * 2.0;
            orthographic(half_w, half_h, near, far)
        }
        // Perspective, and any projection added later.
        _ => {
            let near = (dist - extent).max(extent * 0.01);
            let far = dist + extent * 2.0;
            perspective(fov_y, aspect, near, far)
        }
    };
    let gl_to_wgpu: Mat4 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ];
    mat4_mul(&gl_to_wgpu, &mat4_mul(&proj, &view))
}

/// Unit direction towards the light, from the options' azimuth /
/// elevation (same convention as the scanline backend).
pub(crate) fn light_direction(opts: &RenderOptions) -> [f32; 3] {
    let az = opts.light.azimuth_deg.to_radians();
    let el = opts.light.elevation_deg.to_radians();
    vec3_normalise([el.cos() * az.sin(), el.sin(), el.cos() * az.cos()])
}

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
    let f = vec3_normalise(vec3_sub(target, eye));
    let s = vec3_normalise(vec3_cross(f, up));
    let u = vec3_cross(s, f);
    [
        [s[0], s[1], s[2], -vec3_dot(s, eye)],
        [u[0], u[1], u[2], -vec3_dot(u, eye)],
        [-f[0], -f[1], -f[2], vec3_dot(f, eye)],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, 2.0 * far * near * nf],
        [0.0, 0.0, -1.0, 0.0],
    ]
}

fn orthographic(half_w: f32, half_h: f32, near: f32, far: f32) -> Mat4 {
    let fne = far - near;
    [
        [1.0 / half_w, 0.0, 0.0, 0.0],
        [0.0, 1.0 / half_h, 0.0, 0.0],
        [0.0, 0.0, -2.0 / fne, -(far + near) / fne],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
