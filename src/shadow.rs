//! Shadow-map light frusta. Mirrors the scanline backend's maps
//! exactly (same fit, same texel metric) so GPU and CPU shadows agree:
//!
//! * directional: orthographic box of half-size `r` (scene bounding
//!   radius) looking at the bounds centre from `2r` up-light, depth
//!   range `[0.5r, 3.5r]`;
//! * spot: perspective from the light position along its axis, FOV
//!   `2·outer + 0.05` (≤ 170°), far plane past the farthest bounds
//!   corner;
//! * point lights are not shadowed.

use oxideav_render::prepare::{LightKind, PreparedLight};

pub(crate) type Mat4 = [[f32; 4]; 4];

/// One light's shadow frustum.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ShadowSetup {
    /// Row-major view-projection with `[0, 1]` clip depth.
    pub(crate) view_proj: Mat4,
    pub(crate) eye: [f32; 3],
    pub(crate) dir: [f32; 3],
    /// World texel size (perspective: per unit distance along `dir`).
    pub(crate) texel: f32,
    pub(crate) perspective: bool,
}

pub(crate) fn setup(
    light: &PreparedLight,
    bounds: ([f32; 3], [f32; 3]),
    size: u32,
) -> Option<ShadowSetup> {
    if light.kind == LightKind::Point {
        return None;
    }
    let (mn, mx) = bounds;
    let c: [f32; 3] = std::array::from_fn(|i| (mn[i] + mx[i]) * 0.5);
    let diag = sub(mx, mn);
    let r = (dot(diag, diag).sqrt() * 0.5).max(1.0e-3);
    let dir = normalise(light.direction);
    let up = if dir[1].abs() > 0.99 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let (eye, view, proj, texel, perspective) = if light.kind == LightKind::Directional {
        let eye = std::array::from_fn(|i| c[i] - dir[i] * 2.0 * r);
        let proj = orthographic(r, r, r * 0.5, r * 3.5);
        (eye, look_at(eye, c, up), proj, 2.0 * r / size as f32, false)
    } else {
        let eye = light.position;
        let target = std::array::from_fn(|i| eye[i] + dir[i]);
        let fov = (2.0 * light.outer_cone_angle + 0.05).min(170f32.to_radians());
        let mut far: f32 = 0.0;
        for i in 0..8 {
            let k = [
                if i & 1 == 0 { mn[0] } else { mx[0] },
                if i & 2 == 0 { mn[1] } else { mx[1] },
                if i & 4 == 0 { mn[2] } else { mx[2] },
            ];
            let d = sub(k, eye);
            far = far.max(dot(d, d).sqrt());
        }
        let near = (r * 1.0e-3).max(1.0e-4);
        let far = (far * 1.01).max(near * 4.0);
        let texel = 2.0 * (fov * 0.5).tan() / size as f32;
        (
            eye,
            look_at(eye, target, up),
            perspective_proj(fov, near, far),
            texel,
            true,
        )
    };
    // GL [-1, 1] clip depth → [0, 1].
    let remap: Mat4 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ];
    Some(ShadowSetup {
        view_proj: mul(&remap, &mul(&proj, &view)),
        eye,
        dir,
        texel,
        perspective,
    })
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalise(v: [f32; 3]) -> [f32; 3] {
    let l = dot(v, v).sqrt();
    if l > 0.0 && l.is_finite() {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, -1.0]
    }
}

fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..4).map(|k| a[i][k] * b[k][j]).sum()))
}

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
    let f = normalise(sub(target, eye));
    let s = normalise(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], s[1], s[2], -dot(s, eye)],
        [u[0], u[1], u[2], -dot(u, eye)],
        [-f[0], -f[1], -f[2], dot(f, eye)],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn perspective_proj(fov_y: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        [f, 0.0, 0.0, 0.0],
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

/// Transpose into WGSL's column-major layout.
pub(crate) fn column_major(m: &Mat4) -> Mat4 {
    std::array::from_fn(|c| std::array::from_fn(|r| m[r][c]))
}
