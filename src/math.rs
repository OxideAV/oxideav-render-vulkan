//! Small row-major 4×4 / 3-vector helpers (`m[row][col]`, column
//! vectors), the same convention as `oxideav_mesh3d::Transform`.

pub(crate) type Mat4 = [[f32; 4]; 4];

pub(crate) fn identity4() -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

pub(crate) fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// Transform a point (w = 1) with perspective divide when `w` is
/// meaningful; affine matrices leave `w = 1`.
pub(crate) fn mat4_mul_point(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
    let v: [f32; 4] =
        std::array::from_fn(|i| m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3]);
    if (v[3] - 1.0).abs() > f32::EPSILON && v[3].abs() > f32::EPSILON {
        [v[0] / v[3], v[1] / v[3], v[2] / v[3]]
    } else {
        [v[0], v[1], v[2]]
    }
}

/// Inverse-transpose of the upper-left 3×3 (the normal matrix). Falls
/// back to the plain linear part for a singular matrix so normals stay
/// finite.
pub(crate) fn mat3_inverse_transpose(m: &Mat4) -> [[f32; 3]; 3] {
    let a = [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ];
    // Cofactor matrix C; inverse-transpose = C / det.
    let c = [
        [
            a[1][1] * a[2][2] - a[1][2] * a[2][1],
            a[1][2] * a[2][0] - a[1][0] * a[2][2],
            a[1][0] * a[2][1] - a[1][1] * a[2][0],
        ],
        [
            a[0][2] * a[2][1] - a[0][1] * a[2][2],
            a[0][0] * a[2][2] - a[0][2] * a[2][0],
            a[0][1] * a[2][0] - a[0][0] * a[2][1],
        ],
        [
            a[0][1] * a[1][2] - a[0][2] * a[1][1],
            a[0][2] * a[1][0] - a[0][0] * a[1][2],
            a[0][0] * a[1][1] - a[0][1] * a[1][0],
        ],
    ];
    let det = a[0][0] * c[0][0] + a[0][1] * c[0][1] + a[0][2] * c[0][2];
    if det.abs() <= f32::EPSILON || !det.is_finite() {
        return a;
    }
    let inv = 1.0 / det;
    c.map(|row| row.map(|v| v * inv))
}

pub(crate) fn mat3_mul_vec3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

pub(crate) fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalise; a zero-length or non-finite input yields the zero vector.
pub(crate) fn vec3_normalise(v: [f32; 3]) -> [f32; 3] {
    let len = vec3_dot(v, v).sqrt();
    if len > 0.0 && len.is_finite() {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0; 3]
    }
}

/// Transpose into the column-major layout WGSL's `mat4x4<f32>` uses.
pub(crate) fn to_column_major(m: &Mat4) -> [[f32; 4]; 4] {
    std::array::from_fn(|c| std::array::from_fn(|r| m[r][c]))
}
