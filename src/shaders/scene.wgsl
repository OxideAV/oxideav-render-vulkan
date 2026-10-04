// Forward scene shader. Vertex data arrives already in world space
// (the CPU flattener bakes node transforms), so the vertex stage only
// applies the camera's view-projection.
//
// Shading modes mirror oxideav-render's scanline backend:
//   0 Flat        — unlit base colour
//   1 Gouraud     — Lambert + ambient evaluated per vertex
//   2 Phong       — Lambert + ambient evaluated per fragment
//   3 Wireframe   — unlit base colour (line-list pipeline)
//   4 NormalDebug — (n + 1) / 2 colour key
//   5 DepthDebug  — 1 - depth grayscale
//
// The colour target is a plain UNORM texture: the fragment stage
// performs the IEC 61966-2-1 sRGB encode itself so the debug modes
// can write raw byte values.

struct Globals {
    view_proj: mat4x4<f32>,
    // xyz = unit direction towards the light, w = intensity.
    light: vec4<f32>,
    // x = ambient term.
    params: vec4<f32>,
    // x = shading mode.
    mode: vec4<u32>,
};

@group(0) @binding(0) var<uniform> g: Globals;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
};

fn lambert(base: vec4<f32>, n: vec3<f32>) -> vec4<f32> {
    let ambient = g.params.x;
    let cos_theta = max(dot(n, g.light.xyz), 0.0);
    let factor = clamp(ambient + (1.0 - ambient) * cos_theta * g.light.w, 0.0, 1.0);
    return vec4<f32>(base.rgb * factor, base.a);
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if (len2 > 0.0) {
        return v * inverseSqrt(len2);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = g.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    if (g.mode.x == 1u) {
        out.color = lambert(in.color, safe_normalize(in.normal));
    } else {
        out.color = in.color;
    }
    return out;
}

fn srgb_encode(c: f32) -> f32 {
    let x = clamp(c, 0.0, 1.0);
    if (x <= 0.0031308) {
        return 12.92 * x;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn encode(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(srgb_encode(c.r), srgb_encode(c.g), srgb_encode(c.b), clamp(c.a, 0.0, 1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    switch g.mode.x {
        case 2u: {
            return encode(lambert(in.color, safe_normalize(in.normal)));
        }
        case 4u: {
            let n = safe_normalize(in.normal);
            return vec4<f32>(clamp(n * 0.5 + 0.5, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
        }
        case 5u: {
            let v = clamp(1.0 - in.clip.z, 0.0, 1.0);
            return vec4<f32>(v, v, v, 1.0);
        }
        default: {
            return encode(in.color);
        }
    }
}
