// Forward scene pass. Vertex data arrives in world space (prepared on
// the CPU by oxideav-render's scene-preparation layer), so the vertex
// stage only applies the camera's view-projection.
//
// Output 0 is an Rgba16Float scene-linear target; output 1 is an R8
// coverage mask the resolve pass uses to keep uncovered background
// bytes untouched. What output 0 holds depends on the shading mode:
//
//   0 Flat        legacy: unlit base factor, straight alpha
//   1 Gouraud     legacy: Lambert + ambient per vertex, straight alpha
//   2 Phong       legacy: Lambert + ambient per fragment, straight alpha
//   3 Wireframe   legacy: unlit base factor (line pipeline)
//   4 NormalDebug display values (n + 1) / 2, passed through
//   5 DepthDebug  display value 1 - depth, passed through
//   6 Pbr         glTF 2.0 Appendix B metallic-roughness, premultiplied
//
// Pbr follows the glTF 2.0 specification, Appendix B (BRDF
// implementation): Schlick Fresnel on V·H, Trowbridge-Reitz / GGX
// distribution with α = roughness², height-correlated Smith
// visibility, Lambert diffuse weighted by (1 − F). Punctual light
// attenuation per KHR_lights_punctual.

const MAX_LIGHTS: u32 = 16u;
const MAX_SHADOWS: u32 = 4u;
const PI: f32 = 3.141592653589793;

struct Light {
    // xyz = position (world), w = kind (0 directional, 1 point, 2 spot)
    position: vec4<f32>,
    // xyz = travel direction (unit), w = range (<= 0: infinite)
    direction: vec4<f32>,
    // rgb = colour * intensity
    radiance: vec4<f32>,
    // x = cos(inner cone), y = cos(outer cone), z = shadow layer (< 0: none)
    cone: vec4<f32>,
};

// Shadow map of one light: linear depth along `dir` from `eye` per
// texel (+inf = nothing), mirroring oxideav-render's scanline maps.
struct ShadowInfo {
    view_proj: mat4x4<f32>,
    // xyz = light eye, w = world texel size (perspective: per unit distance)
    eye_texel: vec4<f32>,
    // xyz = light direction, w = 1 for a perspective (spot) map
    dir_persp: vec4<f32>,
};

struct Globals {
    view_proj: mat4x4<f32>,
    // xyz = camera position (world)
    eye: vec4<f32>,
    // Legacy modes: xyz = unit direction towards the light, w = intensity.
    legacy_light: vec4<f32>,
    // x = legacy ambient, y = Pbr ambient radiance
    params: vec4<f32>,
    // x = shading mode, y = light count
    mode: vec4<u32>,
    lights: array<Light, MAX_LIGHTS>,
    shadows: array<ShadowInfo, MAX_SHADOWS>,
};

// Shadow-map pass parameters (one dynamic-offset slot per light).
struct ShadowPass {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    dir: vec4<f32>,
};

struct Material {
    base_color: vec4<f32>,
    // rgb = emissive (strength applied), w = alpha cutoff (MASK)
    emissive: vec4<f32>,
    // x = metallic, y = roughness, z = normal scale, w = occlusion strength
    factors: vec4<f32>,
    // x = dielectric F0
    extra: vec4<f32>,
    // x = texture presence bits (1 base, 2 metallic-roughness,
    //     4 normal, 8 occlusion, 16 emissive)
    // y = alpha mode (0 opaque, 1 mask, 2 blend)
    // z = unlit flag
    // w = UV-set bits (bit n set: slot n reads TEXCOORD_1)
    flags: vec4<u32>,
    // Per texture slot, the two rows of a 2×3 UV affine transform.
    uv_xform: array<vec4<f32>, 10>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var t_shadow: texture_2d_array<f32>;
@group(2) @binding(0) var<uniform> sp: ShadowPass;
@group(1) @binding(0) var<uniform> m: Material;
@group(1) @binding(1) var t_base: texture_2d<f32>;
@group(1) @binding(2) var s_base: sampler;
@group(1) @binding(3) var t_mr: texture_2d<f32>;
@group(1) @binding(4) var s_mr: sampler;
@group(1) @binding(5) var t_normal: texture_2d<f32>;
@group(1) @binding(6) var s_normal: sampler;
@group(1) @binding(7) var t_occlusion: texture_2d<f32>;
@group(1) @binding(8) var s_occlusion: sampler;
@group(1) @binding(9) var t_emissive: texture_2d<f32>;
@group(1) @binding(10) var s_emissive: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) uv1: vec2<f32>,
    @location(5) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) uv1: vec2<f32>,
    @location(5) color: vec4<f32>,
    // Gouraud: pre-lit colour.
    @location(6) lit: vec4<f32>,
};

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) coverage: vec4<f32>,
};

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if (len2 > 0.0) {
        return v * inverseSqrt(len2);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

fn legacy_lambert(base: vec4<f32>, n: vec3<f32>) -> vec4<f32> {
    let ambient = g.params.x;
    let cos_theta = max(dot(n, g.legacy_light.xyz), 0.0);
    let factor = clamp(ambient + (1.0 - ambient) * cos_theta * g.legacy_light.w, 0.0, 1.0);
    return vec4<f32>(base.rgb * factor, base.a);
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = g.view_proj * vec4<f32>(in.position, 1.0);
    out.world = in.position;
    out.normal = in.normal;
    out.tangent = in.tangent;
    out.uv0 = in.uv0;
    out.uv1 = in.uv1;
    out.color = in.color;
    out.lit = vec4<f32>(0.0);
    if (g.mode.x == 1u) {
        out.lit = legacy_lambert(m.base_color, safe_normalize(in.normal));
    }
    return out;
}

fn slot_uv(slot: u32, in: VsOut) -> vec2<f32> {
    var uv = in.uv0;
    if ((m.flags.w & (1u << slot)) != 0u) {
        uv = in.uv1;
    }
    let r0 = m.uv_xform[slot * 2u];
    let r1 = m.uv_xform[slot * 2u + 1u];
    return vec2<f32>(dot(r0.xyz, vec3<f32>(uv, 1.0)), dot(r1.xyz, vec3<f32>(uv, 1.0)));
}

fn has(bit: u32) -> bool {
    return (m.flags.x & bit) != 0u;
}

// Appendix B terms.
fn d_ggx(n_dot_h: f32, a2: f32) -> f32 {
    let f = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * f * f);
}

fn v_smith_correlated(n_dot_l: f32, n_dot_v: f32, a2: f32) -> f32 {
    let gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-8);
}

fn light_sample(i: u32, p: vec3<f32>, l_out: ptr<function, vec3<f32>>) -> vec3<f32> {
    let light = g.lights[i];
    let kind = u32(light.position.w);
    if (kind == 0u) {
        *l_out = -light.direction.xyz;
        return light.radiance.rgb;
    }
    let to = light.position.xyz - p;
    let d2 = dot(to, to);
    if (d2 <= 1e-12) {
        return vec3<f32>(0.0);
    }
    let d = sqrt(d2);
    let l = to / d;
    *l_out = l;
    var att = 1.0 / d2;
    let range = light.direction.w;
    if (range > 0.0) {
        let x = d / range;
        att *= clamp(1.0 - x * x * x * x, 0.0, 1.0);
    }
    if (kind == 2u) {
        let cos_inner = light.cone.x;
        let cos_outer = light.cone.y;
        let scale = 1.0 / max(cos_inner - cos_outer, 0.001);
        let offset = -cos_outer * scale;
        let cd = dot(light.direction.xyz, -l);
        let a = clamp(cd * scale + offset, 0.0, 1.0);
        att *= a * a;
    }
    return light.radiance.rgb * att;
}

// Fraction of light `i` reaching `p` (geometric normal `ng`, unit
// direction to the light `l`): normal-offset 3×3 bilinear PCF over
// linear light depths (Williams 1978; Reeves et al. 1987).
fn shadow_visibility(layer: u32, p: vec3<f32>, ng_in: vec3<f32>, l: vec3<f32>) -> f32 {
    let sh = g.shadows[layer];
    let eye = sh.eye_texel.xyz;
    let dir = sh.dir_persp.xyz;
    var texel = sh.eye_texel.w;
    if (sh.dir_persp.w > 0.5) {
        texel *= max(dot(p - eye, dir), 1e-4);
    }
    var ng = ng_in;
    if (dot(ng, l) < 0.0) {
        ng = -ng;
    }
    let cos_t = clamp(dot(ng, l), 0.0, 1.0);
    let q = p + ng * (texel * (1.0 + 2.0 * (1.0 - cos_t)));
    let c = sh.view_proj * vec4<f32>(q, 1.0);
    if (c.w <= 0.0) {
        return 1.0;
    }
    let size = i32(textureDimensions(t_shadow).x);
    let s = f32(size);
    let u = (c.x / c.w * 0.5 + 0.5) * s - 0.5;
    let v = (1.0 - (c.y / c.w * 0.5 + 0.5)) * s - 0.5;
    if (!(u > -1.0 && v > -1.0 && u < s && v < s)) {
        return 1.0;
    }
    let d_recv = dot(q - eye, dir) - texel;
    var sum = 0.0;
    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            let fu = u + f32(ox);
            let fv = v + f32(oy);
            let x0 = i32(floor(fu));
            let y0 = i32(floor(fv));
            let tx = fu - floor(fu);
            let ty = fv - floor(fv);
            let a = shadow_lit(x0, y0, size, layer, d_recv) * (1.0 - tx)
                + shadow_lit(x0 + 1, y0, size, layer, d_recv) * tx;
            let b = shadow_lit(x0, y0 + 1, size, layer, d_recv) * (1.0 - tx)
                + shadow_lit(x0 + 1, y0 + 1, size, layer, d_recv) * tx;
            sum += a * (1.0 - ty) + b * ty;
        }
    }
    return sum / 9.0;
}

fn shadow_lit(x: i32, y: i32, size: i32, layer: u32, d_recv: f32) -> f32 {
    if (x < 0 || y < 0 || x >= size || y >= size) {
        return 1.0;
    }
    if (textureLoad(t_shadow, vec2<i32>(x, y), i32(layer), 0).r >= d_recv) {
        return 1.0;
    }
    return 0.0;
}

fn shade_pbr(in: VsOut, front: bool) -> vec4<f32> {
    // Geometric normal from screen-space derivatives (sign is resolved
    // towards each light in `shadow_visibility`).
    let ng = safe_normalize(cross(dpdx(in.world), dpdy(in.world)));
    // Sample every slot up front, in uniform control flow (implicit-
    // derivative sampling must not sit behind per-fragment branches).
    // Absent slots are bound to a 1×1 white texture and ignored.
    let tex_base = textureSample(t_base, s_base, slot_uv(0u, in));
    let tex_mr = textureSample(t_mr, s_mr, slot_uv(1u, in));
    let tex_normal = textureSample(t_normal, s_normal, slot_uv(2u, in)).rgb;
    let tex_occ = textureSample(t_occlusion, s_occlusion, slot_uv(3u, in)).r;
    let tex_emissive = textureSample(t_emissive, s_emissive, slot_uv(4u, in)).rgb;

    var base = m.base_color * in.color;
    if (has(1u)) {
        base *= tex_base;
    }
    var alpha = base.a;
    if (m.flags.y == 1u) {
        if (alpha < m.emissive.w) {
            // MASK cut-out. Signalled to the entry point (negative
            // alpha) rather than `discard`ed here: FXC (the D3D12 HLSL
            // compiler) rejects a helper whose only exit on a path is
            // a `discard` ("not all control paths return a value").
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
        alpha = 1.0;
    } else if (m.flags.y == 0u) {
        alpha = 1.0;
    }
    if (m.flags.z != 0u) {
        return vec4<f32>(base.rgb * alpha, alpha);
    }

    var metallic = m.factors.x;
    var roughness = m.factors.y;
    if (has(2u)) {
        roughness *= tex_mr.g;
        metallic *= tex_mr.b;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    roughness = clamp(roughness, 0.0, 1.0);

    var n = safe_normalize(in.normal);
    let t = in.tangent;
    if (!front) {
        n = -n;
    }
    if (has(4u) && dot(t.xyz, t.xyz) > 0.0) {
        let n_ts = (2.0 * tex_normal - 1.0) * vec3<f32>(m.factors.z, m.factors.z, 1.0);
        let tt = safe_normalize(t.xyz);
        let bb = t.w * cross(n, tt);
        n = safe_normalize(tt * n_ts.x + bb * n_ts.y + n * n_ts.z);
    }
    var ao = 1.0;
    if (has(8u)) {
        ao = 1.0 + m.factors.w * (tex_occ - 1.0);
    }
    var emissive = m.emissive.rgb;
    if (has(16u)) {
        emissive *= tex_emissive;
    }

    let v = safe_normalize(g.eye.xyz - in.world);
    let n_dot_v = max(dot(n, v), 1e-4);
    let c_diff = base.rgb * (1.0 - metallic);
    let f0 = mix(vec3<f32>(m.extra.x), base.rgb, metallic);
    let a = max(roughness * roughness, 1e-3);
    let a2 = a * a;

    var colour = g.params.y * ao * (c_diff + f0) + emissive;
    for (var i = 0u; i < min(g.mode.y, MAX_LIGHTS); i++) {
        var l = vec3<f32>(0.0);
        let radiance = light_sample(i, in.world, &l);
        let n_dot_l = dot(n, l);
        if (n_dot_l <= 0.0) {
            continue;
        }
        let h = safe_normalize(l + v);
        let v_dot_h = clamp(dot(v, h), 0.0, 1.0);
        let n_dot_h = clamp(dot(n, h), 0.0, 1.0);
        let f = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - v_dot_h, 5.0);
        let spec = f * d_ggx(n_dot_h, a2) * v_smith_correlated(n_dot_l, n_dot_v, a2);
        let diffuse = (vec3<f32>(1.0) - f) * c_diff / PI;
        var vis = 1.0;
        let layer = g.lights[i].cone.z;
        if (layer >= 0.0) {
            vis = shadow_visibility(u32(layer), in.world, ng, l);
        }
        colour += (diffuse + spec) * radiance * n_dot_l * vis;
    }
    return vec4<f32>(colour * alpha, alpha);
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> FsOut {
    var out: FsOut;
    out.coverage = vec4<f32>(1.0);
    switch g.mode.x {
        case 1u: {
            out.color = in.lit;
        }
        case 2u: {
            out.color = legacy_lambert(m.base_color, safe_normalize(in.normal));
        }
        case 4u: {
            let n = safe_normalize(in.normal);
            out.color = vec4<f32>(clamp(n * 0.5 + 0.5, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
        }
        case 5u: {
            let d = clamp(1.0 - in.clip.z, 0.0, 1.0);
            out.color = vec4<f32>(d, d, d, 1.0);
        }
        case 6u: {
            let c = shade_pbr(in, front);
            if (c.a < 0.0) {
                discard;
            }
            out.color = c;
        }
        default: {
            out.color = m.base_color;
        }
    }
    return out;
}

// Lines / points: unlit. Pbr multiplies the vertex colour in and
// writes premultiplied; legacy modes write the base factor.
@fragment
fn fs_unlit(in: VsOut) -> FsOut {
    var out: FsOut;
    out.coverage = vec4<f32>(1.0);
    if (g.mode.x == 6u) {
        let c = m.base_color * in.color;
        out.color = vec4<f32>(c.rgb * c.a, c.a);
    } else if (g.mode.x == 4u) {
        out.color = vec4<f32>(0.5, 0.5, 1.0, 1.0);
    } else if (g.mode.x == 5u) {
        let d = clamp(1.0 - in.clip.z, 0.0, 1.0);
        out.color = vec4<f32>(d, d, d, 1.0);
    } else {
        out.color = m.base_color;
    }
    return out;
}

// ---- Shadow-map pass ------------------------------------------------

struct ShadowVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) uv0: vec2<f32>,
    @location(2) uv1: vec2<f32>,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_shadow(in: VsIn) -> ShadowVsOut {
    var out: ShadowVsOut;
    out.clip = sp.view_proj * vec4<f32>(in.position, 1.0);
    out.world = in.position;
    out.uv0 = in.uv0;
    out.uv1 = in.uv1;
    out.color = in.color;
    return out;
}

// Writes the linear light depth; MASK casters honour their cutoff.
@fragment
fn fs_shadow(in: ShadowVsOut) -> @location(0) vec4<f32> {
    var uv = in.uv0;
    if ((m.flags.w & 1u) != 0u) {
        uv = in.uv1;
    }
    let r0 = m.uv_xform[0];
    let r1 = m.uv_xform[1];
    let tuv = vec2<f32>(dot(r0.xyz, vec3<f32>(uv, 1.0)), dot(r1.xyz, vec3<f32>(uv, 1.0)));
    let tex = textureSample(t_base, s_base, tuv);
    if (m.flags.y == 1u) {
        var a = m.base_color.a * in.color.a;
        if ((m.flags.x & 1u) != 0u) {
            a *= tex.a;
        }
        if (a < m.emissive.w) {
            discard;
        }
    }
    let d = dot(in.world - sp.eye.xyz, sp.dir.xyz);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
}
