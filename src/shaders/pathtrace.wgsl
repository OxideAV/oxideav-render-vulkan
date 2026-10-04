// GPU path tracer — a WGSL port of oxideav-render's `pathtrace` module.
//
// The CPU module docs (§1–5) are the normative estimator spec; every
// function below mirrors its CPU counterpart (same name where possible)
// so the two backends draw the same samples, choose the same lobes and
// lights, and converge to the same image. Differences: f32 everywhere
// (the CPU's f64 solid-angle / Arvo math and the watertight test's f64
// edge fallback are evaluated in f32), BVH traversal is a per-thread
// stack walk (Aila & Laine 2009 "while-while" shape, ordered
// near-first), and texture texels come from an Rgba16Float atlas
// filtered manually with `textureLoad` (exactly the CPU filter code).
//
// One invocation = one pixel; it traces `frame.count` consecutive
// samples starting at `frame.first` and adds the covered radiance and
// the covered-sample count to the Rgba32Float-style accumulator.

struct Params {
    // xyz eye, w = 1 for orthographic
    eye: vec4<f32>,
    // xyz forward, w = half_w
    forward: vec4<f32>,
    // xyz side, w = half_h
    side: vec4<f32>,
    // xyz up, w = pixel spread (cone width per unit distance)
    up: vec4<f32>,
    // width, height, seed, max_bounces
    dims: vec4<u32>,
    // rr_start, strategy (0 MIS, 1 light only, 2 BSDF only),
    // punctual light count, emissive triangle count
    cfg: vec4<u32>,
    // env width, env height, has env, node count
    env: vec4<u32>,
    // word offsets into `tables`: sobol, sheen, materials, punctual
    offs0: vec4<u32>,
    // emissive lights, (unused), env marginal, env conditional
    offs1: vec4<u32>,
    // env mass, env radiance, first triangle vec4 in `geom`, unused
    offs2: vec4<u32>,
    // ambient, clamp, P_env, unused
    fparams: vec4<f32>,
};

struct Frame {
    first: u32,
    count: u32,
    y0: u32,
    rows: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
// BVH nodes (2 vec4 per node) followed by triangle positions (3 vec4
// per leaf slot: p0 + global id, p1 + emissive light index, p2 +
// material index).
@group(0) @binding(1) var<storage, read> geom: array<vec4<f32>>;
// 12 vec4 per leaf slot: normals (w of the first = flags), tangents,
// uv0.xy|uv1.xy, colours.
@group(0) @binding(2) var<storage, read> attrs: array<vec4<f32>>;
// Sobol' matrices, sheen LUT, materials, lights, texture descriptors,
// environment CDFs — flat words.
@group(0) @binding(3) var<storage, read> tables: array<u32>;
// Per pixel: Σ rgb over covered samples, covered count.
@group(0) @binding(4) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(5) var atlas: texture_2d_array<f32>;
@group(1) @binding(0) var<uniform> frame: Frame;

const PI: f32 = 3.14159265358979;
const INF: f32 = 3.0e38;
const NONE: u32 = 0xffffffffu;
const MIN_ALPHA: f32 = 1.0e-3;
const MIN_SPHERICAL_SOLID_ANGLE: f32 = 1.0e-4;
const SHEEN_COS: u32 = 32u;
const SHEEN_ROUGH: u32 = 16u;

// Attribute flags (w of the first normal of a slot).
const F_NORMALS: u32 = 1u;
const F_TANGENTS: u32 = 2u;
const F_COLORS: u32 = 4u;
const F_UV0: u32 = 8u;
const F_UV1: u32 = 16u;

// Material layout (words). See `pathtrace.rs::material_words`.
const MAT_STRIDE: u32 = 136u;
const M_FLAGS: u32 = 7u;
const M_SLOTS: u32 = 32u;
const S_BASE: u32 = 0u;
const S_MR: u32 = 1u;
const S_NORMAL: u32 = 2u;
const S_EMISSIVE: u32 = 3u;
const S_SPEC: u32 = 4u;
const S_SPEC_COLOR: u32 = 5u;
const S_TRANSMISSION: u32 = 6u;
const S_THICKNESS: u32 = 7u;
const S_CC: u32 = 8u;
const S_CC_ROUGH: u32 = 9u;
const S_CC_NORMAL: u32 = 10u;
const S_SHEEN_COLOR: u32 = 11u;
const S_SHEEN_ROUGH: u32 = 12u;

var<workgroup> wg_sobol: array<u32, 128>;
var<workgroup> wg_sheen: array<f32, 512>;

// =====================================================================
// Small helpers.
// =====================================================================

fn tf(i: u32) -> f32 {
    return bitcast<f32>(tables[i]);
}

fn tv3(i: u32) -> vec3<f32> {
    return vec3<f32>(tf(i), tf(i + 1u), tf(i + 2u));
}

fn tv4(i: u32) -> vec4<f32> {
    return vec4<f32>(tf(i), tf(i + 1u), tf(i + 2u), tf(i + 3u));
}

fn finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn finite3(v: vec3<f32>) -> bool {
    return finite(v.x) && finite(v.y) && finite(v.z);
}

fn is_black(a: vec3<f32>) -> bool {
    return !(a.x > 0.0 || a.y > 0.0 || a.z > 0.0);
}

fn max3(a: vec3<f32>) -> f32 {
    return max(max(a.x, a.y), a.z);
}

fn mean3(a: vec3<f32>) -> f32 {
    return (a.x + a.y + a.z) * (1.0 / 3.0);
}

fn nrm(v: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot(v, v));
    if (len < 1.1920929e-7) {
        return vec3<f32>(0.0);
    }
    return v / len;
}

fn sign_of(x: f32) -> f32 {
    // copysign(1, x), honouring the sign of zero.
    if ((bitcast<u32>(x) & 0x80000000u) != 0u) {
        return -1.0;
    }
    return 1.0;
}

// =====================================================================
// §2 Random numbers.
// =====================================================================

fn pcg_hash(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_mix(a: u32, b: u32) -> u32 {
    return pcg_hash(a ^ pcg_hash(b));
}

fn sobol(index: u32, dim: u32) -> u32 {
    var r = 0u;
    var i = index;
    var k = 0u;
    loop {
        if (i == 0u) {
            break;
        }
        if ((i & 1u) == 1u) {
            r ^= wg_sobol[dim * 32u + k];
        }
        i >>= 1u;
        k += 1u;
    }
    return r;
}

fn laine_karras(x0: u32, seed: u32) -> u32 {
    var x = x0 + seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

fn nested_uniform_scramble(x: u32, seed: u32) -> u32 {
    return reverseBits(laine_karras(reverseBits(x), seed));
}

fn to_unit(v: u32) -> f32 {
    return f32(v >> 8u) * (1.0 / 16777216.0);
}

fn sobol_owen_4d(index: u32, pattern_seed: u32) -> vec4<f32> {
    let i = nested_uniform_scramble(index, pattern_seed);
    var out: vec4<f32>;
    for (var d = 0u; d < 4u; d++) {
        out[d] = to_unit(nested_uniform_scramble(sobol(i, d), hash_mix(pattern_seed, d)));
    }
    return out;
}

struct Sampler {
    pixel_seed: u32,
    index: u32,
    path_seed: u32,
};

fn make_sampler(seed: u32, x: u32, y: u32, index: u32) -> Sampler {
    let pixel_seed = hash_mix(hash_mix(seed, y), x);
    return Sampler(pixel_seed, index, hash_mix(pixel_seed, index));
}

fn pattern(s: Sampler, p: u32) -> vec4<f32> {
    return sobol_owen_4d(s.index, hash_mix(s.pixel_seed, p));
}

fn coin(s: Sampler, ray: u32, tri: u32) -> f32 {
    return to_unit(hash_mix(hash_mix(s.path_seed, ray), tri));
}

// =====================================================================
// Geometry access.
// =====================================================================

fn tri_base(slot: u32) -> u32 {
    return P.offs2.z + 3u * slot;
}

fn tri_pos(slot: u32, v: u32) -> vec3<f32> {
    return geom[tri_base(slot) + v].xyz;
}

fn tri_global(slot: u32) -> u32 {
    return bitcast<u32>(geom[tri_base(slot)].w);
}

fn tri_light(slot: u32) -> u32 {
    return bitcast<u32>(geom[tri_base(slot) + 1u].w);
}

fn tri_material(slot: u32) -> u32 {
    return bitcast<u32>(geom[tri_base(slot) + 2u].w);
}

fn attr(slot: u32, k: u32) -> vec4<f32> {
    return attrs[slot * 12u + k];
}

fn attr_flags(slot: u32) -> u32 {
    return bitcast<u32>(attrs[slot * 12u].w);
}

fn interp3a(slot: u32, k: u32, b: vec3<f32>) -> vec3<f32> {
    return attr(slot, k).xyz * b.x + attr(slot, k + 1u).xyz * b.y + attr(slot, k + 2u).xyz * b.z;
}

fn interp4a(slot: u32, k: u32, b: vec3<f32>) -> vec4<f32> {
    return attr(slot, k) * b.x + attr(slot, k + 1u) * b.y + attr(slot, k + 2u) * b.z;
}

fn vertex_uv(slot: u32, uvset: u32, v: u32) -> vec2<f32> {
    let a = attr(slot, 6u + v);
    if (uvset == 0u) {
        return a.xy;
    }
    return a.zw;
}

fn has_uv(slot: u32, uvset: u32) -> bool {
    let f = attr_flags(slot);
    if (uvset == 0u) {
        return (f & F_UV0) != 0u;
    }
    if (uvset == 1u) {
        return (f & F_UV1) != 0u;
    }
    return false;
}

fn mat_off(m: u32) -> u32 {
    return P.offs0.z + m * MAT_STRIDE;
}

fn mat_flags(m: u32) -> u32 {
    return tables[mat_off(m) + M_FLAGS];
}

// alpha mode (0 opaque, 1 mask, 2 blend) in bits 0–1.
fn mat_double_sided(m: u32) -> bool {
    return (mat_flags(m) & 4u) != 0u;
}

fn mat_unlit(m: u32) -> bool {
    return (mat_flags(m) & 8u) != 0u;
}

// =====================================================================
// Textures: manual filtering over the atlas (oxideav-render texture.rs).
// =====================================================================

struct Level {
    layer: i32,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
};

fn tex_level(desc: u32, level: u32) -> Level {
    // Header: full level count, flags, image w, image h, skipped
    // levels, retained levels, 0, 0; then 4 words per retained level.
    let skip = tables[desc + 4u];
    let kept = tables[desc + 5u];
    let l = min(u32(max(i32(level) - i32(skip), 0)), kept - 1u);
    let o = desc + 8u + 4u * l;
    let wh = tables[o + 3u];
    return Level(
        i32(tables[o]),
        i32(tables[o + 1u]),
        i32(tables[o + 2u]),
        i32(wh & 0xffffu),
        i32(wh >> 16u),
    );
}

fn wrap_index(i: i32, n: i32, mode: u32) -> i32 {
    if (mode == 0u) {
        return clamp(i, 0, n - 1);
    }
    if (mode == 1u) {
        let m = ((i % (2 * n)) + 2 * n) % (2 * n);
        if (m >= n) {
            return 2 * n - 1 - m;
        }
        return m;
    }
    return ((i % n) + n) % n;
}

fn fetch(lv: Level, x: i32, y: i32, ws: u32, wt: u32) -> vec4<f32> {
    let xi = wrap_index(x, lv.w, ws);
    let yi = wrap_index(y, lv.h, wt);
    return textureLoad(atlas, vec2<i32>(lv.x + xi, lv.y + yi), lv.layer, 0);
}

fn filter_level(lv: Level, uv: vec2<f32>, linear: bool, ws: u32, wt: u32) -> vec4<f32> {
    var u = uv.x;
    var v = uv.y;
    if (!finite(u)) {
        u = 0.0;
    }
    if (!finite(v)) {
        v = 0.0;
    }
    u = clamp(u, -1.0e6, 1.0e6);
    v = clamp(v, -1.0e6, 1.0e6);
    let x = u * f32(lv.w);
    let y = v * f32(lv.h);
    if (!linear) {
        return fetch(lv, i32(floor(x)), i32(floor(y)), ws, wt);
    }
    let xf = x - 0.5;
    let yf = y - 0.5;
    let x0f = floor(xf);
    let y0f = floor(yf);
    let fx = xf - x0f;
    let fy = yf - y0f;
    let x0 = i32(x0f);
    let y0 = i32(y0f);
    var taps: array<vec4<f32>, 4>;
    for (var t = 0; t < 4; t++) {
        taps[t] = fetch(lv, x0 + (t & 1), y0 + (t >> 1), ws, wt);
    }
    return mix(mix(taps[0], taps[1], fx), mix(taps[2], taps[3], fx), fy);
}

// `PreparedTexture::sample_lod`.
fn sample_lod(desc: u32, uv: vec2<f32>, lod: f32) -> vec4<f32> {
    let levels = tables[desc];
    let flags = tables[desc + 1u];
    let ws = flags & 3u;
    let wt = (flags >> 2u) & 3u;
    let minf = (flags >> 5u) & 7u;
    // 0 Nearest, 1 Linear, 2 NearestMipNearest, 3 LinearMipNearest,
    // 4 NearestMipLinear, 5 LinearMipLinear.
    var lin = minf == 1u || minf == 3u || minf == 5u;
    let max_level = f32(levels - 1u);
    // Up to two levels blended by `f`.
    var l0 = 0u;
    var l1 = 0u;
    var f = 0.0;
    if (!finite(lod) || lod <= 0.0) {
        lin = (flags & 16u) != 0u;
    } else if (minf <= 1u) {
        l0 = 0u;
    } else if (minf <= 3u) {
        l0 = u32(clamp(floor(lod + 0.5), 0.0, max_level));
    } else {
        let l = clamp(lod, 0.0, max_level);
        l0 = u32(floor(l));
        l1 = min(l0 + 1u, levels - 1u);
        f = l - f32(l0);
    }
    var n = 1u;
    if (f > 0.0 && l0 != l1) {
        n = 2u;
    }
    var r: array<vec4<f32>, 2>;
    for (var i = 0u; i < n; i++) {
        var level = l0;
        if (i == 1u) {
            level = l1;
        }
        r[i] = filter_level(tex_level(desc, level), uv, lin, ws, wt);
    }
    if (n == 1u) {
        return r[0];
    }
    return mix(r[0], r[1], f);
}

// A material texture slot: descriptor, uv uvset, KHR_texture_transform.
fn slot_word(m: u32, s: u32) -> u32 {
    return mat_off(m) + M_SLOTS + 8u * s;
}

fn slot_present(m: u32, s: u32, slot: u32) -> bool {
    let o = slot_word(m, s);
    return tables[o] != NONE && has_uv(slot, tables[o + 1u]);
}

fn slot_xform(m: u32, s: u32, uv: vec2<f32>) -> vec2<f32> {
    let o = slot_word(m, s);
    return vec2<f32>(
        tf(o + 2u) * uv.x + tf(o + 3u) * uv.y + tf(o + 4u),
        tf(o + 5u) * uv.x + tf(o + 6u) * uv.y + tf(o + 7u),
    );
}

// LOD mode: `cone_width < 0` = TexLod::Base (level 0).
struct Lod {
    width: f32,
    dir: vec3<f32>,
};

// `TraceScene::lod` (ray-cone footprint).
fn cone_lod(m: u32, s: u32, slot: u32, lod: Lod) -> f32 {
    let o = slot_word(m, s);
    let desc = tables[o];
    let uvset = tables[o + 1u];
    let uv0 = slot_xform(m, s, vertex_uv(slot, uvset, 0u));
    let uv1 = slot_xform(m, s, vertex_uv(slot, uvset, 1u));
    let uv2 = slot_xform(m, s, vertex_uv(slot, uvset, 2u));
    let ta = f32(tables[desc + 2u]) * f32(tables[desc + 3u])
        * abs((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv2.x - uv0.x) * (uv1.y - uv0.y));
    let p0 = tri_pos(slot, 0u);
    let cr = cross(tri_pos(slot, 1u) - p0, tri_pos(slot, 2u) - p0);
    let pa = sqrt(dot(cr, cr));
    if (!(ta > 0.0 && pa > 0.0 && lod.width > 0.0)) {
        return 0.0;
    }
    let n = cr / pa;
    let c = max(abs(dot(n, lod.dir)), 1.0e-3);
    let l = 0.5 * log2(ta / pa) + log2(lod.width / c);
    if (finite(l)) {
        return l;
    }
    return 0.0;
}

fn slot_sample(m: u32, s: u32, slot: u32, b: vec3<f32>, lod: Lod) -> vec4<f32> {
    let o = slot_word(m, s);
    let uvset = tables[o + 1u];
    let uv = vertex_uv(slot, uvset, 0u) * b.x + vertex_uv(slot, uvset, 1u) * b.y
        + vertex_uv(slot, uvset, 2u) * b.z;
    var l = 0.0;
    if (lod.width >= 0.0) {
        l = cone_lod(m, s, slot, lod);
    }
    return sample_lod(tables[o], slot_xform(m, s, uv), l);
}

fn base_lod() -> Lod {
    return Lod(-1.0, vec3<f32>(0.0));
}

// `TraceScene::alpha` (level 0).
fn hit_alpha(slot: u32, b: vec3<f32>) -> f32 {
    let m = tri_material(slot);
    var a = tf(mat_off(m) + 3u);
    if (slot_present(m, S_BASE, slot)) {
        a *= slot_sample(m, S_BASE, slot, b, base_lod()).w;
    }
    if ((attr_flags(slot) & F_COLORS) != 0u) {
        a *= interp4a(slot, 9u, b).w;
    }
    return a;
}

// =====================================================================
// Rays and BVH traversal.
// =====================================================================

struct Ray {
    o: vec3<f32>,
    d: vec3<f32>,
    inv: vec3<f32>,
    kx: u32,
    ky: u32,
    kz: u32,
    shear: vec3<f32>,
    valid: bool,
};

fn make_ray(o: vec3<f32>, d: vec3<f32>) -> Ray {
    var inv: vec3<f32>;
    for (var k = 0u; k < 3u; k++) {
        if (abs(d[k]) < 1.0e-30) {
            inv[k] = sign_of(d[k]) * 1.0e30;
        } else {
            inv[k] = 1.0 / d[k];
        }
    }
    let ad = abs(d);
    var kz = 2u;
    if (ad.x >= ad.y && ad.x >= ad.z) {
        kz = 0u;
    } else if (ad.y >= ad.z) {
        kz = 1u;
    }
    var kx = (kz + 1u) % 3u;
    var ky = (kx + 1u) % 3u;
    if (d[kz] < 0.0) {
        let t = kx;
        kx = ky;
        ky = t;
    }
    var shear = vec3<f32>(0.0);
    if (d[kz] != 0.0) {
        shear = vec3<f32>(d[kx] / d[kz], d[ky] / d[kz], 1.0 / d[kz]);
    }
    let valid = finite3(o) && finite3(d) && (d.x != 0.0 || d.y != 0.0 || d.z != 0.0);
    return Ray(o, d, inv, kx, ky, kz, shear, valid);
}

// Robust slab test (Williams et al. 2005, Ize 2013 far scaling); the
// entry distance, or -1 for a miss.
fn slab(r: Ray, bmin: vec3<f32>, bmax: vec3<f32>, t_max: f32) -> f32 {
    let t0s = (bmin - r.o) * r.inv;
    let t1s = (bmax - r.o) * r.inv;
    let tn = min(t0s, t1s);
    let tf_ = max(t0s, t1s);
    let t0 = max(max(0.0, tn.x), max(tn.y, tn.z));
    let t1 = min(t_max, min(tf_.x, min(tf_.y, tf_.z)) * 1.0000004);
    if (t0 <= t1) {
        return t0;
    }
    return -1.0;
}

struct Hit {
    slot: u32,
    t: f32,
    b: vec3<f32>,
    front: bool,
};

// Woop-Benthin-Wald watertight test, culling disabled. `h.slot ==
// NONE` on a miss.
fn intersect(r: Ray, slot: u32, t_max: f32) -> Hit {
    var h: Hit;
    h.slot = NONE;
    let p0 = tri_pos(slot, 0u);
    let p1 = tri_pos(slot, 1u);
    let p2 = tri_pos(slot, 2u);
    let a = p0 - r.o;
    let b = p1 - r.o;
    let c = p2 - r.o;
    let ax = a[r.kx] - r.shear.x * a[r.kz];
    let ay = a[r.ky] - r.shear.y * a[r.kz];
    let bx = b[r.kx] - r.shear.x * b[r.kz];
    let by = b[r.ky] - r.shear.y * b[r.kz];
    let cx = c[r.kx] - r.shear.x * c[r.kz];
    let cy = c[r.ky] - r.shear.y * c[r.kz];
    let uu = cx * by - cy * bx;
    let vv = ax * cy - ay * cx;
    let ww = bx * ay - by * ax;
    if ((uu < 0.0 || vv < 0.0 || ww < 0.0) && (uu > 0.0 || vv > 0.0 || ww > 0.0)) {
        return h;
    }
    let det = uu + vv + ww;
    if (det == 0.0 || !finite(det)) {
        return h;
    }
    let az = r.shear.z * a[r.kz];
    let bz = r.shear.z * b[r.kz];
    let cz = r.shear.z * c[r.kz];
    let rcp = 1.0 / det;
    let t = (uu * az + vv * bz + ww * cz) * rcp;
    if (!(t >= 0.0 && t <= t_max)) {
        return h;
    }
    let u = vv * rcp;
    let v = ww * rcp;
    h.slot = slot;
    h.t = t;
    h.b = vec3<f32>(1.0 - u - v, u, v);
    h.front = dot(r.d, cross(p1 - p0, p2 - p0)) < 0.0;
    return h;
}

// Any-hit acceptance (MASK, stochastic BLEND, primary culling).
fn accept(h: Hit, s: Sampler, ray_id: u32, primary: bool) -> bool {
    let m = tri_material(h.slot);
    let fl = mat_flags(m);
    if (primary && !h.front && (fl & 4u) == 0u) {
        return false;
    }
    let mode = fl & 3u;
    if (mode == 0u) {
        return true;
    }
    if (mode == 1u) {
        return hit_alpha(h.slot, h.b) >= tf(mat_off(m) + 11u);
    }
    return coin(s, ray_id, tri_global(h.slot)) < hit_alpha(h.slot, h.b);
}

fn node_lo(i: u32) -> vec4<f32> {
    return geom[2u * i];
}

fn node_hi(i: u32) -> vec4<f32> {
    return geom[2u * i + 1u];
}

// Ordered BVH walk (Aila & Laine 2009 shape, near child first, far
// child pushed with its entry distance). `any`: stop at the first
// accepted hit (shadow rays); otherwise return the closest accepted
// one. Candidates on global triangle `skip` are ignored. `slot ==
// NONE` = nothing hit.
fn traverse(r: Ray, t_max_in: f32, s: Sampler, ray_id: u32, primary: bool, any: bool, skip: u32) -> Hit {
    var best: Hit;
    best.slot = NONE;
    if (P.env.w == 0u || !r.valid) {
        return best;
    }
    var t_max = t_max_in;
    if (slab(r, node_lo(0u).xyz, node_hi(0u).xyz, t_max) < 0.0) {
        return best;
    }
    var stack: array<u32, 64>;
    var stack_t: array<f32, 64>;
    var sp = 0u;
    var idx = 0u;
    loop {
        let lo = node_lo(idx);
        let hi = node_hi(idx);
        let count = bitcast<u32>(hi.w);
        let first = bitcast<u32>(lo.w);
        var descended = false;
        if (count > 0u) {
            var done = false;
            for (var slot = first; slot < first + count; slot++) {
                let h = intersect(r, slot, t_max);
                if (h.slot != NONE && tri_global(slot) != skip && accept(h, s, ray_id, primary)) {
                    best = h;
                    if (any) {
                        done = true;
                        break;
                    }
                    t_max = h.t;
                }
            }
            if (done) {
                break;
            }
        } else {
            let l = first;
            let tl = slab(r, node_lo(l).xyz, node_hi(l).xyz, t_max);
            let tr = slab(r, node_lo(l + 1u).xyz, node_hi(l + 1u).xyz, t_max);
            if (tl >= 0.0 && tr >= 0.0) {
                var near = l;
                var far = l + 1u;
                var t_far = tr;
                if (tl > tr) {
                    near = l + 1u;
                    far = l;
                    t_far = tl;
                }
                if (sp < 64u) {
                    stack[sp] = far;
                    stack_t[sp] = t_far;
                    sp += 1u;
                }
                idx = near;
                descended = true;
            } else if (tl >= 0.0) {
                idx = l;
                descended = true;
            } else if (tr >= 0.0) {
                idx = l + 1u;
                descended = true;
            }
        }
        if (descended) {
            continue;
        }
        var found = false;
        loop {
            if (sp == 0u) {
                break;
            }
            sp -= 1u;
            if (stack_t[sp] <= t_max) {
                idx = stack[sp];
                found = true;
                break;
            }
        }
        if (!found) {
            break;
        }
    }
    return best;
}

// Wächter & Binder 2019 ray-origin offset (`trace::offset_ray_origin`).
fn offset_ray_origin(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var out: vec3<f32>;
    for (var k = 0u; k < 3u; k++) {
        let of_i = i32(256.0 * n[k]);
        let bits = bitcast<i32>(p[k]);
        var moved: i32;
        if (p[k] < 0.0) {
            moved = bits - of_i;
        } else {
            moved = bits + of_i;
        }
        let p_i = bitcast<f32>(moved);
        if (abs(p[k]) < 1.0 / 32.0) {
            out[k] = p[k] + (1.0 / 65536.0) * n[k];
        } else if (finite(p_i)) {
            out[k] = p_i;
        } else {
            out[k] = p[k];
        }
    }
    return out;
}

// =====================================================================
// Surface + material (`TraceScene::surface` / `material_oriented`).
// =====================================================================

struct Surface {
    position: vec3<f32>,
    ng: vec3<f32>,
    sn: vec3<f32>,
};

fn surface(h: Hit) -> Surface {
    let p0 = tri_pos(h.slot, 0u);
    let p1 = tri_pos(h.slot, 1u);
    let p2 = tri_pos(h.slot, 2u);
    let b = h.b;
    var s: Surface;
    s.position = p0 * b.x + p1 * b.y + p2 * b.z;
    let cr = cross(p1 - p0, p2 - p0);
    let len = sqrt(dot(cr, cr));
    if (len > 0.0) {
        s.ng = cr / len;
    } else {
        s.ng = vec3<f32>(0.0, 0.0, 1.0);
    }
    s.sn = s.ng;
    if ((attr_flags(h.slot) & F_NORMALS) != 0u) {
        let n = nrm(interp3a(h.slot, 0u, b));
        if (dot(n, n) > 0.5) {
            s.sn = n;
        }
    }
    return s;
}

struct Mat {
    base: vec4<f32>,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,
    unlit: bool,
    // Le at level 0 (factor × texture; 0 for unlit / black factor).
    emission: vec3<f32>,
    ior: f32,
    specular: f32,
    specular_color: vec3<f32>,
    transmission: f32,
    thickness: f32,
    atten_color: vec3<f32>,
    // < 0 = infinite
    atten_dist: f32,
    clearcoat: f32,
    cc_rough: f32,
    cc_normal: vec3<f32>,
    sheen: vec3<f32>,
    sheen_rough: f32,
};

fn perturb(slot: u32, b: vec3<f32>, n: vec3<f32>, t: vec4<f32>, scale: f32) -> vec3<f32> {
    if ((attr_flags(slot) & F_TANGENTS) == 0u) {
        return n;
    }
    let tg = interp4a(slot, 3u, b);
    var w = 1.0;
    if (attr(slot, 3u).w < 0.0) {
        w = -1.0;
    }
    let t3 = tg.xyz;
    let d = dot(n, t3);
    let tt = nrm(t3 - n * d);
    if (dot(tt, tt) < 0.5) {
        return n;
    }
    let bt = cross(n, tt) * w;
    let ts = vec3<f32>((t.x * 2.0 - 1.0) * scale, (t.y * 2.0 - 1.0) * scale, t.z * 2.0 - 1.0);
    let nn = nrm(tt * ts.x + bt * ts.y + n * ts.z);
    if (dot(nn, nn) > 0.5) {
        return nn;
    }
    return n;
}

fn material(h: Hit, surf: Surface, lod: Lod, flip: bool) -> Mat {
    var sn = surf.sn;
    if (flip) {
        sn = -sn;
    }
    let m = tri_material(h.slot);
    let o = mat_off(m);
    let slot = h.slot;
    let b = h.b;
    // Every present texture slot, sampled once (absent = white, a
    // no-op factor). The emissive slot is read at level 0 — it feeds
    // `emission`, which the CPU samples at level 0 even for camera
    // hits.
    var tex: array<vec4<f32>, 13>;
    var present = 0u;
    for (var si = 0u; si < 13u; si++) {
        tex[si] = vec4<f32>(1.0);
        if (slot_present(m, si, slot)) {
            present |= 1u << si;
            var l = lod;
            if (si == S_EMISSIVE) {
                l = base_lod();
            }
            tex[si] = slot_sample(m, si, slot, b, l);
        }
    }
    var r: Mat;
    var col = tv4(o) * tex[S_BASE];
    if ((attr_flags(slot) & F_COLORS) != 0u) {
        col *= interp4a(slot, 9u, b);
    }
    r.base = col;
    r.metallic = clamp(tf(o + 8u) * tex[S_MR].z, 0.0, 1.0);
    r.roughness = clamp(tf(o + 9u) * tex[S_MR].y, 0.0, 1.0);
    r.normal = sn;
    if ((present & (1u << S_NORMAL)) != 0u) {
        r.normal = perturb(slot, b, sn, tex[S_NORMAL], tf(o + 10u));
    }
    r.unlit = (tables[o + M_FLAGS] & 8u) != 0u;
    r.emission = vec3<f32>(0.0);
    let e = tv3(o + 4u);
    if (!r.unlit && !is_black(e)) {
        r.emission = e * tex[S_EMISSIVE].xyz;
    }
    r.ior = tf(o + 12u);
    r.specular = tf(o + 13u) * tex[S_SPEC].w;
    r.specular_color = tv3(o + 14u) * tex[S_SPEC_COLOR].xyz;
    r.transmission = tf(o + 17u) * tex[S_TRANSMISSION].x;
    r.thickness = tf(o + 18u) * tex[S_THICKNESS].y;
    r.atten_color = tv3(o + 19u);
    r.atten_dist = tf(o + 22u);
    r.clearcoat = tf(o + 23u) * tex[S_CC].x;
    r.cc_rough = tf(o + 24u) * tex[S_CC_ROUGH].y;
    r.cc_normal = sn;
    if ((present & (1u << S_CC_NORMAL)) != 0u) {
        r.cc_normal = perturb(slot, b, sn, tex[S_CC_NORMAL], tf(o + 25u));
    }
    r.sheen = tv3(o + 26u) * tex[S_SHEEN_COLOR].xyz;
    r.sheen_rough = tf(o + 29u) * tex[S_SHEEN_ROUGH].w;
    return r;
}

// Emitted radiance of a hit (`Integrator::emission`, level 0).
fn emission(h: Hit) -> vec3<f32> {
    let m = tri_material(h.slot);
    let o = mat_off(m);
    let e = tv3(o + 4u);
    if ((tables[o + M_FLAGS] & 8u) != 0u || is_black(e)) {
        return vec3<f32>(0.0);
    }
    if (slot_present(m, S_EMISSIVE, h.slot)) {
        return e * slot_sample(m, S_EMISSIVE, h.slot, h.b, base_lod()).xyz;
    }
    return e;
}

// =====================================================================
// §4 BSDF.
// =====================================================================

fn d_ggx(n_dot_h: f32, a: f32) -> f32 {
    if (n_dot_h <= 0.0) {
        return 0.0;
    }
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn lambda(c_in: f32, a: f32) -> f32 {
    let c = max(abs(c_in), 1.0e-7);
    let a2 = a * a;
    return (sqrt(a2 + (1.0 - a2) * c * c) / c - 1.0) * 0.5;
}

fn g1(c: f32, a: f32) -> f32 {
    return 1.0 / (1.0 + lambda(c, a));
}

fn g2(cv: f32, cl: f32, a: f32) -> f32 {
    return 1.0 / (1.0 + lambda(cv, a) + lambda(cl, a));
}

fn schlick(f0: vec3<f32>, c: f32) -> vec3<f32> {
    let x = 1.0 - clamp(c, 0.0, 1.0);
    let k = x * x * x * x * x;
    return f0 + (vec3<f32>(1.0) - f0) * k;
}

fn sample_vndf(v: vec3<f32>, a: f32, u0: f32, u1: f32) -> vec3<f32> {
    let vh = nrm(vec3<f32>(a * v.x, a * v.y, v.z));
    let lensq = vh.x * vh.x + vh.y * vh.y;
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if (lensq > 0.0) {
        let il = 1.0 / sqrt(lensq);
        t1 = vec3<f32>(-vh.y * il, vh.x * il, 0.0);
    }
    let t2 = cross(vh, t1);
    let r = sqrt(u0);
    let phi = 2.0 * PI * u1;
    let p1 = r * cos(phi);
    var p2 = r * sin(phi);
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * p2;
    let pz = sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0));
    let nh = t1 * p1 + t2 * p2 + vh * pz;
    return nrm(vec3<f32>(a * nh.x, a * nh.y, max(nh.z, 1.0e-6)));
}

fn vndf_reflect_pdf(n: vec3<f32>, v: vec3<f32>, h: vec3<f32>, a: f32) -> f32 {
    let nv = dot(n, v);
    let nh = dot(n, h);
    if (nv <= 0.0 || nh <= 0.0 || dot(v, h) <= 0.0) {
        return 0.0;
    }
    return g1(nv, a) * d_ggx(nh, a) / (4.0 * nv);
}

fn reflect_h(v: vec3<f32>, h: vec3<f32>) -> vec3<f32> {
    return 2.0 * dot(v, h) * h - v;
}

fn d_charlie(n_dot_h: f32, a: f32) -> f32 {
    let inv = 1.0 / max(a, 1.0e-3);
    let sin2 = max(1.0 - n_dot_h * n_dot_h, 0.0);
    if (sin2 <= 0.0) {
        return 0.0;
    }
    return (2.0 + inv) * pow(sin2, inv * 0.5) / (2.0 * PI);
}

fn v_neubelt(nl: f32, nv: f32) -> f32 {
    return 1.0 / max(4.0 * (nl + nv - nl * nv), 1.0e-6);
}

fn sheen_albedo(c: f32, rough: f32) -> f32 {
    let x = clamp(clamp(c, 0.0, 1.0) * f32(SHEEN_COS) - 0.5, 0.0, f32(SHEEN_COS - 1u));
    let y = clamp(clamp(rough, 0.0, 1.0) * f32(SHEEN_ROUGH - 1u), 0.0, f32(SHEEN_ROUGH - 1u));
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(x0 + 1u, SHEEN_COS - 1u);
    let y1 = min(y0 + 1u, SHEEN_ROUGH - 1u);
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let a00 = wg_sheen[y0 * SHEEN_COS + x0];
    let a10 = wg_sheen[y0 * SHEEN_COS + x1];
    let a01 = wg_sheen[y1 * SHEEN_COS + x0];
    let a11 = wg_sheen[y1 * SHEEN_COS + x1];
    let a = a00 + (a10 - a00) * fx;
    let bb = a01 + (a11 - a01) * fx;
    return a + (bb - a) * fy;
}

fn basis_t(n: vec3<f32>) -> vec3<f32> {
    let sign = sign_of(n.z);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
}

fn basis_b(n: vec3<f32>) -> vec3<f32> {
    let sign = sign_of(n.z);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(b, sign + n.y * n.y * a, -n.y);
}

fn cosine_hemisphere(u0: f32, u1: f32) -> vec3<f32> {
    let a = 2.0 * u0 - 1.0;
    let b = 2.0 * u1 - 1.0;
    var r = 0.0;
    var phi = 0.0;
    if (a == 0.0 && b == 0.0) {
        r = 0.0;
        phi = 0.0;
    } else if (abs(a) > abs(b)) {
        r = a;
        phi = (PI / 4.0) * (b / a);
    } else {
        r = b;
        phi = PI / 2.0 - (PI / 4.0) * (a / b);
    }
    let x = r * cos(phi);
    let y = r * sin(phi);
    return vec3<f32>(x, y, sqrt(max(1.0 - x * x - y * y, 0.0)));
}

struct Bsdf {
    v: vec3<f32>,
    n: vec3<f32>,
    ng: vec3<f32>,
    ncc: vec3<f32>,
    base: vec3<f32>,
    metallic: f32,
    alpha: f32,
    specular: f32,
    f0d: vec3<f32>,
    transmission: f32,
    thin: bool,
    eta_i: f32,
    eta_t: f32,
    clearcoat: f32,
    cc_alpha: f32,
    sheen: vec3<f32>,
    sheen_rough: f32,
    sheen_alpha: f32,
    w_cc: f32,
    // diffuse, specular, transmission, clearcoat
    p: vec4<f32>,
    // sheen
    p4: f32,
};

fn fresnel_d(bs: Bsdf, c: f32) -> vec3<f32> {
    if (bs.eta_i > bs.eta_t) {
        let eta = bs.eta_i / bs.eta_t;
        let s2 = eta * eta * max(1.0 - c * c, 0.0);
        if (s2 >= 1.0) {
            return vec3<f32>(1.0);
        }
        return schlick(bs.f0d, sqrt(1.0 - s2));
    }
    return schlick(bs.f0d, c);
}

fn make_bsdf(m: Mat, v: vec3<f32>, ng_in: vec3<f32>, front: bool) -> Bsdf {
    var fl = 1.0;
    if (!front) {
        fl = -1.0;
    }
    let ng = ng_in * fl;
    var n = m.normal * fl;
    if (dot(n, v) <= 1.0e-4) {
        n = ng;
    }
    var ncc = m.cc_normal * fl;
    if (dot(ncc, v) <= 1.0e-4) {
        ncc = n;
    }
    let thin = m.thickness <= 0.0;
    var ior = 1.5;
    if (finite(m.ior) && m.ior >= 1.0) {
        ior = m.ior;
    }
    var eta_i = ior;
    var eta_t = 1.0;
    if (thin || front) {
        eta_i = 1.0;
        eta_t = ior;
    }
    let inside = !thin && !front;
    let q = (ior - 1.0) / (ior + 1.0);
    let r0 = q * q;
    var b: Bsdf;
    b.v = v;
    b.n = n;
    b.ng = ng;
    b.ncc = ncc;
    b.base = m.base.xyz;
    b.metallic = m.metallic;
    b.alpha = max(m.roughness * m.roughness, MIN_ALPHA);
    b.specular = clamp(m.specular, 0.0, 1.0);
    b.f0d = min(r0 * m.specular_color, vec3<f32>(1.0));
    b.transmission = m.transmission;
    b.thin = thin;
    b.eta_i = eta_i;
    b.eta_t = eta_t;
    b.clearcoat = m.clearcoat;
    b.sheen = m.sheen;
    if (inside) {
        b.clearcoat = 0.0;
        b.sheen = vec3<f32>(0.0);
    }
    b.cc_alpha = max(m.cc_rough * m.cc_rough, MIN_ALPHA);
    b.sheen_rough = m.sheen_rough;
    b.sheen_alpha = max(m.sheen_rough * m.sheen_rough, 1.0e-3);
    let cv = max(dot(n, v), 1.0e-4);
    b.w_cc = 0.0;
    if (b.clearcoat > 0.0) {
        b.w_cc = b.clearcoat * schlick(vec3<f32>(0.04), dot(ncc, v)).x;
    }
    let sheen_max = max3(b.sheen);
    var e_v = 0.0;
    if (sheen_max > 0.0) {
        e_v = sheen_albedo(cv, b.sheen_rough);
    }
    let s_scale = (1.0 - b.w_cc) * (1.0 - sheen_max * e_v);
    let fd = b.specular * max3(fresnel_d(b, cv));
    let fm = mean3(schlick(b.base, cv));
    let mm = b.metallic;
    let base_mean = max(mean3(b.base), 0.0);
    var p = vec4<f32>(
        s_scale * (1.0 - mm) * (1.0 - b.transmission) * (1.0 - fd) * base_mean,
        s_scale * ((1.0 - mm) * fd + mm * fm),
        s_scale * (1.0 - mm) * b.transmission * (1.0 - fd) * base_mean,
        b.w_cc,
    );
    var p4 = (1.0 - b.w_cc) * sheen_max * e_v;
    let sum = p.x + p.y + p.z + p.w + p4;
    if (sum > 0.0 && finite(sum)) {
        p = p / sum;
        p4 = p4 / sum;
    } else {
        p = vec4<f32>(0.0);
        p4 = 0.0;
    }
    b.p = p;
    b.p4 = p4;
    return b;
}

fn bsdf_black(b: Bsdf) -> bool {
    return b.p.x <= 0.0 && b.p.y <= 0.0 && b.p.z <= 0.0 && b.p.w <= 0.0 && b.p4 <= 0.0;
}

fn lobe_p(b: Bsdf, i: u32) -> f32 {
    if (i == 4u) {
        return b.p4;
    }
    return b.p[i];
}

struct Eval {
    f: vec3<f32>,
    pdf: f32,
};

// `f(v, l)·|n·l|` and the mixture pdf of `l`.
fn bsdf_eval(b: Bsdf, l: vec3<f32>) -> Eval {
    let v = b.v;
    let n = b.n;
    let nv = max(dot(n, v), 1.0e-4);
    if (dot(b.ng, l) > 0.0) {
        let nl = dot(n, l);
        let h = nrm(v + l);
        let vh = max(dot(v, h), 0.0);
        let nh = dot(n, h);
        var f = vec3<f32>(0.0);
        var pdf = 0.0;
        if (nl > 0.0) {
            let fd = fresnel_d(b, vh);
            let fm = schlick(b.base, vh);
            let spec_c = d_ggx(nh, b.alpha) * g2(nv, nl, b.alpha) / (4.0 * nv * nl);
            let m = b.metallic;
            let diff_k = (1.0 - m) * (1.0 - b.transmission) * (1.0 - b.specular * max3(fd)) / PI;
            var layer = spec_c * ((1.0 - m) * b.specular * fd + m * fm) + diff_k * b.base;
            let smax = max3(b.sheen);
            if (smax > 0.0) {
                let scale = min(
                    1.0 - smax * sheen_albedo(nv, b.sheen_rough),
                    1.0 - smax * sheen_albedo(nl, b.sheen_rough),
                );
                let sh = d_charlie(nh, b.sheen_alpha) * v_neubelt(nl, nv);
                layer = layer * scale + b.sheen * sh;
            }
            f = layer * (1.0 - b.w_cc);
            pdf += (b.p.x + b.p4) * nl / PI;
            pdf += b.p.y * vndf_reflect_pdf(n, v, h, b.alpha);
        }
        if (b.clearcoat > 0.0) {
            let cnv = max(dot(b.ncc, v), 1.0e-4);
            let cnl = dot(b.ncc, l);
            if (cnl > 0.0) {
                let cnh = dot(b.ncc, h);
                let c = b.w_cc * d_ggx(cnh, b.cc_alpha) * g2(cnv, cnl, b.cc_alpha) / (4.0 * cnv * cnl);
                f += vec3<f32>(c);
                pdf += b.p.w * vndf_reflect_pdf(b.ncc, v, h, b.cc_alpha);
            }
        }
        return Eval(f * max(nl, 0.0), pdf);
    }
    // Transmission.
    if (b.transmission <= 0.0 || b.metallic >= 1.0) {
        return Eval(vec3<f32>(0.0), 0.0);
    }
    let nl = -dot(n, l);
    if (nl <= 0.0) {
        return Eval(vec3<f32>(0.0), 0.0);
    }
    let smax = max3(b.sheen);
    var sheen_scale = 1.0;
    if (smax > 0.0) {
        sheen_scale = 1.0 - smax * sheen_albedo(nv, b.sheen_rough);
    }
    let w = (1.0 - b.metallic) * b.transmission * (1.0 - b.w_cc) * sheen_scale;
    var val = 0.0;
    var pdf_t = 0.0;
    if (b.thin) {
        let lm = l + n * (2.0 * nl);
        let h = nrm(v + lm);
        let vh = dot(v, h);
        let nh = dot(n, h);
        if (vh <= 0.0 || nh <= 0.0) {
            return Eval(vec3<f32>(0.0), 0.0);
        }
        let fd = b.specular * max3(fresnel_d(b, vh));
        val = d_ggx(nh, b.alpha) * g2(nv, nl, b.alpha) / (4.0 * nv * nl) * (1.0 - fd);
        pdf_t = vndf_reflect_pdf(n, v, h, b.alpha);
    } else {
        let ei = b.eta_i;
        let et = b.eta_t;
        var h = nrm(-(ei * v + et * l));
        if (dot(h, n) < 0.0) {
            h = -h;
        }
        let vh = dot(v, h);
        let lh = dot(l, h);
        let nh = dot(n, h);
        if (vh <= 0.0 || lh >= 0.0 || nh <= 0.0) {
            return Eval(vec3<f32>(0.0), 0.0);
        }
        let denom = ei * vh + et * lh;
        if (abs(denom) < 1.0e-6) {
            return Eval(vec3<f32>(0.0), 0.0);
        }
        let d = d_ggx(nh, b.alpha);
        let fd = b.specular * max3(fresnel_d(b, vh));
        let jac = et * et * abs(lh) / (denom * denom);
        val = vh * abs(lh) * d * g2(nv, nl, b.alpha) * et * et * (1.0 - fd) / (nv * nl * denom * denom);
        pdf_t = g1(nv, b.alpha) * vh * d / nv * jac;
    }
    return Eval(b.base * (w * val * nl), b.p.z * pdf_t);
}

struct BSample {
    ok: bool,
    l: vec3<f32>,
    f: vec3<f32>,
    pdf: f32,
};

fn refract_h(v: vec3<f32>, h: vec3<f32>, eta: f32, out: ptr<function, vec3<f32>>) -> bool {
    let ci = dot(v, h);
    let s2 = eta * eta * max(1.0 - ci * ci, 0.0);
    if (s2 >= 1.0) {
        return false;
    }
    let ct = sqrt(1.0 - s2);
    let k = eta * ci - ct;
    *out = nrm(-eta * v + k * h);
    return true;
}

fn bsdf_sample(b: Bsdf, u: vec3<f32>) -> BSample {
    var res: BSample;
    res.ok = false;
    var acc = 0.0;
    var lobe = NONE;
    for (var i = 0u; i < 5u; i++) {
        let p = lobe_p(b, i);
        acc += p;
        if (p > 0.0 && u.z < acc) {
            lobe = i;
            break;
        }
    }
    if (lobe == NONE) {
        for (var i = 0u; i < 5u; i++) {
            if (lobe_p(b, i) > 0.0) {
                lobe = i;
            }
        }
        if (lobe == NONE) {
            return res;
        }
    }
    let v = b.v;
    var l: vec3<f32>;
    if (lobe == 0u || lobe == 4u) {
        let t = basis_t(b.n);
        let bb = basis_b(b.n);
        let c = cosine_hemisphere(u.x, u.y);
        l = t * c.x + bb * c.y + b.n * c.z;
    } else {
        var nn = b.n;
        var a = b.alpha;
        if (lobe == 3u) {
            nn = b.ncc;
            a = b.cc_alpha;
        }
        let t = basis_t(nn);
        let bb = basis_b(nn);
        let vl = vec3<f32>(dot(v, t), dot(v, bb), dot(v, nn));
        if (vl.z <= 0.0) {
            return res;
        }
        let hl = sample_vndf(vl, a, u.x, u.y);
        let h = t * hl.x + bb * hl.y + nn * hl.z;
        if (lobe != 2u) {
            l = reflect_h(v, h);
        } else if (b.thin) {
            let r = reflect_h(v, h);
            let d = dot(r, b.n);
            l = r + b.n * (-2.0 * d);
        } else {
            var rl: vec3<f32>;
            if (!refract_h(v, h, b.eta_i / b.eta_t, &rl)) {
                return res;
            }
            l = rl;
        }
    }
    if (!finite3(l) || dot(l, l) < 0.5) {
        return res;
    }
    let reflection = dot(b.ng, l) > 0.0;
    if (reflection == (lobe == 2u)) {
        return res;
    }
    let e = bsdf_eval(b, l);
    if (e.pdf <= 0.0 || !finite(e.pdf) || is_black(e.f) || !finite3(e.f)) {
        return res;
    }
    res.ok = true;
    res.l = l;
    res.f = e.f;
    res.pdf = e.pdf;
    return res;
}

// =====================================================================
// Lights.
// =====================================================================

struct LightSample {
    ok: bool,
    l: vec3<f32>,
    // < 0 = infinite (directional)
    distance: f32,
    radiance: vec3<f32>,
};

// `PreparedLight::sample`. Layout (16 words): position + kind,
// direction + range (0 = none), colour × intensity, spot scale /
// offset.
fn punctual_sample(j: u32, p: vec3<f32>) -> LightSample {
    let o = P.offs0.w + 16u * j;
    var r: LightSample;
    r.ok = false;
    let kind = tables[o + 3u];
    let base = tv3(o + 8u);
    let dir = tv3(o + 4u);
    if (kind == 0u) {
        r.ok = true;
        r.l = -dir;
        r.distance = -1.0;
        r.radiance = base;
        return r;
    }
    let to = tv3(o) - p;
    let d2 = dot(to, to);
    if (!(d2 > 1.0e-12)) {
        return r;
    }
    let d = sqrt(d2);
    let l = to / d;
    var att = 1.0 / d2;
    let range = tf(o + 7u);
    if (range > 0.0) {
        let x = d / range;
        att *= clamp(1.0 - x * x * x * x, 0.0, 1.0);
    }
    if (kind == 2u) {
        let cd = dot(dir, -l);
        let a = clamp(cd * tf(o + 12u) + tf(o + 13u), 0.0, 1.0);
        att *= a * a;
    }
    if (att <= 0.0) {
        return r;
    }
    r.ok = true;
    r.l = l;
    r.distance = d;
    r.radiance = base * att;
    return r;
}

// partition_point(c <= u) over `n` f32 words at `o` (ascending).
fn partition_point(o: u32, n: u32, u: f32) -> u32 {
    var lo = 0u;
    var hi = n;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2u;
        if (tf(o + mid) <= u) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Emissive light `li`: cdf, pmf, global id, slot.
fn elight_pmf(li: u32) -> f32 {
    return tf(P.offs1.x + 4u * li + 1u);
}

fn elight_slot(li: u32) -> u32 {
    return tables[P.offs1.x + 4u * li + 3u];
}

fn pick_elight(u: f32) -> u32 {
    let n = P.cfg.w;
    // The CDF occupies word 0 of each 4-word record; binary search
    // over records.
    var lo = 0u;
    var hi = n;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2u;
        if (tf(P.offs1.x + 4u * mid) <= u) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return min(lo, n - 1u);
}

fn triangle_solid_angle(p: vec3<f32>, a0: vec3<f32>, b0: vec3<f32>, c0: vec3<f32>) -> f32 {
    let a = nrm_any(a0 - p);
    let b = nrm_any(b0 - p);
    let c = nrm_any(c0 - p);
    let num = abs(dot(a, cross(b, c)));
    let den = 1.0 + dot(a, b) + dot(b, c) + dot(c, a);
    let o = 2.0 * atan2(num, den);
    if (finite(o)) {
        return max(o, 0.0);
    }
    return 0.0;
}

// Normalise without the epsilon cut-off (only exact zero → zero).
fn nrm_any(v: vec3<f32>) -> vec3<f32> {
    let l = sqrt(dot(v, v));
    if (l > 0.0) {
        return v / l;
    }
    return vec3<f32>(0.0);
}

// Arvo 1995: uniform direction over the spherical triangle. Returns
// zero on degeneracy.
fn sample_spherical_triangle(p: vec3<f32>, a0: vec3<f32>, b0: vec3<f32>, c0: vec3<f32>, u0: f32, u1: f32) -> vec3<f32> {
    let zero = vec3<f32>(0.0);
    let a = nrm_any(a0 - p);
    let b = nrm_any(b0 - p);
    let c = nrm_any(c0 - p);
    let nab = nrm_any(cross(a, b));
    let nbc = nrm_any(cross(b, c));
    let nca = nrm_any(cross(c, a));
    if (dot(a, a) == 0.0 || dot(b, b) == 0.0 || dot(c, c) == 0.0
        || dot(nab, nab) == 0.0 || dot(nbc, nbc) == 0.0 || dot(nca, nca) == 0.0) {
        return zero;
    }
    let alpha = acos(clamp(-dot(nab, nca), -1.0, 1.0));
    let beta = acos(clamp(-dot(nbc, nab), -1.0, 1.0));
    let gamma = acos(clamp(-dot(nca, nbc), -1.0, 1.0));
    let area = alpha + beta + gamma - PI;
    if (!(area > 0.0)) {
        return zero;
    }
    let ap = u0 * area;
    let s = sin(ap - alpha);
    let t = cos(ap - alpha);
    let sa = sin(alpha);
    let ca = cos(alpha);
    let cos_c = dot(a, b);
    let uu = t - ca;
    let vv = s + sa * cos_c;
    let den = (vv * s + uu * t) * sa;
    if (abs(den) < 1.0e-30) {
        return zero;
    }
    let q = clamp(((vv * t - uu * s) * ca - vv) / den, -1.0, 1.0);
    let ca_perp = nrm_any(c - dot(c, a) * a);
    if (dot(ca_perp, ca_perp) == 0.0) {
        return zero;
    }
    let r = sqrt(max(1.0 - q * q, 0.0));
    let cp = q * a + r * ca_perp;
    let z = 1.0 - u1 * (1.0 - dot(cp, b));
    let cb_perp = nrm_any(cp - dot(cp, b) * b);
    let w = sqrt(max(1.0 - z * z, 0.0));
    var d = b;
    if (dot(cb_perp, cb_perp) > 0.0) {
        d = z * b + w * cb_perp;
    }
    return nrm_any(d);
}

fn p_env() -> f32 {
    return P.fparams.z;
}

// `Integrator::tri_pdf`.
fn tri_pdf(li: u32, slot: u32, p: vec3<f32>, q: vec3<f32>) -> f32 {
    let a = tri_pos(slot, 0u);
    let b = tri_pos(slot, 1u);
    let c = tri_pos(slot, 2u);
    let sel = (1.0 - p_env()) * elight_pmf(li);
    let omega = triangle_solid_angle(p, a, b, c);
    if (omega >= MIN_SPHERICAL_SOLID_ANGLE) {
        return sel / omega;
    }
    let cr = cross(b - a, c - a);
    let len = sqrt(dot(cr, cr));
    let to = q - p;
    let d2 = dot(to, to);
    if (len <= 0.0 || d2 <= 0.0) {
        return 0.0;
    }
    let cs = abs(dot(cr, to) / (len * sqrt(d2)));
    if (cs <= 0.0) {
        return 0.0;
    }
    return sel * d2 / (0.5 * len * cs);
}

// ---- Environment map (§5).

struct Texel {
    x: u32,
    y: u32,
    st: f32,
};

fn env_texel(d: vec3<f32>) -> Texel {
    let w = P.env.x;
    let h = P.env.y;
    let u = 0.5 + atan2(d.x, -d.z) / (2.0 * PI);
    let th = acos(clamp(d.y, -1.0, 1.0));
    let v = th / PI;
    let x = min(u32(max(u * f32(w), 0.0)), w - 1u);
    let y = min(u32(max(v * f32(h), 0.0)), h - 1u);
    return Texel(x, y, sin(th));
}

fn env_radiance(d: vec3<f32>) -> vec3<f32> {
    let t = env_texel(d);
    return tv3(P.offs2.y + 3u * (t.y * P.env.x + t.x));
}

fn env_pdf(d: vec3<f32>) -> f32 {
    let t = env_texel(d);
    if (t.st <= 1.0e-6) {
        return 0.0;
    }
    let w = P.env.x;
    let h = P.env.y;
    return tf(P.offs2.x + t.y * w + t.x) * f32(w * h) / (2.0 * PI * PI * t.st);
}

struct EnvSample {
    ok: bool,
    d: vec3<f32>,
    le: vec3<f32>,
    pdf: f32,
};

fn env_sample(u0: f32, u1: f32) -> EnvSample {
    var r: EnvSample;
    r.ok = false;
    let w = P.env.x;
    let h = P.env.y;
    let marg = P.offs1.z;
    let y = clamp(partition_point(marg, h + 1u, u0), 1u, h) - 1u;
    let row = P.offs1.w + y * (w + 1u);
    let x = clamp(partition_point(row, w + 1u, u1), 1u, w) - 1u;
    let my = (u0 - tf(marg + y)) / max(tf(marg + y + 1u) - tf(marg + y), 1.0e-12);
    let mx = (u1 - tf(row + x)) / max(tf(row + x + 1u) - tf(row + x), 1.0e-12);
    let u = (f32(x) + clamp(mx, 0.0, 0.99999)) / f32(w);
    let v = (f32(y) + clamp(my, 0.0, 0.99999)) / f32(h);
    let phi = 2.0 * PI * (u - 0.5);
    let th = PI * v;
    let st = sin(th);
    let ct = cos(th);
    let d = vec3<f32>(st * sin(phi), ct, -st * cos(phi));
    let pdf = env_pdf(d);
    if (pdf <= 0.0 || !finite(pdf)) {
        return r;
    }
    r.ok = true;
    r.d = d;
    r.le = env_radiance(d);
    r.pdf = pdf;
    return r;
}

// =====================================================================
// §3 Integrator.
// =====================================================================

fn power_heuristic(a: f32, b: f32) -> f32 {
    let a2 = a * a;
    let b2 = b * b;
    if (a2 + b2 > 0.0 && finite(a2 + b2)) {
        return a2 / (a2 + b2);
    }
    if (!finite(a) && a > 0.0) {
        return 1.0;
    }
    return 0.0;
}

fn clamp_c(c: vec3<f32>) -> vec3<f32> {
    let m = max3(c);
    let cl = P.fparams.y;
    if (cl > 0.0 && m > cl) {
        return c * (cl / m);
    }
    return c;
}

fn spawn(p: vec3<f32>, ng: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    if (dot(dir, ng) >= 0.0) {
        return offset_ray_origin(p, ng);
    }
    return offset_ray_origin(p, -ng);
}

// Light-sampling pdf of reaching emissive hit `h` (at `q`) from `p`.
fn light_pdf(h: Hit, p: vec3<f32>, q: vec3<f32>) -> f32 {
    let li = tri_light(h.slot);
    if (li == NONE) {
        return 0.0;
    }
    return tri_pdf(li, h.slot, p, q);
}

struct PathResult {
    covered: bool,
    radiance: vec3<f32>,
};

// One camera sample. The path is driven as a per-thread state machine
// so the kernel holds a single traversal (and a single BSDF / texture
// evaluation) call site — every loop iteration traces exactly one ray:
//
// * PATH: the continuation ray (camera ray at k = 0); its hit adds
//   emission, builds the BSDF and arms the shadow-ray jobs;
// * SHADOW: one NEE shadow ray (punctual lights j = 0..n−1 in order,
//   then the area / environment sample) whose contribution is added
//   when unoccluded.
//
// When the jobs run out the BSDF is sampled for the next PATH ray.
// Contributions are added in the CPU integrator's order.
fn trace_sample(x: u32, y: u32, index: u32) -> PathResult {
    let s = make_sampler(P.dims.z, x, y, index);
    let p0 = pattern(s, 0u);
    let width = f32(P.dims.x);
    let height = f32(P.dims.y);
    let ndc_x = ((f32(x) + p0.x) / max(width, 1.0)) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((f32(y) + p0.y) / max(height, 1.0)) * 2.0;
    let orthographic = P.eye.w > 0.5;
    let half_w = P.forward.w;
    let half_h = P.side.w;
    var o: vec3<f32>;
    var d: vec3<f32>;
    if (orthographic) {
        o = P.eye.xyz + (P.side.xyz * (ndc_x * half_w) + P.up.xyz * (ndc_y * half_h));
        d = P.forward.xyz;
    } else {
        o = P.eye.xyz;
        d = nrm(P.side.xyz * (ndc_x * half_w) + P.up.xyz * (ndc_y * half_h) + P.forward.xyz);
    }
    var beta = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    var prev_pdf = 0.0;
    var prev_pos = vec3<f32>(0.0);
    var in_medium = false;
    var sigma = vec3<f32>(0.0);
    let strategy = P.cfg.y;
    let has_env = P.env.z != 0u;
    let n_elights = P.cfg.w;
    let n_punct = P.cfg.z;
    var k = 0u;

    // Current ray.
    var shadow = false;
    var r_o = o;
    var r_d = d;
    var r_tmax = INF;
    var r_id = 0u;
    var r_skip = NONE;
    // Pending shadow contribution.
    var pending = vec3<f32>(0.0);
    // Vertex state shared by the jobs and the BSDF sample.
    var bsdf: Bsdf;
    var p = vec3<f32>(0.0);
    var ng = vec3<f32>(0.0);
    var thickness = 0.0;
    var transmission = 0.0;
    var atten_color = vec3<f32>(1.0);
    var atten_dist = -1.0;
    var front = true;
    var pl_u = vec4<f32>(0.0);
    var job = 0u;

    loop {
        let hit = traverse(make_ray(r_o, r_d), r_tmax, s, r_id, !shadow && k == 0u, shadow, r_skip);
        if (shadow) {
            if (hit.slot == NONE) {
                radiance += pending;
            }
            job += 1u;
        } else {
            if (in_medium) {
                if (hit.slot == NONE) {
                    break;
                }
                beta *= exp(-sigma * hit.t);
            }
            if (hit.slot == NONE) {
                if (k == 0u) {
                    return PathResult(false, vec3<f32>(0.0));
                }
                var le = vec3<f32>(P.fparams.x);
                var w = 1.0;
                if (has_env) {
                    le = env_radiance(d);
                    if (strategy == 0u) {
                        w = power_heuristic(prev_pdf, p_env() * env_pdf(d));
                    } else if (strategy == 1u) {
                        w = 0.0;
                    }
                }
                if (w > 0.0) {
                    radiance += clamp_c(beta * le * w);
                }
                break;
            }
            let surf = surface(hit);
            let mflags = mat_flags(tri_material(hit.slot));
            let double_sided = (mflags & 4u) != 0u;
            var lod = base_lod();
            if (k == 0u) {
                var cw = P.up.w;
                if (!orthographic) {
                    cw = P.up.w * hit.t;
                }
                lod = Lod(cw, d);
            }
            let flip = !hit.front && double_sided;
            var mat = material(hit, surf, lod, flip);
            if (flip) {
                mat.normal = -mat.normal;
                mat.cc_normal = -mat.cc_normal;
            }
            // Emission.
            if ((hit.front || double_sided) && !is_black(mat.emission)) {
                let le = mat.emission;
                if (k == 0u) {
                    radiance += beta * le;
                } else {
                    let pl = light_pdf(hit, prev_pos, surf.position);
                    var w = 1.0;
                    if (pl > 0.0) {
                        if (strategy == 0u) {
                            w = power_heuristic(prev_pdf, pl);
                        } else if (strategy == 1u) {
                            w = 0.0;
                        }
                    }
                    if (w > 0.0) {
                        radiance += clamp_c(beta * le * w);
                    }
                }
            }
            if (mat.unlit) {
                let contrib = beta * mat.base.xyz;
                if (k == 0u) {
                    radiance += contrib;
                } else {
                    radiance += clamp_c(contrib);
                }
                break;
            }
            if (k >= P.dims.w) {
                break;
            }
            bsdf = make_bsdf(mat, -d, surf.ng, hit.front);
            if (bsdf_black(bsdf)) {
                break;
            }
            p = surf.position;
            ng = bsdf.ng;
            thickness = mat.thickness;
            transmission = mat.transmission;
            atten_color = mat.atten_color;
            atten_dist = mat.atten_dist;
            front = hit.front;
            pl_u = pattern(s, 2u + 2u * k);
            job = 0u;
        }

        // ---- Next NEE shadow ray (punctual j < n, then the area set).
        var armed = false;
        let ray_id = k << 16u;
        loop {
            if (job > n_punct) {
                break;
            }
            var ok = false;
            var l = vec3<f32>(0.0);
            var tmax = INF;
            var rid = ray_id | 0xffffu;
            var skip = NONE;
            var le = vec3<f32>(0.0);
            // Light pdf (area set); < 0 = punctual (weight 1).
            var pl = -1.0;
            if (job < n_punct) {
                let ls = punctual_sample(job, p);
                if (ls.ok) {
                    ok = true;
                    l = ls.l;
                    if (ls.distance >= 0.0) {
                        tmax = ls.distance * (1.0 - 1.0e-4);
                    }
                    rid = ray_id | 0x8000u | (job & 0x7fffu);
                    le = ls.radiance;
                }
            } else if (strategy != 2u && (has_env || n_elights > 0u)) {
                let pe = p_env();
                if (pl_u.z < pe) {
                    if (has_env) {
                        let es = env_sample(pl_u.x, pl_u.y);
                        if (es.ok && pe * es.pdf > 0.0) {
                            ok = true;
                            l = es.d;
                            le = es.le;
                            pl = pe * es.pdf;
                        }
                    }
                } else if (n_elights > 0u) {
                    var us = pl_u.z;
                    if (pe > 0.0) {
                        us = clamp((pl_u.z - pe) / (1.0 - pe), 0.0, 0.999999);
                    }
                    let li = pick_elight(us);
                    let lslot = elight_slot(li);
                    let a = tri_pos(lslot, 0u);
                    let b = tri_pos(lslot, 1u);
                    let c = tri_pos(lslot, 2u);
                    var has_target = false;
                    var q = vec3<f32>(0.0);
                    if (triangle_solid_angle(p, a, b, c) >= MIN_SPHERICAL_SOLID_ANGLE) {
                        let sl = sample_spherical_triangle(p, a, b, c, pl_u.x, pl_u.y);
                        if (dot(sl, sl) > 0.0) {
                            let n = cross(b - a, c - a);
                            let dn = dot(sl, n);
                            if (dn != 0.0) {
                                let t = dot(a - p, n) / dn;
                                if (t > 0.0 && finite(t)) {
                                    q = p + sl * t;
                                    has_target = true;
                                }
                            }
                        }
                    } else {
                        let su = sqrt(pl_u.x);
                        let w3 = vec3<f32>(1.0 - su, su * (1.0 - pl_u.y), su * pl_u.y);
                        q = a * w3.x + b * w3.y + c * w3.z;
                        has_target = true;
                    }
                    if (has_target) {
                        // barycentric_of.
                        let e1 = b - a;
                        let e2 = c - a;
                        let rr = q - a;
                        let d11 = dot(e1, e1);
                        let d12 = dot(e1, e2);
                        let d22 = dot(e2, e2);
                        let r1 = dot(rr, e1);
                        let r2 = dot(rr, e2);
                        let det = d11 * d22 - d12 * d12;
                        let to = q - p;
                        let dist2 = dot(to, to);
                        if (finite(det) && abs(det) > 1.17549435e-38 && dist2 > 1.0e-12) {
                            let bu = (d22 * r1 - d12 * r2) / det;
                            let bv = (d11 * r2 - d12 * r1) / det;
                            var lh: Hit;
                            lh.slot = lslot;
                            lh.t = sqrt(dist2);
                            lh.b = clamp(vec3<f32>(1.0 - bu - bv, bu, bv), vec3<f32>(0.0), vec3<f32>(1.0));
                            lh.front = true;
                            let lng = surface(lh).ng;
                            let lds = mat_double_sided(tri_material(lslot));
                            let dist = sqrt(dist2);
                            let ld = to / dist;
                            let cos_l = -dot(lng, ld);
                            let tp = tri_pdf(li, lslot, p, q);
                            if ((cos_l > 0.0 || (lds && cos_l < 0.0)) && tp > 0.0 && finite(tp)) {
                                ok = true;
                                l = ld;
                                le = emission(lh);
                                pl = tp;
                                tmax = max(dist * (1.0 - 1.0e-4), 0.0);
                                skip = tri_global(lslot);
                            }
                        }
                    }
                }
            }
            if (ok) {
                let e = bsdf_eval(bsdf, l);
                if (!is_black(e.f) && !is_black(le)) {
                    if (pl < 0.0) {
                        pending = clamp_c(beta * e.f * le);
                    } else {
                        var w = 1.0;
                        if (strategy == 0u) {
                            w = power_heuristic(pl, e.pdf);
                        }
                        pending = clamp_c(beta * e.f * le * (w / pl));
                    }
                    r_o = spawn(p, ng, l);
                    r_d = l;
                    r_tmax = tmax;
                    r_id = rid;
                    r_skip = skip;
                    armed = true;
                    break;
                }
            }
            job += 1u;
        }
        if (armed) {
            shadow = true;
            continue;
        }

        // ---- BSDF sampling → next PATH ray.
        let pb_u = pattern(s, 1u + 2u * k);
        let bs = bsdf_sample(bsdf, pb_u.xyz);
        if (!bs.ok) {
            break;
        }
        beta *= bs.f * (1.0 / bs.pdf);
        if (!finite3(beta)) {
            break;
        }
        prev_pdf = bs.pdf;
        prev_pos = p;
        if (dot(bs.l, ng) < 0.0 && thickness > 0.0 && transmission > 0.0) {
            if (front) {
                if (atten_dist >= 0.0) {
                    sigma = -log(max(atten_color, vec3<f32>(1.0e-6))) / atten_dist;
                    in_medium = true;
                } else {
                    in_medium = false;
                }
            } else {
                in_medium = false;
            }
        }
        if (k + 1u >= P.cfg.x) {
            let q = clamp(max3(beta), 0.05, 1.0);
            if (q < 1.0) {
                if (pb_u.w >= q) {
                    break;
                }
                beta = beta * (1.0 / q);
            }
        }
        k += 1u;
        d = bs.l;
        shadow = false;
        r_o = spawn(p, ng, bs.l);
        r_d = d;
        r_tmax = INF;
        r_id = k << 16u;
        r_skip = NONE;
    }
    if (finite3(radiance)) {
        return PathResult(true, radiance);
    }
    return PathResult(true, vec3<f32>(0.0));
}

@compute @workgroup_size(8, 8, 1)
fn trace_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    for (var i = li; i < 128u; i += 64u) {
        wg_sobol[i] = tables[P.offs0.x + i];
    }
    for (var i = li; i < 512u; i += 64u) {
        wg_sheen[i] = tf(P.offs0.y + i);
    }
    workgroupBarrier();
    let x = gid.x;
    let y = frame.y0 + gid.y;
    if (x >= P.dims.x || gid.y >= frame.rows || y >= P.dims.y) {
        return;
    }
    let idx = y * P.dims.x + x;
    var acc = accum[idx];
    for (var i = 0u; i < frame.count; i++) {
        let r = trace_sample(x, y, frame.first + i);
        if (r.covered) {
            acc += vec4<f32>(r.radiance, 1.0);
        }
    }
    accum[idx] = acc;
}
