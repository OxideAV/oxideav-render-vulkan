// Resolve pass: one fragment per output pixel. With `cfg.w == 1` the
// pass instead produces the scene-linear HDR image (no exposure, no
// tone map, no encode) into an Rgba32Float target.
// Loads the `aa × aa`
// scene samples behind it, turns each into a display-linear value,
// averages them (premultiplied by alpha) and sRGB-encodes the mean
// into the Rgba8Unorm output. Mirrors oxideav-render's scanline
// resolve:
//
// * uncovered samples are the background bytes, untouched;
// * class 0 (Pbr): premultiplied scene-linear, un-premultiplied, then
//   `tone_map(colour * exposure)`;
// * class 1 (legacy Flat/Gouraud/Phong/Wireframe): straight colour,
//   clamped (no exposure);
// * class 2 (NormalDebug/DepthDebug): values are already display
//   values and pass through.

struct Params {
    // x = aa factor, y = class, z = tone-map operator
    //     (0 clamp, 1 Reinhard, 2 ACES fitted), w = 1 for HDR output
    cfg: vec4<u32>,
    // x = exposure
    exposure: vec4<f32>,
    // background, sRGB-encoded bytes / 255
    background: vec4<f32>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var t_scene: texture_2d<f32>;
@group(0) @binding(2) var t_coverage: texture_2d<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn srgb_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn linear_to_srgb(c: f32) -> f32 {
    let x = clamp(c, 0.0, 1.0);
    if (x <= 0.0031308) {
        return 12.92 * x;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn aces(x: f32) -> f32 {
    let y = x * 0.6;
    return clamp((y * (2.51 * y + 0.03)) / (y * (2.43 * y + 0.59) + 0.14), 0.0, 1.0);
}

fn tone_map(rgb_in: vec3<f32>) -> vec3<f32> {
    let rgb = max(rgb_in, vec3<f32>(0.0));
    switch p.cfg.z {
        case 1u: {
            let l = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            if (l <= 0.0) {
                return vec3<f32>(0.0);
            }
            return min(rgb / (1.0 + l), vec3<f32>(1.0));
        }
        case 2u: {
            return vec3<f32>(aces(rgb.r), aces(rgb.g), aces(rgb.b));
        }
        default: {
            return min(rgb, vec3<f32>(1.0));
        }
    }
}

@fragment
fn fs_resolve(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let hdr = p.cfg.w == 1u;
    let aa = max(p.cfg.x, 1u);
    let base = vec2<u32>(pos.xy) * aa;
    let bg_lin = vec3<f32>(
        srgb_to_linear(p.background.r),
        srgb_to_linear(p.background.g),
        srgb_to_linear(p.background.b),
    );
    // Premultiplied and straight sums (the straight mean is the
    // fallback when every sample is fully transparent).
    var pre = vec4<f32>(0.0);
    var straight = vec3<f32>(0.0);
    for (var j = 0u; j < aa; j++) {
        for (var i = 0u; i < aa; i++) {
            let xy = vec2<i32>(base + vec2<u32>(i, j));
            var c: vec3<f32>;
            var alpha: f32;
            if (textureLoad(t_coverage, xy, 0).r < 0.5) {
                c = bg_lin;
                alpha = p.background.a;
            } else {
                let s = textureLoad(t_scene, xy, 0);
                alpha = clamp(s.a, 0.0, 1.0);
                switch p.cfg.y {
                    case 0u: {
                        c = vec3<f32>(0.0);
                        if (s.a > 0.0) {
                            c = s.rgb / s.a;
                        }
                        if (!hdr) {
                            c = tone_map(c * p.exposure.x);
                        }
                    }
                    case 2u: {
                        // Already display values.
                        c = vec3<f32>(srgb_to_linear(s.r), srgb_to_linear(s.g), srgb_to_linear(s.b));
                    }
                    default: {
                        c = clamp(s.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
                    }
                }
            }
            pre += vec4<f32>(c * alpha, alpha);
            straight += c;
        }
    }
    let n = f32(aa * aa);
    var out: vec4<f32>;
    if (pre.a > 0.0) {
        out = vec4<f32>(pre.rgb / pre.a, pre.a / n);
    } else {
        out = vec4<f32>(straight / n, 0.0);
    }
    if (hdr) {
        return out;
    }
    return vec4<f32>(linear_to_srgb(out.r), linear_to_srgb(out.g), linear_to_srgb(out.b), out.a);
}
