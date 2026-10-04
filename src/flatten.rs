//! CPU-side scene flattening: walk the `Scene3D` node forest, bake
//! every primitive into world space and expand it into non-indexed
//! triangle and line vertex streams ready for upload.
//!
//! Semantics mirror `oxideav-render`'s scanline backend so both
//! backends draw the same picture for the same scene + options:
//!
//! * node graph walked depth-first pre-order, each node claimed once at
//!   first arrival (cycles / shared children terminate);
//! * colour = material `base_color` factor, or a neutral grey when the
//!   primitive has no material;
//! * per-vertex normals transformed by the inverse-transpose of the
//!   world matrix's linear part; primitives without normals fall back
//!   to the world-space face normal;
//! * point topologies are not drawn.

use bytemuck::{Pod, Zeroable};
use oxideav_mesh3d::{Indices, NodeId, Primitive, Scene3D, Topology};

use crate::math::{
    identity4, mat3_inverse_transpose, mat3_mul_vec3, mat4_mul, mat4_mul_point, vec3_cross,
    vec3_normalise, vec3_sub, Mat4,
};

/// Default colour (linear RGBA) for primitives without a material.
/// Matches the scanline backend.
pub(crate) const DEFAULT_COLOUR: [f32; 4] = [0.7, 0.7, 0.75, 1.0];

/// One GPU vertex: world-space position, world-space normal, linear
/// RGBA colour. 40 bytes, tightly packed.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct Vertex {
    pub(crate) position: [f32; 3],
    pub(crate) normal: [f32; 3],
    pub(crate) color: [f32; 4],
}

/// World-space axis-aligned bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Bounds {
    pub(crate) min: [f32; 3],
    pub(crate) max: [f32; 3],
}

impl Bounds {
    fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    fn extend(&mut self, p: [f32; 3]) {
        // `min`/`max` drop NaN operands, so a poisoned vertex can never
        // poison the bounds.
        for ((lo, hi), v) in self.min.iter_mut().zip(&mut self.max).zip(p) {
            *lo = lo.min(v);
            *hi = hi.max(v);
        }
    }

    fn is_empty(&self) -> bool {
        // Empty (inverted) or NaN on any axis.
        (0..3).any(|i| {
            !matches!(
                self.min[i].partial_cmp(&self.max[i]),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            )
        })
    }
}

/// Flattened, upload-ready scene.
#[derive(Debug, Default)]
pub(crate) struct FlatScene {
    /// Triangle-list vertices (3 per triangle).
    pub(crate) triangles: Vec<Vertex>,
    /// Line-list vertices (2 per segment).
    pub(crate) lines: Vec<Vertex>,
    /// World-space bounds over every mesh vertex (unit box around the
    /// origin when the scene has no geometry).
    pub(crate) bounds: Option<Bounds>,
}

impl FlatScene {
    /// World bounds, falling back to a unit box centred on the origin.
    pub(crate) fn bounds_or_unit(&self) -> Bounds {
        self.bounds.unwrap_or(Bounds {
            min: [-0.5; 3],
            max: [0.5; 3],
        })
    }
}

/// Flatten `scene`. When `wireframe` is set, triangle primitives are
/// emitted as their three edges into the line stream instead of as
/// filled triangles.
pub(crate) fn flatten(scene: &Scene3D, wireframe: bool) -> FlatScene {
    let mut out = FlatScene::default();
    let mut bounds = Bounds::empty();
    walk_scene_preorder(scene, |node, world| {
        let Some(mesh) = node.mesh.and_then(|id| scene.meshes.get(id.0 as usize)) else {
            return;
        };
        for prim in &mesh.primitives {
            for p in &prim.positions {
                let w = mat4_mul_point(world, *p);
                if w.iter().all(|c| c.is_finite()) {
                    bounds.extend(w);
                }
            }
            let colour = prim
                .material
                .and_then(|m| scene.materials.get(m.0 as usize))
                .map(|m| m.base_color)
                .unwrap_or(DEFAULT_COLOUR);
            emit_primitive(prim, world, colour, wireframe, &mut out);
        }
    });
    out.bounds = (!bounds.is_empty()).then_some(bounds);
    out
}

fn emit_primitive(
    prim: &Primitive,
    world: &Mat4,
    colour: [f32; 4],
    wireframe: bool,
    out: &mut FlatScene,
) {
    let positions: Vec<[f32; 3]> = prim
        .positions
        .iter()
        .map(|p| mat4_mul_point(world, *p))
        .collect();
    let normals: Option<Vec<[f32; 3]>> = prim
        .normals
        .as_ref()
        .filter(|ns| ns.len() >= positions.len())
        .map(|ns| {
            let nm = mat3_inverse_transpose(world);
            ns.iter()
                .map(|n| vec3_normalise(mat3_mul_vec3(&nm, *n)))
                .collect()
        });
    let vertex = |i: usize, n: [f32; 3]| Vertex {
        position: positions[i],
        normal: n,
        color: colour,
    };
    match prim.topology {
        Topology::Triangles | Topology::TriangleStrip | Topology::TriangleFan => {
            for tri in prim.triangle_indices() {
                let [a, b, c] = tri.map(|i| i as usize);
                if a >= positions.len() || b >= positions.len() || c >= positions.len() {
                    continue;
                }
                let face = vec3_normalise(vec3_cross(
                    vec3_sub(positions[b], positions[a]),
                    vec3_sub(positions[c], positions[a]),
                ));
                let n = |i: usize| normals.as_ref().map_or(face, |ns| ns[i]);
                if wireframe {
                    for (p, q) in [(a, b), (b, c), (c, a)] {
                        out.lines.push(vertex(p, n(p)));
                        out.lines.push(vertex(q, n(q)));
                    }
                } else {
                    out.triangles.push(vertex(a, n(a)));
                    out.triangles.push(vertex(b, n(b)));
                    out.triangles.push(vertex(c, n(c)));
                }
            }
        }
        Topology::Lines | Topology::LineStrip | Topology::LineLoop => {
            let seq = index_sequence(prim);
            let pairs: Vec<(usize, usize)> = match prim.topology {
                Topology::Lines => seq.chunks_exact(2).map(|p| (p[0], p[1])).collect(),
                _ => {
                    let mut v: Vec<(usize, usize)> = seq.windows(2).map(|w| (w[0], w[1])).collect();
                    if prim.topology == Topology::LineLoop && seq.len() >= 2 {
                        v.push((seq[seq.len() - 1], seq[0]));
                    }
                    v
                }
            };
            for (p, q) in pairs {
                if p < positions.len() && q < positions.len() {
                    let np = normals.as_ref().map_or([0.0, 0.0, 1.0], |ns| ns[p]);
                    let nq = normals.as_ref().map_or([0.0, 0.0, 1.0], |ns| ns[q]);
                    out.lines.push(vertex(p, np));
                    out.lines.push(vertex(q, nq));
                }
            }
        }
        Topology::Points => {}
    }
}

/// The logical vertex sequence: the index buffer widened to `usize`,
/// or `0..positions.len()` for non-indexed primitives.
fn index_sequence(prim: &Primitive) -> Vec<usize> {
    match &prim.indices {
        Some(Indices::U16(v)) => v.iter().map(|&i| i as usize).collect(),
        Some(Indices::U32(v)) => v.iter().map(|&i| i as usize).collect(),
        None => (0..prim.positions.len()).collect(),
    }
}

/// Depth-first pre-order walk calling `f(node, world)` once per
/// reachable node (first arrival wins; out-of-range ids skipped).
fn walk_scene_preorder(scene: &Scene3D, mut f: impl FnMut(&oxideav_mesh3d::Node, &Mat4)) {
    let mut visited = vec![false; scene.nodes.len()];
    let mut stack: Vec<(NodeId, Mat4)> = scene
        .roots
        .iter()
        .rev()
        .map(|&r| (r, identity4()))
        .collect();
    while let Some((id, parent)) = stack.pop() {
        match visited.get_mut(id.0 as usize) {
            Some(slot) if !*slot => *slot = true,
            _ => continue,
        }
        let Some(node) = scene.nodes.get(id.0 as usize) else {
            continue;
        };
        let world = mat4_mul(&parent, &node.transform.to_matrix());
        f(node, &world);
        for &child in node.children.iter().rev() {
            stack.push((child, world));
        }
    }
}
