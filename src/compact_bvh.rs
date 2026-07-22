//! A memory-compact, query-only BVH over a mesh's simplices.
//!
//! The tree is built with the `bvh` crate (parallel SAH build) and then
//! converted into this representation, which is what the mesh actually keeps:
//!
//! * only *internal* nodes are stored — leaves are encoded as tagged `u32`
//!   child slots pointing directly into the simplex array;
//! * child indices are `u32` instead of `usize`, and the never-queried
//!   `parent_index` is dropped;
//! * subtrees of up to [`MAX_LEAF_SIZE`] simplices are collapsed into a single
//!   leaf, and the simplices are reordered so each leaf's simplices are
//!   contiguous in memory.
//!
//! With `Real = f32` a [`CompactNode`] is 56 bytes and there are at most
//! `⌈n / 2⌉` of them, compared to `2n - 1` nodes of 80 bytes each for the
//! `bvh` crate's tree — roughly a 10× reduction in index overhead per simplex.

use bvh::aabb::Aabb;
use bvh::bvh::{Bvh, BvhNode};
use deepsize::{Context, DeepSizeOf};
use nalgebra::Point3;
use smallvec::SmallVec;

use crate::real::Real;
use crate::simplex::Simplex;

/// Maximum number of simplices collapsed into a single leaf.
///
/// Simplices within a leaf are contiguous, so testing them is a linear scan
/// of adjacent 48-byte records — cheap compared to extra node traversal.
pub const MAX_LEAF_SIZE: u32 = 4;

/// A child slot with this bit set is a leaf reference, otherwise it is an
/// index into [`CompactBvh::nodes`].
const LEAF_BIT: u32 = 1 << 31;
/// Bits 29–30 of a leaf slot hold `count - 1` (1..=MAX_LEAF_SIZE simplices).
const COUNT_SHIFT: u32 = 29;
/// Bits 0–28 of a leaf slot hold the start index into the simplex array,
/// limiting a single mesh to 2^29 (~537 M) simplices.
const START_MASK: u32 = (1 << COUNT_SHIFT) - 1;

const STACK_CAPACITY: usize = 16;

/// One internal node: the AABBs of both children plus their tagged slots.
#[derive(Clone, Copy)]
struct CompactNode {
    aabb_l: Aabb<Real, 3>,
    aabb_r: Aabb<Real, 3>,
    left: u32,
    right: u32,
}

#[cfg(not(feature = "high_precision"))]
const _: () = assert!(std::mem::size_of::<CompactNode>() == 56);

pub struct CompactBvh {
    nodes: Vec<CompactNode>,
    /// Tagged slot of the root (a leaf for meshes of ≤ `MAX_LEAF_SIZE`
    /// simplices), or `None` for an empty mesh.
    root: Option<u32>,
}

impl DeepSizeOf for CompactBvh {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.nodes.capacity() * std::mem::size_of::<CompactNode>()
    }
}

fn leaf_slot(start: u32, count: u32) -> u32 {
    debug_assert!((1..=MAX_LEAF_SIZE).contains(&count));
    LEAF_BIT | ((count - 1) << COUNT_SHIFT) | start
}

/// Number of shapes in each subtree of the source tree, indexed by node.
fn subtree_counts(nodes: &[BvhNode<Real, 3>]) -> Vec<u32> {
    fn count(nodes: &[BvhNode<Real, 3>], counts: &mut [u32], i: usize) -> u32 {
        let c = match nodes[i] {
            BvhNode::Leaf { .. } => 1,
            BvhNode::Node {
                child_l_index,
                child_r_index,
                ..
            } => count(nodes, counts, child_l_index) + count(nodes, counts, child_r_index),
        };
        counts[i] = c;
        c
    }
    let mut counts = vec![0u32; nodes.len()];
    if !nodes.is_empty() {
        count(nodes, &mut counts, 0);
    }
    counts
}

/// Append the shape indices of every leaf under `i` to `order`.
fn collect_shapes(nodes: &[BvhNode<Real, 3>], order: &mut Vec<u32>, i: usize) {
    match nodes[i] {
        BvhNode::Leaf { shape_index, .. } => order.push(shape_index as u32),
        BvhNode::Node {
            child_l_index,
            child_r_index,
            ..
        } => {
            collect_shapes(nodes, order, child_l_index);
            collect_shapes(nodes, order, child_r_index);
        }
    }
}

impl CompactBvh {
    /// Convert a freshly built `bvh` crate tree over `num_shapes` shapes.
    ///
    /// Returns the compact tree together with the leaf-order permutation:
    /// `order[new_index] = old_shape_index`.  The caller must gather its
    /// per-simplex arrays through `order` so that leaf slots index correctly.
    pub fn from_bvh(tree: &Bvh<Real, 3>, num_shapes: usize) -> (Self, Vec<u32>) {
        assert!(
            num_shapes < START_MASK as usize,
            "mesh exceeds the compact BVH limit of 2^29 simplices"
        );
        if num_shapes == 0 {
            return (
                Self {
                    nodes: Vec::new(),
                    root: None,
                },
                Vec::new(),
            );
        }

        let counts = subtree_counts(&tree.nodes);
        let mut nodes = Vec::with_capacity(num_shapes / 2);
        let mut order = Vec::with_capacity(num_shapes);

        fn emit(
            src: &[BvhNode<Real, 3>],
            counts: &[u32],
            nodes: &mut Vec<CompactNode>,
            order: &mut Vec<u32>,
            i: usize,
        ) -> u32 {
            if counts[i] <= MAX_LEAF_SIZE {
                let start = order.len() as u32;
                collect_shapes(src, order, i);
                return leaf_slot(start, counts[i]);
            }
            // A subtree holding more than one shape is always a `Node`.
            let BvhNode::Node {
                child_l_index,
                child_l_aabb,
                child_r_index,
                child_r_aabb,
                ..
            } = src[i]
            else {
                unreachable!("leaf node with subtree count > 1")
            };
            let slot = nodes.len() as u32;
            nodes.push(CompactNode {
                aabb_l: child_l_aabb,
                aabb_r: child_r_aabb,
                left: 0,
                right: 0,
            });
            let left = emit(src, counts, nodes, order, child_l_index);
            let right = emit(src, counts, nodes, order, child_r_index);
            nodes[slot as usize].left = left;
            nodes[slot as usize].right = right;
            slot
        }

        let root = emit(&tree.nodes, &counts, &mut nodes, &mut order, 0);
        nodes.shrink_to_fit();
        (
            Self {
                nodes,
                root: Some(root),
            },
            order,
        )
    }

    /// Iterator over the indices of all simplices that contain `point`.
    pub fn containing_indices<'a>(
        &'a self,
        simplices: &'a [Simplex],
        point: Point3<Real>,
    ) -> ContainingIndices<'a> {
        let mut stack = SmallVec::new();
        if let Some(root) = self.root {
            stack.push(root);
        }
        ContainingIndices {
            nodes: &self.nodes,
            simplices,
            point,
            stack,
            cursor: 0,
            end: 0,
        }
    }

    /// Maximum depth of the tree (0 when the root is a single leaf).
    pub fn depth(&self) -> usize {
        fn go(nodes: &[CompactNode], slot: u32) -> usize {
            if slot & LEAF_BIT != 0 {
                return 0;
            }
            let node = &nodes[slot as usize];
            1 + go(nodes, node.left).max(go(nodes, node.right))
        }
        self.root.map_or(0, |root| go(&self.nodes, root))
    }
}

/// See [`CompactBvh::containing_indices`].
pub struct ContainingIndices<'a> {
    nodes: &'a [CompactNode],
    simplices: &'a [Simplex],
    point: Point3<Real>,
    stack: SmallVec<[u32; STACK_CAPACITY]>,
    /// Range of the leaf currently being scanned.
    cursor: u32,
    end: u32,
}

impl Iterator for ContainingIndices<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            while self.cursor < self.end {
                let i = self.cursor as usize;
                self.cursor += 1;
                if self.simplices[i].contains(&self.point) {
                    return Some(i);
                }
            }
            let slot = self.stack.pop()?;
            if slot & LEAF_BIT != 0 {
                self.cursor = slot & START_MASK;
                self.end = self.cursor + ((slot >> COUNT_SHIFT) & 0b11) + 1;
            } else {
                let node = &self.nodes[slot as usize];
                if node.aabb_l.contains(&self.point) {
                    self.stack.push(node.left);
                }
                if node.aabb_r.contains(&self.point) {
                    self.stack.push(node.right);
                }
            }
        }
    }
}
