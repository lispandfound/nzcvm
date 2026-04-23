use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHValue,
    bvh::{Bvh, BvhNode},
};
use nalgebra::Point;
use smallvec::SmallVec;

/// A shape that can test whether it contains a query point.
///
/// `Data` is the value produced when the shape *does* contain the point.
/// Returning `Option<Data>` instead of `bool` lets the caller reuse the
/// computation that was performed during the containment test (e.g. the
/// queried quality of a mesh model) without having to repeat it.
pub trait Contains<T: BHValue, const D: usize, Data> {
    fn contains(&self, point: &Point<T, D>) -> Option<Data>;
}

pub struct ContainsIterator<
    'bvh,
    'shape,
    T: BHValue,
    const D: usize,
    Data,
    Shape: Contains<T, D, Data> + Bounded<T, D>,
> {
    bvh: &'bvh Bvh<T, D>,
    point: &'bvh Point<T, D>,
    shapes: &'shape [Shape],
    heap: SmallVec<[usize; 16]>,
    _phantom: std::marker::PhantomData<Data>,
}

impl<'bvh, 'shape, T, const D: usize, Data, Shape>
    ContainsIterator<'bvh, 'shape, T, D, Data, Shape>
where
    T: BHValue,
    Shape: Bounded<T, D> + Contains<T, D, Data>,
{
    fn new(bvh: &'bvh Bvh<T, D>, point: &'bvh Point<T, D>, shapes: &'shape [Shape]) -> Self {
        // To avoid panic! on an empty tree, we only populate with the root node
        // if the shapes array is non-empty.
        let mut heap = SmallVec::new();
        if !shapes.is_empty() {
            heap.push(0);
        }

        Self {
            bvh,
            point,
            shapes,
            heap,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'shape, T, const D: usize, Data, Shape> Iterator
    for ContainsIterator<'_, 'shape, T, D, Data, Shape>
where
    T: BHValue,
    Shape: Bounded<T, D> + Contains<T, D, Data>,
{
    type Item = Data;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(heap_leader) = self.heap.pop() {
            match self.bvh.nodes[heap_leader] {
                BvhNode::Leaf { shape_index, .. } => {
                    let shape = &self.shapes[shape_index];
                    if let Some(data) = shape.contains(self.point) {
                        return Some(data);
                    }
                }
                BvhNode::Node {
                    child_l_index,
                    ref child_l_aabb,
                    child_r_index,
                    ref child_r_aabb,
                    ..
                } => {
                    if child_l_aabb.contains(self.point) {
                        self.heap.push(child_l_index);
                    }
                    if child_r_aabb.contains(self.point) {
                        self.heap.push(child_r_index);
                    }
                }
            }
        }
        None
    }
}

pub fn contains_point_iterator<
    'bvh,
    'shape,
    T: BHValue,
    const D: usize,
    Data,
    Shape: Contains<T, D, Data> + Bounded<T, D>,
>(
    bvh_tree: &'bvh Bvh<T, D>,
    shapes: &'shape [Shape],
    point: &'bvh Point<T, D>,
) -> ContainsIterator<'bvh, 'shape, T, D, Data, Shape> {
    ContainsIterator::new(bvh_tree, point, shapes)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TraversalStats {
    pub aabb_tests: usize,
    pub simplex_tests: usize,
}

// ---------------------------------------------------------------------------
// Priority-ray traversal
// ---------------------------------------------------------------------------

/// Tests whether a `DQ`-dimensional query point lies within the first `DQ`
/// dimensions of a `DBVH`-dimensional AABB.
///
/// Returns `Some(aabb.min[DQ])` — the minimum value along the extra
/// dimension (i.e. the minimum priority reachable through this subtree) —
/// on hit, or `None` on a spatial miss.
///
/// This implements the AABB intersection test for the conceptual ray
/// `origin = (p[0], …, p[DQ-1], 0)`, `direction = (0, …, 0, 1)`.
#[inline]
fn priority_aabb_hit<T: BHValue, const DBVH: usize, const DQ: usize>(
    point: &Point<T, DQ>,
    aabb: &Aabb<T, DBVH>,
) -> Option<T> {
    for i in 0..DQ {
        if point.coords[i] < aabb.min.coords[i] || point.coords[i] > aabb.max.coords[i] {
            return None;
        }
    }
    Some(aabb.min.coords[DQ])
}

/// Iterator over a `DBVH`-dimensional BVH that models a ray from
/// `(point, 0)` along the `DBVH`-th axis (the *priority* dimension).
///
/// Leaves are visited in ascending order of `aabb.min[DQ]` (= priority),
/// so callers receive results from the highest-priority model first.
/// At each internal node the child with the lower `t_min` (= lower minimum
/// priority value in that subtree) is pushed on top of the stack and
/// therefore visited first.
///
/// For each leaf the iterator calls `shape.contains(point_DQ)` and yields
/// the returned `Data` if the shape contains the point.
pub struct PriorityRayIterator<
    'bvh,
    'shape,
    T: BHValue,
    const DBVH: usize,
    const DQ: usize,
    Data,
    Shape: Contains<T, DQ, Data> + Bounded<T, DBVH>,
> {
    bvh: &'bvh Bvh<T, DBVH>,
    point: Point<T, DQ>,
    shapes: &'shape [Shape],
    stack: SmallVec<[usize; 16]>,
    _phantom: std::marker::PhantomData<Data>,
}

impl<'bvh, 'shape, T, const DBVH: usize, const DQ: usize, Data, Shape>
    PriorityRayIterator<'bvh, 'shape, T, DBVH, DQ, Data, Shape>
where
    T: BHValue,
    Shape: Bounded<T, DBVH> + Contains<T, DQ, Data>,
{
    fn new(bvh: &'bvh Bvh<T, DBVH>, point: Point<T, DQ>, shapes: &'shape [Shape]) -> Self {
        let mut stack = SmallVec::new();
        if !shapes.is_empty() {
            stack.push(0usize);
        }
        Self {
            bvh,
            point,
            shapes,
            stack,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'shape, T, const DBVH: usize, const DQ: usize, Data, Shape> Iterator
    for PriorityRayIterator<'_, 'shape, T, DBVH, DQ, Data, Shape>
where
    T: BHValue,
    Shape: Bounded<T, DBVH> + Contains<T, DQ, Data>,
{
    type Item = Data;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node_idx) = self.stack.pop() {
            match self.bvh.nodes[node_idx] {
                BvhNode::Leaf { shape_index, .. } => {
                    let shape = &self.shapes[shape_index];
                    if let Some(data) = shape.contains(&self.point) {
                        return Some(data);
                    }
                }
                BvhNode::Node {
                    child_l_index,
                    ref child_l_aabb,
                    child_r_index,
                    ref child_r_aabb,
                    ..
                } => {
                    let l_t = priority_aabb_hit(&self.point, child_l_aabb);
                    let r_t = priority_aabb_hit(&self.point, child_r_aabb);

                    match (l_t, r_t) {
                        (None, None) => {}
                        (Some(_), None) => self.stack.push(child_l_index),
                        (None, Some(_)) => self.stack.push(child_r_index),
                        (Some(lt), Some(rt)) => {
                            // Push higher-t_min child first so that the lower-t_min
                            // child (= higher-priority subtree) is popped first.
                            if lt <= rt {
                                self.stack.push(child_r_index);
                                self.stack.push(child_l_index);
                            } else {
                                self.stack.push(child_l_index);
                                self.stack.push(child_r_index);
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

pub fn priority_ray_iterator<
    'bvh,
    'shape,
    T: BHValue,
    const DBVH: usize,
    const DQ: usize,
    Data,
    Shape: Contains<T, DQ, Data> + Bounded<T, DBVH>,
>(
    bvh_tree: &'bvh Bvh<T, DBVH>,
    shapes: &'shape [Shape],
    point: Point<T, DQ>,
) -> PriorityRayIterator<'bvh, 'shape, T, DBVH, DQ, Data, Shape> {
    PriorityRayIterator::new(bvh_tree, point, shapes)
}

/// Like [`PriorityRayIterator`] but also tracks AABB and leaf-test counts.
pub struct PriorityRayStatsIterator<
    'bvh,
    'shape,
    T: BHValue,
    const DBVH: usize,
    const DQ: usize,
    Data,
    Shape: Contains<T, DQ, Data> + Bounded<T, DBVH>,
> {
    bvh: &'bvh Bvh<T, DBVH>,
    point: Point<T, DQ>,
    shapes: &'shape [Shape],
    stack: SmallVec<[usize; 16]>,
    pub stats: TraversalStats,
    _phantom: std::marker::PhantomData<Data>,
}

impl<'bvh, 'shape, T, const DBVH: usize, const DQ: usize, Data, Shape>
    PriorityRayStatsIterator<'bvh, 'shape, T, DBVH, DQ, Data, Shape>
where
    T: BHValue,
    Shape: Bounded<T, DBVH> + Contains<T, DQ, Data>,
{
    fn new(bvh: &'bvh Bvh<T, DBVH>, point: Point<T, DQ>, shapes: &'shape [Shape]) -> Self {
        let mut stack = SmallVec::new();
        if !shapes.is_empty() {
            stack.push(0usize);
        }
        Self {
            bvh,
            point,
            shapes,
            stack,
            stats: TraversalStats::default(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'shape, T, const DBVH: usize, const DQ: usize, Data, Shape> Iterator
    for PriorityRayStatsIterator<'_, 'shape, T, DBVH, DQ, Data, Shape>
where
    T: BHValue,
    Shape: Bounded<T, DBVH> + Contains<T, DQ, Data>,
{
    type Item = Data;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node_idx) = self.stack.pop() {
            match self.bvh.nodes[node_idx] {
                BvhNode::Leaf { shape_index, .. } => {
                    self.stats.simplex_tests += 1;
                    let shape = &self.shapes[shape_index];
                    if let Some(data) = shape.contains(&self.point) {
                        return Some(data);
                    }
                }
                BvhNode::Node {
                    child_l_index,
                    ref child_l_aabb,
                    child_r_index,
                    ref child_r_aabb,
                    ..
                } => {
                    self.stats.aabb_tests += 2;

                    let l_t = priority_aabb_hit(&self.point, child_l_aabb);
                    let r_t = priority_aabb_hit(&self.point, child_r_aabb);

                    match (l_t, r_t) {
                        (None, None) => {}
                        (Some(_), None) => self.stack.push(child_l_index),
                        (None, Some(_)) => self.stack.push(child_r_index),
                        (Some(lt), Some(rt)) => {
                            if lt <= rt {
                                self.stack.push(child_r_index);
                                self.stack.push(child_l_index);
                            } else {
                                self.stack.push(child_l_index);
                                self.stack.push(child_r_index);
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

pub fn priority_ray_stats_iterator<
    'bvh,
    'shape,
    T: BHValue,
    const DBVH: usize,
    const DQ: usize,
    Data,
    Shape: Contains<T, DQ, Data> + Bounded<T, DBVH>,
>(
    bvh_tree: &'bvh Bvh<T, DBVH>,
    shapes: &'shape [Shape],
    point: Point<T, DQ>,
) -> PriorityRayStatsIterator<'bvh, 'shape, T, DBVH, DQ, Data, Shape> {
    PriorityRayStatsIterator::new(bvh_tree, point, shapes)
}
