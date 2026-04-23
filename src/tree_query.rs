use bvh::{
    aabb::Bounded,
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

pub struct ContainsStatsIterator<
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
    pub stats: TraversalStats,
    _phantom: std::marker::PhantomData<Data>,
}

impl<'bvh, 'shape, T, const D: usize, Data, Shape>
    ContainsStatsIterator<'bvh, 'shape, T, D, Data, Shape>
where
    T: BHValue,
    Shape: Bounded<T, D> + Contains<T, D, Data>,
{
    fn new(bvh: &'bvh Bvh<T, D>, point: &'bvh Point<T, D>, shapes: &'shape [Shape]) -> Self {
        let mut heap = SmallVec::new();
        if !shapes.is_empty() {
            heap.push(0);
        }

        Self {
            bvh,
            point,
            shapes,
            heap,
            stats: TraversalStats::default(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'shape, T, const D: usize, Data, Shape> Iterator
    for ContainsStatsIterator<'_, 'shape, T, D, Data, Shape>
where
    T: BHValue,
    Shape: Bounded<T, D> + Contains<T, D, Data>,
{
    type Item = Data;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(heap_leader) = self.heap.pop() {
            match self.bvh.nodes[heap_leader] {
                BvhNode::Leaf { shape_index, .. } => {
                    // Track shape intersection test
                    self.stats.simplex_tests += 1;

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
                    // Track AABB bounding box tests (2 per node traversal)
                    self.stats.aabb_tests += 2;

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

pub fn contains_point_stats_iterator<
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
) -> ContainsStatsIterator<'bvh, 'shape, T, D, Data, Shape> {
    ContainsStatsIterator::new(bvh_tree, point, shapes)
}

