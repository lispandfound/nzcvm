use bvh::{
    aabb::Bounded,
    bounding_hierarchy::BHValue,
    bvh::{Bvh, BvhNode},
    point_query::PointDistance,
};
use core::cmp::Ordering;
use nalgebra::Point;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy)]
enum Distance<T: PartialOrd> {
    LowerBound { distance: T, node_index: usize },
    Exact { distance: T, shape_index: usize },
}

impl<T: PartialOrd> Distance<T> {
    fn val(&self) -> &T {
        match self {
            Distance::LowerBound { distance, .. } => distance,
            Distance::Exact { distance, .. } => distance,
        }
    }

    fn is_exact(&self) -> bool {
        matches!(self, Distance::Exact { .. })
    }
}

impl<T: PartialOrd> PartialEq for Distance<T> {
    fn eq(&self, other: &Self) -> bool {
        self.val().partial_cmp(other.val()) == Some(Ordering::Equal)
            && self.is_exact() == other.is_exact()
    }
}

impl<T: PartialOrd> Eq for Distance<T> {}

impl<T: PartialOrd> PartialOrd for Distance<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialOrd> Ord for Distance<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other
            .val()
            .partial_cmp(self.val())
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => match (self.is_exact(), other.is_exact()) {
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ => Ordering::Equal,
            },
            other_ord => other_ord,
        }
    }
}

pub trait Contains<T: BHValue, const D: usize> {
    fn contains(&self, point: &Point<T, D>) -> bool;
}

pub struct ContainsIterator<
    'bvh,
    'shape,
    T: BHValue,
    const D: usize,
    Shape: Contains<T, D> + Bounded<T, D>,
> {
    bvh: &'bvh Bvh<T, D>,
    point: &'bvh Point<T, D>,
    shapes: &'shape [Shape],
    heap: Vec<usize>,
}

impl<'bvh, 'shape, T, const D: usize, Shape> ContainsIterator<'bvh, 'shape, T, D, Shape>
where
    T: BHValue,
    Shape: Bounded<T, D> + Contains<T, D>,
{
    fn new(bvh: &'bvh Bvh<T, D>, point: &'bvh Point<T, D>, shapes: &'shape [Shape]) -> Self {
        // To avoid panic! on an empty tree, we only populate with the root node
        // if the shapes array is non-empty.
        let mut heap = Vec::new();
        if !shapes.is_empty() {
            heap.push(0);
        }

        Self {
            bvh,
            point,
            shapes,
            heap,
        }
    }
}

impl<'shape, T, const D: usize, Shape> Iterator for ContainsIterator<'_, 'shape, T, D, Shape>
where
    T: BHValue,
    Shape: Bounded<T, D> + Contains<T, D>,
{
    type Item = &'shape Shape;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(heap_leader) = self.heap.pop() {
            match self.bvh.nodes[heap_leader] {
                BvhNode::Leaf { shape_index, .. } => {
                    return Some(&self.shapes[shape_index]);
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
    Shape: Contains<T, D> + Bounded<T, D>,
>(
    bvh_tree: &'bvh Bvh<T, D>,
    shapes: &'shape [Shape],
    point: &'bvh Point<T, D>,
) -> ContainsIterator<'bvh, 'shape, T, D, Shape> {
    ContainsIterator::new(bvh_tree, point, shapes)
}

pub struct NearestIterator<
    'bvh,
    'shape,
    T: BHValue,
    const D: usize,
    Shape: PointDistance<T, D> + Bounded<T, D>,
> {
    bvh: &'bvh Bvh<T, D>,
    point: &'bvh Point<T, D>,
    shapes: &'shape [Shape],
    heap: BinaryHeap<Distance<T>>,
    epsilon_sq: T,
}

impl<'bvh, 'shape, T, const D: usize, Shape> NearestIterator<'bvh, 'shape, T, D, Shape>
where
    T: BHValue,
    Shape: PointDistance<T, D> + Bounded<T, D>,
{
    fn new(
        bvh: &'bvh Bvh<T, D>,
        point: &'bvh Point<T, D>,
        shapes: &'shape [Shape],
        epsilon: T,
    ) -> Self {
        // To avoid panic! on an empty tree, we only populate with the root node
        // if the shapes array is non-empty.
        let epsilon_sq = epsilon.powi(2);
        let heap = if shapes.is_empty() {
            BinaryHeap::new()
        } else {
            BinaryHeap::from([Distance::LowerBound {
                distance: T::zero(),
                node_index: 0,
            }])
        };

        Self {
            bvh,
            point,
            shapes,
            heap,
            epsilon_sq,
        }
    }

    fn unpack_node(&mut self, node_index: usize) {
        match self.bvh.nodes[node_index] {
            BvhNode::Node {
                child_l_index,
                ref child_l_aabb,
                child_r_index,
                ref child_r_aabb,
                ..
            } => {
                let left_child_distance = child_l_aabb.min_distance_squared(*self.point);
                if left_child_distance < self.epsilon_sq {
                    self.heap.push(Distance::LowerBound {
                        distance: left_child_distance,
                        node_index: child_l_index,
                    });
                }
                let right_child_distance = child_r_aabb.min_distance_squared(*self.point);
                if right_child_distance < self.epsilon_sq {
                    self.heap.push(Distance::LowerBound {
                        distance: right_child_distance,
                        node_index: child_r_index,
                    });
                }
            }
            BvhNode::Leaf { shape_index, .. } => {
                // Until this point we've only been comparing on lower bounds
                // for distance. But it is possible that a shape has a much
                // larger distance than its bounding box. To resolve this we
                // perform an exact distance check on the shape and push back.
                let dist_sq = self.shapes[shape_index].distance_squared(*self.point);
                if dist_sq < self.epsilon_sq {
                    self.heap.push(Distance::Exact {
                        distance: dist_sq,
                        shape_index,
                    });
                }
            }
        }
    }
}

impl<'shape, T, const D: usize, Shape> Iterator for NearestIterator<'_, 'shape, T, D, Shape>
where
    T: BHValue,
    Shape: Bounded<T, D> + PointDistance<T, D>,
{
    type Item = (&'shape Shape, T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(heap_leader) = self.heap.pop() {
            match heap_leader {
                Distance::Exact {
                    shape_index,
                    distance,
                } => {
                    return Some((&self.shapes[shape_index], distance.sqrt()));
                }
                Distance::LowerBound { node_index, .. } => {
                    self.unpack_node(node_index);
                }
            }
        }
        None
    }
}

pub fn nearest_to_point_iterator<
    'bvh,
    'shape,
    T: BHValue,
    const D: usize,
    Shape: PointDistance<T, D> + Bounded<T, D>,
>(
    bvh_tree: &'bvh Bvh<T, D>,
    shapes: &'shape [Shape],
    point: &'bvh Point<T, D>,
    epsilon: T,
) -> NearestIterator<'bvh, 'shape, T, D, Shape> {
    NearestIterator::new(bvh_tree, point, shapes, epsilon)
}

pub fn nearest_to_point_within<
    'a,
    T: BHValue,
    const D: usize,
    Shape: PointDistance<T, D> + Bounded<T, D>,
>(
    bvh: &Bvh<T, D>,
    shapes: &'a [Shape],
    point: nalgebra::Point<T, D>,
    epsilon: T,
) -> Option<(&'a Shape, T)> {
    // We initialise the distance to epsilon^2.
    // Any shape found must be strictly better than this to be recorded.
    let mut best_candidate: Option<&'a Shape> = None;
    let mut best_candidate_dist = epsilon.powi(2);

    nearest_to_recursive(
        &bvh.nodes,
        0,
        point,
        shapes,
        &mut best_candidate_dist,
        &mut best_candidate,
    );

    // Convert back from squared distance for the final result
    best_candidate.map(|shape| (shape, best_candidate_dist.sqrt()))
}

fn nearest_to_recursive<
    'a,
    T: BHValue,
    const D: usize,
    Shape: Bounded<T, D> + PointDistance<T, D>,
>(
    nodes: &[BvhNode<T, D>],
    node_index: usize,
    query: nalgebra::Point<T, D>,
    shapes: &'a [Shape],
    best_candidate_dist: &mut T,
    best_candidate: &mut Option<&'a Shape>,
) {
    match nodes[node_index] {
        BvhNode::Node {
            ref child_l_aabb,
            child_l_index,
            ref child_r_aabb,
            child_r_index,
            ..
        } => {
            let mut children = [
                (child_l_index, child_l_aabb.min_distance_squared(query)),
                (child_r_index, child_r_aabb.min_distance_squared(query)),
            ];

            if children[0].1 > children[1].1 {
                children.swap(0, 1);
            }

            for (index, child_dist) in children {
                // Get the current upper bound for distance: either the best shape
                // found so far, or the epsilon threshold.

                if child_dist < *best_candidate_dist {
                    nearest_to_recursive(
                        nodes,
                        index,
                        query,
                        shapes,
                        best_candidate_dist,
                        best_candidate,
                    );
                }
            }
        }
        BvhNode::Leaf { shape_index, .. } => {
            let dist_sq = shapes[shape_index].distance_squared(query);

            if dist_sq < *best_candidate_dist {
                *best_candidate_dist = dist_sq;
                *best_candidate = Some(&shapes[shape_index]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real::Real;
    use bvh::aabb::Aabb;
    use bvh::bounding_hierarchy::BHShape;
    use nalgebra::{Point3, Vector3};

    #[derive(Debug, Clone, PartialEq)]
    struct Sphere {
        center: Point3<Real>,
        radius: Real,
        node_index: usize,
        id: usize, // To easily identify shapes in asserts
    }

    impl Bounded<Real, 3> for Sphere {
        fn aabb(&self) -> Aabb<Real, 3> {
            let half_size = Vector3::new(self.radius, self.radius, self.radius);
            Aabb::with_bounds(self.center - half_size, self.center + half_size)
        }
    }

    impl BHShape<Real, 3> for Sphere {
        fn set_bh_node_index(&mut self, index: usize) {
            self.node_index = index;
        }
        fn bh_node_index(&self) -> usize {
            self.node_index
        }
    }

    impl PointDistance<Real, 3> for Sphere {
        fn distance_squared(&self, point: Point3<Real>) -> Real {
            let dist = nalgebra::distance(&self.center, &point);
            // Distance to the surface of the sphere
            let surface_dist = (dist - self.radius).max(0.0);
            surface_dist * surface_dist
        }
    }

    fn setup_test_bvh() -> (Bvh<Real, 3>, Vec<Sphere>) {
        let mut shapes = vec![
            Sphere {
                center: Point3::new(10.0, 0.0, 0.0),
                radius: 1.0,
                node_index: 0,
                id: 1,
            },
            Sphere {
                center: Point3::new(20.0, 0.0, 0.0),
                radius: 1.0,
                node_index: 0,
                id: 2,
            },
            Sphere {
                center: Point3::new(5.0, 0.0, 0.0),
                radius: 1.0,
                node_index: 0,
                id: 3,
            },
            Sphere {
                center: Point3::new(100.0, 0.0, 0.0),
                radius: 1.0,
                node_index: 0,
                id: 4,
            },
        ];

        let bvh = Bvh::build(&mut shapes);
        (bvh, shapes)
    }

    #[test]
    fn test_strict_ordering() {
        let (bvh, shapes) = setup_test_bvh();
        // Query near origin. Expected order based on surface distance:
        // ID 3 (dist: ~4), ID 1 (dist: ~9), ID 2 (dist: ~19), ID 4 (dist: ~99)
        let query_point = Point3::new(0.0, 0.0, 0.0);

        let iterator = NearestIterator::new(&bvh, &query_point, &shapes, Real::MAX);
        let results: Vec<_> = iterator.collect();

        assert_eq!(
            results.len(),
            4,
            "Should exhaustively yield all shapes without deadlocking"
        );
        assert_eq!(results[0].0.id, 3);
        assert_eq!(results[1].0.id, 1);
        assert_eq!(results[2].0.id, 2);
        assert_eq!(results[3].0.id, 4);
    }

    #[test]
    fn test_epsilon_mask() {
        let (bvh, shapes) = setup_test_bvh();
        // Query near origin. Expected order based on surface distance:
        // ID 3 (dist: ~4), ID 1 (dist: ~9), ID 2 (dist: ~19), ID 4 (dist: ~99)
        let query_point = Point3::new(0.0, 0.0, 0.0);
        let epsilon = 10.0;
        let iterator = NearestIterator::new(&bvh, &query_point, &shapes, epsilon);
        let results: Vec<_> = iterator.collect();

        assert_eq!(
            results.len(),
            2,
            "Should exhaustively yield all shapes within distance 10 without deadlocking"
        );
        assert_eq!(results[0].0.id, 3);
        assert_eq!(results[1].0.id, 1);
    }

    #[test]
    fn test_k_retrievals() {
        let (bvh, shapes) = setup_test_bvh();
        let query_point = Point3::new(12.0, 0.0, 0.0);

        // Closest to 12.0 is ID 1 (10.0), then ID 3 (5.0), then ID 2 (20.0)
        let mut iterator = NearestIterator::new(&bvh, &query_point, &shapes, Real::MAX);

        // Retrieve k=2 items
        let first = iterator.next().unwrap().0;
        assert_eq!(first.id, 1);

        let second = iterator.next().unwrap().0;
        assert_eq!(second.id, 3);

        // Iterator is suspended but retains valid state, proving lazy evaluation properties.
    }

    #[test]
    fn test_empty_bvh_does_not_panic() {
        let mut shapes: Vec<Sphere> = vec![];

        let bvh = Bvh::build(&mut shapes);
        let query_point = Point3::new(0.0, 0.0, 0.0);

        let mut iterator = NearestIterator::new(&bvh, &query_point, &shapes, Real::MAX);
        assert!(
            iterator.next().is_none(),
            "Empty BVH should yield None immediately"
        );
    }
}
