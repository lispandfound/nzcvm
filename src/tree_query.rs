use core::cmp::Ordering;
use std::collections::BinaryHeap;

use bvh::{
    aabb::Bounded,
    bounding_hierarchy::BHValue,
    bvh::{Bvh, BvhNode},
    point_query::PointDistance,
};
use nalgebra::Point;

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
}

impl<'bvh, 'shape, T, const D: usize, Shape> NearestIterator<'bvh, 'shape, T, D, Shape>
where
    T: BHValue,
    Shape: PointDistance<T, D> + Bounded<T, D>,
{
    fn new(bvh: &'bvh Bvh<T, D>, point: &'bvh Point<T, D>, shapes: &'shape [Shape]) -> Self {
        // To avoid panic! on an empty tree, we only populate with the root node
        // if the shapes array is non-empty.
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
                self.heap.push(Distance::LowerBound {
                    distance: child_l_aabb.min_distance_squared(*self.point),
                    node_index: child_l_index,
                });
                self.heap.push(Distance::LowerBound {
                    distance: child_r_aabb.min_distance_squared(*self.point),
                    node_index: child_r_index,
                });
            }
            BvhNode::Leaf { shape_index, .. } => {
                // Until this point we've only been comparing on lower bounds
                // for distance. But it is possible that a shape has a much
                // larger distance than its bounding box. To resolve this we
                // perform an exact distance check on the shape and push back.
                let dist_sq = self.shapes[shape_index].distance_squared(*self.point);
                self.heap.push(Distance::Exact {
                    distance: dist_sq,
                    shape_index,
                });
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
) -> NearestIterator<'bvh, 'shape, T, D, Shape> {
    NearestIterator::new(bvh_tree, point, shapes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bvh::aabb::Aabb;
    use bvh::bounding_hierarchy::BHShape;
    use nalgebra::{Point3, Vector3};

    #[derive(Debug, Clone, PartialEq)]
    struct Sphere {
        center: Point3<f32>,
        radius: f32,
        node_index: usize,
        id: usize, // To easily identify shapes in asserts
    }

    impl Bounded<f32, 3> for Sphere {
        fn aabb(&self) -> Aabb<f32, 3> {
            let half_size = Vector3::new(self.radius, self.radius, self.radius);
            Aabb::with_bounds(self.center - half_size, self.center + half_size)
        }
    }

    impl BHShape<f32, 3> for Sphere {
        fn set_bh_node_index(&mut self, index: usize) {
            self.node_index = index;
        }
        fn bh_node_index(&self) -> usize {
            self.node_index
        }
    }

    impl PointDistance<f32, 3> for Sphere {
        fn distance_squared(&self, point: Point3<f32>) -> f32 {
            let dist = nalgebra::distance(&self.center, &point);
            // Distance to the surface of the sphere
            let surface_dist = (dist - self.radius).max(0.0);
            surface_dist * surface_dist
        }
    }

    fn setup_test_bvh() -> (Bvh<f32, 3>, Vec<Sphere>) {
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

        let iterator = NearestIterator::new(&bvh, &query_point, &shapes);
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
    fn test_k_retrievals() {
        let (bvh, shapes) = setup_test_bvh();
        let query_point = Point3::new(12.0, 0.0, 0.0);

        // Closest to 12.0 is ID 1 (10.0), then ID 3 (5.0), then ID 2 (20.0)
        let mut iterator = NearestIterator::new(&bvh, &query_point, &shapes);

        // Retreive k=2 items
        let first = iterator.next().unwrap().0;
        assert_eq!(first.id, 1);

        let second = iterator.next().unwrap().0;
        assert_eq!(second.id, 3);

        // Iterator is suspended but retains valid state, proving lazy evaluation properties.
    }

    #[test]
    fn test_empty_bvh_does_not_panic() {
        let mut shapes: Vec<Sphere> = vec![];
        // Bvh::build_par handles empty vecs by creating a stub node
        let bvh = Bvh::build(&mut shapes);
        let query_point = Point3::new(0.0, 0.0, 0.0);

        let mut iterator = NearestIterator::new(&bvh, &query_point, &shapes);
        assert!(
            iterator.next().is_none(),
            "Empty BVH should yield None immediately"
        );
    }
}
