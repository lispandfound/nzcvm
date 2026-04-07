use crate::layers::LayerTree;
use crate::mesh::MeshModel;
use crate::quality::Quality;
use crate::real::Real;
use nalgebra::Point3;
use std::sync::Arc;

pub enum ModelTree {
    Stack(Arc<ModelTree>, Arc<ModelTree>),
    Blend {
        left: Arc<ModelTree>,
        right: Arc<ModelTree>,
        distance: Real,
    },
    Mesh {
        mesh_model: MeshModel,
    },
    Layers {
        layer_tree: LayerTree,
    },
    // TODO: Ely taper
}

impl ModelTree {
    pub fn layered_model(layer_tree: LayerTree) -> Self {
        Self::Layers { layer_tree }
    }

    pub fn mesh_model(mesh_model: MeshModel) -> Self {
        Self::Mesh { mesh_model }
    }

    pub fn query_within(&self, point: Point3<Real>, epsilon: Real) -> Option<(Quality, Real)> {
        match self {
            Self::Stack(left, right) => left
                .query_within(point, epsilon)
                .or_else(|| right.query_within(point, epsilon)),
            Self::Layers { layer_tree } => layer_tree.query_within(point, epsilon),
            Self::Mesh { mesh_model } => mesh_model.query_within(point, epsilon),
            Self::Blend {
                left,
                right,
                distance,
            } => right
                .query_within(point, epsilon)
                .map(|(quality_right, dist_right)| {
                    left.query_within(point, *distance)
                        .map(|(quality_left, dist_left)| {
                            let alpha = dist_left / distance;
                            (
                                alpha * quality_right + (1.0 - alpha) * quality_left,
                                dist_left,
                            )
                        })
                        .unwrap_or((quality_right, dist_right))
                }),
        }
    }

    pub fn query(&self, point: Point3<Real>) -> Option<(Quality, Real)> {
        self.query_within(point, Real::EPSILON)
    }

    pub fn pretty_print(&self) {
        match self {
            Self::Stack(left, right) => {
                println!("Stacked models, left:");
                left.pretty_print();
                println!("right:");
                right.pretty_print();
            }
            Self::Blend {
                left,
                right,
                distance: _,
            } => {
                println!("Blended models, left:");
                left.pretty_print();
                println!("right:");
                right.pretty_print();
            }
            Self::Layers { layer_tree } => layer_tree.pretty_print(),
            Self::Mesh { mesh_model } => mesh_model.pretty_print(),
        }
    }
}
