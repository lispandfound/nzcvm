use crate::layers::LayerTree;
use crate::mesh::MeshModel;
use crate::quality::Quality;
use nalgebra::Point3;
use std::sync::Arc;

pub enum ModelTree {
    Stack(Arc<ModelTree>, Arc<ModelTree>),
    Blend {
        left: Arc<ModelTree>,
        right: Arc<ModelTree>,
        distance: f32,
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

    pub fn query(&self, point: Point3<f32>) -> Option<(Quality, f32)> {
        match self {
            Self::Stack(left, right) => match left.query(point) {
                Some((quality, dist)) if dist < 1e-6 => Some((quality, dist)),
                _ => right.query(point),
            },
            Self::Layers { layer_tree } => layer_tree.query(point),
            Self::Mesh { mesh_model } => mesh_model.query(point),
            Self::Blend {
                left,
                right,
                distance,
            } => match (left.query(point), right.query(point)) {
                (Some((quality_left, dist_left)), Some((quality_right, _)))
                    if dist_left < *distance =>
                {
                    let alpha = dist_left / distance;
                    Some((
                        alpha * quality_right + (1.0 - alpha) * quality_left,
                        dist_left,
                    ))
                }
                (_, right) => right,
            },
        }
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
