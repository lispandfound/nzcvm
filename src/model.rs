use crate::layers::LayerTree;
use crate::mesh::MeshModel;
use crate::quality::Quality;
use nalgebra::Point3;
use std::sync::Arc;

pub enum ModelTree {
    Stack(Arc<ModelTree>, Arc<ModelTree>),
    Mesh { mesh_model: MeshModel },
    Layers { layer_tree: LayerTree },
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
        }
    }
    pub fn pretty_print(&self) -> () {
        match self {
            Self::Stack(left, right) => {
                println!("Stacked models, left:");
                left.pretty_print();
                println!("right:");
                right.pretty_print();
            }
            Self::Layers { layer_tree } => layer_tree.pretty_print(),
            Self::Mesh { mesh_model } => mesh_model.pretty_print(),
        }
    }
}
