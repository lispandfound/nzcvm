use crate::{
    crs::NZTM,
    geomodelgrid::{Block, GeoModelGrid},
    layers::read_model_data,
    mesh::load_mesh_from_hdf5,
    rfile::write_rfile,
};
use layers::LayerTree;
use model::ModelTree;
use nalgebra::Point3;
use ndarray::parallel::prelude::*;
use ndarray::Array4;
use ndarray::{Axis, Zip};
use rayon::iter::IntoParallelIterator; // Assuming your GeoModelGrid uses ndarray
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;
mod crs;
mod geometry;
mod geomodelgrid;
mod layers;
mod mesh;
mod model;
mod quality;
mod rfile;

fn main() {
    let nx = 500;
    let ny = 500;
    let nz = 20;

    // Grid parameters
    let resolution = 1000.0;
    let resolution_vertical = 25.0;
    let origin_lat = -45.30046120197377;
    let origin_lon = 167.7888330933356;
    let origin_x = 1191438.0;
    let origin_y = 4970446.0;
    let azimuth: f32 = 0.0; // Radians
    let mut vertices_buf = Vec::new();
    let mut qualities_buf = Vec::new();
    println!("Loading EP2020 mesh.");
    let ep2020 = load_mesh_from_hdf5(
        "/home/jake/src/nzcvm/ep2020.h5",
        &mut vertices_buf,
        &mut qualities_buf,
    )
    .expect("Could not read ep2020 mesh");
    println!("Loaded mesh.");
    let tomography = ModelTree::mesh_model(ep2020);
    let model_dir = "/home/jake/src/nzcvm/basins";

    // Containers for our data
    let mut basins = Vec::new();
    let mut models = Vec::new();

    // 1. Walk the directory
    let entries = fs::read_dir(model_dir)
        .expect("Directory not found")
        .filter_map(|res| res.ok()) // Ignore errors on specific files
        .filter(|e| {
            // Only process .h5 files
            e.path().extension().map_or(false, |ext| ext == "h5")
        });

    for entry in entries {
        let path = entry.path();
        println!("Loading: {:?}", path);

        // 2. Read and push data
        match read_model_data(path.to_str().unwrap()) {
            Ok((geometry, model)) => {
                basins.push(geometry);
                models.push(model);
            }
            Err(e) => eprintln!("Failed to read {:?}: {}", path, e),
        }
    }

    // 3. Ensure we actually found models before building the tree
    if basins.is_empty() {
        panic!("No HDF5 models found in {}", model_dir);
    }

    // 4. Initialize the ModelTree
    let basin_models = ModelTree::layered_model(LayerTree::new(&mut basins, &models));
    let model_tree = ModelTree::Stack(&basin_models, &tomography);
    println!("Model tree");
    model_tree.pretty_print();

    let mut block_values = Array4::<f32>::zeros((nx, ny, nz, 7));

    let cos_a = azimuth.cos();
    let sin_a = azimuth.sin();

    let total_start = Instant::now();

    Zip::indexed(block_values.lanes_mut(Axis(3)))
        .into_par_iter()
        .for_each(|((i, j, k), mut lane)| {
            let dy = (i as f32) * resolution;
            let dx = (j as f32) * resolution;

            let global_x = origin_x + (dx * cos_a - dy * sin_a);
            let global_y = origin_y + (dx * sin_a + dy * cos_a);

            let dz = (k as f32) * resolution_vertical;
            let query_point = Point3::new(global_x, global_y, dz);

            let (quality, _) = model_tree
                .query(query_point)
                .expect("Point outside of defined model layers");

            lane[0] = quality.rho;
            lane[1] = quality.vp;
            lane[2] = quality.vs;
            lane[3] = quality.qp;
            lane[4] = quality.qs;
            lane[5] = 0.0;
            lane[6] = 0.0;
        });
    let total_elapsed = total_start.elapsed();
    let total_points = nx * ny * nz;
    println!("--- Timing Results ---");
    println!("Total time for all points: {:?}", total_elapsed);
    println!(
        "Average time per point:    {:?}",
        total_elapsed / total_points as u32
    );
    let width = (nx as f32) * resolution;
    let height = (ny as f32) * resolution;
    let depth = (nz as f32) * resolution_vertical;

    let block = Block {
        name: "block_0".to_string(),
        resolution_horiz: resolution,
        resolution_vert: resolution_vertical,
        z_top: 0.0,
        block: block_values,
    };

    println!("Block Data ({}):\n{:.4?}", block.name, block.block);

    let topo = geomodelgrid::Surface {
        surface: ndarray::Array2::zeros((nx, ny)),
        resolution_horiz: resolution,
        name: "top_surface".to_string(),
    };

    let model_grid = GeoModelGrid::builder()
        .title("New Zealand Regional Model")
        .data_layout("vertex")
        .data_units(vec![
            "kg/m**3".to_string(),
            "m/s".to_string(),
            "m/s".to_string(),
            "None".to_string(),
            "None".to_string(),
            "None".to_string(),
            "None".to_string(),
        ])
        .data_values(vec![
            "density".to_string(),
            "Vp".to_string(),
            "Vs".to_string(),
            "Qp".to_string(),
            "Qs".to_string(),
            "fault_block_id".to_string(),
            "zone_id".to_string(),
        ])
        .crs(NZTM)
        .origin_x(origin_lon.into())
        .origin_y(origin_lat.into())
        .y_azimuth(azimuth.into())
        .dim_x(width.into())
        .dim_y(height.into())
        .dim_z(depth.into())
        .add_block(block)
        .add_surface(topo)
        .build();

    write_rfile(&model_grid, PathBuf::from("velocity_model.rfile")).expect("Cannot write rfile");
}
