use crate::{
    crs::NZTM,
    geomodelgrid::{Block, GeoModelGrid},
    model::{LayerGeometry, Model},
    quality::Quality,
    rfile::write_rfile,
};
use geo::polygon;
use model::ModelTree;
use nalgebra::Point3;
use ndarray::Array4; // Assuming your GeoModelGrid uses ndarray
use std::path::PathBuf;

mod crs;
mod geomodelgrid;
mod model;
mod quality;
mod quality_interpolator;
mod rfile;

fn main() {
    let nx = 500;
    let ny = 500;
    let nz = 100;

    // Grid parameters
    let resolution = 125.0;
    let resolution_vertical = 125.0;
    let origin_lon = 172.0; // These usually represent NZTM Easting/Northing in your CRS
    let origin_x = 1518491.0;
    let origin_lat = -43.0;
    let origin_y = 5238700.0;
    let azimuth: f32 = 0.1; // Radians
    let vp = 3500.0;

    let vs = 1860.0;

    let rho = 2320.0;

    let qp = 208.48;

    let qs = 104.24;

    let bounding_polygon = polygon![
        (x: 1_000_000.0, y: 4_700_000.0), // South-West (roughly below Stewart Island)
        (x: 2_500_000.0, y: 4_700_000.0), // South-East
        (x: 2_500_000.0, y: 6_300_000.0), // North-East (above Raoul Island/Kermadecs is further, but this hits Northland)
        (x: 1_000_000.0, y: 6_300_000.0)  // North-West
    ];
    let layer1 = LayerGeometry::new_with_flat_surface(&bounding_polygon, 0.0, 3000.0);

    let model1 = Model::Uniform(Quality {
        rho: rho,
        vp: vp,
        vs: vs,
        qp: qp,
        qs: qs,
    });

    let layer2 = LayerGeometry::new_with_flat_surface(&bounding_polygon, 3000.0, 20000.0);

    let model2 = Model::Uniform(Quality {
        rho: 2360.0,
        vp: 5590.0,
        vs: 3330.0,
        qp: 333.0,
        qs: 166.5,
    });

    let mut prisms = [layer1, layer2];
    let models = &[model1, model2];
    let model_tree = ModelTree::new(&mut prisms, models);

    // 1. Iterate over the volume and rasterise the model
    // Assuming GeoModelGrid expects an Array4 (nx, ny, nz, num_fields) or similar
    let mut block_values = Array4::<f32>::zeros((nx, ny, nz, 7));

    let cos_a = azimuth.cos();
    let sin_a = azimuth.sin();
    println!("Assigning velocities.");
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // Local offsets from origin
                let dy = (i as f32) * resolution; // Outer axis points north
                let dx = (j as f32) * resolution;
                let dz = (k as f32) * resolution_vertical; // Assuming z is depth (negative)

                // Rotate local offsets by azimuth and translate to origin
                // Assuming azimuth is relative to the Y axis (North)
                let global_x = origin_x + (dx * cos_a - dy * sin_a);
                let global_y = origin_y + (dx * sin_a + dy * cos_a);

                let query_point = Point3::new(global_x, global_y, dz);

                // Query the tree and extract quality
                let quality = model_tree
                    .query(query_point, 1e-6)
                    .expect("Point outside of defined model layers");

                // Assign to the block array
                block_values[[i, j, k, 0]] = quality.rho;
                block_values[[i, j, k, 1]] = quality.vp;
                block_values[[i, j, k, 2]] = quality.vs;
                block_values[[i, j, k, 3]] = quality.qp;
                block_values[[i, j, k, 4]] = quality.qs;
                block_values[[i, j, k, 5]] = 0.0; // fault_block_id (placeholder)
                block_values[[i, j, k, 6]] = 0.0; // zone_id (placeholder)
            }
        }
    }

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
