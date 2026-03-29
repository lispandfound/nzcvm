use std::path::PathBuf;

use ndarray::Array2;

use crate::{
    crs::NZTM,
    geomodelgrid::{Block, GeoModelGrid},
    rfile::write_rfile,
    uniformmodel::uniform_values,
};

mod crs;
mod geomodelgrid;
mod rfile;
mod uniformmodel;

fn main() {
    let nx = 500;
    let ny = 500;
    let nz = 100;
    let shape = (nx, ny, nz);
    let vp = 3500.0;
    let vs = 1860.0;
    let rho = 2320.0;
    let qp = 208.48;
    let qs = 104.24;

    let resolution = 125.0;
    let resolution_vertical = 125.0;
    let width = (nx as f64) * (resolution as f64);
    let height = (ny as f64) * (resolution as f64);

    let block_values =
        uniform_values(shape, vp, vs, rho, qp, qs).expect("Could not generate the block array");
    let block = Block {
        name: "block_0".to_string(),
        resolution_horiz: resolution,
        resolution_vert: resolution_vertical,
        z_top: 0.0,
        block: block_values,
    };
    let topo = geomodelgrid::Surface {
        surface: Array2::zeros((nx, ny)),
        resolution_horiz: resolution,
        name: "top_surface".to_string(),
    };
    let topo_bathy = geomodelgrid::Surface {
        surface: Array2::zeros((nx, ny)),
        resolution_horiz: resolution,
        name: "topography_bathymetry".to_string(),
    };

    let model_grid = GeoModelGrid::builder()
        // Basic Metadata
        .title("Example")
        // Data
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
        // Coordinates
        .crs(NZTM) // Assuming NZTM is a variable/constant in scope
        .origin_x(172.0) /* TODO: fill in */
        .origin_y(-43.0) /* TODO: fill in */
        .y_azimuth(0.1)
        .dim_x(width) /* TODO: fill in */
        .dim_y(height) /* TODO: fill in */
        .dim_z((nz as f64) * (resolution_vertical as f64)) /* TODO: fill in */
        .add_block(block)
        .add_surface(topo)
        // .add_surface(topo_bathy)
        .build();
    write_rfile(&model_grid, PathBuf::from("velocity_model.rfile")).expect("Cannot write rfile");
}
