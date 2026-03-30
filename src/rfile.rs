use crate::geomodelgrid::GeoModelGrid;
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufWriter, Result, Write};
use std::path::Path; // Assuming these are in the parent crate

pub fn write_rfile<P: AsRef<Path>>(grid: &GeoModelGrid, path: P) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // 1. Determine Global Flags
    // We assume Little Endian for the 'magic' check.
    let magic: i32 = 1;
    let precision: i32 = 4; // We are using f32 (4 bytes) from the GeoModelGrid

    // Determine if attenuation is present (check the first material block)
    let att_flag: i32 = if let Some(first_block) = grid.blocks.first() {
        if first_block.block.shape()[3] == 5 {
            1
        } else {
            0
        }
    } else {
        0
    };

    let mercstr = &grid.metadata.coords.crs;
    let mlen = mercstr.len() as i32;
    let nb = (grid.surfaces.len() + grid.blocks.len()) as i32; // +1 for the topography surface

    // --- Header Part 1 ---
    writer.write_i32::<LittleEndian>(magic)?;
    writer.write_i32::<LittleEndian>(precision)?;
    writer.write_i32::<LittleEndian>(att_flag)?;
    writer.write_f64::<LittleEndian>(grid.metadata.coords.y_azimuth)?;
    writer.write_f64::<LittleEndian>(grid.metadata.coords.origin_x)?; // lon0
    writer.write_f64::<LittleEndian>(grid.metadata.coords.origin_y)?; // lat0
    writer.write_i32::<LittleEndian>(mlen)?;
    writer.write_all(mercstr.as_bytes())?;
    writer.write_i32::<LittleEndian>(nb)?;

    // --- Header Part 2: Block Metadata ---
    // Block 1: Topography (Surface)
    if let Some(topo) = grid.surfaces.first() {
        let (ni, nj) = (topo.surface.shape()[0], topo.surface.shape()[1]);
        writer.write_f64::<LittleEndian>(topo.resolution_horiz as f64)?; // hhb
        writer.write_f64::<LittleEndian>(0.0)?; // hvb (not used for topo)
        writer.write_f64::<LittleEndian>(0.0)?; // z0b (not used for topo)
        writer.write_i32::<LittleEndian>(1)?; // ncb (topo is 1 component)
        writer.write_i32::<LittleEndian>(ni as i32)?;
        writer.write_i32::<LittleEndian>(nj as i32)?;
        writer.write_i32::<LittleEndian>(1)?; // nkb (topo is 2D)
    }
    // Blocks 2..Nb: Material Properties
    for block in &grid.blocks {
        let shape = block.block.shape(); // (ni, nj, nk, nc)
        writer.write_f64::<LittleEndian>(block.resolution_horiz as f64)?;
        writer.write_f64::<LittleEndian>(block.resolution_vert as f64)?;
        writer.write_f64::<LittleEndian>(block.z_top as f64)?;
        writer.write_i32::<LittleEndian>(shape[3] as i32)?; // ncb
        writer.write_i32::<LittleEndian>(shape[0] as i32)?; // nib
        writer.write_i32::<LittleEndian>(shape[1] as i32)?; // njb
        writer.write_i32::<LittleEndian>(shape[2] as i32)?; // nkb
    }
    if let Some(topo) = grid.surfaces.first() {
        for &val in topo.surface.iter() {
            writer.write_f32::<LittleEndian>(val)?;
        }
    }

    for block_data in &grid.blocks {
        for &val in block_data.block.iter() {
            writer.write_f32::<LittleEndian>(val)?;
        }
    }

    writer.flush()?;
    Ok(())
}
