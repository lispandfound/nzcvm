use byteorder::{LittleEndian, WriteBytesExt};
use ndarray::{ArrayView4, Axis};
use std::io::{Result, Write};

use crate::geomodelgrid::{Block, GeoModelGrid};

const MAGIC: i32 = 1;
const PRECISION: i32 = 4;

/// The central trait defining a specific output format.
pub trait ModelFormat {
    fn write_metadata(&mut self, model: &GeoModelGrid) -> Result<()>;
    fn write_chunk(&mut self, chunk: &Chunk, buffer: ArrayView4<f32>) -> Result<()>;
    fn order(&self) -> [usize; 3];
    fn chunksizes(&self, model: &GeoModelGrid, buffer_size: usize) -> (usize, usize, usize);
}

// -----------------------------------------------------------------------------
// RFile Implementation
// -----------------------------------------------------------------------------
pub struct RFileFormat<W: Write> {
    handle: W,
}

impl<W: Write> RFileFormat<W> {
    pub fn new(handle: W) -> Self {
        Self { handle }
    }
}

impl<W: Write> ModelFormat for RFileFormat<W> {
    fn write_metadata(&mut self, model: &GeoModelGrid) -> Result<()> {
        self.handle.write_i32::<LittleEndian>(MAGIC)?;
        self.handle.write_i32::<LittleEndian>(PRECISION)?;

        let att_flag: i32 = model
            .blocks
            .first()
            .map(|block| if block.components() == 5 { 1 } else { 0 })
            .unwrap_or(0);
        self.handle.write_i32::<LittleEndian>(att_flag)?;

        self.handle
            .write_f64::<LittleEndian>(model.metadata.coords.y_azimuth)?;
        self.handle
            .write_f64::<LittleEndian>(model.metadata.coords.origin_x)?;
        self.handle
            .write_f64::<LittleEndian>(model.metadata.coords.origin_y)?;

        let mercstr = &model.metadata.coords.crs;
        self.handle
            .write_i32::<LittleEndian>(mercstr.len() as i32)?;
        self.handle.write_all(mercstr.as_bytes())?;

        let nb = (model.surfaces.len() + model.blocks.len()) as i32;
        self.handle.write_i32::<LittleEndian>(nb)?;

        if let Some(topo) = model.surfaces.first() {
            let (ni, nj) = (topo.shape.0, topo.shape.1);
            self.handle
                .write_f64::<LittleEndian>(topo.resolution_horiz as f64)?; // hhb
            self.handle.write_f64::<LittleEndian>(0.0)?; // hvb (not used for topo)
            self.handle.write_f64::<LittleEndian>(0.0)?; // z0b (not used for topo)
            self.handle.write_i32::<LittleEndian>(1)?; // ncb (topo is 1 component)
            self.handle.write_i32::<LittleEndian>(ni as i32)?;
            self.handle.write_i32::<LittleEndian>(nj as i32)?;
            self.handle.write_i32::<LittleEndian>(1)?; // nkb (topo is 2D)
        }

        for block in &model.blocks {
            let (ni, nj, nk, nc) = block.shape; // (ni, nj, nk, nc)
            self.handle
                .write_f64::<LittleEndian>(block.resolution_horiz as f64)?;
            self.handle
                .write_f64::<LittleEndian>(block.resolution_vert as f64)?;
            self.handle.write_f64::<LittleEndian>(block.z_top as f64)?;
            self.handle.write_i32::<LittleEndian>(nc as i32)?; // ncb
            self.handle.write_i32::<LittleEndian>(ni as i32)?; // nib
            self.handle.write_i32::<LittleEndian>(nj as i32)?; // njb
            self.handle.write_i32::<LittleEndian>(nk as i32)?; // nkb
        }

        Ok(())
    }

    fn write_chunk(&mut self, _chunk: &Chunk, buffer: ArrayView4<f32>) -> Result<()> {
        for el in buffer.iter() {
            self.handle.write_f32::<LittleEndian>(*el)?;
        }
        Ok(())
    }

    fn order(&self) -> [usize; 3] {
        [2, 1, 0]
    }

    fn chunksizes(&self, model: &GeoModelGrid, buffer_size: usize) -> (usize, usize, usize) {
        let max_block = model
            .blocks
            .iter()
            .max_by_key(|block| block.size())
            .expect("Empty blocks");

        let optimal_size = optimal_chunksize_for(
            [max_block.shape.0, max_block.shape.1, max_block.shape.2],
            self.order(),
            buffer_size,
        );
        (optimal_size[0], optimal_size[1], optimal_size[2])
    }
}

// -----------------------------------------------------------------------------
// EMOD3D Implementation
// -----------------------------------------------------------------------------
pub struct Emod3dFormat<W: Write> {
    rho: W,
    vp: W,
    vs: W,
}

impl<W: Write> Emod3dFormat<W> {
    pub fn new(rho: W, vp: W, vs: W) -> Self {
        Self { rho, vp, vs }
    }
}

impl<W: Write> ModelFormat for Emod3dFormat<W> {
    fn write_metadata(&mut self, _model: &GeoModelGrid) -> Result<()> {
        // EMOD3D files have no metadata!
        Ok(())
    }

    fn write_chunk(&mut self, _chunk: &Chunk, buffer: ArrayView4<f32>) -> Result<()> {
        for quality in buffer.lanes(Axis(3)) {
            self.rho.write_f32::<LittleEndian>(quality[0])?;
            self.vp.write_f32::<LittleEndian>(quality[1])?;
            self.vs.write_f32::<LittleEndian>(quality[2])?;
        }
        Ok(())
    }

    fn order(&self) -> [usize; 3] {
        [1, 2, 0]
    }

    fn chunksizes(&self, model: &GeoModelGrid, buffer_size: usize) -> (usize, usize, usize) {
        let block = model.blocks.first().expect("Empty blocks");
        let shape = [block.shape.0, block.shape.1, block.shape.2];
        let optimal_chunk = optimal_chunksize_for(shape, self.order(), buffer_size);
        (optimal_chunk[0], optimal_chunk[1], optimal_chunk[2])
    }
}

// -----------------------------------------------------------------------------
// The Writer Orchestrator
// -----------------------------------------------------------------------------
pub struct VelocityModelWriter<'a, F: ModelFormat> {
    pub model: &'a GeoModelGrid,
    pub buffer_size: usize,
    pub format: F,
}

impl<'a, F: ModelFormat> VelocityModelWriter<'a, F> {
    pub fn write_metadata(&mut self) -> Result<()> {
        self.format.write_metadata(&self.model)
    }

    pub fn write_chunk(&mut self, chunk: &Chunk, buffer: ArrayView4<f32>) -> Result<()> {
        self.format.write_chunk(chunk, buffer)
    }

    pub fn order(&self) -> [usize; 3] {
        self.format.order()
    }

    pub fn chunksizes(&self) -> (usize, usize, usize) {
        self.format.chunksizes(&self.model, self.buffer_size)
    }

    pub fn chunks(&self) -> ChunkIterator {
        let (nx, ny, nz) = self.chunksizes();
        let chunks = [nx, ny, nz];
        ChunkIterator::new(self.model.blocks.clone(), self.format.order(), chunks)
    }
}

// -----------------------------------------------------------------------------
// Core Utilities (Chunking & Iteration)
// -----------------------------------------------------------------------------
pub fn optimal_chunksize_for(
    shape: [usize; 3],
    order: [usize; 3],
    max_elements: usize,
) -> [usize; 3] {
    let mut remaining_capacity = max_elements;
    let mut chunksize = [1; 3];

    for &c in order.iter() {
        if remaining_capacity <= 1 {
            break;
        }
        let length = shape[c].min(remaining_capacity);
        chunksize[c] = length;
        remaining_capacity /= length;
    }
    chunksize
}

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub block: Block,
    pub start: (usize, usize, usize),
    pub shape: (usize, usize, usize, usize),
}

pub struct ChunkIterator {
    blocks: Vec<Block>,
    block_idx: usize,
    order: [usize; 3],
    start: [usize; 3],
    strides: [usize; 3],
}

impl ChunkIterator {
    pub fn new(blocks: Vec<Block>, order: [usize; 3], strides: [usize; 3]) -> Self {
        ChunkIterator {
            blocks,
            block_idx: 0,
            start: [0; 3],
            order,
            strides,
        }
    }
}

impl Iterator for ChunkIterator {
    type Item = Chunk;

    fn next(&mut self) -> Option<Self::Item> {
        if self.block_idx >= self.blocks.len() {
            return None;
        }
        let block = self.blocks[self.block_idx].clone();
        let (nx, ny, nz, nc) = block.shape;
        let size = [nx, ny, nz];
        let mut current_chunk_shape = self.strides;

        for i in 0..3 {
            current_chunk_shape[i] = current_chunk_shape[i].min(size[i] - self.start[i]);
        }

        // This implements the stride-order wrapping logic. It tries
        // to increment each component in the prescribed order and
        // wraps to the next coordinate axis if we wrap around.
        let current = self.start;
        let mut wrapped_all = true;
        for &c in self.order.iter() {
            self.start[c] += self.strides[c];
            if self.start[c] >= size[c] {
                self.start[c] = 0;
            } else {
                wrapped_all = false;
                break;
            }
        }

        // If we wrapped all, then the block index needs to increase by one.
        if wrapped_all {
            self.block_idx += 1;
        }

        Some(Chunk {
            block,
            start: (current[0], current[1], current[2]),
            shape: (
                current_chunk_shape[0],
                current_chunk_shape[1],
                current_chunk_shape[2],
                nc,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::geomodelgrid::{Block, GeoModelGrid, GeoModelGridBuilder};

    use super::*;
    use ndarray::Array4;

    fn dummy_model() -> GeoModelGrid {
        GeoModelGridBuilder::new()
            .origin_x(2.0)
            .origin_y(3.0)
            .y_azimuth(1.0)
            .crs("EPSG:4326")
            .add_block(Block {
                resolution_horiz: 100.0,
                resolution_vert: 100.0,
                z_top: 0.0,
                shape: (10, 10, 10, 3),
                name: "dummy".into(),
            })
            .build()
    }

    // -------------------------------------------------------------------------
    // 1. Test Chunking Math
    // -------------------------------------------------------------------------
    #[test]
    fn test_optimal_chunksize() {
        let shape = [10, 10, 10];
        // Test 1: Buffer is big enough for everything
        let res = optimal_chunksize_for(shape, [2, 1, 0], 1000);
        assert_eq!(res, [10, 10, 10]);

        // Test 2: Row-major priority (Z first in order array [2, 1, 0])
        // With 50 elements, it should grab all of Z(10), then 5 of Y, 1 of X
        let res = optimal_chunksize_for(shape, [2, 1, 0], 50);
        assert_eq!(res, [1, 5, 10]);

        // Test 3: Very small buffer (smaller than 1 dimension)
        let res = optimal_chunksize_for(shape, [2, 1, 0], 3);
        assert_eq!(res, [1, 1, 3]);
    }

    // -------------------------------------------------------------------------
    // 2. Test Iteration Logic
    // -------------------------------------------------------------------------
    #[test]
    fn test_chunk_iterator_edge_clamping() {
        let block = Block {
            resolution_horiz: 1.0,
            resolution_vert: 1.0,
            z_top: 0.0,
            shape: (5, 5, 1, 3),
            name: "dummy".into(),
        };
        let mut iter = ChunkIterator {
            blocks: vec![block],
            block_idx: 0,
            order: [0, 1, 2], // Iterate X, then Y, then Z
            start: [0, 0, 0],
            strides: [3, 3, 1], // Stride is 3, but block is 5 (forces partials)
        };

        // 1st chunk: (0,0,0), full stride in X and Y available but clamped at Y?
        // Wait, start is 0,0. X goes to 3. Y goes to 3.
        let c1 = iter.next().unwrap();
        assert_eq!(c1.start, (0, 0, 0));
        assert_eq!(c1.shape, (3, 3, 1, 3));

        // 2nd chunk: X advances to 3. Y is still 0.
        let c2 = iter.next().unwrap();
        assert_eq!(c2.start, (3, 0, 0));
        assert_eq!(c2.shape, (2, 3, 1, 3)); // X clamped to (5-3)=2

        // 3rd chunk: X wraps. Y advances to 3.
        let c3 = iter.next().unwrap();
        assert_eq!(c3.start, (0, 3, 0));
        assert_eq!(c3.shape, (3, 2, 1, 3)); // Y clamped to (5-3)=2

        // 4th chunk: X advances to 3. Y is 3.
        let c4 = iter.next().unwrap();
        assert_eq!(c4.start, (3, 3, 0));
        assert_eq!(c4.shape, (2, 2, 1, 3)); // Both X and Y clamped

        // Iteration complete
        assert!(iter.next().is_none());
    }

    // -------------------------------------------------------------------------
    // 3. Test RFile Output (Trait specific logic)
    // -------------------------------------------------------------------------
    #[test]
    fn test_rfile_writing_metadata() {
        let mut out_buffer = Vec::new();
        let model = dummy_model();
        {
            let mut writer = VelocityModelWriter {
                model: &model,
                buffer_size: 1000,
                format: RFileFormat::new(&mut out_buffer),
            };

            // Test metadata output
            writer.write_metadata().unwrap();
        }
        // 4 bytes (MAGIC) + 4 bytes (PREC) + 4 bytes (att) + 8 (azim) + 8 (ox) + 8 (oy)
        // + 4 (crs len) + 9 ("EPSG:4326") + 4 (nb) + 8 (horiz res) + 8 (vert res) + 8 (ztop) + 4*4 (nc, ni, nj, nk) = 93 bytes
        assert_eq!(out_buffer.len(), 93);
    }

    #[test]
    fn test_rfile_writing_blocks() {
        // Test chunk output
        let mut out_buffer = Vec::new();
        let data = Array4::<f32>::ones((1, 1, 1, 3)); // 3 elements
        let model = dummy_model();
        let chunk = Chunk {
            block: model.blocks[0].clone(),
            start: (0, 0, 0),
            shape: (1, 1, 1, 3),
        };

        {
            let mut writer = VelocityModelWriter {
                model: &model,
                buffer_size: 1000,
                format: RFileFormat::new(&mut out_buffer),
            };

            // Test metadata output
            writer
                .write_chunk(&chunk, data.view())
                .expect("Should succeed");
        }

        // Should have added 3 f32s (12 bytes)
        assert_eq!(out_buffer.len(), 12);
    }

    // -------------------------------------------------------------------------
    // 4. Test EMOD3D Output (Lane Splitting)
    // -------------------------------------------------------------------------
    #[test]
    fn test_emod3d_writing() {
        let mut rho_buf = Vec::new();
        let mut vp_buf = Vec::new();
        let mut vs_buf = Vec::new();
        let model = dummy_model();
        let mut writer = VelocityModelWriter {
            model: &model,
            buffer_size: 1000,
            format: Emod3dFormat::new(&mut rho_buf, &mut vp_buf, &mut vs_buf),
        };

        // Create an array where the 3 components are distinct: [1.0, 2.0, 3.0]
        let mut data = Array4::<f32>::zeros((1, 1, 2, 3));
        // 2 elements in the Z axis, 3 lanes.
        data[[0, 0, 0, 0]] = 1.0; // rho
        data[[0, 0, 0, 1]] = 2.0; // vp
        data[[0, 0, 0, 2]] = 3.0; // vs

        data[[0, 0, 1, 0]] = 10.0; // rho
        data[[0, 0, 1, 1]] = 20.0; // vp
        data[[0, 0, 1, 2]] = 30.0; // vs

        let chunk = Chunk {
            block: writer.model.blocks[0].clone(),
            start: (0, 0, 0),
            shape: (1, 1, 2, 3),
        };

        writer.write_chunk(&chunk, data.view()).unwrap();

        // Each buffer should have exactly 2 f32s written (8 bytes)
        assert_eq!(rho_buf.len(), 8);
        assert_eq!(vp_buf.len(), 8);
        assert_eq!(vs_buf.len(), 8);
    }
}
