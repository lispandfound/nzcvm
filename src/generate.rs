use std::io::Result;

use crate::coordinates::CoordinateSystem;
use crate::model::ModelTree;
use crate::writer::{ModelFormat, VelocityModelWriter};
use ndarray::{par_azip, Array1, Array4, Axis};

pub fn write_model_parallel<'a, C, F>(
    coordinate_system: &C,
    model_tree: &ModelTree,
    mut writer: VelocityModelWriter<F>,
) -> Result<()>
where
    C: CoordinateSystem + Sync,
    F: ModelFormat,
{
    writer.write_metadata()?;

    let mut buffer: Array4<f32> = Array4::zeros((0, 0, 0, 0));

    for chunk in writer.chunks() {
        let (nx, ny, nz, nc) = chunk.shape;
        if buffer.shape() != [nx, ny, nz, nc] {
            buffer = Array4::zeros((nx, ny, nz, nc));
        }

        let dx = chunk.block.resolution_horiz;
        let dy = chunk.block.resolution_horiz;
        let dz = chunk.block.resolution_vert;
        let (gi, gj, gk) = chunk.start;

        par_azip!((index (i, j, k), mut lane in buffer.lanes_mut(Axis(3))) {
            let global_indices = (gi + i, gj + j, gk + k);
            let point = coordinate_system.coordinates(global_indices, dx, dy, dz);

            if let Some((quality, _)) = model_tree.query(point) {
                let quality_arr: Array1<f32> = quality.into();
                lane.assign(&quality_arr);
            }
        });

        writer.write_chunk(&chunk, buffer.view())?;
    }

    Ok(())
}
