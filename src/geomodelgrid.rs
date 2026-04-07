use crate::real::Real;
use pyo3::prelude::*;
use std::mem::size_of;

/// Grouped Metadata containing descriptive information about the model.
#[derive(Default, Clone, Debug, PartialEq)]
#[pyclass(get_all, set_all, from_py_object)]
pub struct BasicMetadata {
    /// The primary title of the model grid.
    pub title: String,
    /// A unique identifier for the model.
    pub id: String,
    /// A long-form description of the model purpose and content.
    pub description: String,
    /// Semantic versioning string (e.g., "1.0.0").
    pub version: String,
    /// Historical record of changes made to this model.
    pub history: String,
    /// General developer or user comments.
    pub comment: String,
    /// Usage license (e.g., "MIT", "CC-BY-4.0").
    pub license: String,
    /// List of tags or keywords for searching.
    pub keywords: Vec<String>,
    /// Additional auxiliary information in string format.
    pub auxiliary: String,
}

#[pymethods]
impl BasicMetadata {
    #[new]
    #[pyo3(signature = (title=String::new(), id=String::new(), description=String::new(), version=String::new(), history=String::new(), comment=String::new(), license=String::new(), keywords=vec![], auxiliary=String::new()))]
    fn new(
        title: String,
        id: String,
        description: String,
        version: String,
        history: String,
        comment: String,
        license: String,
        keywords: Vec<String>,
        auxiliary: String,
    ) -> Self {
        Self {
            title,
            id,
            description,
            version,
            history,
            comment,
            license,
            keywords,
            auxiliary,
        }
    }
}

/// Metadata regarding the creators and scholarly references of the model.
#[derive(Default, Clone, Debug, PartialEq)]
#[pyclass(get_all, set_all, from_py_object)]
pub struct AttributionMetadata {
    /// Name of the primary creator.
    pub creator_name: String,
    /// Contact email for the creator.
    pub creator_email: String,
    /// Institution associated with the creation.
    pub creator_institution: String,
    /// General acknowledgements for funding or support.
    pub acknowledgement: String,
    /// List of contributing authors.
    pub authors: Vec<String>,
    /// List of academic or technical references (DOIs, URLs).
    pub references: Vec<String>,
}

#[pymethods]
impl AttributionMetadata {
    #[new]
    #[pyo3(signature = (creator_name=String::new(), creator_email=String::new(), creator_institution=String::new(), acknowledgement=String::new(), authors=vec![], references=vec![]))]
    fn new(
        creator_name: String,
        creator_email: String,
        creator_institution: String,
        acknowledgement: String,
        authors: Vec<String>,
        references: Vec<String>,
    ) -> Self {
        Self {
            creator_name,
            creator_email,
            creator_institution,
            acknowledgement,
            authors,
            references,
        }
    }
}

/// Information about where the model data is hosted or archived.
#[derive(Default, Clone, Debug, PartialEq)]
#[pyclass(get_all, set_all, from_py_object)]
pub struct RepositoryMetadata {
    /// DOI for the repository entry.
    pub repository_doi: String,
    /// Name of the hosting repository (e.g., "Zenodo").
    pub repository_name: String,
    /// Direct URL to the repository.
    pub repository_url: String,
}

#[pymethods]
impl RepositoryMetadata {
    #[new]
    #[pyo3(signature = (repository_doi=String::new(), repository_name=String::new(), repository_url=String::new()))]
    fn new(repository_doi: String, repository_name: String, repository_url: String) -> Self {
        Self {
            repository_doi,
            repository_name,
            repository_url,
        }
    }
}

/// Metadata describing the internal data formatting and units.
#[derive(Default, Clone, Debug, PartialEq)]
#[pyclass(get_all, set_all, from_py_object)]
pub struct DataMetadata {
    /// Description of how data is laid out (e.g., "vertex-centered").
    pub data_layout: String,
    /// List of units for each data component (e.g., ["m/s", "kg/m3"]).
    pub data_units: Vec<String>,
    /// Names or labels for the data values stored in blocks.
    pub data_values: Vec<String>,
}

#[pymethods]
impl DataMetadata {
    #[new]
    #[pyo3(signature = (data_layout=String::new(), data_units=vec![], data_values=vec![]))]
    fn new(data_layout: String, data_units: Vec<String>, data_values: Vec<String>) -> Self {
        Self {
            data_layout,
            data_units,
            data_values,
        }
    }
}

/// Spatial reference and coordinate system metadata.
#[derive(Default, Clone, Debug, PartialEq)]
#[pyclass(get_all, set_all, from_py_object)]
pub struct CoordinateMetadata {
    /// Coordinate Reference System string (e.g., "EPSG:2193").
    pub crs: String,
    /// X-coordinate of the grid origin.
    pub origin_x: f64,
    /// Y-coordinate of the grid origin.
    pub origin_y: f64,
    /// Azimuth angle of the Y-axis in degrees.
    pub y_azimuth: f64,
    /// Total dimension of the grid in the X direction.
    pub dim_x: f64,
    /// Total dimension of the grid in the Y direction.
    pub dim_y: f64,
    /// Total dimension of the grid in the Z direction.
    pub dim_z: f64,
}

#[pymethods]
impl CoordinateMetadata {
    #[new]
    #[pyo3(signature = (crs=String::new(), origin_x=0.0, origin_y=0.0, y_azimuth=0.0, dim_x=0.0, dim_y=0.0, dim_z=0.0))]
    fn new(
        crs: String,
        origin_x: f64,
        origin_y: f64,
        y_azimuth: f64,
        dim_x: f64,
        dim_y: f64,
        dim_z: f64,
    ) -> Self {
        Self {
            crs,
            origin_x,
            origin_y,
            y_azimuth,
            dim_x,
            dim_y,
            dim_z,
        }
    }
}

/// Root metadata object that aggregates all sub-metadata categories.
#[derive(Default, Clone, Debug, PartialEq)]
#[pyclass(get_all, set_all, from_py_object)]
pub struct ModelMetadata {
    /// Basic metadata (title, id, description, licence, ...)
    pub basic: BasicMetadata,
    /// Attribution metadata (institution, creator name and email, ...)
    pub attribution: AttributionMetadata,
    /// Repository metadata, if applicable, (DOI, name, email, ...)
    pub repository: RepositoryMetadata,
    /// Metadata describing the layout and semantics of the data (units, column names).
    pub data: DataMetadata,
    /// Metadata describing the coordinate system
    pub coords: CoordinateMetadata,
}

#[pymethods]
impl ModelMetadata {
    #[new]
    #[pyo3(signature = (basic=None, attribution=None, repository=None, data=None, coords=None))]
    fn new(
        basic: Option<BasicMetadata>,
        attribution: Option<AttributionMetadata>,
        repository: Option<RepositoryMetadata>,
        data: Option<DataMetadata>,
        coords: Option<CoordinateMetadata>,
    ) -> Self {
        Self {
            basic: basic.unwrap_or_default(),
            attribution: attribution.unwrap_or_default(),
            repository: repository.unwrap_or_default(),
            data: data.unwrap_or_default(),
            coords: coords.unwrap_or_default(),
        }
    }
}

pub type Dimensions = (usize, usize, usize, usize);

/// Represents a 3D block of model data with specific resolution.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(get_all, set_all, from_py_object)]
pub struct Block {
    /// Horizontal resolution of the grid cells.
    pub resolution_horiz: f32,
    /// Vertical resolution of the grid cells.
    pub resolution_vert: Real,
    /// Depth or elevation at the top of the block.
    pub z_top: Real,
    /// Grid shape defined as (nx, ny, nz, n_components).
    pub shape: Dimensions,
    /// Human-readable name for the block.
    pub name: String,
}

#[pymethods]
impl Block {
    #[new]
    #[pyo3(signature = (resolution_horiz=0.0, resolution_vert=0.0, z_top=0.0, shape=(0,0,0,0), name=String::new()))]
    fn new(
        resolution_horiz: f32,
        resolution_vert: Real,
        z_top: Real,
        shape: Dimensions,
        name: String,
    ) -> Self {
        Self {
            resolution_horiz,
            resolution_vert,
            z_top,
            shape,
            name,
        }
    }

    /// Returns the number of data components per cell.
    pub fn components(&self) -> usize {
        self.shape.3
    }

    /// Returns the total memory size of the block data in bytes.
    pub fn size(&self) -> usize {
        let (nx, ny, nz, nc) = self.shape;
        nx * ny * nz * nc * size_of::<Real>()
    }
}

/// Represents a 2D surface within the model grid.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(get_all, set_all, from_py_object)]
pub struct Surface {
    /// Grid shape defined as (nx, ny).
    pub shape: (usize, usize),
    /// Horizontal resolution of the surface grid.
    pub resolution_horiz: Real,
    /// Human-readable name for the surface.
    pub name: String,
}

#[pymethods]
impl Surface {
    #[new]
    #[pyo3(signature = (shape=(0,0), resolution_horiz=0.0, name=String::new()))]
    fn new(shape: (usize, usize), resolution_horiz: Real, name: String) -> Self {
        Self {
            shape,
            resolution_horiz,
            name,
        }
    }

    /// Returns the total memory size of the surface data in bytes.
    pub fn size(&self) -> usize {
        let (nx, ny) = self.shape;
        nx * ny * size_of::<Real>()
    }
}

/// The top-level GeoModelGrid container.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(get_all, set_all, from_py_object)]
pub struct GeoModelGrid {
    /// All metadata associated with the model.
    pub metadata: ModelMetadata,
    /// List of surfaces included in the model.
    pub surfaces: Vec<Surface>,
    /// List of 3D data blocks included in the model.
    pub blocks: Vec<Block>,
}

#[pymethods]
impl GeoModelGrid {
    #[new]
    #[pyo3(signature = (metadata=None, surfaces=vec![], blocks=vec![]))]
    fn new(metadata: Option<ModelMetadata>, surfaces: Vec<Surface>, blocks: Vec<Block>) -> Self {
        Self {
            metadata: metadata.unwrap_or_default(),
            surfaces,
            blocks,
        }
    }
}
// --- Builder Pattern Implementation ---

#[derive(Default)]
pub struct GeoModelGridBuilder {
    metadata: ModelMetadata,
    surfaces: Vec<Surface>,
    blocks: Vec<Block>,
}

#[pymodule]
pub fn geomodelgrid(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BasicMetadata>()?;
    m.add_class::<AttributionMetadata>()?;
    m.add_class::<RepositoryMetadata>()?;
    m.add_class::<DataMetadata>()?;
    m.add_class::<CoordinateMetadata>()?;
    m.add_class::<ModelMetadata>()?;

    m.add_class::<Block>()?;
    m.add_class::<Surface>()?;
    m.add_class::<GeoModelGrid>()?;

    Ok(())
}

impl GeoModelGridBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    // Basic Metadata
    pub fn title(mut self, val: impl Into<String>) -> Self {
        self.metadata.basic.title = val.into();
        self
    }
    pub fn id(mut self, val: impl Into<String>) -> Self {
        self.metadata.basic.id = val.into();
        self
    }
    pub fn description(mut self, val: impl Into<String>) -> Self {
        self.metadata.basic.description = val.into();
        self
    }
    pub fn version(mut self, val: impl Into<String>) -> Self {
        self.metadata.basic.version = val.into();
        self
    }
    pub fn history(mut self, val: impl Into<String>) -> Self {
        self.metadata.basic.history = val.into();
        self
    }
    pub fn comment(mut self, val: impl Into<String>) -> Self {
        self.metadata.basic.comment = val.into();
        self
    }
    pub fn license(mut self, val: impl Into<String>) -> Self {
        self.metadata.basic.license = val.into();
        self
    }
    pub fn auxiliary(mut self, val: impl Into<String>) -> Self {
        self.metadata.basic.auxiliary = val.into();
        self
    }
    pub fn keywords(mut self, val: Vec<String>) -> Self {
        self.metadata.basic.keywords = val;
        self
    }

    // Attribution Metadata
    pub fn creator_name(mut self, val: impl Into<String>) -> Self {
        self.metadata.attribution.creator_name = val.into();
        self
    }
    pub fn creator_email(mut self, val: impl Into<String>) -> Self {
        self.metadata.attribution.creator_email = val.into();
        self
    }
    pub fn creator_institution(mut self, val: impl Into<String>) -> Self {
        self.metadata.attribution.creator_institution = val.into();
        self
    }
    pub fn acknowledgement(mut self, val: impl Into<String>) -> Self {
        self.metadata.attribution.acknowledgement = val.into();
        self
    }
    pub fn authors(mut self, val: Vec<String>) -> Self {
        self.metadata.attribution.authors = val;
        self
    }
    pub fn references(mut self, val: Vec<String>) -> Self {
        self.metadata.attribution.references = val;
        self
    }

    // Repository Metadata
    pub fn repository_doi(mut self, val: impl Into<String>) -> Self {
        self.metadata.repository.repository_doi = val.into();
        self
    }
    pub fn repository_name(mut self, val: impl Into<String>) -> Self {
        self.metadata.repository.repository_name = val.into();
        self
    }
    pub fn repository_url(mut self, val: impl Into<String>) -> Self {
        self.metadata.repository.repository_url = val.into();
        self
    }

    // Data Metadata
    pub fn data_layout(mut self, val: impl Into<String>) -> Self {
        self.metadata.data.data_layout = val.into();
        self
    }
    pub fn data_units(mut self, val: Vec<String>) -> Self {
        self.metadata.data.data_units = val;
        self
    }
    pub fn data_values(mut self, val: Vec<String>) -> Self {
        self.metadata.data.data_values = val;
        self
    }

    // Coordinate Metadata
    pub fn crs(mut self, val: impl Into<String>) -> Self {
        self.metadata.coords.crs = val.into();
        self
    }
    pub fn origin_x(mut self, val: f64) -> Self {
        self.metadata.coords.origin_x = val;
        self
    }
    pub fn origin_y(mut self, val: f64) -> Self {
        self.metadata.coords.origin_y = val;
        self
    }
    pub fn y_azimuth(mut self, val: f64) -> Self {
        self.metadata.coords.y_azimuth = val;
        self
    }
    pub fn dim_x(mut self, val: f64) -> Self {
        self.metadata.coords.dim_x = val;
        self
    }
    pub fn dim_y(mut self, val: f64) -> Self {
        self.metadata.coords.dim_y = val;
        self
    }
    pub fn dim_z(mut self, val: f64) -> Self {
        self.metadata.coords.dim_z = val;
        self
    }

    // Payload Data
    pub fn add_surface(mut self, surf: Surface) -> Self {
        self.surfaces.push(surf);
        self
    }

    pub fn add_block(mut self, block: Block) -> Self {
        self.blocks.push(block);
        self
    }

    pub fn build(self) -> GeoModelGrid {
        GeoModelGrid {
            metadata: self.metadata,
            surfaces: self.surfaces,
            blocks: self.blocks,
        }
    }
}
