use crate::real::Real;

/// Grouped Metadata Sub-structs
#[derive(Default, Clone, Debug)]
pub struct BasicMetadata {
    pub title: String,
    pub id: String,
    pub description: String,
    pub version: String,
    pub history: String,
    pub comment: String,
    pub license: String,
    pub keywords: Vec<String>,
    pub auxiliary: String,
}

#[derive(Default, Clone, Debug)]
pub struct AttributionMetadata {
    pub creator_name: String,
    pub creator_email: String,
    pub creator_institution: String,
    pub acknowledgement: String,
    pub authors: Vec<String>,
    pub references: Vec<String>,
}

#[derive(Default, Clone, Debug)]
pub struct RepositoryMetadata {
    pub repository_doi: String,
    pub repository_name: String,
    pub repository_url: String,
}

#[derive(Default, Clone, Debug)]
pub struct DataMetadata {
    pub data_layout: String,
    pub data_units: Vec<String>,
    pub data_values: Vec<String>,
}

#[derive(Default, Clone, Debug)]
pub struct CoordinateMetadata {
    pub crs: String,
    pub origin_x: f64,
    pub origin_y: f64,
    pub y_azimuth: f64,
    pub dim_x: f64,
    pub dim_y: f64,
    pub dim_z: f64,
}

/// Represents the Root attributes, now cleanly grouped
#[derive(Default, Clone, Debug)]
pub struct ModelMetadata {
    pub basic: BasicMetadata,
    pub attribution: AttributionMetadata,
    pub repository: RepositoryMetadata,
    pub data: DataMetadata,
    pub coords: CoordinateMetadata,
}

pub type Dimensions = (usize, usize, usize, usize);

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub resolution_horiz: f32,
    pub resolution_vert: Real,
    pub z_top: Real,
    pub shape: Dimensions,
    pub name: String,
}

impl Block {
    pub fn components(&self) -> usize {
        let (_, _, _, nc) = self.shape;
        nc
    }

    pub fn size(&self) -> usize {
        let (nx, ny, nz, nc) = self.shape;
        nx * ny * nz * nc * size_of::<Real>()
    }
}

pub struct Surface {
    pub shape: (usize, usize),
    pub resolution_horiz: Real,
    pub name: String,
}

impl Surface {
    pub fn size(&self) -> usize {
        let (nx, ny) = self.shape;
        nx * ny * size_of::<Real>()
    }
}

pub struct GeoModelGrid {
    pub metadata: ModelMetadata,
    pub surfaces: Vec<Surface>,
    pub blocks: Vec<Block>,
}

// --- Builder Pattern Implementation ---

#[derive(Default)]
pub struct GeoModelGridBuilder {
    metadata: ModelMetadata,
    surfaces: Vec<Surface>,
    blocks: Vec<Block>,
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
