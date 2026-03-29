use hdf5_metno::types::{FixedAscii, FixedUnicode};
use hdf5_metno::{
    Attribute, Error, File, Group, Result, types::TypeDescriptor, types::VarLenAscii,
    types::VarLenUnicode,
};
use ndarray::{Array2, Array4};
use std::{path::PathBuf, str::FromStr};

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

pub struct Block {
    pub resolution_horiz: f32,
    pub resolution_vert: f32,
    pub z_top: f32,
    pub block: Array4<f32>,
    pub name: String,
}

pub struct Surface {
    pub surface: Array2<f32>,
    pub resolution_horiz: f32,
    pub name: String,
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

// --- End Builder Pattern ---

const N: usize = 2048;

fn read_h5_string(attr: &Attribute) -> Result<String> {
    let dtype = attr.dtype()?;
    match dtype.to_descriptor()? {
        TypeDescriptor::VarLenUnicode => {
            let v = attr.read_scalar::<VarLenUnicode>()?;
            Ok(v.as_str().to_string())
        }
        TypeDescriptor::VarLenAscii => {
            let v = attr.read_scalar::<VarLenAscii>()?;
            Ok(v.as_str().to_string())
        }
        TypeDescriptor::FixedAscii(_len) => {
            let str: hdf5_metno::types::FixedAscii<N> = attr.read_scalar::<FixedAscii<_>>()?;
            Ok(str.to_string())
        }
        TypeDescriptor::FixedUnicode(_len) => {
            let str: hdf5_metno::types::FixedUnicode<N> = attr.read_scalar::<FixedUnicode<_>>()?;
            Ok(str.to_string())
        }
        _ => Err(Error::from("Expected a string datatype".to_string())),
    }
}

fn read_h5_string_array(attr: &Attribute) -> Result<Vec<String>> {
    let dtype = attr.dtype()?;
    match dtype.to_descriptor()? {
        TypeDescriptor::VarLenUnicode => {
            let v = attr.read_1d::<VarLenUnicode>()?;
            Ok(v.iter().map(|s| s.as_str().to_string()).collect())
        }
        TypeDescriptor::VarLenAscii => {
            let v = attr.read_1d::<VarLenAscii>()?;
            Ok(v.iter().map(|s| s.as_str().to_string()).collect())
        }
        TypeDescriptor::FixedAscii(_len) => {
            let raw_vec = attr.read_1d::<FixedAscii<N>>()?;
            Ok(raw_vec.iter().map(|s| s.to_string()).collect())
        }
        TypeDescriptor::FixedUnicode(_len) => {
            let raw_vec = attr.read_1d::<FixedUnicode<N>>()?;
            Ok(raw_vec.iter().map(|s| s.to_string()).collect())
        }
        _ => Err(Error::from("Expected a string array datatype".to_string())),
    }
}

impl GeoModelGrid {
    pub fn builder() -> GeoModelGridBuilder {
        GeoModelGridBuilder::new()
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let root = file.as_group()?;

        self.write_metadata(&root)?;

        let surfaces_group = root.create_group("surfaces")?;
        for surface_data in self.surfaces.iter() {
            let ds = surfaces_group
                .new_dataset::<f32>()
                .shape(surface_data.surface.shape())
                .create(surface_data.name.as_str())?;
            ds.write(&surface_data.surface)?;

            ds.new_attr::<f32>()
                .create("resolution_horiz")?
                .write_scalar(&surface_data.resolution_horiz)?;
        }

        let blocks_group = root.create_group("blocks")?;
        for block_data in self.blocks.iter() {
            let ds = blocks_group
                .new_dataset::<f32>()
                .shape(block_data.block.shape())
                .create(block_data.name.as_str())?;
            ds.write(&block_data.block)?;

            ds.new_attr::<f32>()
                .create("resolution_horiz")?
                .write_scalar(&block_data.resolution_horiz)?;
            ds.new_attr::<f32>()
                .create("resolution_vert")?
                .write_scalar(&block_data.resolution_vert)?;
            ds.new_attr::<f32>()
                .create("z_top")?
                .write_scalar(&block_data.z_top)?;
        }

        Ok(())
    }

    fn write_metadata(&self, group: &Group) -> Result<()> {
        let write_str = |name: &str, val: &str| -> Result<()> {
            group
                .new_attr::<VarLenUnicode>()
                .create(name)?
                .write_scalar(&VarLenUnicode::from_str(val).unwrap())
        };

        let write_str_vec = |name: &str, vals: &[String]| -> Result<()> {
            let vlu_data: Vec<VarLenUnicode> = vals
                .iter()
                .map(|s| VarLenUnicode::from_str(s).unwrap())
                .collect();
            group
                .new_attr::<VarLenUnicode>()
                .shape(vlu_data.len())
                .create(name)?
                .write(vlu_data.as_slice())
        };

        let write_f64 = |name: &str, val: f64| -> Result<()> {
            group.new_attr::<f64>().create(name)?.write_scalar(&val)
        };

        // Basic Metadata
        write_str("title", &self.metadata.basic.title)?;
        write_str("id", &self.metadata.basic.id)?;
        write_str("description", &self.metadata.basic.description)?;
        write_str("version", &self.metadata.basic.version)?;
        write_str("history", &self.metadata.basic.history)?;
        write_str("comment", &self.metadata.basic.comment)?;
        write_str("license", &self.metadata.basic.license)?;
        write_str("auxiliary", &self.metadata.basic.auxiliary)?;
        write_str_vec("keywords", &self.metadata.basic.keywords)?;

        // Attribution
        write_str("creator_name", &self.metadata.attribution.creator_name)?;
        write_str("creator_email", &self.metadata.attribution.creator_email)?;
        write_str(
            "creator_institution",
            &self.metadata.attribution.creator_institution,
        )?;
        write_str(
            "acknowledgement",
            &self.metadata.attribution.acknowledgement,
        )?;
        write_str_vec("authors", &self.metadata.attribution.authors)?;
        write_str_vec("references", &self.metadata.attribution.references)?;

        // Repository
        write_str("repository_doi", &self.metadata.repository.repository_doi)?;
        write_str("repository_name", &self.metadata.repository.repository_name)?;
        write_str("repository_url", &self.metadata.repository.repository_url)?;

        // Data
        write_str("data_layout", &self.metadata.data.data_layout)?;
        write_str_vec("data_units", &self.metadata.data.data_units)?;
        write_str_vec("data_values", &self.metadata.data.data_values)?;

        // Coordinates
        write_str("crs", &self.metadata.coords.crs)?;
        write_f64("origin_x", self.metadata.coords.origin_x)?;
        write_f64("origin_y", self.metadata.coords.origin_y)?;
        write_f64("y_azimuth", self.metadata.coords.y_azimuth)?;
        write_f64("dim_x", self.metadata.coords.dim_x)?;
        write_f64("dim_y", self.metadata.coords.dim_y)?;
        write_f64("dim_z", self.metadata.coords.dim_z)?;

        Ok(())
    }

    pub fn load(path: PathBuf) -> Result<Self> {
        let file = File::open(path)?;
        let root = file.as_group()?;

        let read_str = |name: &str| -> Result<String> { read_h5_string(&root.attr(name)?) };
        let read_str_vec =
            |name: &str| -> Result<Vec<String>> { read_h5_string_array(&root.attr(name)?) };

        let metadata = ModelMetadata {
            basic: BasicMetadata {
                title: read_str("title")?,
                id: read_str("id")?,
                description: read_str("description")?,
                version: read_str("version")?,
                history: read_str("history")?,
                comment: read_str("comment")?,
                license: read_str("license")?,
                auxiliary: read_str("auxiliary")?,
                keywords: read_str_vec("keywords")?,
            },
            attribution: AttributionMetadata {
                creator_name: read_str("creator_name")?,
                creator_email: read_str("creator_email")?,
                creator_institution: read_str("creator_institution")?,
                acknowledgement: read_str("acknowledgement")?,
                authors: read_str_vec("authors")?,
                references: read_str_vec("references")?,
            },
            repository: RepositoryMetadata {
                repository_doi: read_str("repository_doi")?,
                repository_name: read_str("repository_name")?,
                repository_url: read_str("repository_url")?,
            },
            data: DataMetadata {
                data_layout: read_str("data_layout")?,
                data_units: read_str_vec("data_units")?,
                data_values: read_str_vec("data_values")?,
            },
            coords: CoordinateMetadata {
                crs: read_str("crs")?,
                origin_x: root.attr("origin_x")?.read_scalar::<f64>()?,
                origin_y: root.attr("origin_y")?.read_scalar::<f64>()?,
                y_azimuth: root.attr("y_azimuth")?.read_scalar::<f64>()?,
                dim_x: root.attr("dim_x")?.read_scalar::<f64>()?,
                dim_y: root.attr("dim_y")?.read_scalar::<f64>()?,
                dim_z: root.attr("dim_z")?.read_scalar::<f64>()?,
            },
        };

        let mut surfaces = Vec::new();
        if root.link_exists("surfaces") {
            let surfaces_group = root.group("surfaces")?;
            let names: Vec<_> = surfaces_group.member_names()?;

            for name in names {
                let ds = surfaces_group.dataset(&name)?;
                surfaces.push(Surface {
                    resolution_horiz: ds.attr("resolution_horiz")?.read_scalar()?,
                    surface: ds.read()?,
                    name: name,
                });
            }
        }

        let mut blocks = Vec::new();
        if root.link_exists("blocks") {
            let blocks_group = root.group("blocks")?;
            let names: Vec<_> = blocks_group.member_names()?;

            for name in names {
                let ds = blocks_group.dataset(&name)?;
                blocks.push(Block {
                    resolution_horiz: ds.attr("resolution_horiz")?.read_scalar()?,
                    resolution_vert: ds.attr("resolution_vert")?.read_scalar()?,
                    z_top: ds.attr("z_top")?.read_scalar()?,
                    block: ds.read()?,
                    name: name,
                });
            }
        }

        Ok(GeoModelGrid {
            metadata,
            surfaces,
            blocks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, Array4};
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    fn get_fixture_path(filename: &str) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/fixtures");
        path.push(filename);
        path
    }

    // Notice how much cleaner constructing mock data is now!
    fn create_mock_grid(with_topography: bool) -> GeoModelGrid {
        let mut builder = GeoModelGrid::builder()
            .title("Test Model")
            .id("UUID-1234")
            .description("A unit test model")
            .version("1.0.0")
            .history("First version")
            .comment("Test comment")
            .license("CC0")
            .auxiliary(r#"{"0": "zero"}"#)
            .creator_name("Alice Scientist")
            .creator_email("alice@example.com")
            .creator_institution("Geo Lab")
            .acknowledgement("Thanks to everyone")
            .repository_doi("this.is.a.doi")
            .repository_name("Some Repo")
            .repository_url("http://example.com")
            .data_layout("vertex")
            .crs("EPSG:4326")
            .authors(vec!["Alice".to_string(), "Bob".to_string()])
            .references(vec!["Ref 1".to_string()])
            .keywords(vec!["key1".to_string()])
            .data_units(vec!["m".to_string(), "m/s".to_string()])
            .data_values(vec!["one".to_string(), "two".to_string()])
            .origin_x(0.0)
            .origin_y(0.0)
            .y_azimuth(90.0)
            .dim_x(100.0)
            .dim_y(100.0)
            .dim_z(50.0)
            .add_block(Block {
                name: "test1".to_string(),
                resolution_horiz: 1.0,
                resolution_vert: 0.5,
                z_top: 10.0,
                block: Array4::from_elem((10, 10, 5, 1), 1.1),
            })
            .add_block(Block {
                name: "test2".to_string(),
                resolution_horiz: 1.0,
                resolution_vert: 0.5,
                z_top: 5.0,
                block: Array4::from_elem((10, 10, 5, 1), 2.2),
            });

        if with_topography {
            builder = builder.add_surface(Surface {
                surface: Array2::from_elem((10, 10), 5.5),
                resolution_horiz: 1.0,
                name: "surface".to_string(),
            });
        }

        builder.build()
    }

    #[test]
    fn test_san_francisco() -> Result<()> {
        let path = get_fixture_path("USGS_SFCVM_v21-0_detailed.berkeley.h5");
        let loaded = GeoModelGrid::load(path)?;

        // Assertions mapped to the newly grouped sub-structs
        assert_eq!(loaded.metadata.attribution.acknowledgement, "");
        assert_eq!(loaded.metadata.attribution.authors.len(), 2);
        assert_eq!(loaded.metadata.attribution.authors[0], "Aagaard, Brad T.");
        assert_eq!(loaded.metadata.attribution.authors[1], "Hirakawa, Evan T.");
        assert_eq!(
            loaded.metadata.basic.description,
            "USGS 3D seismic velocity model for the San Francisco Bay region (detailed domain)"
        );
        assert_eq!(loaded.metadata.basic.version, "21.0");
        assert_eq!(loaded.metadata.basic.id, "usgs-sfcvm-detailed");

        assert_eq!(
            loaded.metadata.coords.crs,
            "PROJCS[\"unnamed\",GEOGCS[\"NAD83\",DATUM[\"North_American_Datum_1983\",SPHEROID[\"GRS 1980\",6378137,298.257222101,AUTHORITY[\"EPSG\",\"7019\"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY[\"EPSG\",\"6269\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4269\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",35],PARAMETER[\"central_meridian\",-123],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"Meter\",1]]"
        );

        assert_eq!(loaded.blocks.len(), 4);

        let first_block = &loaded.blocks[0];
        assert_abs_diff_eq!(first_block.block[[0, 0, 0, 0]], 2659.72, epsilon = 0.1);
        assert_abs_diff_eq!(first_block.block[[0, 0, 0, 1]], 5459.42, epsilon = 0.1);
        assert_abs_diff_eq!(first_block.block[[0, 0, 0, 5]], 22.0, epsilon = 0.1);
        assert_abs_diff_eq!(first_block.block[[0, 0, 0, 6]], 12.0, epsilon = 0.1);

        assert_abs_diff_eq!(first_block.block[[0, 0, 1, 0]], 2660.61, epsilon = 0.1);

        Ok(())
    }

    #[test]
    fn test_full_round_trip() -> Result<()> {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_str().unwrap();

        let original = create_mock_grid(true);

        original.save(path)?;
        let loaded = GeoModelGrid::load(PathBuf::from(path))?;

        // Assertions mapped to grouped structs
        assert_eq!(loaded.metadata.basic.title, original.metadata.basic.title);
        assert_eq!(
            loaded.metadata.coords.origin_x,
            original.metadata.coords.origin_x
        );
        assert_eq!(loaded.metadata.coords.crs, original.metadata.coords.crs);
        assert_eq!(loaded.metadata.attribution.authors.len(), 2);

        assert!(loaded.surfaces.len() == 1);
        let loaded_topo = &loaded.surfaces[0];
        let orig_topo = &original.surfaces[0];
        assert_eq!(loaded_topo.surface, orig_topo.surface);
        assert_eq!(loaded_topo.name, orig_topo.name);
        assert_eq!(loaded_topo.resolution_horiz, orig_topo.resolution_horiz);

        assert_eq!(loaded.blocks.len(), 2);
        assert_eq!(loaded.blocks[0].z_top, 10.0);
        assert_eq!(loaded.blocks[1].block[[0, 0, 0, 0]], 2.2);

        Ok(())
    }

    #[test]
    fn test_no_topography_round_trip() -> Result<()> {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_str().unwrap();

        let original = create_mock_grid(false);

        original.save(path)?;
        let loaded = GeoModelGrid::load(PathBuf::from(path))?;

        assert!(loaded.surfaces.is_empty());
        assert_eq!(loaded.blocks.len(), 2);

        Ok(())
    }

    #[test]
    fn test_empty_blocks_round_trip() -> Result<()> {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_str().unwrap();

        // Easily override the builder to have no blocks
        let mut original = create_mock_grid(false);
        original.blocks = vec![];

        original.save(path)?;
        let loaded = GeoModelGrid::load(PathBuf::from(path))?;

        assert!(loaded.blocks.is_empty());
        Ok(())
    }
}
