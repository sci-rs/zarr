//! TODO.

#![deny(missing_debug_implementations)]
#![forbid(unsafe_code)]

// TODO: this does not run the test for recent stable rust because `test`
// is no longer set during doc tests. When 1.40 stabilizes and is the MSRV
// this can be changed from `test` to `doctest` and will work correctly.
#[cfg(all(test, feature = "filesystem"))]
doc_comment::doctest!("../README.md");

#[macro_use]
pub extern crate smallvec;

use std::io::Error;
use std::path::PathBuf;
use std::time::SystemTime;

use byteorder::BigEndian;
use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;
use smallvec::SmallVec;

use crate::chunk::{
    DataChunk,
    ReadableDataChunk,
    ReinitDataChunk,
    SliceDataChunk,
    VecDataChunk,
    WriteableDataChunk,
};

pub mod chunk;
pub mod compression;
#[macro_use]
pub mod data_type;
pub use data_type::*;
#[cfg(feature = "filesystem")]
pub mod filesystem;
#[cfg(feature = "use_ndarray")]
pub mod ndarray;
pub mod prelude;
pub mod storage;

#[cfg(test)]
#[macro_use]
pub(crate) mod tests;

pub use semver::{
    Version,
    VersionReq,
};

const COORD_SMALLVEC_SIZE: usize = 6;
pub type CoordVec<T> = SmallVec<[T; COORD_SMALLVEC_SIZE]>;
pub type ChunkCoord = CoordVec<u32>;
pub type GridCoord = CoordVec<u64>;

type ZarrEndian = BigEndian;

/// Version of the Zarr spec supported by this library.
pub const VERSION: Version = Version {
    major: 3,
    minor: 0,
    patch: 0,
    pre: Vec::new(),
    build: Vec::new(),
};

// TODO: from https://users.rust-lang.org/t/append-an-additional-extension/23586/12
fn add_extension(path: &mut std::path::PathBuf, extension: impl AsRef<std::path::Path>) {
    match path.extension() {
        Some(ext) => {
            let mut ext = ext.to_os_string();
            ext.push(".");
            ext.push(extension.as_ref());
            path.set_extension(ext)
        }
        None => path.set_extension(extension.as_ref()),
    };
}

const ENTRY_POINT_KEY: &str = "zarr.json";
const DATA_ROOT_PATH: &str = "/data/root";
const META_ROOT_PATH: &str = "/meta/root";
const ARRAY_METADATA_KEY_EXT: &str = ".array";
const GROUP_METADATA_KEY_EXT: &str = ".group";

/// Store metadata about a node.
///
/// This is metadata from the persistence layer of the hierarchy, such as
/// filesystem access times and on-disk sizes, and is not to be confused with
/// semantic metadata stored as attributes in the hierarchy.
#[derive(Clone, Debug)]
pub struct StoreNodeMetadata {
    pub created: Option<SystemTime>,
    pub accessed: Option<SystemTime>,
    pub modified: Option<SystemTime>,
    pub size: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntryPointMetadata {
    zarr_format: String,
    metadata_encoding: String,
    metadata_key_suffix: String,
    // TODO: extensions
}

impl Default for EntryPointMetadata {
    fn default() -> Self {
        Self {
            zarr_format: "https://purl.org/zarr/spec/protocol/core/3.0".to_owned(),
            metadata_encoding: "https://purl.org/zarr/spec/protocol/core/3.0".to_owned(),
            metadata_key_suffix: ".json".to_owned(),
        }
    }
}

pub trait Hierarchy {
    fn get_entry_point_metadata(&self) -> &EntryPointMetadata;

    fn array_metadata_key(&self, path_name: &str) -> PathBuf {
        let mut key = PathBuf::from(META_ROOT_PATH).join(path_name);
        add_extension(&mut key, ARRAY_METADATA_KEY_EXT);
        add_extension(
            &mut key,
            &self.get_entry_point_metadata().metadata_key_suffix,
        );
        key
    }

    fn group_metadata_key(&self, path_name: &str) -> PathBuf {
        let mut key = PathBuf::from(META_ROOT_PATH).join(path_name);
        add_extension(&mut key, GROUP_METADATA_KEY_EXT);
        add_extension(
            &mut key,
            &self.get_entry_point_metadata().metadata_key_suffix,
        );
        key
    }

    fn data_path_key(&self, path_name: &str) -> PathBuf {
        PathBuf::from(DATA_ROOT_PATH).join(path_name)
    }
}

/// Non-mutating operations on Zarr hierarchys.
pub trait HierarchyReader: Hierarchy {
    /// Get the Zarr specification version of the hierarchy.
    fn get_version(&self) -> Result<VersionReq, Error>;

    /// Get metadata for an array.
    fn get_array_metadata(&self, path_name: &str) -> Result<ArrayMetadata, Error>;

    /// Test whether a group or array exists.
    fn exists(&self, path_name: &str) -> Result<bool, Error>;

    /// Test whether a array exists.
    fn array_exists(&self, path_name: &str) -> Result<bool, Error> {
        Ok(self.exists(path_name)? && self.get_array_metadata(path_name).is_ok())
    }

    /// Get a URI string for a data chunk.
    ///
    /// Whether this requires that the array and chunk exist is currently
    /// implementation dependent. Whether this URI is a URL is implementation
    /// dependent.
    fn get_chunk_uri(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: &[u64],
    ) -> Result<String, Error> {
        todo!()
    }

    /// Read a single array chunk into a linear vec.
    fn read_chunk<T>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataChunk<T>>, Error>
    where
        VecDataChunk<T>: DataChunk<T> + ReadableDataChunk,
        T: ReflectedType;

    /// Read a single array chunk into an existing buffer.
    fn read_chunk_into<T: ReflectedType, B: DataChunk<T> + ReinitDataChunk<T> + ReadableDataChunk>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: GridCoord,
        chunk: &mut B,
    ) -> Result<Option<()>, Error>;

    /// Read store metadata about a chunk.
    fn store_chunk_metadata(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: &[u64],
    ) -> Result<Option<StoreNodeMetadata>, Error>;

    /// List all attributes of a group.
    fn list_attributes(&self, path_name: &str) -> Result<serde_json::Value, Error>;
}

/// Non-mutating operations on Zarr hierarchys that support group discoverability.
pub trait HierarchyLister: HierarchyReader {
    /// List all groups (including arrays) in a group.
    fn list(&self, path_name: &str) -> Result<Vec<String>, Error>;
}

/// Mutating operations on Zarr hierarchys.
pub trait HierarchyWriter: HierarchyReader {
    /// Set a single attribute.
    fn set_attribute<T: Serialize>(
        &self,
        path_name: &str,
        key: String,
        attribute: T,
    ) -> Result<(), Error> {
        self.set_attributes(
            path_name,
            vec![(key, serde_json::to_value(attribute)?)]
                .into_iter()
                .collect(),
        )
    }

    /// Set a map of attributes.
    fn set_attributes(
        &self,
        path_name: &str,
        attributes: serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), Error>;

    /// Set mandatory array attributes.
    fn set_array_attributes(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
    ) -> Result<(), Error> {
        if let serde_json::Value::Object(map) = serde_json::to_value(array_meta)? {
            self.set_attributes(path_name, map)
        } else {
            panic!("Impossible: ArrayMetadata serializes to object")
        }
    }

    /// Create a group (directory).
    fn create_group(&self, path_name: &str) -> Result<(), Error>;

    /// Create a array. This will create the array group and attributes,
    /// but not populate any chunk data.
    fn create_array(&self, path_name: &str, array_meta: &ArrayMetadata) -> Result<(), Error>;

    /// Remove the Zarr hierarchy.
    fn remove_all(&self) -> Result<(), Error> {
        self.remove("")
    }

    /// Remove a group or array (directory and all contained files).
    ///
    /// This will wait on locks acquired by other writers or readers.
    fn remove(&self, path_name: &str) -> Result<(), Error>;

    fn write_chunk<T: ReflectedType, B: DataChunk<T> + WriteableDataChunk>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        chunk: &B,
    ) -> Result<(), Error>;

    /// Delete a chunk from a array.
    ///
    /// Returns `true` if the chunk does not exist on the backend at the
    /// completion of the call.
    fn delete_chunk(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: &[u64],
    ) -> Result<bool, Error>;
}

fn u64_ceil_div(a: u64, b: u64) -> u64 {
    (a + 1) / b + (if a % b != 0 { 1 } else { 0 })
}

/// Metadata for groups.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct GroupMetadata {
    extensions: Vec<serde_json::Value>,
    attributes: serde_json::Map<String, serde_json::Value>,
}

impl Default for GroupMetadata {
    fn default() -> Self {
        GroupMetadata {
            extensions: Vec::new(),
            attributes: serde_json::Map::new(),
        }
    }
}

/// Attributes of a tensor array.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct ArrayMetadata {
    /// Dimensions of the entire array, in voxels.
    shape: GridCoord,
    /// Element data type.
    data_type: DataType,
    /// Compression scheme for voxel data in each chunk.
    compressor: compression::CompressionType,
    /// TODO
    chunk_grid: ChunkGridMetadata,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct ChunkGridMetadata {
    /// Shape of each chunk, in voxels.
    chunk_shape: ChunkCoord,
    /// TODO
    separator: String,
}

impl ArrayMetadata {
    pub fn new(
        shape: GridCoord,
        chunk_shape: ChunkCoord,
        data_type: DataType,
        compressor: compression::CompressionType,
    ) -> ArrayMetadata {
        assert_eq!(
            shape.len(),
            chunk_shape.len(),
            "Number of array dimensions must match number of chunk size dimensions."
        );
        ArrayMetadata {
            shape,
            data_type,
            compressor,
            // TODO
            chunk_grid: ChunkGridMetadata {
                chunk_shape,
                separator: "/".to_owned(),
            },
        }
    }

    pub fn get_shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn get_chunk_shape(&self) -> &[u32] {
        &self.chunk_grid.chunk_shape
    }

    pub fn get_data_type(&self) -> &DataType {
        &self.data_type
    }

    pub fn get_compressor(&self) -> &compression::CompressionType {
        &self.compressor
    }

    pub fn get_ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements possible given the shape.
    pub fn get_num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Get the total number of elements possible in a chunk.
    pub fn get_chunk_num_elements(&self) -> usize {
        self.chunk_grid
            .chunk_shape
            .iter()
            .map(|&d| d as usize)
            .product()
    }

    /// Get the upper bound extent of grid coordinates.
    pub fn get_grid_extent(&self) -> GridCoord {
        self.shape
            .iter()
            .zip(self.chunk_grid.chunk_shape.iter().cloned().map(u64::from))
            .map(|(d, b)| u64_ceil_div(*d, b))
            .collect()
    }

    /// Get the total number of chunks.
    /// ```
    /// use zarr::prelude::*;
    /// use zarr::smallvec::smallvec;
    /// let attrs = ArrayMetadata::new(
    ///     smallvec![50, 40, 30],
    ///     smallvec![11, 10, 10],
    ///     DataType::UINT8,
    ///     zarr::compression::CompressionType::default(),
    /// );
    /// assert_eq!(attrs.get_num_chunks(), 60);
    /// ```
    pub fn get_num_chunks(&self) -> u64 {
        self.get_grid_extent().iter().product()
    }

    /// Check whether a chunk grid position is in the bounds of this array.
    /// ```
    /// use zarr::prelude::*;
    /// use zarr::smallvec::smallvec;
    /// let attrs = ArrayMetadata::new(
    ///     smallvec![50, 40, 30],
    ///     smallvec![11, 10, 10],
    ///     DataType::UINT8,
    ///     zarr::compression::CompressionType::default(),
    /// );
    /// assert!(attrs.in_bounds(&smallvec![4, 3, 2]));
    /// assert!(!attrs.in_bounds(&smallvec![5, 3, 2]));
    /// ```
    pub fn in_bounds(&self, grid_position: &GridCoord) -> bool {
        self.shape.len() == grid_position.len()
            && self
                .get_grid_extent()
                .iter()
                .zip(grid_position.iter())
                .all(|(&bound, &coord)| coord < bound)
    }
}
