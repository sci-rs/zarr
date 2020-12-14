//! Interfaces for the [N5 "Not HDF5" n-dimensional tensor file system storage
//! format](https://github.com/saalfeldlab/n5) created by the Saalfeld lab at
//! Janelia Research Campus.

#![deny(missing_debug_implementations)]
#![forbid(unsafe_code)]

// TODO: this does not run the test for recent stable rust because `test`
// is no longer set during doc tests. When 1.40 stabilizes and is the MSRV
// this can be changed from `test` to `doctest` and will work correctly.
#[cfg(all(test, feature = "filesystem"))]
doc_comment::doctest!("../README.md");

#[macro_use]
pub extern crate smallvec;

use std::io::{
    Error,
    ErrorKind,
    Read,
    Write,
};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::time::SystemTime;

use byteorder::{
    BigEndian,
    ByteOrder,
    ReadBytesExt,
    WriteBytesExt,
};
use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;
use smallvec::SmallVec;

use crate::compression::Compression;

pub mod compression;
#[macro_use]
pub mod data_type;
pub use data_type::*;
#[cfg(feature = "filesystem")]
pub mod filesystem;
#[cfg(feature = "use_ndarray")]
pub mod ndarray;
pub mod prelude;

#[cfg(test)]
#[macro_use]
pub(crate) mod tests;

pub use semver::{
    Version,
    VersionReq,
};

const COORD_SMALLVEC_SIZE: usize = 6;
pub type CoordVec<T> = SmallVec<[T; COORD_SMALLVEC_SIZE]>;
pub type BlockCoord = CoordVec<u32>;
pub type GridCoord = CoordVec<u64>;

type N5Endian = BigEndian;

/// Version of the Java N5 spec supported by this library.
pub const VERSION: Version = Version {
    major: 3,
    minor: 0,
    patch: 0,
    pre: Vec::new(),
    build: Vec::new(),
};

/// Determines whether a version of an N5 implementation is capable of accessing
/// a version of an N5 container (`other`).
pub fn is_version_compatible(s: &Version, other: &Version) -> bool {
    other.major <= s.major
}

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

/// Key name for the version attribute in the container root.
pub const VERSION_ATTRIBUTE_KEY: &str = "n5";
const DATA_ROOT_PATH: &str = "/data/root";
const META_ROOT_PATH: &str = "/meta/root";
const ARRAY_METADATA_PATH: &str = ".array";
const GROUP_METADATA_PATH: &str = ".group";

/// Container metadata about a data block.
///
/// This is metadata from the persistence layer of the container, such as
/// filesystem access times and on-disk sizes, and is not to be confused with
/// semantic metadata stored as attributes in the container.
#[derive(Clone, Debug)]
pub struct DataBlockMetadata {
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

pub trait ReadableStore {
    type GetReader: Read;

    /// TODO: not in zarr spec
    fn exists(&self, key: &str) -> Result<bool, Error>;

    fn get(&self, key: &str) -> Result<Option<Self::GetReader>, Error>;
}

pub trait WriteableStore {
    type SetWriter: Write;

    fn set<F: FnOnce(Self::SetWriter) -> Result<(), Error>>(
        &self,
        key: &str,
        value: F,
    ) -> Result<(), Error>;

    fn delete(&self, key: &str) -> Result<(), Error>;
}

pub trait Hierarchy {
    fn get_entry_point_metadata(&self) -> &EntryPointMetadata;

    fn array_metadata_key(&self, path_name: &str) -> PathBuf {
        let mut key = PathBuf::from(META_ROOT_PATH).join(path_name);
        add_extension(&mut key, ARRAY_METADATA_PATH);
        add_extension(
            &mut key,
            &self.get_entry_point_metadata().metadata_key_suffix,
        );
        key
    }

    fn group_metadata_key(&self, path_name: &str) -> PathBuf {
        let mut key = PathBuf::from(META_ROOT_PATH).join(path_name);
        add_extension(&mut key, GROUP_METADATA_PATH);
        add_extension(
            &mut key,
            &self.get_entry_point_metadata().metadata_key_suffix,
        );
        key
    }
}

/// Non-mutating operations on N5 containers.
pub trait N5Reader: Hierarchy {
    /// Get the N5 specification version of the container.
    fn get_version(&self) -> Result<VersionReq, Error>;

    /// Get attributes for a dataset.
    fn get_dataset_attributes(&self, path_name: &str) -> Result<DatasetAttributes, Error>;

    /// Test whether a group or dataset exists.
    fn exists(&self, path_name: &str) -> Result<bool, Error>;

    /// Test whether a dataset exists.
    fn dataset_exists(&self, path_name: &str) -> Result<bool, Error> {
        Ok(self.exists(path_name)? && self.get_dataset_attributes(path_name).is_ok())
    }

    /// Get a URI string for a data block.
    ///
    /// Whether this requires that the dataset and block exist is currently
    /// implementation dependent. Whether this URI is a URL is implementation
    /// dependent.
    fn get_block_uri(&self, path_name: &str, grid_position: &[u64]) -> Result<String, Error>;

    /// Read a single dataset block into a linear vec.
    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataBlock<T>>, Error>
    where
        VecDataBlock<T>: DataBlock<T> + ReadableDataBlock,
        T: ReflectedType;

    /// Read a single dataset block into an existing buffer.
    fn read_block_into<T: ReflectedType, B: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
        block: &mut B,
    ) -> Result<Option<()>, Error>;

    /// Read metadata about a block.
    fn block_metadata(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: &[u64],
    ) -> Result<Option<DataBlockMetadata>, Error>;

    /// List all attributes of a group.
    fn list_attributes(&self, path_name: &str) -> Result<serde_json::Value, Error>;
}

impl<S: ReadableStore + Hierarchy> N5Reader for S {
    fn get_version(&self) -> Result<VersionReq, Error> {
        let vers_str = self
            .get_entry_point_metadata()
            .zarr_format
            .rsplit('/')
            .next()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    "Entry point metadata zarr format URI does not have version",
                )
            })?;
        VersionReq::parse(vers_str).map_err(|_| {
            Error::new(
                ErrorKind::InvalidData,
                "Entry point metadata zarr format URI does not have version",
            )
        })
    }

    fn get_dataset_attributes(&self, path_name: &str) -> Result<DatasetAttributes, Error> {
        let dataset_path = self.array_metadata_key(path_name);
        let value_reader = ReadableStore::get(self, &dataset_path.to_str().expect("TODO"))?
            .ok_or_else(|| Error::from(std::io::ErrorKind::NotFound))?;
        Ok(serde_json::from_reader(value_reader)?)
    }

    fn exists(&self, path_name: &str) -> Result<bool, Error> {
        self.exists(path_name)
    }

    fn get_block_uri(&self, path_name: &str, grid_position: &[u64]) -> Result<String, Error> {
        todo!()
    }

    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataBlock<T>>, Error>
    where
        VecDataBlock<T>: DataBlock<T> + ReadableDataBlock,
        T: ReflectedType,
    {
        // TODO convert asserts to errors
        assert!(data_attrs.in_bounds(&grid_position));

        // Construct block path string
        let block_key = get_block_key(path_name, data_attrs, &grid_position);

        // Get key from store
        let value_reader = ReadableStore::get(self, &block_key)?;

        // Read value into container
        value_reader
            .map(|reader| {
                <crate::DefaultBlock as DefaultBlockReader<T, _>>::read_block(
                    reader,
                    data_attrs,
                    grid_position,
                )
            })
            .transpose()
    }

    fn read_block_into<
        T: ReflectedType,
        B: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock,
    >(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
        block: &mut B,
    ) -> Result<Option<()>, Error> {
        // TODO convert asserts to errors
        assert!(data_attrs.in_bounds(&grid_position));

        // Construct block path string
        let block_key = get_block_key(path_name, data_attrs, &grid_position);

        // Get key from store
        let value_reader = ReadableStore::get(self, &block_key)?;

        // Read value into container
        value_reader
            .map(|reader| {
                <crate::DefaultBlock as DefaultBlockReader<T, _>>::read_block_into(
                    reader,
                    data_attrs,
                    grid_position,
                    block,
                )
            })
            .transpose()
    }

    fn block_metadata(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: &[u64],
    ) -> Result<Option<DataBlockMetadata>, Error> {
        todo!()
    }

    fn list_attributes(&self, path_name: &str) -> Result<serde_json::Value, Error> {
        // TODO: wasteful path recomputation
        let metadata_key =
            if self.exists(self.array_metadata_key(path_name).to_str().expect("TODO"))? {
                self.array_metadata_key(path_name)
            } else if self.exists(self.group_metadata_key(path_name).to_str().expect("TODO"))? {
                self.group_metadata_key(path_name)
            } else {
                return Err(Error::new(
                    std::io::ErrorKind::NotFound,
                    "Node does not exist at path",
                ));
            };

        // TODO: race condition
        let value_reader = ReadableStore::get(self, &metadata_key.to_str().expect("TODO"))?
            .ok_or_else(|| Error::from(std::io::ErrorKind::NotFound))?;
        Ok(serde_json::from_reader(value_reader)?)
    }
}

fn get_block_key(base_path: &str, data_attrs: &DatasetAttributes, grid_position: &[u64]) -> String {
    use std::fmt::Write;
    // TODO remove allocs and cleanup
    let mut block_key = match grid_position.len() {
        0 => base_path.to_owned(),
        _ => format!("{}/", base_path),
    };
    write!(
        block_key,
        "c{}",
        grid_position
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(&data_attrs.chunk_grid.chunk_separator)
    )
    .unwrap();

    block_key
}

/// Non-mutating operations on N5 containers that support group discoverability.
pub trait N5Lister: N5Reader {
    /// List all groups (including datasets) in a group.
    fn list(&self, path_name: &str) -> Result<Vec<String>, Error>;
}

/// Mutating operations on N5 containers.
pub trait N5Writer: N5Reader {
    /// Set a single attribute.
    fn set_attribute<T: Serialize>(
        &self, // TODO: should this be mut for semantics?
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
        &self, // TODO: should this be mut for semantics?
        path_name: &str,
        attributes: serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), Error>;

    /// Set mandatory dataset attributes.
    fn set_dataset_attributes(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
    ) -> Result<(), Error> {
        if let serde_json::Value::Object(map) = serde_json::to_value(data_attrs)? {
            self.set_attributes(path_name, map)
        } else {
            panic!("Impossible: DatasetAttributes serializes to object")
        }
    }

    /// Create a group (directory).
    fn create_group(&self, path_name: &str) -> Result<(), Error>;

    /// Create a dataset. This will create the dataset group and attributes,
    /// but not populate any block data.
    fn create_dataset(&self, path_name: &str, data_attrs: &DatasetAttributes) -> Result<(), Error> {
        self.create_group(path_name)?;
        self.set_dataset_attributes(path_name, data_attrs)
    }

    /// Remove the N5 container.
    fn remove_all(&self) -> Result<(), Error> {
        self.remove("")
    }

    /// Remove a group or dataset (directory and all contained files).
    ///
    /// This will wait on locks acquired by other writers or readers.
    fn remove(&self, path_name: &str) -> Result<(), Error>;

    fn write_block<T: ReflectedType, B: DataBlock<T> + WriteableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        block: &B,
    ) -> Result<(), Error>;

    // TODO
    // /// Delete a block from a dataset.
    // ///
    // /// Returns `true` if the block does not exist on the backend at the
    // /// completion of the call.
    // fn delete_block(&self, path_name: &str, grid_position: &[u64]) -> Result<bool, Error>;
}

// From: https://github.com/serde-rs/json/issues/377
// TODO: Could be much better.
// TODO: n5 filesystem later settled on top-level key merging only.
fn merge(a: &mut Value, b: &Value) {
    match (a, b) {
        (&mut Value::Object(ref mut a), &Value::Object(ref b)) => {
            for (k, v) in b {
                merge(a.entry(k.clone()).or_insert(Value::Null), v);
            }
        }
        (a, b) => {
            *a = b.clone();
        }
    }
}

impl<S: ReadableStore + WriteableStore + Hierarchy> N5Writer for S {
    fn set_attributes(
        &self, // TODO: should this be mut for semantics?
        path_name: &str,
        attributes: serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), Error> {
        // TODO: wasteful path recomputation
        let metadata_key =
            if self.exists(self.array_metadata_key(path_name).to_str().expect("TODO"))? {
                self.array_metadata_key(path_name)
            } else if self.exists(self.group_metadata_key(path_name).to_str().expect("TODO"))? {
                self.group_metadata_key(path_name)
            } else {
                return Err(Error::new(
                    std::io::ErrorKind::NotFound,
                    "Node does not exist at path",
                ));
            };

        // TODO: race condition
        let value_reader = ReadableStore::get(self, &metadata_key.to_str().expect("TODO"))?
            .ok_or_else(|| Error::from(std::io::ErrorKind::NotFound))?;
        let existing: Value = serde_json::from_reader(value_reader)?;

        // TODO: determine whether attribute merging is still necessary for zarr
        let mut merged = existing.clone();
        let new: Value = attributes.into();
        merge(&mut merged, &new);
        if merged != existing {
            self.set(metadata_key.to_str().expect("TODO"), |writer| {
                Ok(serde_json::to_writer(writer, &merged)?)
            })?;
        }
        Ok(())
    }

    fn create_group(&self, path_name: &str) -> Result<(), Error> {
        let path_buf = PathBuf::from(path_name);
        if let Some(parent) = path_buf.parent() {
            self.create_group(parent.to_str().expect("TODO"))?;
        }
        let metadata_key = self.group_metadata_key(path_name);
        if self.exists(self.array_metadata_key(path_name).to_str().expect("TODO"))? {
            Err(Error::new(
                std::io::ErrorKind::AlreadyExists,
                "Array already exists at group path",
            ))
        } else if self.exists(metadata_key.to_str().expect("TODO"))? {
            Ok(())
        } else {
            self.set(metadata_key.to_str().expect("TODO"), |writer| {
                Ok(serde_json::to_writer(writer, &GroupMetadata::default())?)
            })
        }
    }

    fn create_dataset(&self, path_name: &str, data_attrs: &DatasetAttributes) -> Result<(), Error> {
        let path_buf = PathBuf::from(path_name);
        if let Some(parent) = path_buf.parent() {
            self.create_group(parent.to_str().expect("TODO"))?;
        }
        let metadata_key = self.array_metadata_key(path_name);
        if self.exists(self.group_metadata_key(path_name).to_str().expect("TODO"))?
            || self.exists(metadata_key.to_str().expect("TODO"))?
        {
            Err(Error::new(
                std::io::ErrorKind::AlreadyExists,
                "Node already exists at array path",
            ))
        } else {
            self.set(metadata_key.to_str().expect("TODO"), |writer| {
                Ok(serde_json::to_writer(writer, data_attrs)?)
            })
        }
    }

    fn remove(&self, path_name: &str) -> Result<(), Error> {
        self.delete(path_name)
    }

    fn write_block<T: ReflectedType, B: DataBlock<T> + WriteableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        block: &B,
    ) -> Result<(), Error> {
        // TODO convert assert
        // assert!(data_attrs.in_bounds(block.get_grid_position()));
        let block_key = get_block_key(path_name, data_attrs, block.get_grid_position());
        self.set(&block_key, |writer| {
            <DefaultBlock as DefaultBlockWriter<T, _, _>>::write_block(writer, data_attrs, block)
        })
    }
}

fn u64_ceil_div(a: u64, b: u64) -> u64 {
    (a + 1) / b + (if a % b != 0 { 1 } else { 0 })
}

/// Metadata for groups.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(rename_all = "camelCase")]
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

/// Attributes of a tensor dataset.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DatasetAttributes {
    /// Dimensions of the entire dataset, in voxels.
    dimensions: GridCoord,
    /// Element data type.
    data_type: DataType,
    /// Compression scheme for voxel data in each block.
    compression: compression::CompressionType,
    /// TODO
    chunk_grid: ChunkGridMetadata,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ChunkGridMetadata {
    /// Size of each block, in voxels.
    block_size: BlockCoord,
    /// TODO
    chunk_separator: String,
}

impl DatasetAttributes {
    pub fn new(
        dimensions: GridCoord,
        block_size: BlockCoord,
        data_type: DataType,
        compression: compression::CompressionType,
    ) -> DatasetAttributes {
        assert_eq!(
            dimensions.len(),
            block_size.len(),
            "Number of dataset dimensions must match number of block size dimensions."
        );
        DatasetAttributes {
            dimensions,
            data_type,
            compression,
            // TODO
            chunk_grid: ChunkGridMetadata {
                block_size,
                chunk_separator: "/".to_owned(),
            },
        }
    }

    pub fn get_dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    pub fn get_block_size(&self) -> &[u32] {
        &self.chunk_grid.block_size
    }

    pub fn get_data_type(&self) -> &DataType {
        &self.data_type
    }

    pub fn get_compression(&self) -> &compression::CompressionType {
        &self.compression
    }

    pub fn get_ndim(&self) -> usize {
        self.dimensions.len()
    }

    /// Get the total number of elements possible given the dimensions.
    pub fn get_num_elements(&self) -> usize {
        self.dimensions.iter().map(|&d| d as usize).product()
    }

    /// Get the total number of elements possible in a block.
    pub fn get_block_num_elements(&self) -> usize {
        self.chunk_grid
            .block_size
            .iter()
            .map(|&d| d as usize)
            .product()
    }

    /// Get the upper bound extent of grid coordinates.
    pub fn get_grid_extent(&self) -> GridCoord {
        self.dimensions
            .iter()
            .zip(self.chunk_grid.block_size.iter().cloned().map(u64::from))
            .map(|(d, b)| u64_ceil_div(*d, b))
            .collect()
    }

    /// Get the total number of blocks.
    /// ```
    /// use n5::prelude::*;
    /// use n5::smallvec::smallvec;
    /// let attrs = DatasetAttributes::new(
    ///     smallvec![50, 40, 30],
    ///     smallvec![11, 10, 10],
    ///     DataType::UINT8,
    ///     n5::compression::CompressionType::default(),
    /// );
    /// assert_eq!(attrs.get_num_blocks(), 60);
    /// ```
    pub fn get_num_blocks(&self) -> u64 {
        self.get_grid_extent().iter().product()
    }

    /// Check whether a block grid position is in the bounds of this dataset.
    /// ```
    /// use n5::prelude::*;
    /// use n5::smallvec::smallvec;
    /// let attrs = DatasetAttributes::new(
    ///     smallvec![50, 40, 30],
    ///     smallvec![11, 10, 10],
    ///     DataType::UINT8,
    ///     n5::compression::CompressionType::default(),
    /// );
    /// assert!(attrs.in_bounds(&smallvec![4, 3, 2]));
    /// assert!(!attrs.in_bounds(&smallvec![5, 3, 2]));
    /// ```
    pub fn in_bounds(&self, grid_position: &GridCoord) -> bool {
        self.dimensions.len() == grid_position.len()
            && self
                .get_grid_extent()
                .iter()
                .zip(grid_position.iter())
                .all(|(&bound, &coord)| coord < bound)
    }
}

/// Unencoded, non-payload header of a data block.
#[derive(Debug)]
pub struct BlockHeader {
    size: BlockCoord,
    grid_position: GridCoord,
    num_el: usize,
}

/// Traits for data blocks that can be reused as a different blocks after
/// construction.
pub trait ReinitDataBlock<T> {
    /// Reinitialize this data block with a new header, reallocating as
    /// necessary.
    fn reinitialize(&mut self, header: BlockHeader);

    /// Reinitialize this data block with the header and data of another block.
    fn reinitialize_with<B: DataBlock<T>>(&mut self, other: &B);
}

/// Traits for data blocks that can read in data.
pub trait ReadableDataBlock {
    /// Read data into this block from a source, overwriting any existing data.
    ///
    /// Unlike Java N5, read the stream directly into the block data instead
    /// of creating a copied byte buffer.
    fn read_data<R: std::io::Read>(&mut self, source: R) -> std::io::Result<()>;
}

/// Traits for data blocks that can write out data.
pub trait WriteableDataBlock {
    /// Write the data from this block into a target.
    fn write_data<W: std::io::Write>(&self, target: W) -> std::io::Result<()>;
}

/// Common interface for data blocks of element (rust) type `T`.
///
/// To enable custom types to be written to N5 volumes, implement this trait.
pub trait DataBlock<T> {
    fn get_size(&self) -> &[u32];

    fn get_grid_position(&self) -> &[u64];

    fn get_data(&self) -> &[T];

    fn get_num_elements(&self) -> u32;

    fn get_header(&self) -> BlockHeader {
        BlockHeader {
            size: self.get_size().into(),
            grid_position: self.get_grid_position().into(),
            num_el: self.get_num_elements() as usize,
        }
    }
}

/// A generic data block container wrapping any type that can be taken as a
/// slice ref.
#[derive(Clone, Debug)]
pub struct SliceDataBlock<T: ReflectedType, C> {
    data_type: PhantomData<T>,
    size: BlockCoord,
    grid_position: GridCoord,
    data: C,
}

/// A linear vector storing a data block. All read data blocks are returned as
/// this type.
pub type VecDataBlock<T> = SliceDataBlock<T, Vec<T>>;

impl<T: ReflectedType, C> SliceDataBlock<T, C> {
    pub fn new(size: BlockCoord, grid_position: GridCoord, data: C) -> SliceDataBlock<T, C> {
        SliceDataBlock {
            data_type: PhantomData,
            size,
            grid_position,
            data,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }
}

impl<T: ReflectedType> ReinitDataBlock<T> for VecDataBlock<T> {
    fn reinitialize(&mut self, header: BlockHeader) {
        self.size = header.size;
        self.grid_position = header.grid_position;
        self.data.resize_with(header.num_el, Default::default);
    }

    fn reinitialize_with<B: DataBlock<T>>(&mut self, other: &B) {
        self.size = other.get_size().into();
        self.grid_position = other.get_grid_position().into();
        self.data.clear();
        self.data.extend_from_slice(other.get_data());
    }
}

macro_rules! vec_data_block_impl {
    ($ty_name:ty, $bo_read_fn:ident, $bo_write_fn:ident) => {
        impl<C: AsMut<[$ty_name]>> ReadableDataBlock for SliceDataBlock<$ty_name, C> {
            fn read_data<R: std::io::Read>(&mut self, mut source: R) -> std::io::Result<()> {
                source.$bo_read_fn::<N5Endian>(self.data.as_mut())
            }
        }

        impl<C: AsRef<[$ty_name]>> WriteableDataBlock for SliceDataBlock<$ty_name, C> {
            fn write_data<W: std::io::Write>(&self, mut target: W) -> std::io::Result<()> {
                const CHUNK: usize = 256;
                let mut buf: [u8; CHUNK * std::mem::size_of::<$ty_name>()] =
                    [0; CHUNK * std::mem::size_of::<$ty_name>()];

                for c in self.data.as_ref().chunks(CHUNK) {
                    let byte_len = c.len() * std::mem::size_of::<$ty_name>();
                    N5Endian::$bo_write_fn(c, &mut buf[..byte_len]);
                    target.write_all(&buf[..byte_len])?;
                }

                Ok(())
            }
        }
    };
}

// Wrapper trait to erase a generic trait argument for consistent ByteOrder
// signatures.
trait ReadBytesExtI8: ReadBytesExt {
    fn read_i8_into_wrapper<B: ByteOrder>(&mut self, dst: &mut [i8]) -> std::io::Result<()> {
        self.read_i8_into(dst)
    }
}
impl<T: ReadBytesExt> ReadBytesExtI8 for T {}

vec_data_block_impl!(u16, read_u16_into, write_u16_into);
vec_data_block_impl!(u32, read_u32_into, write_u32_into);
vec_data_block_impl!(u64, read_u64_into, write_u64_into);
vec_data_block_impl!(i8, read_i8_into_wrapper, write_i8_into);
vec_data_block_impl!(i16, read_i16_into, write_i16_into);
vec_data_block_impl!(i32, read_i32_into, write_i32_into);
vec_data_block_impl!(i64, read_i64_into, write_i64_into);
vec_data_block_impl!(f32, read_f32_into, write_f32_into);
vec_data_block_impl!(f64, read_f64_into, write_f64_into);

impl<C: AsMut<[u8]>> ReadableDataBlock for SliceDataBlock<u8, C> {
    fn read_data<R: std::io::Read>(&mut self, mut source: R) -> std::io::Result<()> {
        source.read_exact(self.data.as_mut())
    }
}

impl<C: AsRef<[u8]>> WriteableDataBlock for SliceDataBlock<u8, C> {
    fn write_data<W: std::io::Write>(&self, mut target: W) -> std::io::Result<()> {
        target.write_all(self.data.as_ref())
    }
}

impl<T: ReflectedType, C: AsRef<[T]>> DataBlock<T> for SliceDataBlock<T, C> {
    fn get_size(&self) -> &[u32] {
        &self.size
    }

    fn get_grid_position(&self) -> &[u64] {
        &self.grid_position
    }

    fn get_data(&self) -> &[T] {
        self.data.as_ref()
    }

    fn get_num_elements(&self) -> u32 {
        self.data.as_ref().len() as u32
    }
}

const BLOCK_FIXED_LEN: u16 = 0;
const BLOCK_VAR_LEN: u16 = 1;

pub trait DefaultBlockHeaderReader<R: std::io::Read> {
    fn read_block_header(buffer: &mut R, grid_position: GridCoord) -> std::io::Result<BlockHeader> {
        let mode = buffer.read_u16::<N5Endian>()?;
        let ndim = buffer.read_u16::<N5Endian>()?;
        let mut size = smallvec![0; ndim as usize];
        buffer.read_u32_into::<N5Endian>(&mut size)?;
        let num_el = match mode {
            BLOCK_FIXED_LEN => size.iter().product(),
            BLOCK_VAR_LEN => buffer.read_u32::<N5Endian>()?,
            _ => return Err(Error::new(ErrorKind::InvalidData, "Unsupported block mode")),
        };

        Ok(BlockHeader {
            size,
            grid_position,
            num_el: num_el as usize,
        })
    }
}

/// Reads blocks from rust readers.
pub trait DefaultBlockReader<T: ReflectedType, R: std::io::Read>:
    DefaultBlockHeaderReader<R>
{
    fn read_block(
        mut buffer: R,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> std::io::Result<VecDataBlock<T>>
    where
        VecDataBlock<T>: DataBlock<T> + ReadableDataBlock,
    {
        if data_attrs.data_type != T::VARIANT {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Attempt to create data block for wrong type.",
            ));
        }
        let header = Self::read_block_header(&mut buffer, grid_position)?;

        let mut block = T::create_data_block(header);
        let mut decompressed = data_attrs.compression.decoder(buffer);
        block.read_data(&mut decompressed)?;

        Ok(block)
    }

    fn read_block_into<B: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock>(
        mut buffer: R,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
        block: &mut B,
    ) -> std::io::Result<()> {
        if data_attrs.data_type != T::VARIANT {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Attempt to create data block for wrong type.",
            ));
        }
        let header = Self::read_block_header(&mut buffer, grid_position)?;

        block.reinitialize(header);
        let mut decompressed = data_attrs.compression.decoder(buffer);
        block.read_data(&mut decompressed)?;

        Ok(())
    }
}

/// Writes blocks to rust writers.
pub trait DefaultBlockWriter<
    T: ReflectedType,
    W: std::io::Write,
    B: DataBlock<T> + WriteableDataBlock,
>
{
    fn write_block(
        mut buffer: W,
        data_attrs: &DatasetAttributes,
        block: &B,
    ) -> std::io::Result<()> {
        if data_attrs.data_type != T::VARIANT {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Attempt to write data block for wrong type.",
            ));
        }

        let mode: u16 = if block.get_num_elements() == block.get_size().iter().product::<u32>() {
            BLOCK_FIXED_LEN
        } else {
            BLOCK_VAR_LEN
        };
        buffer.write_u16::<N5Endian>(mode)?;
        buffer.write_u16::<N5Endian>(data_attrs.get_ndim() as u16)?;
        for i in block.get_size() {
            buffer.write_u32::<N5Endian>(*i)?;
        }

        if mode != BLOCK_FIXED_LEN {
            buffer.write_u32::<N5Endian>(block.get_num_elements())?;
        }

        let mut compressor = data_attrs.compression.encoder(buffer);
        block.write_data(&mut compressor)?;

        Ok(())
    }
}

// TODO: needed because cannot invoke type parameterized static trait methods
// directly from trait name in Rust. Symptom of design problems with
// `DefaultBlockReader`, etc.
#[derive(Debug)]
pub struct DefaultBlock;
impl<R: std::io::Read> DefaultBlockHeaderReader<R> for DefaultBlock {}
impl<T: ReflectedType, R: std::io::Read> DefaultBlockReader<T, R> for DefaultBlock {}
impl<T: ReflectedType, W: std::io::Write, B: DataBlock<T> + WriteableDataBlock>
    DefaultBlockWriter<T, W, B> for DefaultBlock
{
}
