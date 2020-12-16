//! Zarr prelude.
//!
//! This module contains the most used import targets for easy import into
//! client libraries.
//!
//! ```
//! extern crate zarr;
//!
//! use zarr::prelude::*;
//! ```

#[doc(no_inline)]
pub use crate::compression::{
    self,
    CompressionType,
};
#[cfg(feature = "filesystem")]
#[doc(no_inline)]
pub use crate::filesystem::FilesystemHierarchy;
#[doc(no_inline)]
pub use crate::{
    ChunkCoord,
    DataChunk,
    DataChunkMetadata,
    DataType,
    DatasetAttributes,
    GridCoord,
    HierarchyLister,
    HierarchyReader,
    HierarchyWriter,
    ReflectedType,
    SliceDataChunk,
    VecDataChunk,
};
