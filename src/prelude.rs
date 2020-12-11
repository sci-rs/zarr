//! N5 prelude.
//!
//! This module contains the most used import targets for easy import into
//! client libraries.
//!
//! ```
//! extern crate n5;
//!
//! use n5::prelude::*;
//! ```

#[doc(no_inline)]
pub use crate::compression::{
    self,
    CompressionType,
};
#[cfg(feature = "filesystem")]
#[doc(no_inline)]
pub use crate::filesystem::N5Filesystem;
#[doc(no_inline)]
pub use crate::{
    BlockCoord,
    DataBlock,
    DataBlockMetadata,
    DataType,
    DatasetAttributes,
    GridCoord,
    N5Lister,
    N5Reader,
    N5Writer,
    ReflectedType,
    SliceDataBlock,
    VecDataBlock,
};
