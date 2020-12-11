# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [0.7.6] - 2020-10-26
### Added
- `N5NdarrayReader::read_ndarray_into(_with_buffer)?` allow reading into
  existing ndarrays and with existing block buffers for fewer allocations.

### Changed
- `N5NdarrayWriter::write_ndarray` performs fewer per-block allocations.

### Fixed
- Fixed a performance regression around path canonicalization in `N5Filesystem`.

## [0.7.5] - 2020-10-21
### Fixed
- `N5Filesystem` incorrectly handled absolute data paths for many operations.

## [0.7.4] - 2020-10-20
### Changed
- Dataset paths can now be absolute (with a leading `/`)

### Fixed
- `N5Filesystem::list` now traverses symlinks.

## [0.7.3] - 2020-07-17
### Added
- An alternative pure Rust Lz4 backend is available under the `lz_pure` feature
  flag. This is not the default as it does not yet match the performance of the
  default `lz` feature backend.

### Changed
- `Bzip2Compression` now respects exact `block_size` value in [1, 9] rather than
  just a three-tier behavior.
- `N5Filesystem::open` and `N5Filesystem::open_or_create` now accept any
  `AsRef<Path>` base path. This is backwards compatible with the previous `&str`
  type.

### Fixed
- Fixed a panic in `N5NdarrayReader::read_ndarray` when a block's dataset-
  level bounds overlapped the request bounds, but its actual bounds did not.
- Removed the ability to write blocks with a different datatype than the
  attributes' to a dataset.
- Dependency updates include fixes for upstream vulnerabilities and bugs in
  bzip2 and gzip compressions.

## [0.7.2] - 2020-05-23
### Added
- `CompressionType` implements `FromStr` for default construction from plain
  type names.
- All public types now implement `Debug`.

### Changed
- `get_bounds` is now implemented for all `SliceDataBlock`, not just
  `VecDataBlock`.

## [0.7.1] - 2020-01-19
### Fixed
- `DatasetAttributes::get_grid_extent` no longer undercalculates extents.

## [0.7.0] - 2020-01-13
### Added
- `N5Writer::deleteBlock` to remove blocks from a dataset.
- `DatasetAttributes::get_num_blocks` to get the total number of blocks possible
  in a dataset.
- Data block metadata now includes the size of the block on the backend.

### Changed
- Setting an attribute or attributes by key will now overwrite those attributes
  rather than recursively merging with them.
- Data block metadata fields are now optional, to support more platforms.
- `N5Reader::list` is now in a new trait, `N5Lister`.
- `N5NdarrayWriter::write_ndarray` now accepts any `ndarray::AsArray` input type.

## [0.6.1] - 2020-01-08
### Fixed
- The filesystem could corrupt attributes files when setting attributes,
  including the root version attributes when opening, because trailing data
  was not truncated.
- The filesystem would leave trailing data in blocks that were overwritten with
  shorter data.

## [0.6.0] - 2020-01-02
### Added
- `DatasetAttribues` gained coordinate checking methods.

### Changed
- `N5Reader::exists` and `N5Reader::dataset_exists` return `Result`s.
- `Version` is now a `semver` type.
- No longer depend on `regex`.

## [0.5.0] - 2019-11-15
### Changed
- The minimum supported Rust version is now 1.39.
- All coordinates are now unsigned rather than signed integers, since Java N5
  has adopted this recommendation as of spec version 2.1.3.
- `SliceDataBlock` trait allows using slices for `write_block` and
  `read_block_into`.
- `ReadableDataBlock`, `ReinitDataBlock`, and `WriteableDataBlock` traits and
  bounds have been refactored to allow writing of const slices, reinitialization
  without reserialization, and other features.
- `ReflectedType` now has more bounds for thread safety.
- LZ4 blocks are now written in independent mode, to more closely match the
  behavior of Java N5.
- `read_ndarray` now performs fewer allocations.


## [0.4.0] - 2019-06-05
### Added
- `N5Reader::read_block_into`: read a block into an existing `VecDataBlock`
  without allocating a new block.
- `data_type_match!`: a macro to dispatch a primitive-type generic block of
  code based on a `DataType`.
- `ReflectedType` trait, which supercedes the `TypeReflection` trait,
  `DataBlockCreator` trait, and `Clone` bound on primitive types.
- `N5NdarrayWriter` supertrait that provides a `write_ndarray` method to write
  ndarrays serially to blocks.

### Changed
- All coordinates are now `SmallVec` types `GridCoord` and `BlockCoord`. This
  avoids allocations for datasets with <= 6 dimensions.
- ndarray reading is now in a `N5NdarrayReader` supertrait.

### Removed
- `TypeReflection` trait.
- `DataBlockCreator` trait.


## [0.3.0] - 2019-01-16
### Changed
- `DataType` implements `Display`.
- `VecDataBlock<T>` implements `Clone` and requires `T: Clone`.


## [0.2.4] - 2018-10-17
### Changed
- Updated the `flate2-rs` GZIP dependency to be compatible with WebAssembly.


## [0.2.3] - 2018-10-11
### Added
- `N5Reader::block_metadata`: retrieve block metadata (currently timestamps)
  without reading the block.


## [0.2.2] - 2018-10-07
### Added
- `N5Reader::get_block_uri`: implementation-specific URI strings for data
  blocks.
- `DatasetAttributes::get_[block_]num_elements`: convenient access to
  dataset and block element counts.
- `DatasetAttributes::coord_iter`: convenient iteration over all possible
  coordinates (requires `use_ndarray` feature).

### Changed
- Filesystem implementation is now behind a `filesystem` feature flag, which is
  default.


## [0.2.1] - 2018-06-18
### Added
- Easy import prelude: `use n5::prelude::*;`

### Fixed
- Mode flag was inverted from correct setting for default and varlength blocks.


## [0.2.0] - 2018-03-10
### Added
- Dataset and container removal methods for `N5Writer`.
- `N5Reader::read_ndarray` to read arbitrary bounding box column-major
  `ndarray` arrays from datasets.

### Fixed
- Performance issues with some data types, especially writes.


## [0.1.0] - 2018-02-28
