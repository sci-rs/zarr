use std::io::{
    Error,
    ErrorKind,
    Read,
    Write,
};

use semver::VersionReq;
use serde_json::Value;

use crate::{
    canonicalize_path,
    chunk::{
        DataChunk,
        ReadableDataChunk,
        ReinitDataChunk,
        VecDataChunk,
        WriteableDataChunk,
    },
    ArrayMetadata,
    GridCoord,
    GroupMetadata,
    Hierarchy,
    HierarchyReader,
    HierarchyWriter,
    ReflectedType,
    StoreNodeMetadata,
};

pub trait ReadableStore {
    type GetReader: Read;

    /// TODO: not in zarr spec
    fn exists(&self, key: &str) -> Result<bool, Error>;

    fn get(&self, key: &str) -> Result<Option<Self::GetReader>, Error>;

    /// TODO: not in zarr spec
    fn uri(&self, key: &str) -> Result<String, Error>;
}

pub trait WriteableStore {
    type SetWriter: Write;

    fn set<F: FnOnce(Self::SetWriter) -> Result<(), Error>>(
        &self,
        key: &str,
        value: F,
    ) -> Result<(), Error>;

    // TODO differs from spec in that it returns a bool indicating existence of the key at the end of the operation.
    fn erase(&self, key: &str) -> Result<bool, Error>;

    // TODO
    fn erase_prefix(&self, key_prefix: &str) -> Result<bool, Error>;
}

/// TODO
///
/// ```
/// use zarr::prelude::*;
/// use zarr::storage::get_chunk_key;
/// use zarr::smallvec::smallvec;
/// let meta = ArrayMetadata::new(
///     smallvec![50, 40, 30],
///     smallvec![11, 10, 10],
///     i8::ZARR_TYPE,
///     zarr::compression::CompressionType::default(),
/// );
/// assert_eq!(get_chunk_key("/foo/baz", &meta, &[0, 0, 0]), "/data/root/foo/baz/c0/0/0");
/// assert_eq!(get_chunk_key("/foo/baz", &meta, &[1, 2, 3]), "/data/root/foo/baz/c1/2/3");
///
/// let meta = ArrayMetadata::new(
///     smallvec![],
///     smallvec![],
///     i8::ZARR_TYPE,
///     zarr::compression::CompressionType::default(),
/// );
/// assert_eq!(get_chunk_key("/foo/baz", &meta, &[]), "/data/root/foo/baz/c");
/// ```
pub fn get_chunk_key(base_path: &str, array_meta: &ArrayMetadata, grid_position: &[u64]) -> String {
    use std::fmt::Write;
    // TODO: normalize relative or absolute paths
    let canon_path = canonicalize_path(base_path);
    let mut chunk_key = if canon_path.is_empty() {
        format!("{}/c", crate::DATA_ROOT_PATH,)
    } else {
        format!("{}/{}/c", crate::DATA_ROOT_PATH, canon_path)
    };

    for (i, coord) in grid_position.iter().enumerate() {
        write!(chunk_key, "{}", coord).unwrap();
        if i < grid_position.len() - 1 {
            chunk_key.push_str(&array_meta.chunk_grid.separator)
        }
    }

    chunk_key
}

// From: https://github.com/serde-rs/json/issues/377
// TODO: Could be much better.
// TODO: zarr filesystem later settled on top-level key merging only.
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

impl<S: ReadableStore + Hierarchy> HierarchyReader for S {
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

    fn get_array_metadata(&self, path_name: &str) -> Result<ArrayMetadata, Error> {
        let array_path = self.array_metadata_key(path_name);
        let value_reader = ReadableStore::get(self, &array_path.to_str().expect("TODO"))?
            .ok_or_else(|| Error::from(std::io::ErrorKind::NotFound))?;
        Ok(serde_json::from_reader(value_reader)?)
    }

    fn exists(&self, path_name: &str) -> Result<bool, Error> {
        // TODO: needless path allocs
        // TODO: should follow spec more closely by using `list_dir` for implicit groups.
        Ok(
            self.exists(self.array_metadata_key(path_name).to_str().expect("TODO"))?
                || self.exists(self.group_metadata_key(path_name).to_str().expect("TODO"))?
                || self.exists(
                    self.group_metadata_key(path_name)
                        .with_extension("")
                        .with_extension("")
                        .to_str()
                        .expect("TODO"),
                )?,
        )
    }

    fn get_chunk_uri(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: &[u64],
    ) -> Result<String, Error> {
        let chunk_key = get_chunk_key(path_name, array_meta, &grid_position);
        self.uri(&chunk_key)
    }

    fn read_chunk<T>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataChunk<T>>, Error>
    where
        VecDataChunk<T>: DataChunk<T> + ReadableDataChunk,
        T: ReflectedType,
    {
        // TODO convert asserts to errors
        assert!(array_meta.in_bounds(&grid_position));

        // Construct chunk path string
        let chunk_key = get_chunk_key(path_name, array_meta, &grid_position);

        // Get key from store
        let value_reader = ReadableStore::get(self, &chunk_key)?;

        // Read value into container
        value_reader
            .map(|reader| {
                <crate::chunk::DefaultChunk as crate::chunk::DefaultChunkReader<T, _>>::read_chunk(
                    reader,
                    array_meta,
                    grid_position,
                )
            })
            .transpose()
    }

    fn read_chunk_into<
        T: ReflectedType,
        B: DataChunk<T> + ReinitDataChunk<T> + ReadableDataChunk,
    >(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: GridCoord,
        chunk: &mut B,
    ) -> Result<Option<()>, Error> {
        // TODO convert asserts to errors
        assert!(array_meta.in_bounds(&grid_position));

        // Construct chunk path string
        let chunk_key = get_chunk_key(path_name, array_meta, &grid_position);

        // Get key from store
        let value_reader = ReadableStore::get(self, &chunk_key)?;

        // Read value into container
        value_reader
            .map(|reader| {
                <crate::chunk::DefaultChunk as crate::chunk::DefaultChunkReader<T, _>>::read_chunk_into(
                    reader,
                    array_meta,
                    grid_position,
                    chunk,
                )
            })
            .transpose()
    }

    fn store_chunk_metadata(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: &[u64],
    ) -> Result<Option<StoreNodeMetadata>, Error> {
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

impl<S: ReadableStore + WriteableStore + Hierarchy> HierarchyWriter for S {
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
        // Because of implicit hierarchy rules, it is not necessary to create
        // the parent group.
        // let path_buf = PathBuf::from(path_name);
        // if let Some(parent) = path_buf.parent() {
        //     self.create_group(parent.to_str().expect("TODO"))?;
        // }
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

    fn create_array(&self, path_name: &str, array_meta: &ArrayMetadata) -> Result<(), Error> {
        // Because of implicit hierarchy rules, it is not necessary to create
        // the parent group.
        // let path_buf = PathBuf::from(path_name);
        // if let Some(parent) = path_buf.parent() {
        //     self.create_group(parent.to_str().expect("TODO"))?;
        // }
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
                Ok(serde_json::to_writer(writer, array_meta)?)
            })
        }
    }

    fn remove(&self, path_name: &str) -> Result<(), Error> {
        // TODO: needless allocs
        let metadata_key = self.group_metadata_key(path_name);
        self.erase(metadata_key.to_str().expect("TODO"))?;
        let mut metadata_key = self.array_metadata_key(path_name);
        self.erase(metadata_key.to_str().expect("TODO"))?;
        metadata_key.set_extension("");
        metadata_key.set_extension("");
        self.erase_prefix(self.data_path_key(path_name).to_str().expect("TODO"))?;
        Ok(())
    }

    fn write_chunk<T: ReflectedType, B: DataChunk<T> + WriteableDataChunk>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        chunk: &B,
    ) -> Result<(), Error> {
        // TODO convert assert
        // assert!(array_meta.in_bounds(chunk.get_grid_position()));
        let chunk_key = get_chunk_key(path_name, array_meta, chunk.get_grid_position());
        self.set(&chunk_key, |writer| {
            <crate::chunk::DefaultChunk as crate::chunk::DefaultChunkWriter<T, _, _>>::write_chunk(
                writer, array_meta, chunk,
            )
        })
    }

    fn delete_chunk(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: &[u64],
    ) -> Result<bool, Error> {
        let chunk_key = get_chunk_key(path_name, array_meta, grid_position);
        self.erase(&chunk_key)
    }
}
