//! A filesystem-backed Zarr hierarchy.

use std::fs::{
    self,
    File,
};
use std::io::{
    BufReader,
    BufWriter,
    Error,
    ErrorKind,
    Result,
};
use std::path::PathBuf;

use fs2::FileExt;
use serde_json::{
    self,
    json,
    Value,
};
use walkdir::WalkDir;

use crate::{
    storage::{
        ReadableStore,
        WriteableStore,
    },
    EntryPointMetadata,
    Hierarchy,
    HierarchyLister,
    HierarchyReader,
};

/// A filesystem-backed Zarr hierarchy.
#[derive(Clone, Debug)]
pub struct FilesystemHierarchy {
    base_path: PathBuf,
    entry_point_metadata: EntryPointMetadata,
}

impl Hierarchy for FilesystemHierarchy {
    fn get_entry_point_metadata(&self) -> &EntryPointMetadata {
        &self.entry_point_metadata
    }
}

impl FilesystemHierarchy {
    fn read_entry_point_metadata(base_path: &PathBuf) -> Result<EntryPointMetadata> {
        let entry_point_path = base_path.join(crate::ENTRY_POINT_KEY);
        let reader = BufReader::new(File::open(entry_point_path)?);
        Ok(serde_json::from_reader(reader)?)
    }

    /// Open an existing Zarr hierarchy by path.
    pub fn open<P: AsRef<std::path::Path>>(base_path: P) -> Result<FilesystemHierarchy> {
        let base_path = PathBuf::from(base_path.as_ref());
        let entry_point_metadata = Self::read_entry_point_metadata(&base_path)?;

        let reader = FilesystemHierarchy {
            base_path,
            entry_point_metadata,
        };

        let version = reader.get_version()?;

        if !version.matches(&crate::VERSION) {
            return Err(Error::new(ErrorKind::Other, "TODO: Incompatible version"));
        }

        Ok(reader)
    }

    /// Open an existing Zarr hierarchy by path or create one if none exists.
    ///
    /// Note this will update the version attribute for existing hierarchys.
    pub fn open_or_create<P: AsRef<std::path::Path>>(base_path: P) -> Result<FilesystemHierarchy> {
        let base_path = PathBuf::from(base_path.as_ref());
        let entry_point_path = base_path.join(crate::ENTRY_POINT_KEY);

        let entry_point_metadata = if entry_point_path.exists() {
            Self::read_entry_point_metadata(&base_path)?
        } else {
            fs::create_dir_all(&base_path)?;
            let metadata = EntryPointMetadata::default();
            let file = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(entry_point_path)?;
            file.lock_exclusive()?;

            let writer = BufWriter::new(file);
            serde_json::to_writer(writer, &metadata)?;
            metadata
        };

        let reader = FilesystemHierarchy {
            base_path,
            entry_point_metadata,
        };

        let version = reader.get_version()?;

        if !version.matches(&crate::VERSION) {
            return Err(Error::new(ErrorKind::Other, "TODO: Incompatible version"));
        }

        Ok(reader)
    }

    pub fn get_attributes(&self, key: &str) -> Result<Value> {
        // TODO: no longer used, but should be adapted for getting array/group user attributes
        let path = self.get_path(key)?;
        if path.exists() && path.is_file() {
            let file = File::open(path)?;
            file.lock_shared()?;
            let reader = BufReader::new(file);
            Ok(serde_json::from_reader(reader)?)
        } else {
            let mut path = path;
            // TODO
            path.set_extension("");
            path.set_extension("");
            // Check for implicit group.
            if path.exists() && path.is_file() {
                Ok(json!({}))
            } else {
                Err(Error::new(ErrorKind::NotFound, "Path does not exist"))
            }
        }
    }

    /// Get the filesystem path for a given Zarr key.
    fn get_path(&self, key: &str) -> Result<PathBuf> {
        // Note: cannot use `canonicalize` on both the constructed array path
        // and `base_path` and check `starts_with`, because `canonicalize` also
        // requires the path exist.
        use std::path::{
            Component,
            Path,
        };

        // Normalize the path to be relative.
        let mut components = Path::new(key).components();
        while components.as_path().has_root() {
            match components.next() {
                Some(Component::Prefix(_)) => {
                    return Err(Error::new(
                        ErrorKind::NotFound,
                        "Path name is outside this Zarr filesystem on a prefix path",
                    ))
                }
                Some(Component::RootDir) => (),
                // This should be unreachable.
                _ => return Err(Error::new(ErrorKind::NotFound, "Path is malformed")),
            }
        }
        let unrooted_path = components.as_path();

        // Check that the path is inside the hierarchy's base path.
        let mut nest: i32 = 0;
        for component in unrooted_path.components() {
            match component {
                // This should be unreachable.
                Component::Prefix(_) | Component::RootDir => {
                    return Err(Error::new(ErrorKind::NotFound, "Path is malformed"))
                }
                Component::CurDir => continue,
                Component::ParentDir => nest -= 1,
                Component::Normal(_) => nest += 1,
            };
        }

        if nest < 0 {
            Err(Error::new(
                ErrorKind::NotFound,
                "Path name is outside this Zarr filesystem",
            ))
        } else {
            Ok(self.base_path.join(unrooted_path))
        }
    }
}

impl ReadableStore for FilesystemHierarchy {
    type GetReader = BufReader<File>;

    fn exists(&self, key: &str) -> Result<bool> {
        let target = self.get_path(key)?;
        Ok(target.is_file())
    }

    fn get(&self, key: &str) -> Result<Option<Self::GetReader>> {
        let target = self.get_path(key)?;
        if target.is_file() {
            let file = File::open(target)?;
            file.lock_shared()?;
            Ok(Some(BufReader::new(file)))
        } else {
            Ok(None)
        }
    }

    fn uri(&self, key: &str) -> Result<String> {
        self.get_path(key).and_then(|p| {
            p.into_os_string()
                .into_string()
                .map(|mut s| {
                    s.insert_str(0, "file://");
                    s
                })
                .map_err(|_| Error::new(ErrorKind::NotFound, "TODO: non-unicode path"))
        })
    }
}

impl HierarchyLister for FilesystemHierarchy {
    fn list(&self, path_name: &str) -> Result<Vec<String>> {
        // TODO: shouldn't do this in a closure to not equivocate errors with Nones.
        Ok(fs::read_dir(self.get_path(path_name)?)?
            .filter_map(|e| {
                if let Ok(file) = e {
                    if fs::metadata(file.path())
                        .map(|f| f.file_type().is_dir())
                        .ok()
                        == Some(true)
                    {
                        file.file_name().into_string().ok()
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect())
    }
}

fn merge_top_level(a: &mut Value, b: serde_json::Map<String, Value>) {
    match a {
        &mut Value::Object(ref mut a) => {
            for (k, v) in b {
                a.insert(k, v);
            }
        }
        a => {
            *a = b.into();
        }
    }
}

impl WriteableStore for FilesystemHierarchy {
    type SetWriter = BufWriter<File>;

    fn set<F: FnOnce(Self::SetWriter) -> Result<()>>(&self, key: &str, value: F) -> Result<()> {
        let target = self.get_path(key)?;
        if let Some(parent) = target.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(target)?;
        file.lock_exclusive()?;
        // Truncate after the lock is acquired, rather than on opening.
        file.set_len(0)?;

        let writer = BufWriter::new(file);

        value(writer)
    }

    fn erase(&self, key: &str) -> Result<bool> {
        let path = self.get_path(key)?;

        if path.exists() {
            let file = File::open(&path)?;
            file.lock_exclusive()?;
            fs::remove_file(&path)?;
        }

        Ok(!path.exists())
    }

    fn erase_prefix(&self, key_prefix: &str) -> Result<bool> {
        let path = self.get_path(key_prefix)?;

        if path.exists() {
            for entry in WalkDir::new(&path).contents_first(true) {
                let entry = entry?;

                if entry.file_type().is_dir() {
                    fs::remove_dir(entry.path())?;
                } else {
                    let file = File::open(entry.path())?;
                    file.lock_exclusive()?;
                    fs::remove_file(entry.path())?;
                }
            }
        }

        Ok(!path.exists())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_backend;
    use crate::tests::{
        ContextWrapper,
        ZarrTestable,
    };
    use crate::{
        chunk::DataChunk,
        ArrayMetadata,
        HierarchyWriter,
    };
    use tempdir::TempDir;

    impl crate::tests::ZarrTestable for FilesystemHierarchy {
        type Wrapper = ContextWrapper<TempDir, FilesystemHierarchy>;

        fn temp_new_rw() -> Self::Wrapper {
            let dir = TempDir::new("rust_zarr_tests").unwrap();
            let zarr = FilesystemHierarchy::open_or_create(dir.path())
                .expect("Failed to create Zarr filesystem");

            ContextWrapper { context: dir, zarr }
        }

        fn open_reader(&self) -> Self {
            FilesystemHierarchy::open(&self.base_path).unwrap()
        }
    }

    test_backend!(FilesystemHierarchy);

    #[test]
    fn reject_exterior_paths() {
        let wrapper = FilesystemHierarchy::temp_new_rw();
        let create = wrapper.as_ref();

        assert!(create.get_path("/").is_ok());
        assert_eq!(create.get_path("/").unwrap(), create.get_path("").unwrap());
        assert!(create.get_path("/foo/bar").is_ok());
        assert_eq!(
            create.get_path("/foo/bar").unwrap(),
            create.get_path("foo/bar").unwrap()
        );
        assert!(create.get_path("//").is_ok());
        assert_eq!(create.get_path("//").unwrap(), create.get_path("").unwrap());
        assert!(create.get_path("/..").is_err());
        assert!(create.get_path("..").is_err());
        assert!(create.get_path("foo/bar/baz/../../..").is_ok());
        assert!(create.get_path("foo/bar/baz/../../../..").is_err());
    }

    // TODO
    // #[test]
    // fn accept_hardlink_attributes() {
    //     let wrapper = FilesystemHierarchy::temp_new_rw();
    //     let dir = TempDir::new("rust_zarr_tests_dupe").unwrap();
    //     let mut attr_path = dir.path().to_path_buf();
    //     attr_path.push(ATTRIBUTES_FILE);

    //     std::fs::hard_link(wrapper.zarr.get_attributes_path("").unwrap(), &attr_path).unwrap();

    //     wrapper.zarr.set_attribute("", "foo".into(), "bar").unwrap();

    //     let dupe = FilesystemHierarchy::open(dir.path()).unwrap();
    //     assert_eq!(dupe.get_attributes("").unwrap()["foo"], "bar");
    // }

    #[test]
    fn list_symlinked_arrays() {
        let wrapper = FilesystemHierarchy::temp_new_rw();
        let dir = TempDir::new("rust_zarr_tests_dupe").unwrap();
        let mut linked_path = wrapper.context.path().to_path_buf();
        linked_path.push("linked_array");

        #[cfg(target_family = "unix")]
        std::os::unix::fs::symlink(dir.path(), &linked_path).unwrap();
        #[cfg(target_family = "windows")]
        std::os::windows::fs::symlink_dir(dir.path(), &linked_path).unwrap();

        assert_eq!(wrapper.zarr.list("").unwrap(), vec!["linked_array"]);
        // TODO
        // assert!(wrapper.zarr.exists("linked_array").unwrap());

        let array_meta = ArrayMetadata::new(
            smallvec![10, 10, 10],
            smallvec![5, 5, 5],
            crate::DataType::INT32,
            crate::compression::CompressionType::Raw(
                crate::compression::raw::RawCompression::default(),
            ),
        );
        wrapper
            .zarr
            .create_array("linked_array", &array_meta)
            .expect("Failed to create array");
        assert!(wrapper.zarr.array_exists("linked_array").unwrap());
    }

    #[test]
    fn test_get_chunk_uri() {
        let dir = TempDir::new("rust_zarr_tests").unwrap();
        let path_str = dir.path().to_str().unwrap();

        let create = FilesystemHierarchy::open_or_create(path_str)
            .expect("Failed to create Zarr filesystem");
        let array_meta = ArrayMetadata::new(
            smallvec![10, 10, 10],
            smallvec![5, 5, 5],
            crate::DataType::INT32,
            crate::compression::CompressionType::Raw(
                crate::compression::raw::RawCompression::default(),
            ),
        );
        create
            .create_array("/foo/bar", &array_meta)
            .expect("Failed to create array");
        let uri = create
            .get_chunk_uri("/foo/bar", &array_meta, &vec![1, 2, 3])
            .unwrap();
        assert_eq!(uri, format!("file://{}/data/root/foo/bar/c1/2/3", path_str));
    }

    #[test]
    pub(crate) fn short_chunk_truncation() {
        let wrapper = FilesystemHierarchy::temp_new_rw();
        let create = wrapper.as_ref();
        let array_meta = ArrayMetadata::new(
            smallvec![10, 10, 10],
            smallvec![5, 5, 5],
            crate::DataType::INT32,
            crate::compression::CompressionType::Raw(
                crate::compression::raw::RawCompression::default(),
            ),
        );
        let chunk_data: Vec<i32> = (0..125_i32).collect();
        let chunk_in = crate::SliceDataChunk::new(
            array_meta.chunk_grid.chunk_shape.clone(),
            smallvec![0, 0, 0],
            &chunk_data,
        );

        create
            .create_array("foo/bar", &array_meta)
            .expect("Failed to create array");
        create
            .write_chunk("foo/bar", &array_meta, &chunk_in)
            .expect("Failed to write chunk");

        let read = create.open_reader();
        let chunk_out = read
            .read_chunk::<i32>("foo/bar", &array_meta, smallvec![0, 0, 0])
            .expect("Failed to read chunk")
            .expect("Chunk is empty");
        let missing_chunk_out = read
            .read_chunk::<i32>("foo/bar", &array_meta, smallvec![0, 0, 1])
            .expect("Failed to read chunk");

        assert_eq!(chunk_out.get_data(), &chunk_data[..]);
        assert!(missing_chunk_out.is_none());

        // Shorten data (this still will not catch trailing data less than the length).
        let chunk_data: Vec<i32> = (0..10_i32).collect();
        let chunk_in = crate::SliceDataChunk::new(
            array_meta.chunk_grid.chunk_shape.clone(),
            smallvec![0, 0, 0],
            &chunk_data,
        );
        create
            .write_chunk("foo/bar", &array_meta, &chunk_in)
            .expect("Failed to write chunk");

        let chunk_file = create
            .get_chunk_uri("foo/bar", &array_meta, &[0, 0, 0])
            .unwrap();
        let file = File::open(chunk_file.strip_prefix("file://").unwrap()).unwrap();
        let metadata = file.metadata().unwrap();

        let header_len = 2 * std::mem::size_of::<u16>() + 4 * std::mem::size_of::<u32>();
        assert_eq!(
            metadata.len(),
            (header_len + chunk_data.len() * std::mem::size_of::<i32>()) as u64
        );
    }
}
