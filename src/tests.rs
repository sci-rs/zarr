use super::*;
use std::io::{
    Cursor,
    Result,
};

use serde_json::json;

const DOC_SPEC_CHUNK_DATA: [i16; 6] = [1, 2, 3, 4, 5, 6];

pub(crate) trait ZarrTestable: HierarchyReader + HierarchyWriter {
    type Wrapper: AsRef<Self>;

    fn temp_new_rw() -> Self::Wrapper;

    fn open_reader(&self) -> Self;
}

/// Wrapper type for holding a context from dropping during the lifetime of an
/// Zarr backend. This is useful for things like tempdirs.
pub struct ContextWrapper<C, N: HierarchyReader + HierarchyWriter> {
    pub context: C,
    pub zarr: N,
}

impl<C, N: HierarchyReader + HierarchyWriter> AsRef<N> for ContextWrapper<C, N> {
    fn as_ref(&self) -> &N {
        &self.zarr
    }
}

fn doc_spec_dataset_attributes(compression: compression::CompressionType) -> DatasetAttributes {
    DatasetAttributes::new(
        smallvec![5, 6, 7],
        smallvec![1, 2, 3],
        DataType::INT16,
        compression,
    )
}

pub(crate) fn test_read_doc_spec_chunk(chunk: &[u8], compression: compression::CompressionType) {
    let buff = Cursor::new(chunk);
    let data_attrs = doc_spec_dataset_attributes(compression);

    let chunk = <DefaultChunk as DefaultChunkReader<i16, std::io::Cursor<&[u8]>>>::read_chunk(
        buff,
        &data_attrs,
        smallvec![0, 0, 0],
    )
    .expect("read_chunk failed");

    assert_eq!(chunk.get_size(), data_attrs.get_chunk_size());
    assert_eq!(chunk.get_grid_position(), &[0, 0, 0]);
    assert_eq!(chunk.get_data(), &DOC_SPEC_CHUNK_DATA);
}

pub(crate) fn test_write_doc_spec_chunk(
    expected_chunk: &[u8],
    compression: compression::CompressionType,
) {
    let data_attrs = doc_spec_dataset_attributes(compression);
    let chunk_in = SliceDataChunk::new(
        data_attrs.chunk_grid.chunk_size.clone(),
        smallvec![0, 0, 0],
        DOC_SPEC_CHUNK_DATA,
    );
    let mut buff: Vec<u8> = Vec::new();

    <DefaultChunk as DefaultChunkWriter<i16, _, _>>::write_chunk(&mut buff, &data_attrs, &chunk_in)
        .expect("read_chunk failed");

    assert_eq!(buff, expected_chunk);
}

pub(crate) fn test_chunk_compression_rw(compression: compression::CompressionType) {
    let data_attrs = DatasetAttributes::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        DataType::INT32,
        compression,
    );
    let chunk_data: Vec<i32> = (0..125_i32).collect();
    let chunk_in = SliceDataChunk::new(
        data_attrs.chunk_grid.chunk_size.clone(),
        smallvec![0, 0, 0],
        &chunk_data,
    );

    let mut inner: Vec<u8> = Vec::new();

    <DefaultChunk as DefaultChunkWriter<i32, _, _>>::write_chunk(
        &mut inner,
        &data_attrs,
        &chunk_in,
    )
    .expect("write_chunk failed");

    let chunk_out = <DefaultChunk as DefaultChunkReader<i32, _>>::read_chunk(
        &inner[..],
        &data_attrs,
        smallvec![0, 0, 0],
    )
    .expect("read_chunk failed");

    assert_eq!(chunk_out.get_size(), &[5, 5, 5]);
    assert_eq!(chunk_out.get_grid_position(), &[0, 0, 0]);
    assert_eq!(chunk_out.get_data(), &chunk_data[..]);
}

pub(crate) fn test_varlength_chunk_rw(compression: compression::CompressionType) {
    let data_attrs = DatasetAttributes::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        DataType::INT32,
        compression,
    );
    let chunk_data: Vec<i32> = (0..100_i32).collect();
    let chunk_in = SliceDataChunk::new(
        data_attrs.chunk_grid.chunk_size.clone(),
        smallvec![0, 0, 0],
        &chunk_data,
    );

    let mut inner: Vec<u8> = Vec::new();

    <DefaultChunk as DefaultChunkWriter<i32, _, _>>::write_chunk(
        &mut inner,
        &data_attrs,
        &chunk_in,
    )
    .expect("write_chunk failed");

    let chunk_out = <DefaultChunk as DefaultChunkReader<i32, _>>::read_chunk(
        &inner[..],
        &data_attrs,
        smallvec![0, 0, 0],
    )
    .expect("read_chunk failed");

    assert_eq!(chunk_out.get_size(), &[5, 5, 5]);
    assert_eq!(chunk_out.get_grid_position(), &[0, 0, 0]);
    assert_eq!(chunk_out.get_data(), &chunk_data[..]);
}

pub(crate) fn create_backend<N: ZarrTestable>() {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    create.create_group("").unwrap();
    create
        .set_attribute("", "foo".to_owned(), "bar")
        .expect("Failed to set attribute");

    let read = create.open_reader();

    assert!(read
        .get_version()
        .expect("Cannot read version")
        .matches(&crate::VERSION));
    assert_eq!(read.list_attributes("").unwrap()["foo"], "bar");
}

pub(crate) fn create_dataset<N: ZarrTestable>() {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    let data_attrs = DatasetAttributes::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        DataType::INT32,
        crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
    );
    create
        .create_dataset("foo/bar", &data_attrs)
        .expect("Failed to create dataset");

    let read = create.open_reader();

    assert_eq!(read.get_dataset_attributes("foo/bar").unwrap(), data_attrs);
}

pub(crate) fn absolute_relative_paths<N: ZarrTestable>() -> Result<()> {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    let data_attrs = DatasetAttributes::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        DataType::INT32,
        crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
    );
    create
        .create_dataset("foo/bar", &data_attrs)
        .expect("Failed to create dataset");

    let read = create.open_reader();

    assert_eq!(read.get_dataset_attributes("foo/bar")?, data_attrs);
    assert!(read.exists("/foo/bar")?);
    assert_eq!(read.get_dataset_attributes("/foo/bar")?, data_attrs);
    assert!(read.dataset_exists("/foo/bar")?);
    // Repeated slashes are combined in Rust, not roots.
    assert!(!read.exists("/foo//foo/bar")?);
    assert!(read.exists("/foo//bar")?);
    assert_eq!(read.get_dataset_attributes("/foo//bar")?, data_attrs);
    assert!(read.dataset_exists("/foo//bar")?);

    Ok(())
}

pub(crate) fn attributes_rw<N: ZarrTestable>() {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    let group = "foo";
    create.create_group(group).expect("Failed to create group");

    // Currently reading attributes that have not been set is an error.
    // Whether this should be the case is still open for decision.
    assert!(create.list_attributes(group).is_err());

    let attrs_1 = json!({
        "foo": {"bar": 42},
        "baz": [1, 2, 3],
    })
    .as_object()
    .unwrap()
    .clone();
    create
        .set_attributes(group, attrs_1.clone())
        .expect("Failed to set attributes");
    assert_eq!(
        create.list_attributes(group).unwrap(),
        serde_json::Value::Object(attrs_1)
    );

    let attrs_2 = json!({
        "baz": [4, 5, 6],
    })
    .as_object()
    .unwrap()
    .clone();
    create
        .set_attributes(group, attrs_2)
        .expect("Failed to set attributes");
    assert_eq!(
        create.list_attributes(group).unwrap(),
        json!({
            "foo": {"bar": 42},
            "baz": [4, 5, 6],
        })
    );

    // Test that key merging is at top-level only.
    let attrs_2 = json!({
        "foo": {"moo": 7},
    })
    .as_object()
    .unwrap()
    .clone();
    create
        .set_attributes(group, attrs_2)
        .expect("Failed to set attributes");
    assert_eq!(
        create.list_attributes(group).unwrap(),
        json!({
            "foo": {"moo": 7},
            "baz": [4, 5, 6],
        })
    );

    let attrs_3 = json!({
        "foo": null,
        "baz": null,
    })
    .as_object()
    .unwrap()
    .clone();
    create
        .set_attributes(group, attrs_3.clone())
        .expect("Failed to set attributes");
    assert_eq!(
        create.list_attributes(group).unwrap(),
        serde_json::Value::Object(attrs_3)
    );
}

pub(crate) fn create_chunk_rw<N: ZarrTestable>() {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    let data_attrs = DatasetAttributes::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        DataType::INT32,
        crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
    );
    let chunk_data: Vec<i32> = (0..125_i32).collect();
    let chunk_in = crate::SliceDataChunk::new(
        data_attrs.chunk_grid.chunk_size.clone(),
        smallvec![0, 0, 0],
        &chunk_data,
    );

    create
        .create_dataset("foo/bar", &data_attrs)
        .expect("Failed to create dataset");
    create
        .write_chunk("foo/bar", &data_attrs, &chunk_in)
        .expect("Failed to write chunk");

    let read = create.open_reader();
    let chunk_out = read
        .read_chunk::<i32>("foo/bar", &data_attrs, smallvec![0, 0, 0])
        .expect("Failed to read chunk")
        .expect("Chunk is empty");
    let missing_chunk_out = read
        .read_chunk::<i32>("foo/bar", &data_attrs, smallvec![0, 0, 1])
        .expect("Failed to read chunk");

    assert_eq!(chunk_out.get_data(), &chunk_data[..]);
    assert!(missing_chunk_out.is_none());

    // Shorten data (this still will not catch trailing data less than the length).
    let chunk_data: Vec<i32> = (0..10_i32).collect();
    let chunk_in = crate::SliceDataChunk::new(
        data_attrs.chunk_grid.chunk_size.clone(),
        smallvec![0, 0, 0],
        &chunk_data,
    );
    create
        .write_chunk("foo/bar", &data_attrs, &chunk_in)
        .expect("Failed to write chunk");
    let chunk_out = read
        .read_chunk::<i32>("foo/bar", &data_attrs, smallvec![0, 0, 0])
        .expect("Failed to read chunk")
        .expect("Chunk is empty");

    assert_eq!(chunk_out.get_data(), &chunk_data[..]);
}

pub(crate) fn delete_chunk<N: ZarrTestable>() {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    let data_attrs = DatasetAttributes::new(
        smallvec![10, 100, 100],
        smallvec![5, 5, 5],
        DataType::INT32,
        crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
    );

    let coord_a = smallvec![1, 2, 3];
    let coord_b: GridCoord = smallvec![1, 2, 4];

    let dataset = "foo/bar";
    let chunk_data: Vec<i32> = (0..125_i32).collect();
    let chunk_in = crate::SliceDataChunk::new(
        data_attrs.chunk_grid.chunk_size.clone(),
        coord_a.clone(),
        &chunk_data,
    );

    create
        .create_dataset(dataset, &data_attrs)
        .expect("Failed to create dataset");
    create
        .write_chunk(dataset, &data_attrs, &chunk_in)
        .expect("Failed to write chunk");

    assert!(create
        .read_chunk::<i32>(dataset, &data_attrs, coord_a.clone())
        .expect("Failed to read chunk")
        .is_some());

    assert!(create.delete_chunk(dataset, &data_attrs, &coord_a).unwrap());
    assert!(create.delete_chunk(dataset, &data_attrs, &coord_a).unwrap());
    assert!(create.delete_chunk(dataset, &data_attrs, &coord_b).unwrap());

    assert!(create
        .read_chunk::<i32>(dataset, &data_attrs, coord_a.clone())
        .expect("Failed to read chunk")
        .is_none());
}

#[macro_export]
macro_rules! test_backend {
    ($backend:ty) => {
        #[test]
        fn create_backend() {
            $crate::tests::create_backend::<$backend>()
        }

        #[test]
        fn create_dataset() {
            $crate::tests::create_dataset::<$backend>()
        }

        #[test]
        fn absolute_relative_paths() -> Result<()> {
            $crate::tests::absolute_relative_paths::<$backend>()
        }

        #[test]
        fn attributes_rw() {
            $crate::tests::attributes_rw::<$backend>()
        }

        #[test]
        fn create_chunk_rw() {
            $crate::tests::create_chunk_rw::<$backend>()
        }

        #[test]
        fn delete_chunk() {
            $crate::tests::delete_chunk::<$backend>()
        }
    };
}
