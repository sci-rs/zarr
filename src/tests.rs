use super::chunk::{
    DefaultChunk,
    DefaultChunkReader,
    DefaultChunkWriter,
};
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

fn doc_spec_array_metadata(compression: compression::CompressionType) -> ArrayMetadata {
    ArrayMetadata::new(
        smallvec![5, 6, 7],
        smallvec![1, 2, 3],
        DataType::Int {
            size: IntSize::B2,
            endian: Endian::Big,
        },
        compression,
    )
}

pub(crate) fn test_read_doc_spec_chunk(chunk: &[u8], compression: compression::CompressionType) {
    let buff = Cursor::new(chunk);
    let array_meta = doc_spec_array_metadata(compression);

    let chunk = <DefaultChunk as DefaultChunkReader<i16, std::io::Cursor<&[u8]>>>::read_chunk(
        buff,
        &array_meta,
        smallvec![0, 0, 0],
    )
    .expect("read_chunk failed");

    assert_eq!(chunk.get_grid_position(), &[0, 0, 0]);
    assert_eq!(chunk.get_data(), &DOC_SPEC_CHUNK_DATA);
}

pub(crate) fn test_write_doc_spec_chunk(
    expected_chunk: &[u8],
    compression: compression::CompressionType,
) {
    let array_meta = doc_spec_array_metadata(compression);
    let chunk_in = SliceDataChunk::new(smallvec![0, 0, 0], DOC_SPEC_CHUNK_DATA);
    let mut buff: Vec<u8> = Vec::new();

    <DefaultChunk as DefaultChunkWriter<i16, _, _>>::write_chunk(&mut buff, &array_meta, &chunk_in)
        .expect("read_chunk failed");

    assert_eq!(buff, expected_chunk);
}

pub(crate) fn test_chunk_compression_rw(compression: compression::CompressionType) {
    let array_meta = ArrayMetadata::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        i32::ZARR_TYPE,
        compression,
    );
    let chunk_data: Vec<i32> = (0..125_i32).collect();
    let chunk_in = SliceDataChunk::new(smallvec![0, 0, 0], &chunk_data);

    let mut inner: Vec<u8> = Vec::new();

    <DefaultChunk as DefaultChunkWriter<i32, _, _>>::write_chunk(
        &mut inner,
        &array_meta,
        &chunk_in,
    )
    .expect("write_chunk failed");

    let chunk_out = <DefaultChunk as DefaultChunkReader<i32, _>>::read_chunk(
        &inner[..],
        &array_meta,
        smallvec![0, 0, 0],
    )
    .expect("read_chunk failed");

    assert_eq!(chunk_out.get_grid_position(), &[0, 0, 0]);
    assert_eq!(chunk_out.get_data(), &chunk_data[..]);
}

pub(crate) fn test_varlength_chunk_rw(compression: compression::CompressionType) {
    let array_meta = ArrayMetadata::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        i32::ZARR_TYPE,
        compression,
    );
    let chunk_data: Vec<i32> = (0..100_i32).collect();
    let chunk_in = SliceDataChunk::new(smallvec![0, 0, 0], &chunk_data);

    let mut inner: Vec<u8> = Vec::new();

    <DefaultChunk as DefaultChunkWriter<i32, _, _>>::write_chunk(
        &mut inner,
        &array_meta,
        &chunk_in,
    )
    .expect("write_chunk failed");

    let chunk_out = <DefaultChunk as DefaultChunkReader<i32, _>>::read_chunk(
        &inner[..],
        &array_meta,
        smallvec![0, 0, 0],
    )
    .expect("read_chunk failed");

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

pub(crate) fn create_array<N: ZarrTestable>() {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    let array_meta = ArrayMetadata::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        i32::ZARR_TYPE,
        crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
    );
    create
        .create_array("foo/bar", &array_meta)
        .expect("Failed to create array");

    let read = create.open_reader();

    assert_eq!(read.get_array_metadata("foo/bar").unwrap(), array_meta);
}

pub(crate) fn absolute_relative_paths<N: ZarrTestable>() -> Result<()> {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    let array_meta = ArrayMetadata::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        i32::ZARR_TYPE,
        crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
    );
    create
        .create_array("foo/bar", &array_meta)
        .expect("Failed to create array");

    let read = create.open_reader();

    assert_eq!(read.get_array_metadata("foo/bar")?, array_meta);
    assert!(read.exists("/foo/bar")?);
    assert_eq!(read.get_array_metadata("/foo/bar")?, array_meta);
    assert!(read.array_exists("/foo/bar")?);
    // Repeated slashes are combined in Rust, not roots.
    assert!(!read.exists("/foo//foo/bar")?);
    assert!(read.exists("/foo//bar")?);
    assert_eq!(read.get_array_metadata("/foo//bar")?, array_meta);
    assert!(read.array_exists("/foo//bar")?);

    Ok(())
}

pub(crate) fn attributes_rw<N: ZarrTestable>() {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    let group = "foo/bar";
    create.create_group(group).expect("Failed to create group");

    assert_eq!(
        Value::Object(create.list_attributes(group).unwrap()),
        json!({})
    );

    // Currently reading attributes of an implicit group is an error.
    // Whether this should be the case is still open for decision.
    assert!(create.list_attributes("foo").is_err());

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
    assert_eq!(create.list_attributes(group).unwrap(), attrs_1);

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
        Value::Object(create.list_attributes(group).unwrap()),
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
        Value::Object(create.list_attributes(group).unwrap()),
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
    assert_eq!(create.list_attributes(group).unwrap(), attrs_3);

    // Test attributes for arrays also.
    let array = "baz";
    let array_meta = ArrayMetadata::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        i32::ZARR_TYPE,
        crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
    );
    create
        .create_array(array, &array_meta)
        .expect("Failed to create array");

    assert_eq!(
        Value::Object(create.list_attributes(array).unwrap()),
        json!({})
    );
    create
        .set_attributes(array, attrs_1.clone())
        .expect("Failed to set attributes");
    assert_eq!(create.list_attributes(array).unwrap(), attrs_1);
}

pub(crate) fn create_chunk_rw<N: ZarrTestable>() {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    let array_meta = ArrayMetadata::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        i32::ZARR_TYPE,
        crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
    );
    let chunk_data: Vec<i32> = (0..125_i32).collect();
    let chunk_in = crate::SliceDataChunk::new(smallvec![0, 0, 0], &chunk_data);

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
    let chunk_in = crate::SliceDataChunk::new(smallvec![0, 0, 0], &chunk_data);
    create
        .write_chunk("foo/bar", &array_meta, &chunk_in)
        .expect("Failed to write chunk");
    let chunk_out = read
        .read_chunk::<i32>("foo/bar", &array_meta, smallvec![0, 0, 0])
        .expect("Failed to read chunk")
        .expect("Chunk is empty");

    assert_eq!(chunk_out.get_data(), &chunk_data[..]);
}

pub(crate) fn delete_chunk<N: ZarrTestable>() {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    let array_meta = ArrayMetadata::new(
        smallvec![10, 100, 100],
        smallvec![5, 5, 5],
        i32::ZARR_TYPE,
        crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
    );

    let coord_a = smallvec![1, 2, 3];
    let coord_b: GridCoord = smallvec![1, 2, 4];

    let array = "foo/bar";
    let chunk_data: Vec<i32> = (0..125_i32).collect();
    let chunk_in = crate::SliceDataChunk::new(coord_a.clone(), &chunk_data);

    create
        .create_array(array, &array_meta)
        .expect("Failed to create array");
    create
        .write_chunk(array, &array_meta, &chunk_in)
        .expect("Failed to write chunk");

    assert!(create
        .read_chunk::<i32>(array, &array_meta, coord_a.clone())
        .expect("Failed to read chunk")
        .is_some());

    assert!(create.delete_chunk(array, &array_meta, &coord_a).unwrap());
    assert!(create.delete_chunk(array, &array_meta, &coord_a).unwrap());
    assert!(create.delete_chunk(array, &array_meta, &coord_b).unwrap());

    assert!(create
        .read_chunk::<i32>(array, &array_meta, coord_a.clone())
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
        fn create_array() {
            $crate::tests::create_array::<$backend>()
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
