use super::*;
use std::io::{
    Cursor,
    Result,
};

use serde_json::json;

const DOC_SPEC_BLOCK_DATA: [i16; 6] = [1, 2, 3, 4, 5, 6];

pub(crate) trait N5Testable: N5Reader + N5Writer {
    type Wrapper: AsRef<Self>;

    fn temp_new_rw() -> Self::Wrapper;

    fn open_reader(&self) -> Self;
}

/// Wrapper type for holding a context from dropping during the lifetime of an
/// N5 backend. This is useful for things like tempdirs.
pub struct ContextWrapper<C, N: N5Reader + N5Writer> {
    pub context: C,
    pub n5: N,
}

impl<C, N: N5Reader + N5Writer> AsRef<N> for ContextWrapper<C, N> {
    fn as_ref(&self) -> &N {
        &self.n5
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

pub(crate) fn test_read_doc_spec_block(block: &[u8], compression: compression::CompressionType) {
    let buff = Cursor::new(block);
    let data_attrs = doc_spec_dataset_attributes(compression);

    let block = <DefaultBlock as DefaultBlockReader<i16, std::io::Cursor<&[u8]>>>::read_block(
        buff,
        &data_attrs,
        smallvec![0, 0, 0],
    )
    .expect("read_block failed");

    assert_eq!(block.get_size(), data_attrs.get_block_size());
    assert_eq!(block.get_grid_position(), &[0, 0, 0]);
    assert_eq!(block.get_data(), &DOC_SPEC_BLOCK_DATA);
}

pub(crate) fn test_write_doc_spec_block(
    expected_block: &[u8],
    compression: compression::CompressionType,
) {
    let data_attrs = doc_spec_dataset_attributes(compression);
    let block_in = SliceDataBlock::new(
        data_attrs.chunk_grid.block_size.clone(),
        smallvec![0, 0, 0],
        DOC_SPEC_BLOCK_DATA,
    );
    let mut buff: Vec<u8> = Vec::new();

    <DefaultBlock as DefaultBlockWriter<i16, _, _>>::write_block(&mut buff, &data_attrs, &block_in)
        .expect("read_block failed");

    assert_eq!(buff, expected_block);
}

pub(crate) fn test_block_compression_rw(compression: compression::CompressionType) {
    let data_attrs = DatasetAttributes::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        DataType::INT32,
        compression,
    );
    let block_data: Vec<i32> = (0..125_i32).collect();
    let block_in = SliceDataBlock::new(
        data_attrs.chunk_grid.block_size.clone(),
        smallvec![0, 0, 0],
        &block_data,
    );

    let mut inner: Vec<u8> = Vec::new();

    <DefaultBlock as DefaultBlockWriter<i32, _, _>>::write_block(
        &mut inner,
        &data_attrs,
        &block_in,
    )
    .expect("write_block failed");

    let block_out = <DefaultBlock as DefaultBlockReader<i32, _>>::read_block(
        &inner[..],
        &data_attrs,
        smallvec![0, 0, 0],
    )
    .expect("read_block failed");

    assert_eq!(block_out.get_size(), &[5, 5, 5]);
    assert_eq!(block_out.get_grid_position(), &[0, 0, 0]);
    assert_eq!(block_out.get_data(), &block_data[..]);
}

pub(crate) fn test_varlength_block_rw(compression: compression::CompressionType) {
    let data_attrs = DatasetAttributes::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        DataType::INT32,
        compression,
    );
    let block_data: Vec<i32> = (0..100_i32).collect();
    let block_in = SliceDataBlock::new(
        data_attrs.chunk_grid.block_size.clone(),
        smallvec![0, 0, 0],
        &block_data,
    );

    let mut inner: Vec<u8> = Vec::new();

    <DefaultBlock as DefaultBlockWriter<i32, _, _>>::write_block(
        &mut inner,
        &data_attrs,
        &block_in,
    )
    .expect("write_block failed");

    let block_out = <DefaultBlock as DefaultBlockReader<i32, _>>::read_block(
        &inner[..],
        &data_attrs,
        smallvec![0, 0, 0],
    )
    .expect("read_block failed");

    assert_eq!(block_out.get_size(), &[5, 5, 5]);
    assert_eq!(block_out.get_grid_position(), &[0, 0, 0]);
    assert_eq!(block_out.get_data(), &block_data[..]);
}

pub(crate) fn create_backend<N: N5Testable>() {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    create
        .set_attribute("", "foo".to_owned(), "bar")
        .expect("Failed to set attribute");

    let read = create.open_reader();

    assert_eq!(
        read.get_version().expect("Cannot read version"),
        crate::VERSION
    );
    assert_eq!(read.list_attributes("").unwrap()["foo"], "bar");
}

pub(crate) fn create_dataset<N: N5Testable>() {
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

pub(crate) fn absolute_relative_paths<N: N5Testable>() -> Result<()> {
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

pub(crate) fn attributes_rw<N: N5Testable>() {
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

pub(crate) fn create_block_rw<N: N5Testable>() {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    let data_attrs = DatasetAttributes::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        DataType::INT32,
        crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
    );
    let block_data: Vec<i32> = (0..125_i32).collect();
    let block_in = crate::SliceDataBlock::new(
        data_attrs.chunk_grid.block_size.clone(),
        smallvec![0, 0, 0],
        &block_data,
    );

    create
        .create_dataset("foo/bar", &data_attrs)
        .expect("Failed to create dataset");
    create
        .write_block("foo/bar", &data_attrs, &block_in)
        .expect("Failed to write block");

    let read = create.open_reader();
    let block_out = read
        .read_block::<i32>("foo/bar", &data_attrs, smallvec![0, 0, 0])
        .expect("Failed to read block")
        .expect("Block is empty");
    let missing_block_out = read
        .read_block::<i32>("foo/bar", &data_attrs, smallvec![0, 0, 1])
        .expect("Failed to read block");

    assert_eq!(block_out.get_data(), &block_data[..]);
    assert!(missing_block_out.is_none());

    // Shorten data (this still will not catch trailing data less than the length).
    let block_data: Vec<i32> = (0..10_i32).collect();
    let block_in = crate::SliceDataBlock::new(
        data_attrs.chunk_grid.block_size.clone(),
        smallvec![0, 0, 0],
        &block_data,
    );
    create
        .write_block("foo/bar", &data_attrs, &block_in)
        .expect("Failed to write block");
    let block_out = read
        .read_block::<i32>("foo/bar", &data_attrs, smallvec![0, 0, 0])
        .expect("Failed to read block")
        .expect("Block is empty");

    assert_eq!(block_out.get_data(), &block_data[..]);
}

pub(crate) fn delete_block<N: N5Testable>() {
    let wrapper = N::temp_new_rw();
    let create = wrapper.as_ref();
    let data_attrs = DatasetAttributes::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        DataType::INT32,
        crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
    );

    let coord_a = smallvec![1, 2, 3];
    let coord_b: GridCoord = smallvec![1, 2, 4];

    let dataset = "foo/bar";
    let block_data: Vec<i32> = (0..125_i32).collect();
    let block_in = crate::SliceDataBlock::new(
        data_attrs.chunk_grid.block_size.clone(),
        coord_a.clone(),
        &block_data,
    );

    create
        .create_dataset(dataset, &data_attrs)
        .expect("Failed to create dataset");
    create
        .write_block(dataset, &data_attrs, &block_in)
        .expect("Failed to write block");

    assert!(create
        .read_block::<i32>(dataset, &data_attrs, coord_a.clone())
        .expect("Failed to read block")
        .is_some());

    // TODO
    // assert!(create.delete_block(dataset, &coord_a).unwrap());
    // assert!(create.delete_block(dataset, &coord_a).unwrap());
    // assert!(create.delete_block(dataset, &coord_b).unwrap());

    assert!(create
        .read_block::<i32>(dataset, &data_attrs, coord_a.clone())
        .expect("Failed to read block")
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
        fn create_block_rw() {
            $crate::tests::create_block_rw::<$backend>()
        }

        #[test]
        fn delete_block() {
            $crate::tests::delete_block::<$backend>()
        }
    };
}
