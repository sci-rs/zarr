#![cfg(feature = "use_ndarray")]
#![cfg(any(feature = "gzip", feature = "gzip_pure"))]
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;

use ndarray::{
    Array,
    IxDyn,
};
use smallvec::smallvec;

use zarr::ndarray::prelude::*;
use zarr::prelude::*;

fn zarrita_test_data<T>(array_meta: &ArrayMetadata) -> Array<T, IxDyn>
where
    usize: TryInto<T>,
    <usize as TryInto<T>>::Error: std::fmt::Debug,
{
    let expected: Vec<T> = (0..(array_meta.get_num_elements()))
        .map(TryInto::try_into)
        .collect::<Result<Vec<T>, _>>()
        .unwrap();
    let array_shape_usize: Vec<usize> =
        array_meta.get_shape().iter().map(|s| *s as usize).collect();
    Array::from_shape_vec(array_shape_usize, expected).unwrap()
}

#[test]
fn test_read_zarrita_i16() {
    type ExpectedType = i16;
    let h =
        FilesystemHierarchy::open("tests/data/zarrita.zr3").expect("Failed to open Zarr hierarchy");

    let path = "/seq/i2";
    let array_meta = h.get_array_metadata(path).expect("Failed to read metadata");
    let bbox = array_meta.get_bounds();
    let array = h
        .read_ndarray::<ExpectedType>(path, &array_meta, &bbox)
        .unwrap();

    let expected = zarrita_test_data::<ExpectedType>(&array_meta);

    assert_eq!(array, expected);
}

#[test]
fn test_write_zarrita_i16() {
    type ExpectedType = i16;
    let h =
        FilesystemHierarchy::open("tests/data/zarrita.zr3").expect("Failed to open Zarr hierarchy");

    let path = "/seq/i2";
    let array_meta = h.get_array_metadata(path).expect("Failed to read metadata");
    let bbox = array_meta.get_bounds();

    let dir = tempdir::TempDir::new("rust_zarr_ndarray_tests").unwrap();

    let hw =
        FilesystemHierarchy::open_or_create(dir.path()).expect("Failed to create Zarr filesystem");
    hw.create_array(path, &array_meta).unwrap();

    let expected = zarrita_test_data::<ExpectedType>(&array_meta);
    hw.write_ndarray(path, &array_meta, smallvec![0, 0, 0], &expected)
        .unwrap();

    let array = hw
        .read_ndarray::<ExpectedType>(path, &array_meta, &bbox)
        .unwrap();

    assert_eq!(array, expected);

    fn read_to_buffer<S: AsRef<str>>(filename: S) -> std::io::Result<Vec<u8>> {
        let mut buffer = Vec::new();
        let mut f = File::open(filename.as_ref().strip_prefix("file://").unwrap())?;
        f.read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    for coord in array_meta.bounded_coord_iter(&bbox) {
        let grid_pos = GridCoord::from(&coord[..]);
        let zarrita_arr = h
            .read_chunk::<ExpectedType>(path, &array_meta, grid_pos.clone())
            .unwrap()
            .map(SliceDataChunk::into_data);
        let rust_arr = hw
            .read_chunk::<ExpectedType>(path, &array_meta, grid_pos.clone())
            .unwrap()
            .map(SliceDataChunk::into_data);
        assert_eq!(zarrita_arr, rust_arr);

        let _zarrita_chunk = h
            .get_chunk_uri(path, &array_meta, &grid_pos)
            .and_then(read_to_buffer)
            .unwrap();
        let _rust_chunk = hw
            .get_chunk_uri(path, &array_meta, &grid_pos)
            .and_then(read_to_buffer)
            .unwrap();
        // TODO: This does not produced identically compressed blobs because different gzip compressors.
        // assert_eq!(zarrita_chunk, rust_chunk);
    }
}
