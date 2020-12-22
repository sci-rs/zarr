#![cfg(feature = "use_ndarray")]

use ndarray::Array;

use zarr::ndarray::prelude::*;
use zarr::prelude::*;

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

    let expected: Vec<ExpectedType> =
        (0..(array_meta.get_num_elements() as ExpectedType)).collect();
    let array_shape_usize: Vec<usize> =
        array_meta.get_shape().iter().map(|s| *s as usize).collect();
    let expected = Array::from_shape_vec(array_shape_usize, expected).unwrap();

    assert_eq!(array, expected);
}
