#![cfg(feature = "use_ndarray")]

use std::iter::FromIterator;

use ndarray::Array;
use rand::distributions::Standard;
use rand::Rng;
use smallvec::smallvec;

use zarr::ndarray::prelude::*;
use zarr::prelude::*;

#[test]
fn test_read_ndarray() {
    let dir = tempdir::TempDir::new("rust_zarr_ndarray_tests").unwrap();

    let n =
        FilesystemHierarchy::open_or_create(dir.path()).expect("Failed to create Zarr filesystem");

    let chunk_shape = smallvec![3, 4, 2, 1];
    let array_meta = ArrayMetadata::new(
        smallvec![3, 300, 200, 100],
        chunk_shape.clone(),
        i32::ZARR_TYPE,
        CompressionType::default(),
    );
    let numel = array_meta.get_chunk_num_elements();

    let path_name = "test/array/group";
    n.create_array(path_name, &array_meta)
        .expect("Failed to create array");

    for k in 0..10 {
        let z = chunk_shape[3] * k;
        for j in 0..10 {
            let y = chunk_shape[2] * j;
            for i in 0..10 {
                let x = chunk_shape[1] * i;

                let mut chunk_data = Vec::<i32>::with_capacity(numel);

                for zo in 0..chunk_shape[3] {
                    for yo in 0..chunk_shape[2] {
                        for xo in 0..chunk_shape[1] {
                            chunk_data.push(1000 + x as i32 + xo as i32);
                            chunk_data.push(2000 + y as i32 + yo as i32);
                            chunk_data.push(3000 + z as i32 + zo as i32);
                        }
                    }
                }

                let chunk_in = VecDataChunk::new(
                    smallvec![0, u64::from(i), u64::from(j), u64::from(k)],
                    chunk_data,
                );
                n.write_chunk(path_name, &array_meta, &chunk_in)
                    .expect("Failed to write chunk");
            }
        }
    }

    let bbox = BoundingBox::new(smallvec![0, 5, 4, 3], smallvec![3, 35, 15, 7]);
    let a = n
        .read_ndarray::<i32>(path_name, &array_meta, &bbox)
        .unwrap();

    for z in 0..a.shape()[3] {
        for y in 0..a.shape()[2] {
            for x in 0..a.shape()[1] {
                assert_eq!(
                    a[[0, x, y, z]],
                    1005 + x as i32,
                    "0 {} {} {}: {}",
                    x,
                    y,
                    z,
                    a[[0, x, y, z]]
                );
                assert_eq!(
                    a[[1, x, y, z]],
                    2004 + y as i32,
                    "1 {} {} {}: {}",
                    x,
                    y,
                    z,
                    a[[1, x, y, z]]
                );
                assert_eq!(
                    a[[2, x, y, z]],
                    3003 + z as i32,
                    "2 {} {} {}: {}",
                    x,
                    y,
                    z,
                    a[[2, x, y, z]]
                );
            }
        }
    }
}

#[test]
fn test_read_ndarray_oob() {
    let dir = tempdir::TempDir::new("rust_zarr_ndarray_tests").unwrap();

    let n =
        FilesystemHierarchy::open_or_create(dir.path()).expect("Failed to create Zarr filesystem");

    let chunk_shape = smallvec![50, 100];
    let array_meta = ArrayMetadata::new(
        smallvec![100, 200],
        chunk_shape.clone(),
        i32::ZARR_TYPE,
        CompressionType::default(),
    );

    let path_name = "test/array/group";
    n.create_array(path_name, &array_meta)
        .expect("Failed to create array");

    let chunk_in = VecDataChunk::new(smallvec![1, 1], vec![1]);
    n.write_chunk(path_name, &array_meta, &chunk_in)
        .expect("Failed to write chunk");

    let bbox = BoundingBox::new(smallvec![45, 175], smallvec![50, 50]);
    let a = n
        .read_ndarray::<i32>(path_name, &array_meta, &bbox)
        .unwrap();
    assert!(a.iter().all(|v| *v == 0));
}

#[test]
fn test_write_read_ndarray() {
    let dir = tempdir::TempDir::new("rust_zarr_ndarray_tests").unwrap();

    let n =
        FilesystemHierarchy::open_or_create(dir.path()).expect("Failed to create Zarr filesystem");

    let chunk_shape = smallvec![3, 4, 2, 1];
    let array_meta = ArrayMetadata::new(
        smallvec![3, 300, 200, 100],
        chunk_shape.clone(),
        i32::ZARR_TYPE,
        CompressionType::default(),
    );

    let path_name = "test/array/group";
    n.create_array(path_name, &array_meta)
        .expect("Failed to create array");

    let rng = rand::thread_rng();
    let arr_shape = [3, 35, 15, 7];
    let array: Array<i32, _> =
        Array::from_iter(rng.sample_iter(&Standard).take(arr_shape.iter().product()))
            .into_shape(arr_shape.clone())
            .unwrap()
            .into_dyn();
    let offset = smallvec![0, 5, 4, 3];

    n.write_ndarray(path_name, &array_meta, offset.clone(), &array, 0)
        .unwrap();

    let bbox = BoundingBox::new(offset, arr_shape.iter().map(|s| *s as u64).collect());
    let a = n
        .read_ndarray::<i32>("test/array/group", &array_meta, &bbox)
        .unwrap();
    // Also test c-order.
    let mut a_c = Array::zeros(bbox.shape_ndarray_shape().as_slice());
    n.read_ndarray_into::<i32>("test/array/group", &array_meta, &bbox, a_c.view_mut())
        .unwrap();

    assert_eq!(array, a);
    assert_eq!(array, a_c);
    assert_eq!(a, a_c);
}
