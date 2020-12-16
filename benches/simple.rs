//! # Simple In-memory Read/Write Benchmarks
#![feature(test)]

extern crate test;

use rand::{
    distributions::Standard,
    Rng,
};
use test::Bencher;

use zarr::prelude::*;
use zarr::smallvec::smallvec;
use zarr::{
    DefaultChunk,
    DefaultChunkReader,
    DefaultChunkWriter,
};

fn test_chunk_compression_rw<T>(compression: compression::CompressionType, b: &mut Bencher)
where
    T: 'static + std::fmt::Debug + ReflectedType + PartialEq + Default,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
    VecDataChunk<T>: zarr::ReadableDataChunk + zarr::WriteableDataChunk,
{
    let data_attrs = DatasetAttributes::new(
        smallvec![1024, 1024, 1024],
        smallvec![64, 64, 64],
        T::VARIANT,
        compression,
    );
    let numel = data_attrs.get_chunk_num_elements();
    let rng = rand::thread_rng();
    let chunk_data: Vec<T> = rng.sample_iter(&Standard).take(numel).collect();

    let chunk_in = VecDataChunk::new(
        data_attrs.get_chunk_size().into(),
        smallvec![0, 0, 0],
        chunk_data.clone(),
    );

    let mut inner: Vec<u8> = Vec::new();

    b.iter(|| {
        DefaultChunk::write_chunk(&mut inner, &data_attrs, &chunk_in).expect("write_chunk failed");

        let _chunk_out = <DefaultChunk as DefaultChunkReader<T, _>>::read_chunk(
            &inner[..],
            &data_attrs,
            smallvec![0, 0, 0],
        )
        .expect("read_chunk failed");
    });

    b.bytes = (data_attrs.get_chunk_num_elements() * data_attrs.get_data_type().size_of()) as u64;
}

#[bench]
fn simple_rw_i8_raw(b: &mut Bencher) {
    test_chunk_compression_rw::<i8>(compression::raw::RawCompression.into(), b);
}

#[bench]
fn simple_rw_i16_raw(b: &mut Bencher) {
    test_chunk_compression_rw::<i16>(compression::raw::RawCompression.into(), b);
}

#[bench]
fn simple_rw_i32_raw(b: &mut Bencher) {
    test_chunk_compression_rw::<i32>(compression::raw::RawCompression.into(), b);
}

#[bench]
fn simple_rw_i64_raw(b: &mut Bencher) {
    test_chunk_compression_rw::<i64>(compression::raw::RawCompression.into(), b);
}
