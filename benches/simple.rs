//! # Simple In-memory Read/Write Benchmarks
#![feature(test)]

extern crate test;

use rand::{
    distributions::Standard,
    Rng,
};
use test::Bencher;

use n5::prelude::*;
use n5::smallvec::smallvec;
use n5::{
    DefaultBlock,
    DefaultBlockReader,
    DefaultBlockWriter,
};

fn test_block_compression_rw<T>(compression: compression::CompressionType, b: &mut Bencher)
where
    T: 'static + std::fmt::Debug + ReflectedType + PartialEq + Default,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
    VecDataBlock<T>: n5::ReadableDataBlock + n5::WriteableDataBlock,
{
    let data_attrs = DatasetAttributes::new(
        smallvec![1024, 1024, 1024],
        smallvec![64, 64, 64],
        T::VARIANT,
        compression,
    );
    let numel = data_attrs.get_block_num_elements();
    let rng = rand::thread_rng();
    let block_data: Vec<T> = rng.sample_iter(&Standard).take(numel).collect();

    let block_in = VecDataBlock::new(
        data_attrs.get_block_size().into(),
        smallvec![0, 0, 0],
        block_data.clone(),
    );

    let mut inner: Vec<u8> = Vec::new();

    b.iter(|| {
        DefaultBlock::write_block(&mut inner, &data_attrs, &block_in).expect("write_block failed");

        let _block_out = <DefaultBlock as DefaultBlockReader<T, _>>::read_block(
            &inner[..],
            &data_attrs,
            smallvec![0, 0, 0],
        )
        .expect("read_block failed");
    });

    b.bytes = (data_attrs.get_block_num_elements() * data_attrs.get_data_type().size_of()) as u64;
}

#[bench]
fn simple_rw_i8_raw(b: &mut Bencher) {
    test_block_compression_rw::<i8>(compression::raw::RawCompression.into(), b);
}

#[bench]
fn simple_rw_i16_raw(b: &mut Bencher) {
    test_block_compression_rw::<i16>(compression::raw::RawCompression.into(), b);
}

#[bench]
fn simple_rw_i32_raw(b: &mut Bencher) {
    test_block_compression_rw::<i32>(compression::raw::RawCompression.into(), b);
}

#[bench]
fn simple_rw_i64_raw(b: &mut Bencher) {
    test_block_compression_rw::<i64>(compression::raw::RawCompression.into(), b);
}
