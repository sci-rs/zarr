//! # Parallel Writing Benchmarks
//!
//! Provides parallel writing benchmarks. For parity with Java N5's
//! `N5Benchmark`, the benchmark dataset is `i16`, with a 5x5x5 grid of blocks
//! of 64x64x64. The blocks are loaded with data from this file, which
//! must be manually downloaded and extracted to the `benches` directory:
//!
//! [Test data](https://imagej.nih.gov/ij/images/t1-head-raw.zip)
//!
//! Note that since this uses the default cargo bencher, even though each
//! iteration only takes seconds they will run hundreds of times. Hence this
//! will take several hours to run.
#![feature(test)]

extern crate test;

use std::fs::File;
use std::io::BufReader;

use futures::{
    self,
    Future,
};
use futures_cpupool::{
    CpuFuture,
    CpuPool,
};
use lazy_static::lazy_static;
use tempdir;
use test::Bencher;
use tiff::decoder::{
    Decoder,
    DecodingResult,
};

use n5::prelude::*;
use n5::smallvec::smallvec;

lazy_static! {
    static ref TEST_IMAGE: Vec<i8> = {
        let mut pixels = Vec::with_capacity(163 * 163 * 93);
        let fin = File::open("benches/JeffT1_le.tif").unwrap();
        let fin = BufReader::new(fin);

        let mut decoder = Decoder::new(fin).unwrap();

        while decoder.more_images() {
            match decoder.read_image().unwrap() {
                DecodingResult::U8(_) => panic!("Expect u16 image!"),
                DecodingResult::U16(img) => {
                    for p in img {
                        pixels.push(p as i8);
                    }
                }
            }

            decoder.next_image().unwrap();
        }

        pixels
    };
}
const BLOCK_DIM: u32 = 64;
const N_BLOCKS: u64 = 5;

fn write<T, N5>(n: &N5, compression: &CompressionType, block_data: &[T], pool_size: usize)
where
    T: 'static + std::fmt::Debug + ReflectedType + PartialEq + Default + Sync + Send,
    N5: N5Writer + Sync + Send + Clone + 'static,
    SliceDataBlock<T, std::sync::Arc<[T]>>: n5::WriteableDataBlock,
{
    let block_size = smallvec![BLOCK_DIM; 3];
    let data_attrs = DatasetAttributes::new(
        smallvec![u64::from(BLOCK_DIM) * N_BLOCKS; 3],
        block_size.clone(),
        T::VARIANT,
        compression.clone(),
    );

    let path_name = format!(
        "dataset.{:?}.{}",
        data_attrs.get_data_type(),
        data_attrs.get_compression()
    );

    n.create_dataset(&path_name, &data_attrs)
        .expect("Failed to create dataset");

    let mut all_jobs: Vec<CpuFuture<usize, std::io::Error>> =
        Vec::with_capacity((N_BLOCKS * N_BLOCKS * N_BLOCKS) as usize);
    let pool = CpuPool::new(pool_size);
    let bd: std::sync::Arc<[T]> = block_data.to_owned().into();

    for x in 0..N_BLOCKS {
        for y in 0..N_BLOCKS {
            for z in 0..N_BLOCKS {
                let bs = block_size.clone();
                let bd = bd.clone();
                let ni = n.clone();
                let pn = path_name.clone();
                let da = data_attrs.clone();
                all_jobs.push(pool.spawn_fn(move || {
                    let block_in = SliceDataBlock::new(bs, smallvec![x, y, z], bd);
                    ni.write_block(&pn, &da, &block_in)
                        .expect("Failed to write block");
                    Ok(0)
                }));
            }
        }
    }

    futures::future::join_all(all_jobs).wait().unwrap();
}

fn bench_write_dtype_compression<T, C>(b: &mut Bencher, pool_size: usize)
where
    T: 'static
        + ReflectedType
        + Default
        + PartialEq
        + std::fmt::Debug
        + std::convert::From<i8>
        + Sync
        + Send,
    C: compression::Compression,
    CompressionType: std::convert::From<C>,
    SliceDataBlock<T, std::sync::Arc<[T]>>: n5::WriteableDataBlock,
{
    let dir = tempdir::TempDir::new("rust_n5_integration_tests").unwrap();

    let n = N5Filesystem::open_or_create(dir.path()).expect("Failed to create N5 filesystem");
    let compression = CompressionType::new::<C>();
    // TODO: load the test image data.
    // let block_data: Vec<T> = vec![T::default(); (BLOCK_DIM * BLOCK_DIM * BLOCK_DIM) as usize];
    let block_data = TEST_IMAGE
        .iter()
        .take((BLOCK_DIM * BLOCK_DIM * BLOCK_DIM) as usize)
        .map(|&v| T::from(v))
        .collect::<Vec<T>>();

    b.iter(|| write(&n, &compression, &block_data, pool_size));

    b.bytes = (BLOCK_DIM * BLOCK_DIM * BLOCK_DIM) as u64
        * (N_BLOCKS * N_BLOCKS * N_BLOCKS) as u64
        * std::mem::size_of::<T>() as u64;
}

// 1 Thread. Can't macro this because of the concat_idents! limitation.
#[bench]
fn bench_write_i16_raw_1(b: &mut Bencher) {
    bench_write_dtype_compression::<i16, compression::raw::RawCompression>(b, 1);
}

#[bench]
fn bench_write_i16_bzip2_1(b: &mut Bencher) {
    bench_write_dtype_compression::<i16, compression::bzip::Bzip2Compression>(b, 1);
}

#[bench]
fn bench_write_i16_gzip_1(b: &mut Bencher) {
    bench_write_dtype_compression::<i16, compression::gzip::GzipCompression>(b, 1);
}

#[bench]
fn bench_write_i16_xz_1(b: &mut Bencher) {
    bench_write_dtype_compression::<i16, compression::xz::XzCompression>(b, 1);
}

#[bench]
fn bench_write_i16_lz4_1(b: &mut Bencher) {
    bench_write_dtype_compression::<i16, compression::lz::Lz4Compression>(b, 1);
}

// 2 Threads.
#[bench]
fn bench_write_i16_raw_2(b: &mut Bencher) {
    bench_write_dtype_compression::<i16, compression::raw::RawCompression>(b, 2);
}

#[bench]
fn bench_write_i16_bzip2_2(b: &mut Bencher) {
    bench_write_dtype_compression::<i16, compression::bzip::Bzip2Compression>(b, 2);
}

#[bench]
fn bench_write_i16_gzip_2(b: &mut Bencher) {
    bench_write_dtype_compression::<i16, compression::gzip::GzipCompression>(b, 2);
}

#[bench]
fn bench_write_i16_xz_2(b: &mut Bencher) {
    bench_write_dtype_compression::<i16, compression::xz::XzCompression>(b, 2);
}

#[bench]
fn bench_write_i16_lz4_2(b: &mut Bencher) {
    bench_write_dtype_compression::<i16, compression::lz::Lz4Compression>(b, 2);
}
