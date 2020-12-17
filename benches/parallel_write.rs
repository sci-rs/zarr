//! # Parallel Writing Benchmarks
//!
//! Provides parallel writing benchmarks. For parity with Java Zarr's
//! `ZarrBenchmark`, the benchmark array is `i16`, with a 5x5x5 grid of chunks
//! of 64x64x64. The chunks are loaded with data from this file, which
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

use zarr::chunk::WriteableDataChunk;
use zarr::prelude::*;
use zarr::smallvec::smallvec;

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
const CHUNK_DIM: u32 = 64;
const N_CHUNKS: u64 = 5;

fn write<T, Zarr>(n: &Zarr, compression: &CompressionType, chunk_data: &[T], pool_size: usize)
where
    T: 'static + std::fmt::Debug + ReflectedType + PartialEq + Default + Sync + Send,
    Zarr: HierarchyWriter + Sync + Send + Clone + 'static,
    SliceDataChunk<T, std::sync::Arc<[T]>>: WriteableDataChunk,
{
    let chunk_shape = smallvec![CHUNK_DIM; 3];
    let array_meta = ArrayMetadata::new(
        smallvec![u64::from(CHUNK_DIM) * N_CHUNKS; 3],
        chunk_shape.clone(),
        T::ZARR_TYPE,
        compression.clone(),
    );

    let path_name = format!(
        "array.{:?}.{}",
        array_meta.get_data_type(),
        array_meta.get_compressor()
    );

    n.create_array(&path_name, &array_meta)
        .expect("Failed to create array");

    let mut all_jobs: Vec<CpuFuture<usize, std::io::Error>> =
        Vec::with_capacity((N_CHUNKS * N_CHUNKS * N_CHUNKS) as usize);
    let pool = CpuPool::new(pool_size);
    let bd: std::sync::Arc<[T]> = chunk_data.to_owned().into();

    for x in 0..N_CHUNKS {
        for y in 0..N_CHUNKS {
            for z in 0..N_CHUNKS {
                let bd = bd.clone();
                let ni = n.clone();
                let pn = path_name.clone();
                let da = array_meta.clone();
                all_jobs.push(pool.spawn_fn(move || {
                    let chunk_in = SliceDataChunk::new(smallvec![x, y, z], bd);
                    ni.write_chunk(&pn, &da, &chunk_in)
                        .expect("Failed to write chunk");
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
    SliceDataChunk<T, std::sync::Arc<[T]>>: WriteableDataChunk,
{
    let dir = tempdir::TempDir::new("rust_zarr_integration_tests").unwrap();

    let n =
        FilesystemHierarchy::open_or_create(dir.path()).expect("Failed to create Zarr filesystem");
    let compression = CompressionType::new::<C>();
    // TODO: load the test image data.
    // let chunk_data: Vec<T> = vec![T::default(); (CHUNK_DIM * CHUNK_DIM * CHUNK_DIM) as usize];
    let chunk_data = TEST_IMAGE
        .iter()
        .take((CHUNK_DIM * CHUNK_DIM * CHUNK_DIM) as usize)
        .map(|&v| T::from(v))
        .collect::<Vec<T>>();

    b.iter(|| write(&n, &compression, &chunk_data, pool_size));

    b.bytes = (CHUNK_DIM * CHUNK_DIM * CHUNK_DIM) as u64
        * (N_CHUNKS * N_CHUNKS * N_CHUNKS) as u64
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
