use rand::{
    distributions::Standard,
    Rng,
};
use smallvec::smallvec;

use zarr::prelude::*;

fn test_read_write<T, Zarr: HierarchyReader + HierarchyWriter>(n: &Zarr, compression: &CompressionType, dim: usize)
where
    T: 'static + std::fmt::Debug + ReflectedType + PartialEq + Default,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
    VecDataChunk<T>: zarr::ReadableDataChunk + zarr::WriteableDataChunk,
{
    let chunk_size: ChunkCoord = (1..=dim as u32).rev().map(|d| d * 5).collect();
    let array_meta = ArrayMetadata::new(
        (1..=dim as u64).map(|d| d * 100).collect(),
        chunk_size.clone(),
        T::VARIANT,
        compression.clone(),
    );
    let numel = array_meta.get_chunk_num_elements();
    let rng = rand::thread_rng();
    let chunk_data: Vec<T> = rng.sample_iter(&Standard).take(numel).collect();

    let chunk_in = SliceDataChunk::new(chunk_size, smallvec![0; dim], chunk_data);

    let path_name = "test/array/group";

    n.create_array(path_name, &array_meta)
        .expect("Failed to create array");
    n.write_chunk(path_name, &array_meta, &chunk_in)
        .expect("Failed to write chunk");

    let chunk_data = chunk_in.into_data();

    let chunk_out = n
        .read_chunk::<T>(path_name, &array_meta, smallvec![0; dim])
        .expect("Failed to read chunk")
        .expect("Chunk is empty");
    assert_eq!(chunk_out.get_data(), &chunk_data[..]);

    let mut into_chunk = VecDataChunk::new(smallvec![0; dim], smallvec![0; dim], vec![]);
    n.read_chunk_into(path_name, &array_meta, smallvec![0; dim], &mut into_chunk)
        .expect("Failed to read chunk")
        .expect("Chunk is empty");
    assert_eq!(into_chunk.get_data(), &chunk_data[..]);

    n.remove(path_name).unwrap();
}

fn test_all_types<Zarr: HierarchyReader + HierarchyWriter>(n: &Zarr, compression: &CompressionType, dim: usize) {
    test_read_write::<u8, _>(n, compression, dim);
    test_read_write::<u16, _>(n, compression, dim);
    test_read_write::<u32, _>(n, compression, dim);
    test_read_write::<u64, _>(n, compression, dim);
    test_read_write::<i8, _>(n, compression, dim);
    test_read_write::<i16, _>(n, compression, dim);
    test_read_write::<i32, _>(n, compression, dim);
    test_read_write::<i64, _>(n, compression, dim);
    test_read_write::<f32, _>(n, compression, dim);
    test_read_write::<f64, _>(n, compression, dim);
}

fn test_zarr_filesystem_dim(dim: usize) {
    let dir = tempdir::TempDir::new("rust_zarr_integration_tests").unwrap();

    let n = FilesystemHierarchy::open_or_create(dir.path()).expect("Failed to create Zarr filesystem");
    test_all_types(
        &n,
        &CompressionType::Raw(compression::raw::RawCompression::default()),
        dim,
    );
}

#[test]
fn test_zarr_filesystem_dims() {
    for dim in 1..=5 {
        test_zarr_filesystem_dim(dim);
    }
}

fn test_all_compressions<Zarr: HierarchyReader + HierarchyWriter>(n: &Zarr) {
    test_all_types(
        n,
        &CompressionType::Raw(compression::raw::RawCompression::default()),
        3,
    );
    #[cfg(feature = "bzip")]
    test_all_types(
        n,
        &CompressionType::Bzip2(compression::bzip::Bzip2Compression::default()),
        3,
    );
    #[cfg(feature = "gzip")]
    test_all_types(
        n,
        &CompressionType::Gzip(compression::gzip::GzipCompression::default()),
        3,
    );
    #[cfg(feature = "lz")]
    test_all_types(
        n,
        &CompressionType::Lz4(compression::lz::Lz4Compression::default()),
        3,
    );
    #[cfg(feature = "xz")]
    test_all_types(
        n,
        &CompressionType::Xz(compression::xz::XzCompression::default()),
        3,
    );
}

#[test]
fn test_zarr_filesystem_compressions() {
    let dir = tempdir::TempDir::new("rust_zarr_integration_tests").unwrap();

    let n = FilesystemHierarchy::open_or_create(dir.path()).expect("Failed to create Zarr filesystem");
    test_all_compressions(&n)
}
