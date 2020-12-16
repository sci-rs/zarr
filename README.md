# Zarr [![Build Status](https://travis-ci.org/sci-rs/zarr.svg?branch=main)](https://travis-ci.org/sci-rs/zarr) [![Coverage](https://codecov.io/gh/sci-rs/zarr/branch/main/graph/badge.svg)](https://codecov.io/gh/sci-rs/zarr)

TODO

## Minimum supported Rust version (MSRV)

Stable 1.39

## Quick start

```toml
[dependencies]
zarr = "0.0.1"
```

```rust
use zarr::prelude::*;
use zarr::smallvec::smallvec;

fn zarr_roundtrip(root_path: &str) -> std::io::Result<()> {
    let n = FilesystemHierarchy::open_or_create(root_path)?;

    let chunk_size = smallvec![44, 33, 22];
    let data_attrs = DatasetAttributes::new(
        smallvec![100, 200, 300],
        chunk_size.clone(),
        DataType::INT16,
        CompressionType::default(),
    );
    let chunk_data = vec![0i16; data_attrs.get_chunk_num_elements()];

    let chunk_in = SliceDataChunk::new(
        chunk_size,
        smallvec![0, 0, 0],
        &chunk_data);

    let path_name = "/test/dataset/group";

    n.create_dataset(path_name, &data_attrs)?;
    n.write_chunk(path_name, &data_attrs, &chunk_in)?;

    let chunk_out = n.read_chunk::<i16>(path_name, &data_attrs, smallvec![0, 0, 0])?
        .expect("Chunk is empty");
    assert_eq!(chunk_out.get_data(), &chunk_data[..]);

    Ok(())
}

fn main() {
    zarr_roundtrip("tmp.zarr").expect("Zarr roundtrip failed!");
}
```

## Status

This library is compatible with all Zarr datasets the authors have encountered and is used in production services. However, some aspects of the library are still unergonomic and interfaces may still undergo rapid breaking changes.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
