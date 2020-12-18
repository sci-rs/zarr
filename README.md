# Zarr [![Build Status](https://travis-ci.org/sci-rs/zarr.svg?branch=main)](https://travis-ci.org/sci-rs/zarr) [![Coverage](https://codecov.io/gh/sci-rs/zarr/branch/main/graph/badge.svg)](https://codecov.io/gh/sci-rs/zarr)

A pure rust implementation of [version 3.0-dev of the Zarr core protocol for storage and retrieval of N-dimensional typed arrays](https://zarr-specs.readthedocs.io/en/core-protocol-v3.0-dev/protocol/core/v3.0.html).

## Minimum supported Rust version (MSRV)

Stable 1.41

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

    let chunk_shape = smallvec![44, 33, 22];
    let array_meta = ArrayMetadata::new(
        smallvec![100, 200, 300],
        chunk_shape,
        i16::ZARR_TYPE,
        CompressionType::default(),
    );
    let chunk_data = vec![0i16; array_meta.get_chunk_num_elements()];

    let chunk_in = SliceDataChunk::new(
        smallvec![0, 0, 0],
        &chunk_data);

    let path_name = "/test/array/group";

    n.create_array(path_name, &array_meta)?;
    n.write_chunk(path_name, &array_meta, &chunk_in)?;

    let chunk_out = n.read_chunk::<i16>(path_name, &array_meta, smallvec![0, 0, 0])?
        .expect("Chunk is empty");
    assert_eq!(chunk_out.get_data(), &chunk_data[..]);

    Ok(())
}

fn main() {
    zarr_roundtrip("tmp.zarr").expect("Zarr roundtrip failed!");
    std::fs::remove_dir_all("tmp.zarr").expect("Failed to delete temporary zarr hierarchy");
}
```

## Status

TODO

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
