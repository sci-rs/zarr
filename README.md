# N5 [![Build Status](https://travis-ci.org/aschampion/rust-n5.svg?branch=master)](https://travis-ci.org/aschampion/rust-n5) [![Coverage](https://codecov.io/gh/aschampion/rust-n5/branch/master/graph/badge.svg)](https://codecov.io/gh/aschampion/rust-n5)

A (mostly pure) rust implementation of the [N5 "Not HDF5" n-dimensional tensor file system storage format](https://github.com/saalfeldlab/n5) created by the Saalfeld lab at Janelia Research Campus.

Compatible with Java N5 Version 2.1.3.

## Differences from Java N5
- Dataset paths may be relative. The root path in a dataset is addressable both by `"/"` and `""`.
- Dataset paths are more strict. Calling methods with paths outside the dataset, e.g., `".."`, will return a `Result::Err`.

## Minimum supported Rust version (MSRV)

Stable 1.39

## Quick start

```toml
[dependencies]
n5 = "0.7"
```

```rust
use n5::prelude::*;
use n5::smallvec::smallvec;

fn n5_roundtrip(root_path: &str) -> std::io::Result<()> {
    let n = N5Filesystem::open_or_create(root_path)?;

    let block_size = smallvec![44, 33, 22];
    let data_attrs = DatasetAttributes::new(
        smallvec![100, 200, 300],
        block_size.clone(),
        DataType::INT16,
        CompressionType::default(),
    );
    let block_data = vec![0i16; data_attrs.get_block_num_elements()];

    let block_in = SliceDataBlock::new(
        block_size,
        smallvec![0, 0, 0],
        &block_data);

    let path_name = "/test/dataset/group";

    n.create_dataset(path_name, &data_attrs)?;
    n.write_block(path_name, &data_attrs, &block_in)?;

    let block_out = n.read_block::<i16>(path_name, &data_attrs, smallvec![0, 0, 0])?
        .expect("Block is empty");
    assert_eq!(block_out.get_data(), &block_data[..]);

    Ok(())
}

fn main() {
    n5_roundtrip("tmp.n5").expect("N5 roundtrip failed!");
}
```

## Status

This library is compatible with all N5 datasets the authors have encountered and is used in production services. However, some aspects of the library are still unergonomic and interfaces may still undergo rapid breaking changes.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
