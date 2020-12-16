use std::io::{
    Read,
    Write,
};

use serde::{
    Deserialize,
    Serialize,
};

use super::Compression;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
pub struct RawCompression;

impl Compression for RawCompression {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(r)
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        Box::new(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionType;

    // Example from the zarr documentation spec.
    #[rustfmt::skip]
    const TEST_CHUNK_I16_RAW: [u8; 28] = [
        0x00, 0x00,
        0x00, 0x03,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x03,
        0x00, 0x01,
        0x00, 0x02,
        0x00, 0x03,
        0x00, 0x04,
        0x00, 0x05,
        0x00, 0x06,
    ];

    #[test]
    fn test_read_doc_spec_chunk() {
        crate::tests::test_read_doc_spec_chunk(
            TEST_CHUNK_I16_RAW.as_ref(),
            CompressionType::Raw(RawCompression),
        );
    }

    #[test]
    fn test_write_doc_spec_chunk() {
        crate::tests::test_write_doc_spec_chunk(
            TEST_CHUNK_I16_RAW.as_ref(),
            CompressionType::Raw(RawCompression),
        );
    }

    #[test]
    fn test_rw() {
        crate::tests::test_chunk_compression_rw(CompressionType::Raw(RawCompression));
    }

    #[test]
    fn test_varlength_rw() {
        crate::tests::test_varlength_chunk_rw(CompressionType::Raw(RawCompression));
    }
}
