use std::io::{
    Read,
    Write,
};

use serde::{
    Deserialize,
    Serialize,
};
use xz2::read::XzDecoder;
use xz2::write::XzEncoder;

use super::Compression;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct XzCompression {
    #[serde(default = "default_xz_preset")]
    preset: i32,
}

fn default_xz_preset() -> i32 {
    6
}

impl Default for XzCompression {
    fn default() -> XzCompression {
        XzCompression {
            preset: default_xz_preset(),
        }
    }
}

impl Compression for XzCompression {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(XzDecoder::new(r))
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        // TODO: check that preset is non-negative.s
        Box::new(XzEncoder::new(w, self.preset as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionType;

    // Example from the zarr documentation spec.
    #[rustfmt::skip]
    const TEST_CHUNK_I16_XZ: [u8; 84] = [
        0x00, 0x00,
        0x00, 0x03,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x03,
        0xfd, 0x37, 0x7a, 0x58,
        0x5a, 0x00, 0x00, 0x04,
        0xe6, 0xd6, 0xb4, 0x46,
        0x02, 0x00, 0x21, 0x01,
        0x16, 0x00, 0x00, 0x00,
        0x74, 0x2f, 0xe5, 0xa3,
        0x01, 0x00, 0x0b, 0x00,
        0x01, 0x00, 0x02, 0x00,
        0x03, 0x00, 0x04, 0x00,
        0x05, 0x00, 0x06, 0x00,
        0x0d, 0x03, 0x09, 0xca,
        0x34, 0xec, 0x15, 0xa7,
        0x00, 0x01, 0x24, 0x0c,
        0xa6, 0x18, 0xd8, 0xd8,
        0x1f, 0xb6, 0xf3, 0x7d,
        0x01, 0x00, 0x00, 0x00,
        0x00, 0x04, 0x59, 0x5a,
    ];

    #[test]
    fn test_read_doc_spec_chunk() {
        crate::tests::test_read_doc_spec_chunk(
            TEST_CHUNK_I16_XZ.as_ref(),
            CompressionType::Xz(XzCompression::default()),
        );
    }

    #[test]
    fn test_write_doc_spec_chunk() {
        crate::tests::test_write_doc_spec_chunk(
            TEST_CHUNK_I16_XZ.as_ref(),
            CompressionType::Xz(XzCompression::default()),
        );
    }

    #[test]
    fn test_rw() {
        crate::tests::test_chunk_compression_rw(CompressionType::Xz(XzCompression::default()));
    }
}
