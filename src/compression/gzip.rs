use std::io::{
    Read,
    Write,
};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression as GzCompression;
use serde::{
    Deserialize,
    Serialize,
};

use super::Compression;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GzipCompression {
    #[serde(default = "default_gzip_level")]
    level: i32,
}

impl GzipCompression {
    /// Java has -1 as the default compression level for Gzip
    /// despite this not being a valid compression level.
    ///
    /// Use `flate2`'s default level if the configured level is not in [0, 9].
    /// (At the time of writing this is 6.)
    fn get_effective_level(&self) -> GzCompression {
        if self.level < 0 || self.level > 9 {
            GzCompression::default()
        } else {
            GzCompression::new(self.level as u32)
        }
    }
}

fn default_gzip_level() -> i32 {
    -1
}

impl Default for GzipCompression {
    fn default() -> GzipCompression {
        GzipCompression {
            level: default_gzip_level(),
        }
    }
}

impl Compression for GzipCompression {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(GzDecoder::new(r))
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        Box::new(GzEncoder::new(w, self.get_effective_level()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionType;

    // Example from the n5 documentation spec.
    #[rustfmt::skip]
    const TEST_CHUNK_I16_GZIP: [u8; 48] = [
        0x00, 0x00,
        0x00, 0x03,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x03,
        0x1f, 0x8b, 0x08, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x63, 0x60,
        0x64, 0x60, 0x62, 0x60,
        0x66, 0x60, 0x61, 0x60,
        0x65, 0x60, 0x03, 0x00,
        0xaa, 0xea, 0x6d, 0xbf,
        0x0c, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn test_read_doc_spec_chunk() {
        crate::tests::test_read_doc_spec_chunk(
            TEST_CHUNK_I16_GZIP.as_ref(),
            CompressionType::Gzip(GzipCompression::default()),
        );
    }

    #[test]
    fn test_write_doc_spec_chunk() {
        // The compressed stream differs from Java.
        // The difference is one byte: the operating system ID.
        // Java uses 0 (FAT) while flate2 usese 255 (unknown).
        let mut fudge_test_chunk = TEST_CHUNK_I16_GZIP.clone();
        fudge_test_chunk[25] = 255;
        crate::tests::test_write_doc_spec_chunk(
            &fudge_test_chunk,
            CompressionType::Gzip(GzipCompression::default()),
        );
    }

    #[test]
    fn test_rw() {
        crate::tests::test_chunk_compression_rw(CompressionType::Gzip(GzipCompression::default()));
    }
}
