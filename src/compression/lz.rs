use std::io::{
    Read,
    Result,
    Write,
};

use lz4::{
    BlockMode,
    BlockSize,
    Decoder,
    Encoder,
    EncoderBuilder,
};
use serde::{
    Deserialize,
    Serialize,
};

use super::Compression;

// From: https://github.com/bozaro/lz4-rs/issues/9
// Kludge to finish Lz4 encoder on Drop.
struct Wrapper<W: Write> {
    s: Option<Encoder<W>>,
}

impl<W: Write> Write for Wrapper<W> {
    fn write(&mut self, buffer: &[u8]) -> Result<usize> {
        self.s.as_mut().unwrap().write(buffer)
    }

    fn flush(&mut self) -> Result<()> {
        self.s.as_mut().unwrap().flush()
    }
}

impl<W: Write> Drop for Wrapper<W> {
    fn drop(&mut self) {
        if let Some(s) = self.s.take() {
            s.finish().1.unwrap();
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Lz4Compression {
    #[serde(default = "default_lz4_block_size")]
    block_size: i32,
}

impl Lz4Compression {
    /// `lz4` uses an enum for specifying block size, so choose the smallest
    /// larger size from that enum.
    fn get_effective_block_size(&self) -> BlockSize {
        if self.block_size <= BlockSize::Max64KB.get_size() as i32 {
            BlockSize::Max64KB
        } else if self.block_size as usize <= BlockSize::Max256KB.get_size() {
            BlockSize::Max256KB
        } else if self.block_size as usize <= BlockSize::Max1MB.get_size() {
            BlockSize::Max1MB
        } else {
            BlockSize::Max4MB
        }
    }
}

fn default_lz4_block_size() -> i32 {
    65_536
}

impl Default for Lz4Compression {
    fn default() -> Lz4Compression {
        Lz4Compression {
            block_size: default_lz4_block_size(),
        }
    }
}

impl Compression for Lz4Compression {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(Decoder::new(r).expect("TODO: LZ4 returns a result here"))
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        let encoder = EncoderBuilder::new()
            .block_size(self.get_effective_block_size())
            .block_mode(BlockMode::Independent)
            .build(w)
            .expect("TODO");
        Box::new(Wrapper { s: Some(encoder) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionType;

    #[rustfmt::skip]
    const TEST_BLOCK_I16_LZ4: [u8; 47] = [
        0x00, 0x00,
        0x00, 0x03,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x03,
        0x04, 0x22, 0x4d, 0x18,
        0x64, 0x40, 0xa7, 0x0c,
        0x00, 0x00, 0x80, 0x00,
        0x01, 0x00, 0x02, 0x00,
        0x03, 0x00, 0x04, 0x00,
        0x05, 0x00, 0x06, 0x00,
        0x00, 0x00, 0x00, 0x41,
        0x37, 0x33, 0x08,
    ];

    #[test]
    fn test_read_doc_spec_block() {
        crate::tests::test_read_doc_spec_block(
            TEST_BLOCK_I16_LZ4.as_ref(),
            CompressionType::Lz4(Lz4Compression::default()),
        );
    }

    #[test]
    fn test_write_doc_spec_block() {
        crate::tests::test_write_doc_spec_block(
            TEST_BLOCK_I16_LZ4.as_ref(),
            CompressionType::Lz4(Lz4Compression::default()),
        );
    }

    #[test]
    fn test_rw() {
        crate::tests::test_block_compression_rw(CompressionType::Lz4(Lz4Compression::default()));
    }
}
