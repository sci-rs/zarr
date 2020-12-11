use std::io::{
    Read,
    Result,
    Write,
};

use lz_fear::framed::{
    CompressionSettings,
    LZ4FrameReader,
};
use serde::{
    Deserialize,
    Serialize,
};

use super::Compression;

struct Wrapper<W: Write> {
    writer: W,
    settings: CompressionSettings<'static>,
}

impl<W: Write> Write for Wrapper<W> {
    fn write(&mut self, buffer: &[u8]) -> Result<usize> {
        let len = buffer.len();
        self.settings.compress(buffer, &mut self.writer)?;
        Ok(len)
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Lz4Compression {
    #[serde(default = "default_lz4_block_size")]
    block_size: i32,
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
        Box::new(
            LZ4FrameReader::new(r)
                .expect("TODO: LZ4 returns a result here")
                .into_read(),
        )
    }

    fn encoder<'a, W: Write + 'a>(&self, writer: W) -> Box<dyn Write + 'a> {
        let mut settings = CompressionSettings::default();
        settings
            .block_size(self.block_size as usize)
            .independent_blocks(true);
        Box::new(Wrapper { writer, settings })
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
