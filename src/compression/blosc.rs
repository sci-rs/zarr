extern crate blosc;

use std::io::{
  Read,
  Write,
  Cursor,
};

use serde::{
  Deserialize,
  Serialize,
};

use blosc::{
  Clevel,
  Compressor,
  Context,
  ShuffleMode,
  decompress_bytes,
};

use super::Compression;

const COMPRESSOR_BLOSCLZ: &str = "blosclz";
const COMPRESSOR_LZ4: &str = "lz4";
const COMPRESSOR_ZLIB: &str = "zlib";
const COMPRESSOR_ZSTD: &str = "zstd";

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub struct BloscCompression {
  #[serde(default = "default_blosc_blocksize")]
  blocksize: usize,
  clevel: u8,  // serialize index into enum by index
  cname: String,
  #[serde(default = "default_blosc_shufflemode")]
  shuffle: u8, // serialize shuffle mode into enum by index
}

fn default_blosc_blocksize() -> usize {
  0
}

fn default_blosc_shufflemode() -> u8 {
  1
}

impl Default for BloscCompression {
  fn default() -> BloscCompression {
      BloscCompression {
          blocksize: default_blosc_blocksize(),
          clevel: 5,
          cname: String::from(COMPRESSOR_BLOSCLZ),
          shuffle: default_blosc_shufflemode(),
      }
  }
}

impl BloscCompression {
  // convert cname to compressor (todo: possible to do this in serde?)
  fn compressor(&self) -> Compressor {
    match self.cname.as_str() {
      COMPRESSOR_BLOSCLZ => Compressor::BloscLZ,
      COMPRESSOR_LZ4 => Compressor::LZ4,
      COMPRESSOR_ZLIB => Compressor::Zlib,
      COMPRESSOR_ZSTD => Compressor::Zstd,
      _ => Compressor::Invalid,
    }
  }

  fn clevel_enum(&self) -> Clevel {
    // there has to be a better way to do this but the enum
    // was not marked FromPrimitive so *shrug*
    match self.clevel {
        0 => Clevel::None,
        1 => Clevel::L1,
        2 => Clevel::L2,
        3 => Clevel::L3,
        4 => Clevel::L4,
        5 => Clevel::L5,
        6 => Clevel::L6,
        7 => Clevel::L7,
        8 => Clevel::L8,
        9 => Clevel::L9,
        _ => Clevel::None,
    }
  }

  // convert serialized shuffle val to mode
  fn shuffle_enum(&self) -> ShuffleMode {
    match self.shuffle {
      0 => ShuffleMode::None,
      1 => ShuffleMode::Byte,
      2 => ShuffleMode::Bit,
      _ => ShuffleMode::None, // defensive here
    }
  }
}

impl Compression for BloscCompression {
  fn decoder<'a, R: Read + 'a>(&self, mut r: R) -> Box<dyn Read + 'a> {
      // blosc is all at the same time...
      let mut bytes: Vec<u8> = Vec::new();
      r.read_to_end(&mut bytes);
      println!("{:?}", bytes);
      let decompressed = unsafe { decompress_bytes(&bytes) }.unwrap();
      println!("{:?}", decompressed);
      Box::new(Cursor::new(decompressed))
  }

  // TODO not currently supported
  fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
      // TODO: need wrapper that only does the compression when
      // the end of the data/EOF is reached.
      Box::new(w)

      /*
      let ctx = Context::new()
          .blocksize(Some(self.blocksize))
          .clevel(self.clevel_enum())
          .compressor(self.compressor()).unwrap()
          .shuffle(self.shuffle_enum());

      // TODO write wrapper that calls ctx.compress(w) when buffer
      // is complete
      */
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::compression::CompressionType;

  #[rustfmt::skip]
  const TEST_CHUNK_I16_BLOSC: [u8; 28] = [
      // 0x00, 0x00,
      // 0x00, 0x03,
      // 0x00, 0x00, 0x00, 0x01,
      // 0x00, 0x00, 0x00, 0x02,
      // 0x00, 0x00, 0x00, 0x03,
      0x02, 0x01, 0x33, 0x02,
      0x0c, 0x00, 0x00, 0x00,
      0x0c, 0x00, 0x00, 0x00,
      0x1c, 0x00, 0x00, 0x00,
      0x00, 0x01, 0x00, 0x02,  // big endian
      0x00, 0x03, 0x00, 0x04,
      0x00, 0x05, 0x00, 0x06, // not very compressed now is it
  ];

  #[test]
  fn test_read_doc_spec_chunk() {
      let blosc_lz4: BloscCompression = BloscCompression {
        blocksize: 0,
        clevel: 5,
        cname: COMPRESSOR_LZ4.to_string(),
        shuffle: 1,
      };
      crate::tests::test_read_doc_spec_chunk(
          TEST_CHUNK_I16_BLOSC.as_ref(),
          CompressionType::Blosc(blosc_lz4),
      );
  }

  #[test]
  // This test is ignored since the compressed stream differs from Java.
  fn test_write_doc_spec() {
      let data: [i16; 6] = [1, 2, 3, 4, 5, 6];
      let ctx = Context::new()
        .blocksize(Some(0))
        .clevel(Clevel::L5)
        .compressor(Compressor::LZ4).unwrap()
        .shuffle(ShuffleMode::Byte);
      let buffer = ctx.compress(&data);
      let bytes: Vec<u8> = buffer.into();

      println!("{:?}", bytes);
  }

  #[test]
  #[ignore]
  fn test_rw() {
      let blosc_lz4: BloscCompression = BloscCompression {
        blocksize: 0,
        clevel: 5,
        cname: COMPRESSOR_LZ4.to_string(),
        shuffle: 1,
      };
      crate::tests::test_chunk_compression_rw(
          CompressionType::Blosc(blosc_lz4),
      );
  }
}
