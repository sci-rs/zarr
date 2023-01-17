//! Compression for chunk voxel data.

use std::io::{
    Read,
    Write,
};

use serde::{
    Deserialize,
    Serialize,
};

#[cfg(feature = "blosc")]
pub mod blosc;
#[cfg(feature = "bzip")]
pub mod bzip;
#[cfg(any(feature = "gzip", feature = "gzip_pure"))]
pub mod gzip;
#[cfg(all(feature = "lz", not(feature = "lz_pure")))]
pub mod lz;
#[cfg(feature = "lz_pure")]
pub(self) mod lz_pure;
pub mod raw;
#[cfg(feature = "lz_pure")]
pub mod lz {
    pub use super::lz_pure::*;
}
#[cfg(feature = "xz")]
pub mod xz;

/// Common interface for compressing writers and decompressing readers.
pub trait Compression: Default {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a>;

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a>;
}

/// Enumeration of known compression schemes.
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
#[serde(tag = "codec", content = "configuration")]
pub enum CompressionType {
    Raw(raw::RawCompression),
    #[cfg(feature = "blosc")]
    Blosc(blosc::BloscCompression),
    #[cfg(feature = "bzip")]
    Bzip2(bzip::Bzip2Compression),
    #[cfg(any(feature = "gzip", feature = "gzip_pure"))]
    #[serde(rename = "https://purl.org/zarr/spec/codec/gzip/1.0")]
    Gzip(gzip::GzipCompression),
    #[cfg(any(feature = "lz", feature = "lz_pure"))]
    Lz4(lz::Lz4Compression),
    #[cfg(feature = "xz")]
    Xz(xz::XzCompression),
}

impl CompressionType {
    pub fn new<T: Compression>() -> CompressionType
    where
        CompressionType: std::convert::From<T>,
    {
        T::default().into()
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

impl Default for CompressionType {
    fn default() -> CompressionType {
        CompressionType::new::<raw::RawCompression>()
    }
}

impl Compression for CompressionType {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        match *self {
            CompressionType::Raw(ref c) => c.decoder(r),

            #[cfg(feature = "blosc")]
            CompressionType::Blosc(ref c) => c.decoder(r),

            #[cfg(feature = "bzip")]
            CompressionType::Bzip2(ref c) => c.decoder(r),

            #[cfg(any(feature = "gzip", feature = "gzip_pure"))]
            CompressionType::Gzip(ref c) => c.decoder(r),

            #[cfg(feature = "xz")]
            CompressionType::Xz(ref c) => c.decoder(r),

            #[cfg(any(feature = "lz", feature = "lz_pure"))]
            CompressionType::Lz4(ref c) => c.decoder(r),
        }
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        match *self {
            CompressionType::Raw(ref c) => c.encoder(w),

            #[cfg(feature = "blosc")]
            CompressionType::Blosc(ref c) => c.encoder(w),

            #[cfg(feature = "bzip")]
            CompressionType::Bzip2(ref c) => c.encoder(w),

            #[cfg(any(feature = "gzip", feature = "gzip_pure"))]
            CompressionType::Gzip(ref c) => c.encoder(w),

            #[cfg(feature = "xz")]
            CompressionType::Xz(ref c) => c.encoder(w),

            #[cfg(any(feature = "lz", feature = "lz_pure"))]
            CompressionType::Lz4(ref c) => c.encoder(w),
        }
    }
}

impl std::fmt::Display for CompressionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match *self {
                CompressionType::Raw(_) => "Raw",

                #[cfg(feature = "blosc")]
                CompressionType::Blosc(_) => "Blosc",

                #[cfg(feature = "bzip")]
                CompressionType::Bzip2(_) => "Bzip2",

                #[cfg(any(feature = "gzip", feature = "gzip_pure"))]
                CompressionType::Gzip(_) => "Gzip",

                #[cfg(feature = "xz")]
                CompressionType::Xz(_) => "Xz",

                #[cfg(any(feature = "lz", feature = "lz_pure"))]
                CompressionType::Lz4(_) => "Lz4",
            }
        )
    }
}

impl std::str::FromStr for CompressionType {
    type Err = std::io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "raw" => Ok(Self::new::<raw::RawCompression>()),

            #[cfg(feature = "bzip")]
            "bzip2" => Ok(Self::new::<bzip::Bzip2Compression>()),

            #[cfg(any(feature = "gzip", feature = "gzip_pure"))]
            "gzip" => Ok(Self::new::<gzip::GzipCompression>()),

            #[cfg(feature = "xz")]
            "xz" => Ok(Self::new::<xz::XzCompression>()),

            #[cfg(feature = "lz")]
            "lz4" => Ok(Self::new::<lz::Lz4Compression>()),

            _ => Err(std::io::ErrorKind::InvalidInput.into()),
        }
    }
}

macro_rules! compression_from_impl {
    ($variant:ident, $c_type:ty) => {
        impl std::convert::From<$c_type> for CompressionType {
            fn from(c: $c_type) -> Self {
                CompressionType::$variant(c)
            }
        }
    };
}

compression_from_impl!(Raw, raw::RawCompression);
#[cfg(feature = "bzip")]
compression_from_impl!(Bzip2, bzip::Bzip2Compression);
#[cfg(any(feature = "gzip", feature = "gzip_pure"))]
compression_from_impl!(Gzip, gzip::GzipCompression);
#[cfg(feature = "xz")]
compression_from_impl!(Xz, xz::XzCompression);
#[cfg(any(feature = "lz", feature = "lz_pure"))]
compression_from_impl!(Lz4, lz::Lz4Compression);
