use std::io::{
    Error,
    ErrorKind,
};
use std::marker::PhantomData;

use byteorder::{
    ByteOrder,
    ReadBytesExt,
    WriteBytesExt,
};

use crate::compression::Compression;
use crate::{
    ArrayMetadata,
    ChunkCoord,
    GridCoord,
    ReflectedType,
    ZarrEndian,
};

/// Unencoded, non-payload header of a data chunk.
#[derive(Debug)]
pub struct ChunkHeader {
    pub(crate) size: ChunkCoord,
    pub(crate) grid_position: GridCoord,
    pub(crate) num_el: usize,
}

/// Traits for data chunks that can be reused as a different chunks after
/// construction.
pub trait ReinitDataChunk<T> {
    /// Reinitialize this data chunk with a new header, reallocating as
    /// necessary.
    fn reinitialize(&mut self, header: ChunkHeader);

    /// Reinitialize this data chunk with the header and data of another chunk.
    fn reinitialize_with<B: DataChunk<T>>(&mut self, other: &B);
}

/// Traits for data chunks that can read in data.
pub trait ReadableDataChunk {
    /// Read data into this chunk from a source, overwriting any existing data.
    ///
    /// Unlike Java Zarr, read the stream directly into the chunk data instead
    /// of creating a copied byte buffer.
    fn read_data<R: std::io::Read>(&mut self, source: R) -> std::io::Result<()>;
}

/// Traits for data chunks that can write out data.
pub trait WriteableDataChunk {
    /// Write the data from this chunk into a target.
    fn write_data<W: std::io::Write>(&self, target: W) -> std::io::Result<()>;
}

/// Common interface for data chunks of element (rust) type `T`.
///
/// To enable custom types to be written to Zarr volumes, implement this trait.
pub trait DataChunk<T> {
    fn get_size(&self) -> &[u32];

    fn get_grid_position(&self) -> &[u64];

    fn get_data(&self) -> &[T];

    fn get_num_elements(&self) -> u32;

    fn get_header(&self) -> ChunkHeader {
        ChunkHeader {
            size: self.get_size().into(),
            grid_position: self.get_grid_position().into(),
            num_el: self.get_num_elements() as usize,
        }
    }
}

/// A generic data chunk container wrapping any type that can be taken as a
/// slice ref.
#[derive(Clone, Debug)]
pub struct SliceDataChunk<T: ReflectedType, C> {
    data_type: PhantomData<T>,
    size: ChunkCoord,
    grid_position: GridCoord,
    data: C,
}

impl<T: ReflectedType, C> SliceDataChunk<T, C> {
    pub fn new(size: ChunkCoord, grid_position: GridCoord, data: C) -> SliceDataChunk<T, C> {
        SliceDataChunk {
            data_type: PhantomData,
            size,
            grid_position,
            data,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }
}

/// A linear vector storing a data chunk. All read data chunks are returned as
/// this type.
pub type VecDataChunk<T> = SliceDataChunk<T, Vec<T>>;

impl<T: ReflectedType> ReinitDataChunk<T> for VecDataChunk<T> {
    fn reinitialize(&mut self, header: ChunkHeader) {
        self.size = header.size;
        self.grid_position = header.grid_position;
        self.data.resize_with(header.num_el, Default::default);
    }

    fn reinitialize_with<B: DataChunk<T>>(&mut self, other: &B) {
        self.size = other.get_size().into();
        self.grid_position = other.get_grid_position().into();
        self.data.clear();
        self.data.extend_from_slice(other.get_data());
    }
}

macro_rules! vec_data_chunk_impl {
    ($ty_name:ty, $bo_read_fn:ident, $bo_write_fn:ident) => {
        impl<C: AsMut<[$ty_name]>> ReadableDataChunk for SliceDataChunk<$ty_name, C> {
            fn read_data<R: std::io::Read>(&mut self, mut source: R) -> std::io::Result<()> {
                source.$bo_read_fn::<ZarrEndian>(self.data.as_mut())
            }
        }

        impl<C: AsRef<[$ty_name]>> WriteableDataChunk for SliceDataChunk<$ty_name, C> {
            fn write_data<W: std::io::Write>(&self, mut target: W) -> std::io::Result<()> {
                const CHUNK: usize = 256;
                let mut buf: [u8; CHUNK * std::mem::size_of::<$ty_name>()] =
                    [0; CHUNK * std::mem::size_of::<$ty_name>()];

                for c in self.data.as_ref().chunks(CHUNK) {
                    let byte_len = c.len() * std::mem::size_of::<$ty_name>();
                    ZarrEndian::$bo_write_fn(c, &mut buf[..byte_len]);
                    target.write_all(&buf[..byte_len])?;
                }

                Ok(())
            }
        }
    };
}

// Wrapper trait to erase a generic trait argument for consistent ByteOrder
// signatures.
trait ReadBytesExtI8: ReadBytesExt {
    fn read_i8_into_wrapper<B: ByteOrder>(&mut self, dst: &mut [i8]) -> std::io::Result<()> {
        self.read_i8_into(dst)
    }
}
impl<T: ReadBytesExt> ReadBytesExtI8 for T {}

vec_data_chunk_impl!(u16, read_u16_into, write_u16_into);
vec_data_chunk_impl!(u32, read_u32_into, write_u32_into);
vec_data_chunk_impl!(u64, read_u64_into, write_u64_into);
vec_data_chunk_impl!(i8, read_i8_into_wrapper, write_i8_into);
vec_data_chunk_impl!(i16, read_i16_into, write_i16_into);
vec_data_chunk_impl!(i32, read_i32_into, write_i32_into);
vec_data_chunk_impl!(i64, read_i64_into, write_i64_into);
vec_data_chunk_impl!(f32, read_f32_into, write_f32_into);
vec_data_chunk_impl!(f64, read_f64_into, write_f64_into);

impl<C: AsMut<[u8]>> ReadableDataChunk for SliceDataChunk<u8, C> {
    fn read_data<R: std::io::Read>(&mut self, mut source: R) -> std::io::Result<()> {
        source.read_exact(self.data.as_mut())
    }
}

impl<C: AsRef<[u8]>> WriteableDataChunk for SliceDataChunk<u8, C> {
    fn write_data<W: std::io::Write>(&self, mut target: W) -> std::io::Result<()> {
        target.write_all(self.data.as_ref())
    }
}

impl<T: ReflectedType, C: AsRef<[T]>> DataChunk<T> for SliceDataChunk<T, C> {
    fn get_size(&self) -> &[u32] {
        &self.size
    }

    fn get_grid_position(&self) -> &[u64] {
        &self.grid_position
    }

    fn get_data(&self) -> &[T] {
        self.data.as_ref()
    }

    fn get_num_elements(&self) -> u32 {
        self.data.as_ref().len() as u32
    }
}

const CHUNK_FIXED_LEN: u16 = 0;
const CHUNK_VAR_LEN: u16 = 1;

pub trait DefaultChunkHeaderReader<R: std::io::Read> {
    fn read_chunk_header(buffer: &mut R, grid_position: GridCoord) -> std::io::Result<ChunkHeader> {
        let mode = buffer.read_u16::<ZarrEndian>()?;
        let ndim = buffer.read_u16::<ZarrEndian>()?;
        let mut size = smallvec![0; ndim as usize];
        buffer.read_u32_into::<ZarrEndian>(&mut size)?;
        let num_el = match mode {
            CHUNK_FIXED_LEN => size.iter().product(),
            CHUNK_VAR_LEN => buffer.read_u32::<ZarrEndian>()?,
            _ => return Err(Error::new(ErrorKind::InvalidData, "Unsupported chunk mode")),
        };

        Ok(ChunkHeader {
            size,
            grid_position,
            num_el: num_el as usize,
        })
    }
}

/// Reads chunks from rust readers.
pub trait DefaultChunkReader<T: ReflectedType, R: std::io::Read>:
    DefaultChunkHeaderReader<R>
{
    fn read_chunk(
        mut buffer: R,
        array_meta: &ArrayMetadata,
        grid_position: GridCoord,
    ) -> std::io::Result<VecDataChunk<T>>
    where
        VecDataChunk<T>: DataChunk<T> + ReadableDataChunk,
    {
        if array_meta.data_type != T::VARIANT {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Attempt to create data chunk for wrong type.",
            ));
        }
        let header = Self::read_chunk_header(&mut buffer, grid_position)?;

        let mut chunk = T::create_data_chunk(header);
        let mut decompressed = array_meta.compression.decoder(buffer);
        chunk.read_data(&mut decompressed)?;

        Ok(chunk)
    }

    fn read_chunk_into<B: DataChunk<T> + ReinitDataChunk<T> + ReadableDataChunk>(
        mut buffer: R,
        array_meta: &ArrayMetadata,
        grid_position: GridCoord,
        chunk: &mut B,
    ) -> std::io::Result<()> {
        if array_meta.data_type != T::VARIANT {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Attempt to create data chunk for wrong type.",
            ));
        }
        let header = Self::read_chunk_header(&mut buffer, grid_position)?;

        chunk.reinitialize(header);
        let mut decompressed = array_meta.compression.decoder(buffer);
        chunk.read_data(&mut decompressed)?;

        Ok(())
    }
}

/// Writes chunks to rust writers.
pub trait DefaultChunkWriter<
    T: ReflectedType,
    W: std::io::Write,
    B: DataChunk<T> + WriteableDataChunk,
>
{
    fn write_chunk(mut buffer: W, array_meta: &ArrayMetadata, chunk: &B) -> std::io::Result<()> {
        if array_meta.data_type != T::VARIANT {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Attempt to write data chunk for wrong type.",
            ));
        }

        let mode: u16 = if chunk.get_num_elements() == chunk.get_size().iter().product::<u32>() {
            CHUNK_FIXED_LEN
        } else {
            CHUNK_VAR_LEN
        };
        buffer.write_u16::<ZarrEndian>(mode)?;
        buffer.write_u16::<ZarrEndian>(array_meta.get_ndim() as u16)?;
        for i in chunk.get_size() {
            buffer.write_u32::<ZarrEndian>(*i)?;
        }

        if mode != CHUNK_FIXED_LEN {
            buffer.write_u32::<ZarrEndian>(chunk.get_num_elements())?;
        }

        let mut compressor = array_meta.compression.encoder(buffer);
        chunk.write_data(&mut compressor)?;

        Ok(())
    }
}

// TODO: needed because cannot invoke type parameterized static trait methods
// directly from trait name in Rust. Symptom of design problems with
// `DefaultChunkReader`, etc.
#[derive(Debug)]
pub struct DefaultChunk;
impl<R: std::io::Read> DefaultChunkHeaderReader<R> for DefaultChunk {}
impl<T: ReflectedType, R: std::io::Read> DefaultChunkReader<T, R> for DefaultChunk {}
impl<T: ReflectedType, W: std::io::Write, B: DataChunk<T> + WriteableDataChunk>
    DefaultChunkWriter<T, W, B> for DefaultChunk
{
}
