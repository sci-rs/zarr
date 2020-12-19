use std::io::{
    Error,
    ErrorKind,
    Read,
    Result,
    Write,
};
use std::marker::PhantomData;

use byteorder::{
    BigEndian,
    ByteOrder,
    LittleEndian,
    ReadBytesExt,
};
use half::f16;

use crate::compression::Compression;
use crate::{
    data_type::Endian,
    ArrayMetadata,
    GridCoord,
    ReflectedType,
};

/// Traits for data chunks that can be reused as a different chunks after
/// construction.
pub trait ReinitDataChunk<T> {
    /// Reinitialize this data chunk with a new header, reallocating as
    /// necessary.
    fn reinitialize(&mut self, grid_position: &GridCoord, num_el: u32);

    /// Reinitialize this data chunk with the header and data of another chunk.
    fn reinitialize_with<B: DataChunk<T>>(&mut self, other: &B);
}

/// Traits for data chunks that can read in data.
pub trait ReadableDataChunk {
    /// Read data into this chunk from a source, overwriting any existing data.
    ///
    /// Read the stream directly into the chunk data instead
    /// of creating a copied byte buffer.
    fn read_data<R: Read>(&mut self, source: R, array_meta: &ArrayMetadata) -> Result<()>;
}

/// Traits for data chunks that can write out data.
pub trait WriteableDataChunk {
    /// Write the data from this chunk into a target.
    fn write_data<W: Write>(&self, target: W, array_meta: &ArrayMetadata) -> Result<()>;
}

/// Common interface for data chunks of element (rust) type `T`.
///
/// To enable custom types to be written to Zarr volumes, implement this trait.
pub trait DataChunk<T> {
    fn get_grid_position(&self) -> &[u64];

    fn get_data(&self) -> &[T];

    fn get_num_elements(&self) -> u32;
}

/// A generic data chunk container wrapping any type that can be taken as a
/// slice ref.
#[derive(Clone, Debug)]
pub struct SliceDataChunk<T: ReflectedType, C> {
    data_type: PhantomData<T>,
    grid_position: GridCoord,
    data: C,
}

impl<T: ReflectedType, C> SliceDataChunk<T, C> {
    pub fn new(grid_position: GridCoord, data: C) -> SliceDataChunk<T, C> {
        SliceDataChunk {
            data_type: PhantomData,
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
    fn reinitialize(&mut self, grid_position: &GridCoord, num_el: u32) {
        self.grid_position = grid_position.clone();
        self.data.resize_with(num_el as usize, Default::default);
    }

    fn reinitialize_with<B: DataChunk<T>>(&mut self, other: &B) {
        self.grid_position = other.get_grid_position().into();
        self.data.clear();
        self.data.extend_from_slice(other.get_data());
    }
}

macro_rules! vec_data_chunk_impl {
    ($ty_name:ty, $bo_read_fn:ident, $bo_write_fn:ident) => {
        impl<C: AsMut<[$ty_name]>> ReadableDataChunk for SliceDataChunk<$ty_name, C> {
            fn read_data<R: Read>(
                &mut self,
                mut source: R,
                array_meta: &ArrayMetadata,
            ) -> Result<()> {
                match array_meta.data_type.effective_type()?.endian() {
                    Endian::Big => source.$bo_read_fn::<BigEndian>(self.data.as_mut()),
                    Endian::Little => source.$bo_read_fn::<LittleEndian>(self.data.as_mut()),
                }
            }
        }

        impl<C: AsRef<[$ty_name]>> WriteableDataChunk for SliceDataChunk<$ty_name, C> {
            fn write_data<W: Write>(
                &self,
                mut target: W,
                array_meta: &ArrayMetadata,
            ) -> Result<()> {
                const CHUNK: usize = 256;
                let mut buf: [u8; CHUNK * std::mem::size_of::<$ty_name>()] =
                    [0; CHUNK * std::mem::size_of::<$ty_name>()];

                let endian = array_meta.data_type.effective_type()?.endian();
                for c in self.data.as_ref().chunks(CHUNK) {
                    let byte_len = c.len() * std::mem::size_of::<$ty_name>();
                    match endian {
                        Endian::Big => BigEndian::$bo_write_fn(c, &mut buf[..byte_len]),
                        Endian::Little => LittleEndian::$bo_write_fn(c, &mut buf[..byte_len]),
                    }
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
    fn read_i8_into_wrapper<B: ByteOrder>(&mut self, dst: &mut [i8]) -> Result<()> {
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
    fn read_data<R: Read>(&mut self, mut source: R, _array_meta: &ArrayMetadata) -> Result<()> {
        source.read_exact(self.data.as_mut())
    }
}

impl<C: AsRef<[u8]>> WriteableDataChunk for SliceDataChunk<u8, C> {
    fn write_data<W: Write>(&self, mut target: W, _array_meta: &ArrayMetadata) -> Result<()> {
        target.write_all(self.data.as_ref())
    }
}

impl<C: AsMut<[bool]>> ReadableDataChunk for SliceDataChunk<bool, C> {
    fn read_data<R: Read>(&mut self, mut source: R, _array_meta: &ArrayMetadata) -> Result<()> {
        const CHUNK: usize = 256;
        let mut buf: [u8; CHUNK] = [0; CHUNK];

        for c in self.data.as_mut().chunks_mut(CHUNK) {
            let len = c.len();
            source.read_exact(&mut buf[..len])?;
            for (i, &j) in c.iter_mut().zip(buf[..len].iter()) {
                *i = j != 0;
            }
        }

        Ok(())
    }
}

impl<C: AsRef<[bool]>> WriteableDataChunk for SliceDataChunk<bool, C> {
    fn write_data<W: Write>(&self, mut target: W, _array_meta: &ArrayMetadata) -> Result<()> {
        const CHUNK: usize = 256;
        let mut buf: [u8; CHUNK] = [0; CHUNK];

        for c in self.data.as_ref().chunks(CHUNK) {
            for (&i, j) in c.iter().zip(buf[..c.len()].iter_mut()) {
                *j = i.into();
            }
            target.write_all(&buf[..c.len()])?;
        }

        Ok(())
    }
}

impl<C: AsMut<[f16]>> ReadableDataChunk for SliceDataChunk<f16, C> {
    fn read_data<R: Read>(&mut self, mut source: R, array_meta: &ArrayMetadata) -> Result<()> {
        // TODO: no chunking
        let endian = array_meta.data_type.effective_type()?.endian();
        for n in self.data.as_mut() {
            let mut bytes = [0; 2];
            source.read_exact(&mut bytes[..])?;
            *n = match endian {
                Endian::Big => f16::from_be_bytes(bytes),
                Endian::Little => f16::from_le_bytes(bytes),
            };
        }
        Ok(())
    }
}

impl<C: AsRef<[f16]>> WriteableDataChunk for SliceDataChunk<f16, C> {
    fn write_data<W: Write>(&self, mut target: W, array_meta: &ArrayMetadata) -> Result<()> {
        // TODO: no chunking
        let endian = array_meta.data_type.effective_type()?.endian();
        for n in self.data.as_ref() {
            let bytes = match endian {
                Endian::Big => n.to_be_bytes(),
                Endian::Little => n.to_le_bytes(),
            };
            target.write_all(&bytes[..])?;
        }
        Ok(())
    }
}

impl<T: ReflectedType, C: AsRef<[T]>> DataChunk<T> for SliceDataChunk<T, C> {
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

fn check_array_type<T: ReflectedType>(array_meta: &ArrayMetadata) -> Result<()> {
    if array_meta
        .data_type
        .effective_type()?
        .eq_modulo_endian(&T::ZARR_TYPE)
    {
        Ok(())
    } else {
        Err(Error::new(
            ErrorKind::InvalidInput,
            "Attempt to create data chunk for wrong type.",
        ))
    }
}

/// Reads chunks from rust readers.
pub trait DefaultChunkReader<T: ReflectedType, R: Read> {
    fn read_chunk(
        buffer: R,
        array_meta: &ArrayMetadata,
        grid_position: GridCoord,
    ) -> Result<VecDataChunk<T>>
    where
        VecDataChunk<T>: DataChunk<T> + ReadableDataChunk,
    {
        check_array_type::<T>(array_meta)?;

        let mut chunk =
            T::create_data_chunk(&grid_position, array_meta.get_chunk_num_elements() as u32);
        let mut decompressed = array_meta.compressor.decoder(buffer);
        chunk.read_data(&mut decompressed, array_meta)?;

        Ok(chunk)
    }

    fn read_chunk_into<B: DataChunk<T> + ReinitDataChunk<T> + ReadableDataChunk>(
        buffer: R,
        array_meta: &ArrayMetadata,
        grid_position: GridCoord,
        chunk: &mut B,
    ) -> Result<()> {
        check_array_type::<T>(array_meta)?;

        chunk.reinitialize(&grid_position, array_meta.get_chunk_num_elements() as u32);
        let mut decompressed = array_meta.compressor.decoder(buffer);
        chunk.read_data(&mut decompressed, array_meta)?;

        Ok(())
    }
}

/// Writes chunks to rust writers.
pub trait DefaultChunkWriter<T: ReflectedType, W: Write, B: DataChunk<T> + WriteableDataChunk> {
    fn write_chunk(buffer: W, array_meta: &ArrayMetadata, chunk: &B) -> Result<()> {
        check_array_type::<T>(array_meta)?;

        if chunk.get_num_elements() as usize != array_meta.get_chunk_num_elements() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Can not write chunk with too few elements. Expected {} given {}",
                    array_meta.get_chunk_num_elements(),
                    chunk.get_num_elements()
                ),
            ));
        }
        let mut compressor = array_meta.compressor.encoder(buffer);
        chunk.write_data(&mut compressor, array_meta)?;

        Ok(())
    }
}

// TODO: needed because cannot invoke type parameterized static trait methods
// directly from trait name in Rust. Symptom of design problems with
// `DefaultChunkReader`, etc.
#[derive(Debug)]
pub struct DefaultChunk;
impl<T: ReflectedType, R: Read> DefaultChunkReader<T, R> for DefaultChunk {}
impl<T: ReflectedType, W: Write, B: DataChunk<T> + WriteableDataChunk> DefaultChunkWriter<T, W, B>
    for DefaultChunk
{
}
