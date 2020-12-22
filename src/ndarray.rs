use std::cmp;
use std::io::{
    Error,
    ErrorKind,
};
use std::ops::Sub;

use itertools::Itertools;
use ndarray::{
    Array,
    ArrayView,
    IxDyn,
    ShapeBuilder,
    SliceInfo,
};

use crate::{
    ArrayMetadata,
    ChunkCoord,
    CoordVec,
    DataChunk,
    GridCoord,
    HierarchyReader,
    HierarchyWriter,
    Order,
    ReadableDataChunk,
    ReflectedType,
    ReinitDataChunk,
    SliceDataChunk,
    VecDataChunk,
    WriteableDataChunk,
};

pub mod prelude {
    pub use super::{
        BoundingBox,
        ZarrNdarrayReader,
        ZarrNdarrayWriter,
    };
}

/// Specifes the extents of an axis-aligned bounding box.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundingBox {
    offset: GridCoord,
    shape: GridCoord,
}

impl BoundingBox {
    pub fn new(offset: GridCoord, shape: GridCoord) -> BoundingBox {
        assert_eq!(offset.len(), shape.len());

        BoundingBox { offset, shape }
    }

    pub fn shape_chunk(&self) -> ChunkCoord {
        self.shape.iter().map(|n| *n as u32).collect()
    }

    pub fn shape_ndarray_shape(&self) -> CoordVec<usize> {
        self.shape.iter().map(|n| *n as usize).collect()
    }

    /// ```
    /// # use zarr::ndarray::BoundingBox;
    /// # use zarr::smallvec::smallvec;
    /// let mut a = BoundingBox::new(smallvec![0, 0], smallvec![5, 8]);
    /// let b = BoundingBox::new(smallvec![3, 3], smallvec![5, 3]);
    /// let c = BoundingBox::new(smallvec![3, 3], smallvec![2, 3]);
    /// a.intersect(&b);
    /// assert_eq!(a, c);
    /// ```
    pub fn intersect(&mut self, other: &BoundingBox) {
        assert_eq!(self.offset.len(), other.offset.len());

        self.shape
            .iter_mut()
            .zip(self.offset.iter_mut())
            .zip(other.shape.iter())
            .zip(other.offset.iter())
            .for_each(|(((s, o), os), oo)| {
                let new_o = std::cmp::max(*oo, *o);
                *s = std::cmp::min(*s + *o, *oo + *os).saturating_sub(new_o);
                *o = new_o;
            });
    }

    /// ```
    /// # use zarr::ndarray::BoundingBox;
    /// # use zarr::smallvec::smallvec;
    /// let mut a = BoundingBox::new(smallvec![0, 0], smallvec![5, 8]);
    /// let b = BoundingBox::new(smallvec![3, 3], smallvec![5, 3]);
    /// let c = BoundingBox::new(smallvec![0, 0], smallvec![8, 8]);
    /// a.union(&b);
    /// assert_eq!(a, c);
    /// ```
    pub fn union(&mut self, other: &BoundingBox) {
        assert_eq!(self.offset.len(), other.offset.len());

        self.shape
            .iter_mut()
            .zip(self.offset.iter_mut())
            .zip(other.shape.iter())
            .zip(other.offset.iter())
            .for_each(|(((s, o), os), oo)| {
                let new_o = std::cmp::min(*oo, *o);
                *s = std::cmp::max(*s + *o, *oo + *os) - new_o;
                *o = new_o;
            });
    }

    pub fn end(&self) -> impl Iterator<Item = u64> + '_ {
        self.offset
            .iter()
            .zip(self.shape.iter())
            .map(|(o, s)| o + s)
    }

    pub fn to_ndarray_slice(&self) -> CoordVec<ndarray::SliceOrIndex> {
        self.offset
            .iter()
            .zip(self.end())
            .map(|(&start, end)| ndarray::SliceOrIndex::Slice {
                start: start as isize,
                end: Some(end as isize),
                step: 1,
            })
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.shape.contains(&0)
    }
}

impl Sub<&GridCoord> for BoundingBox {
    type Output = Self;

    fn sub(self, other: &GridCoord) -> Self::Output {
        Self {
            offset: self
                .offset
                .iter()
                .zip(other.iter())
                .map(|(s, o)| s.checked_sub(*o).unwrap())
                .collect(),
            shape: self.shape.clone(),
        }
    }
}

pub trait ZarrNdarrayReader: HierarchyReader {
    /// Read an arbitrary bounding box from an Zarr volume into an ndarray,
    /// reading chunks in serial as necessary.
    fn read_ndarray<T>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        bbox: &BoundingBox,
    ) -> Result<ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>, Error>
    where
        VecDataChunk<T>: DataChunk<T> + ReinitDataChunk<T> + ReadableDataChunk,
        T: ReflectedType,
    {
        let chunk_shape = match array_meta.get_chunk_memory_layout() {
            Order::ColumnMajor => bbox.shape_ndarray_shape().f(),
            Order::RowMajor => bbox.shape_ndarray_shape()[..].into_shape(),
        };
        let fill_value = array_meta.get_effective_fill_value()?;
        let mut arr = Array::from_elem(chunk_shape, fill_value);

        self.read_ndarray_into(path_name, array_meta, bbox, arr.view_mut())?;

        Ok(arr)
    }

    /// Read an arbitrary bounding box from an Zarr volume into an existing
    /// ndarray view, reading chunks in serial as necessary.
    fn read_ndarray_into<'a, T>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        bbox: &BoundingBox,
        arr: ndarray::ArrayViewMut<'a, T, ndarray::Dim<ndarray::IxDynImpl>>,
    ) -> Result<(), Error>
    where
        VecDataChunk<T>: DataChunk<T> + ReinitDataChunk<T> + ReadableDataChunk,
        T: ReflectedType,
    {
        self.read_ndarray_into_with_buffer(path_name, array_meta, bbox, arr, &mut None)
    }

    /// Read an arbitrary bounding box from an Zarr volume into an existing
    /// ndarray view, reading chunks in serial as necessary into a provided
    /// buffer.
    fn read_ndarray_into_with_buffer<'a, T>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        bbox: &BoundingBox,
        mut arr: ndarray::ArrayViewMut<'a, T, ndarray::Dim<ndarray::IxDynImpl>>,
        chunk_buff_opt: &mut Option<VecDataChunk<T>>,
    ) -> Result<(), Error>
    where
        VecDataChunk<T>: DataChunk<T> + ReinitDataChunk<T> + ReadableDataChunk,
        T: ReflectedType,
    {
        if bbox.offset.len() != array_meta.get_ndim() || array_meta.get_ndim() != arr.ndim() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Wrong number of dimensions",
            ));
        }

        if bbox.shape_ndarray_shape().as_slice() != arr.shape() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Bounding box and array have different shape",
            ));
        }

        for coord in array_meta.bounded_coord_iter(bbox) {
            let grid_pos = GridCoord::from(&coord[..]);
            let is_chunk = match chunk_buff_opt {
                None => {
                    *chunk_buff_opt = self.read_chunk(path_name, array_meta, grid_pos)?;
                    chunk_buff_opt.is_some()
                }
                Some(ref mut chunk_buff) => self
                    .read_chunk_into(path_name, array_meta, grid_pos, chunk_buff)?
                    .is_some(),
            };

            // TODO: cannot combine this into condition below until `let_chains` stabilizes.
            if !is_chunk {
                continue;
            }

            if let Some(ref chunk) = chunk_buff_opt {
                let chunk_bb = chunk.get_bounds(array_meta);
                let mut read_bb = bbox.clone();
                read_bb.intersect(&chunk_bb);

                // It may be the case the while the chunk's potential bounds are
                // in the request region, the chunk is smaller such that it does
                // not intersect.
                if read_bb.is_empty() {
                    continue;
                }

                let arr_read_bb = read_bb.clone() - &bbox.offset;
                let chunk_read_bb = read_bb.clone() - &chunk_bb.offset;

                let arr_slice = arr_read_bb.to_ndarray_slice();
                let mut arr_view =
                    arr.slice_mut(SliceInfo::<_, IxDyn>::new(arr_slice).unwrap().as_ref());

                let chunk_slice = chunk_read_bb.to_ndarray_slice();

                let chunk_data = chunk.as_ndarray(array_meta);
                let chunk_view =
                    chunk_data.slice(SliceInfo::<_, IxDyn>::new(chunk_slice).unwrap().as_ref());

                arr_view.assign(&chunk_view);
            }
        }

        Ok(())
    }
}

impl<T: HierarchyReader> ZarrNdarrayReader for T {}

pub trait ZarrNdarrayWriter: HierarchyWriter {
    /// Write an arbitrary bounding box from an ndarray into an Zarr volume,
    /// writing chunks in serial as necessary.
    fn write_ndarray<'a, T, A>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        offset: GridCoord,
        array: A,
    ) -> Result<(), Error>
    // TODO: Next breaking version, refactor to use `SliceDataChunk` bounds.
    where
        VecDataChunk<T>: DataChunk<T> + ReadableDataChunk + WriteableDataChunk,
        T: ReflectedType,
        A: ndarray::AsArray<'a, T, ndarray::Dim<ndarray::IxDynImpl>>,
    {
        let array = array.into();
        if array.ndim() != array_meta.get_ndim() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Wrong number of dimensions",
            ));
        }
        let bbox = BoundingBox {
            offset,
            shape: array.shape().iter().map(|n| *n as u64).collect(),
        };
        let fill_value: T = array_meta.get_effective_fill_value()?;

        let mut chunk_vec: Vec<T> = Vec::new();
        let mut existing_chunk_vec: Vec<T> = Vec::new();

        for coord in array_meta.bounded_coord_iter(&bbox) {
            let grid_coord = GridCoord::from(&coord[..]);
            let nom_chunk_bb = array_meta.get_chunk_bounds(&grid_coord);
            let mut write_bb = nom_chunk_bb.clone();
            write_bb.intersect(&bbox);
            let arr_bb = write_bb.clone() - &bbox.offset;

            let arr_slice = arr_bb.to_ndarray_slice();
            let arr_view = array.slice(SliceInfo::<_, IxDyn>::new(arr_slice).unwrap().as_ref());

            if write_bb == nom_chunk_bb {
                // No need to read whether there is an extant chunk if it is
                // going to be entirely overwriten.
                chunk_vec.clear();
                // TODO: need to adjust `t` based on array ordering?
                chunk_vec.extend(arr_view.t().iter().cloned());
                let chunk = VecDataChunk::new(coord.into(), chunk_vec);

                self.write_chunk(path_name, array_meta, &chunk)?;
                chunk_vec = chunk.into_data();
            } else {
                let mut existing_chunk = VecDataChunk::new(grid_coord.clone(), existing_chunk_vec);
                let chunk_opt = self.read_chunk_into(
                    path_name,
                    array_meta,
                    grid_coord.clone(),
                    &mut existing_chunk,
                )?;

                let (chunk_bb, mut chunk_array) = match chunk_opt {
                    Some(()) => {
                        let chunk_bb = existing_chunk.get_bounds(array_meta);
                        let chunk_array = existing_chunk.into_ndarray(array_meta);
                        (chunk_bb, chunk_array)
                    }
                    None => {
                        // If no chunk exists, need to write from its origin.
                        // In Zarr this simply means the chunk must be full.
                        let chunk_shape_usize = nom_chunk_bb.shape_ndarray_shape();
                        existing_chunk_vec = existing_chunk.into_data();
                        existing_chunk_vec.clear();
                        existing_chunk_vec
                            .resize(chunk_shape_usize.iter().product(), fill_value.clone());

                        let chunk_array =
                            Array::from_shape_vec(&chunk_shape_usize[..], existing_chunk_vec)
                                .expect("TODO: chunk ndarray failed");
                        (nom_chunk_bb, chunk_array)
                    }
                };

                let chunk_write_bb = write_bb.clone() - &chunk_bb.offset;
                let chunk_slice = chunk_write_bb.to_ndarray_slice();
                let mut chunk_view = chunk_array
                    .slice_mut(SliceInfo::<_, IxDyn>::new(chunk_slice).unwrap().as_ref());

                chunk_view.assign(&arr_view);

                chunk_vec.clear();
                // TODO: need to adjust `t` based on array ordering?
                chunk_vec.extend(chunk_array.t().iter().cloned());
                let chunk = VecDataChunk::new(coord.into(), chunk_vec);

                self.write_chunk(path_name, array_meta, &chunk)?;
                chunk_vec = chunk.into_data();
                existing_chunk_vec = chunk_array.into_raw_vec();
            }
        }

        Ok(())
    }
}

impl<T: HierarchyWriter> ZarrNdarrayWriter for T {}

impl ArrayMetadata {
    pub fn coord_iter(&self) -> impl Iterator<Item = Vec<u64>> + ExactSizeIterator {
        let coord_ceil = self
            .get_shape()
            .iter()
            .zip(self.get_chunk_shape().iter())
            .map(|(&d, &s)| (d + u64::from(s) - 1) / u64::from(s))
            .collect::<GridCoord>();

        CoordIterator::new(&coord_ceil)
    }

    pub fn bounded_coord_iter(
        &self,
        bbox: &BoundingBox,
    ) -> impl Iterator<Item = Vec<u64>> + ExactSizeIterator {
        let floor_coord: GridCoord = bbox
            .offset
            .iter()
            .zip(&self.chunk_grid.chunk_shape)
            .map(|(&o, &bs)| o / u64::from(bs))
            .collect();
        let ceil_coord: GridCoord = bbox
            .offset
            .iter()
            .zip(&bbox.shape)
            .zip(self.chunk_grid.chunk_shape.iter().cloned().map(u64::from))
            .map(|((&o, &s), bs)| (o + s + bs - 1) / bs)
            .collect();

        CoordIterator::floor_ceil(&floor_coord, &ceil_coord)
    }

    pub fn get_bounds(&self) -> BoundingBox {
        BoundingBox {
            offset: smallvec![0; self.shape.len()],
            shape: self.shape.clone(),
        }
    }

    pub fn get_chunk_bounds(&self, coord: &[u64]) -> BoundingBox {
        let mut shape: GridCoord = self
            .get_chunk_shape()
            .iter()
            .cloned()
            .map(u64::from)
            .collect();
        let offset: GridCoord = coord.iter().zip(shape.iter()).map(|(c, s)| c * s).collect();
        shape
            .iter_mut()
            .zip(offset.iter())
            .zip(self.get_shape().iter())
            .for_each(|((s, o), d)| *s = cmp::min(*s + *o, *d) - *o);
        BoundingBox { offset, shape }
    }
}

impl<T: ReflectedType, C: AsRef<[T]>> SliceDataChunk<T, C> {
    /// Get the bounding box of the occupied extent of this chunk.
    /// In Zarr all chunks in an array are the same shape.
    pub fn get_bounds(&self, array_meta: &ArrayMetadata) -> BoundingBox {
        array_meta.get_chunk_bounds(self.get_grid_position())
    }

    fn shape_ndarray_shape(
        &self,
        array_meta: &ArrayMetadata,
    ) -> ndarray::Shape<ndarray::Dim<ndarray::IxDynImpl>> {
        let chunk_bb = self.get_bounds(array_meta);
        match array_meta.get_chunk_memory_layout() {
            Order::ColumnMajor => chunk_bb.shape_ndarray_shape().f(),
            Order::RowMajor => chunk_bb.shape_ndarray_shape()[..].into_shape(),
        }
    }

    pub fn as_ndarray(
        &self,
        array_meta: &ArrayMetadata,
    ) -> ArrayView<T, ndarray::Dim<ndarray::IxDynImpl>> {
        let chunk_shape = self.shape_ndarray_shape(array_meta);
        ArrayView::from_shape(chunk_shape, self.get_data()).expect("TODO: chunk ndarray failed")
    }
}

impl<T: ReflectedType> VecDataChunk<T> {
    pub fn into_ndarray(
        self,
        array_meta: &ArrayMetadata,
    ) -> Array<T, ndarray::Dim<ndarray::IxDynImpl>> {
        let chunk_shape = self.shape_ndarray_shape(array_meta);
        Array::from_shape_vec(chunk_shape, self.into_data()).expect("TODO: chunk ndarray failed")
    }
}

/// Iterator wrapper to provide exact size when iterating over coordinate
/// ranges.
struct CoordIterator<T: Iterator<Item = Vec<u64>>> {
    iter: T,
    accumulator: usize,
    total_coords: usize,
}

impl CoordIterator<itertools::MultiProduct<std::ops::Range<u64>>> {
    fn new(ceil: &[u64]) -> Self {
        CoordIterator {
            iter: ceil.iter().map(|&c| 0..c).multi_cartesian_product(),
            accumulator: 0,
            total_coords: ceil.iter().product::<u64>() as usize,
        }
    }

    fn floor_ceil(floor: &[u64], ceil: &[u64]) -> Self {
        let total_coords = floor
            .iter()
            .zip(ceil.iter())
            .map(|(&f, &c)| c - f)
            .product::<u64>() as usize;
        CoordIterator {
            iter: floor
                .iter()
                .zip(ceil.iter())
                .map(|(&f, &c)| f..c)
                .multi_cartesian_product(),
            accumulator: 0,
            total_coords,
        }
    }
}

impl<T: Iterator<Item = Vec<u64>>> Iterator for CoordIterator<T> {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        self.accumulator += 1;
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_coords - self.accumulator;
        (remaining, Some(remaining))
    }
}

impl<T: Iterator<Item = Vec<u64>>> ExactSizeIterator for CoordIterator<T> {}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::data_type::ReflectedType;

    #[test]
    fn test_array_metadata_coord_iter() {
        use std::collections::HashSet;

        let array_meta = ArrayMetadata::new(
            smallvec![1, 4, 5],
            smallvec![1, 2, 3],
            i16::ZARR_TYPE,
            crate::compression::CompressionType::default(),
        );

        let coords: HashSet<Vec<u64>> = array_meta.coord_iter().collect();
        let expected: HashSet<Vec<u64>> =
            vec![vec![0, 0, 0], vec![0, 0, 1], vec![0, 1, 0], vec![0, 1, 1]]
                .into_iter()
                .collect();

        assert_eq!(coords, expected);
    }
}
