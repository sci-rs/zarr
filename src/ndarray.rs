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
    ///
    /// Assumes chunks are column-major and returns a column-major ndarray.
    fn read_ndarray<T>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        bbox: &BoundingBox,
    ) -> Result<ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>, Error>
    where
        VecDataChunk<T>: DataChunk<T> + ReinitDataChunk<T> + ReadableDataChunk,
        T: ReflectedType + num_traits::identities::Zero,
    {
        let mut arr = Array::zeros(bbox.shape_ndarray_shape().f());

        self.read_ndarray_into(path_name, array_meta, bbox, arr.view_mut())?;

        Ok(arr)
    }

    /// Read an arbitrary bounding box from an Zarr volume into an existing
    /// ndarray view, reading chunks in serial as necessary.
    ///
    /// Assumes chunks are column-major. The array can be any order, but column-
    /// major will be more efficient.
    fn read_ndarray_into<'a, T>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        bbox: &BoundingBox,
        arr: ndarray::ArrayViewMut<'a, T, ndarray::Dim<ndarray::IxDynImpl>>,
    ) -> Result<(), Error>
    where
        VecDataChunk<T>: DataChunk<T> + ReinitDataChunk<T> + ReadableDataChunk,
        T: ReflectedType + num_traits::identities::Zero,
    {
        self.read_ndarray_into_with_buffer(path_name, array_meta, bbox, arr, &mut None)
    }

    /// Read an arbitrary bounding box from an Zarr volume into an existing
    /// ndarray view, reading chunks in serial as necessary into a provided
    /// buffer.
    ///
    /// Assumes chunks are column-major. The array can be any order, but column-
    /// major will be more efficient.
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
        T: ReflectedType + num_traits::identities::Zero,
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

                // Zarr arrays are stored f-order/column-major.
                let chunk_data =
                    ArrayView::from_shape(chunk_bb.shape_ndarray_shape().f(), chunk.get_data())
                        .expect("TODO: chunk ndarray failed");
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
        fill_val: T,
    ) -> Result<(), Error>
    // TODO: Next breaking version, refactor to use `SliceDataChunk` bounds.
    where
        VecDataChunk<T>: DataChunk<T> + ReadableDataChunk + WriteableDataChunk,
        T: ReflectedType + num_traits::identities::Zero,
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

        let mut chunk_vec: Vec<T> = Vec::new();

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
                chunk_vec.extend(arr_view.t().iter().cloned());
                let chunk = VecDataChunk::new(coord.into(), chunk_vec);

                self.write_chunk(path_name, array_meta, &chunk)?;
                chunk_vec = chunk.into_data();
            } else {
                let chunk_opt = self.read_chunk(path_name, array_meta, grid_coord.clone())?;

                let (chunk_bb, mut chunk_array) = match chunk_opt {
                    Some(chunk) => {
                        let chunk_bb = chunk.get_bounds(array_meta);
                        let chunk_array = Array::from_shape_vec(
                            chunk_bb.shape_ndarray_shape().f(),
                            chunk.into_data(),
                        )
                        .expect("TODO: chunk ndarray failed");
                        (chunk_bb, chunk_array)
                    }
                    None => {
                        // If no chunk exists, need to write from its origin.
                        let mut chunk_bb = write_bb.clone();
                        chunk_bb
                            .shape
                            .iter_mut()
                            .zip(write_bb.offset.iter())
                            .zip(nom_chunk_bb.offset.iter())
                            .for_each(|((s, o), g)| *s += *o - *g);
                        chunk_bb.offset = nom_chunk_bb.offset.clone();
                        let chunk_shape_usize = chunk_bb.shape_ndarray_shape();

                        let chunk_array =
                            Array::from_elem(&chunk_shape_usize[..], fill_val.clone()).into_dyn();
                        (chunk_bb, chunk_array)
                    }
                };

                let chunk_write_bb = write_bb.clone() - &chunk_bb.offset;
                let chunk_slice = chunk_write_bb.to_ndarray_slice();
                let mut chunk_view = chunk_array
                    .slice_mut(SliceInfo::<_, IxDyn>::new(chunk_slice).unwrap().as_ref());

                chunk_view.assign(&arr_view);

                chunk_vec.clear();
                chunk_vec.extend(chunk_array.t().iter().cloned());
                let chunk = VecDataChunk::new(coord.into(), chunk_vec);

                self.write_chunk(path_name, array_meta, &chunk)?;
                chunk_vec = chunk.into_data();
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
    /// Get the bounding box of the occupied extent of this chunk, which may
    /// be smaller than the nominal bounding box expected from the array.
    pub fn get_bounds(&self, array_meta: &ArrayMetadata) -> BoundingBox {
        let mut bbox = array_meta.get_chunk_bounds(self.get_grid_position());
        bbox.shape = array_meta
            .get_chunk_shape()
            .iter()
            .cloned()
            .map(u64::from)
            .collect();
        bbox
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
    fn test_array_attributes_coord_iter() {
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
