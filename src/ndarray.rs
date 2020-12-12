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
    BlockCoord,
    CoordVec,
    DataBlock,
    DatasetAttributes,
    GridCoord,
    N5Reader,
    N5Writer,
    ReadableDataBlock,
    ReflectedType,
    ReinitDataBlock,
    SliceDataBlock,
    VecDataBlock,
    WriteableDataBlock,
};

pub mod prelude {
    pub use super::{
        BoundingBox,
        N5NdarrayReader,
        N5NdarrayWriter,
    };
}

/// Specifes the extents of an axis-aligned bounding box.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundingBox {
    offset: GridCoord,
    size: GridCoord,
}

impl BoundingBox {
    pub fn new(offset: GridCoord, size: GridCoord) -> BoundingBox {
        assert_eq!(offset.len(), size.len());

        BoundingBox { offset, size }
    }

    pub fn size_block(&self) -> BlockCoord {
        self.size.iter().map(|n| *n as u32).collect()
    }

    pub fn size_ndarray_shape(&self) -> CoordVec<usize> {
        self.size.iter().map(|n| *n as usize).collect()
    }

    /// ```
    /// # use n5::ndarray::BoundingBox;
    /// # use n5::smallvec::smallvec;
    /// let mut a = BoundingBox::new(smallvec![0, 0], smallvec![5, 8]);
    /// let b = BoundingBox::new(smallvec![3, 3], smallvec![5, 3]);
    /// let c = BoundingBox::new(smallvec![3, 3], smallvec![2, 3]);
    /// a.intersect(&b);
    /// assert_eq!(a, c);
    /// ```
    pub fn intersect(&mut self, other: &BoundingBox) {
        assert_eq!(self.offset.len(), other.offset.len());

        self.size
            .iter_mut()
            .zip(self.offset.iter_mut())
            .zip(other.size.iter())
            .zip(other.offset.iter())
            .for_each(|(((s, o), os), oo)| {
                let new_o = std::cmp::max(*oo, *o);
                *s = std::cmp::min(*s + *o, *oo + *os).saturating_sub(new_o);
                *o = new_o;
            });
    }

    /// ```
    /// # use n5::ndarray::BoundingBox;
    /// # use n5::smallvec::smallvec;
    /// let mut a = BoundingBox::new(smallvec![0, 0], smallvec![5, 8]);
    /// let b = BoundingBox::new(smallvec![3, 3], smallvec![5, 3]);
    /// let c = BoundingBox::new(smallvec![0, 0], smallvec![8, 8]);
    /// a.union(&b);
    /// assert_eq!(a, c);
    /// ```
    pub fn union(&mut self, other: &BoundingBox) {
        assert_eq!(self.offset.len(), other.offset.len());

        self.size
            .iter_mut()
            .zip(self.offset.iter_mut())
            .zip(other.size.iter())
            .zip(other.offset.iter())
            .for_each(|(((s, o), os), oo)| {
                let new_o = std::cmp::min(*oo, *o);
                *s = std::cmp::max(*s + *o, *oo + *os) - new_o;
                *o = new_o;
            });
    }

    pub fn end(&self) -> impl Iterator<Item = u64> + '_ {
        self.offset.iter().zip(self.size.iter()).map(|(o, s)| o + s)
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
        self.size.contains(&0)
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
            size: self.size.clone(),
        }
    }
}

pub trait N5NdarrayReader: N5Reader {
    /// Read an arbitrary bounding box from an N5 volume into an ndarray,
    /// reading blocks in serial as necessary.
    ///
    /// Assumes blocks are column-major and returns a column-major ndarray.
    fn read_ndarray<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        bbox: &BoundingBox,
    ) -> Result<ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>, Error>
    where
        VecDataBlock<T>: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock,
        T: ReflectedType + num_traits::identities::Zero,
    {
        let mut arr = Array::zeros(bbox.size_ndarray_shape().f());

        self.read_ndarray_into(path_name, data_attrs, bbox, arr.view_mut())?;

        Ok(arr)
    }

    /// Read an arbitrary bounding box from an N5 volume into an existing
    /// ndarray view, reading blocks in serial as necessary.
    ///
    /// Assumes blocks are column-major. The array can be any order, but column-
    /// major will be more efficient.
    fn read_ndarray_into<'a, T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        bbox: &BoundingBox,
        arr: ndarray::ArrayViewMut<'a, T, ndarray::Dim<ndarray::IxDynImpl>>,
    ) -> Result<(), Error>
    where
        VecDataBlock<T>: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock,
        T: ReflectedType + num_traits::identities::Zero,
    {
        self.read_ndarray_into_with_buffer(path_name, data_attrs, bbox, arr, &mut None)
    }

    /// Read an arbitrary bounding box from an N5 volume into an existing
    /// ndarray view, reading blocks in serial as necessary into a provided
    /// buffer.
    ///
    /// Assumes blocks are column-major. The array can be any order, but column-
    /// major will be more efficient.
    fn read_ndarray_into_with_buffer<'a, T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        bbox: &BoundingBox,
        mut arr: ndarray::ArrayViewMut<'a, T, ndarray::Dim<ndarray::IxDynImpl>>,
        block_buff_opt: &mut Option<VecDataBlock<T>>,
    ) -> Result<(), Error>
    where
        VecDataBlock<T>: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock,
        T: ReflectedType + num_traits::identities::Zero,
    {
        if bbox.offset.len() != data_attrs.get_ndim() || data_attrs.get_ndim() != arr.ndim() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Wrong number of dimensions",
            ));
        }

        if bbox.size_ndarray_shape().as_slice() != arr.shape() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Bounding box and array have different shape",
            ));
        }

        for coord in data_attrs.bounded_coord_iter(bbox) {
            let grid_pos = GridCoord::from(&coord[..]);
            let is_block = match block_buff_opt {
                None => {
                    *block_buff_opt = self.read_block(path_name, data_attrs, grid_pos)?;
                    block_buff_opt.is_some()
                }
                Some(ref mut block_buff) => self
                    .read_block_into(path_name, data_attrs, grid_pos, block_buff)?
                    .is_some(),
            };

            // TODO: cannot combine this into condition below until `let_chains` stabilizes.
            if !is_block {
                continue;
            }

            if let Some(ref block) = block_buff_opt {
                let block_bb = block.get_bounds(data_attrs);
                let mut read_bb = bbox.clone();
                read_bb.intersect(&block_bb);

                // It may be the case the while the block's potential bounds are
                // in the request region, the block is smaller such that it does
                // not intersect.
                if read_bb.is_empty() {
                    continue;
                }

                let arr_read_bb = read_bb.clone() - &bbox.offset;
                let block_read_bb = read_bb.clone() - &block_bb.offset;

                let arr_slice = arr_read_bb.to_ndarray_slice();
                let mut arr_view =
                    arr.slice_mut(SliceInfo::<_, IxDyn>::new(arr_slice).unwrap().as_ref());

                let block_slice = block_read_bb.to_ndarray_slice();

                // N5 datasets are stored f-order/column-major.
                let block_data =
                    ArrayView::from_shape(block_bb.size_ndarray_shape().f(), block.get_data())
                        .expect("TODO: block ndarray failed");
                let block_view =
                    block_data.slice(SliceInfo::<_, IxDyn>::new(block_slice).unwrap().as_ref());

                arr_view.assign(&block_view);
            }
        }

        Ok(())
    }
}

impl<T: N5Reader> N5NdarrayReader for T {}

pub trait N5NdarrayWriter: N5Writer {
    /// Write an arbitrary bounding box from an ndarray into an N5 volume,
    /// writing blocks in serial as necessary.
    fn write_ndarray<'a, T, A>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        offset: GridCoord,
        array: A,
        fill_val: T,
    ) -> Result<(), Error>
    // TODO: Next breaking version, refactor to use `SliceDataBlock` bounds.
    where
        VecDataBlock<T>: DataBlock<T> + ReadableDataBlock + WriteableDataBlock,
        T: ReflectedType + num_traits::identities::Zero,
        A: ndarray::AsArray<'a, T, ndarray::Dim<ndarray::IxDynImpl>>,
    {
        let array = array.into();
        if array.ndim() != data_attrs.get_ndim() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Wrong number of dimensions",
            ));
        }
        let bbox = BoundingBox {
            offset,
            size: array.shape().iter().map(|n| *n as u64).collect(),
        };

        let mut block_vec: Vec<T> = Vec::new();

        for coord in data_attrs.bounded_coord_iter(&bbox) {
            let grid_coord = GridCoord::from(&coord[..]);
            let nom_block_bb = data_attrs.get_block_bounds(&grid_coord);
            let mut write_bb = nom_block_bb.clone();
            write_bb.intersect(&bbox);
            let arr_bb = write_bb.clone() - &bbox.offset;

            let arr_slice = arr_bb.to_ndarray_slice();
            let arr_view = array.slice(SliceInfo::<_, IxDyn>::new(arr_slice).unwrap().as_ref());

            if write_bb == nom_block_bb {
                // No need to read whether there is an extant block if it is
                // going to be entirely overwriten.
                block_vec.clear();
                block_vec.extend(arr_view.t().iter().cloned());
                let block = VecDataBlock::new(write_bb.size_block(), coord.into(), block_vec);

                self.write_block(path_name, data_attrs, &block)?;
                block_vec = block.into_data();
            } else {
                let block_opt = self.read_block(path_name, data_attrs, grid_coord.clone())?;

                let (block_bb, mut block_array) = match block_opt {
                    Some(block) => {
                        let block_bb = block.get_bounds(data_attrs);
                        let block_array = Array::from_shape_vec(
                            block_bb.size_ndarray_shape().f(),
                            block.into_data(),
                        )
                        .expect("TODO: block ndarray failed");
                        (block_bb, block_array)
                    }
                    None => {
                        // If no block exists, need to write from its origin.
                        let mut block_bb = write_bb.clone();
                        block_bb
                            .size
                            .iter_mut()
                            .zip(write_bb.offset.iter())
                            .zip(nom_block_bb.offset.iter())
                            .for_each(|((s, o), g)| *s += *o - *g);
                        block_bb.offset = nom_block_bb.offset.clone();
                        let block_size_usize = block_bb.size_ndarray_shape();

                        let block_array =
                            Array::from_elem(&block_size_usize[..], fill_val.clone()).into_dyn();
                        (block_bb, block_array)
                    }
                };

                let block_write_bb = write_bb.clone() - &block_bb.offset;
                let block_slice = block_write_bb.to_ndarray_slice();
                let mut block_view = block_array
                    .slice_mut(SliceInfo::<_, IxDyn>::new(block_slice).unwrap().as_ref());

                block_view.assign(&arr_view);

                block_vec.clear();
                block_vec.extend(block_array.t().iter().cloned());
                let block = VecDataBlock::new(block_bb.size_block(), coord.into(), block_vec);

                self.write_block(path_name, data_attrs, &block)?;
                block_vec = block.into_data();
            }
        }

        Ok(())
    }
}

impl<T: N5Writer> N5NdarrayWriter for T {}

impl DatasetAttributes {
    pub fn coord_iter(&self) -> impl Iterator<Item = Vec<u64>> + ExactSizeIterator {
        let coord_ceil = self
            .get_dimensions()
            .iter()
            .zip(self.get_block_size().iter())
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
            .zip(&self.chunk_grid.block_size)
            .map(|(&o, &bs)| o / u64::from(bs))
            .collect();
        let ceil_coord: GridCoord = bbox
            .offset
            .iter()
            .zip(&bbox.size)
            .zip(self.chunk_grid.block_size.iter().cloned().map(u64::from))
            .map(|((&o, &s), bs)| (o + s + bs - 1) / bs)
            .collect();

        CoordIterator::floor_ceil(&floor_coord, &ceil_coord)
    }

    pub fn get_bounds(&self) -> BoundingBox {
        BoundingBox {
            offset: smallvec![0; self.dimensions.len()],
            size: self.dimensions.clone(),
        }
    }

    pub fn get_block_bounds(&self, coord: &GridCoord) -> BoundingBox {
        let mut size: GridCoord = self
            .get_block_size()
            .iter()
            .cloned()
            .map(u64::from)
            .collect();
        let offset: GridCoord = coord.iter().zip(size.iter()).map(|(c, s)| c * s).collect();
        size.iter_mut()
            .zip(offset.iter())
            .zip(self.get_dimensions().iter())
            .for_each(|((s, o), d)| *s = cmp::min(*s + *o, *d) - *o);
        BoundingBox { offset, size }
    }
}

impl<T: ReflectedType, C> SliceDataBlock<T, C> {
    /// Get the bounding box of the occupied extent of this block, which may
    /// be smaller than the nominal bounding box expected from the dataset.
    pub fn get_bounds(&self, data_attrs: &DatasetAttributes) -> BoundingBox {
        let mut bbox = data_attrs.get_block_bounds(&self.grid_position);
        bbox.size = self.size.iter().cloned().map(u64::from).collect();
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
    use crate::DataType;

    #[test]
    fn test_dataset_attributes_coord_iter() {
        use std::collections::HashSet;

        let data_attrs = DatasetAttributes::new(
            smallvec![1, 4, 5],
            smallvec![1, 2, 3],
            DataType::INT16,
            crate::compression::CompressionType::default(),
        );

        let coords: HashSet<Vec<u64>> = data_attrs.coord_iter().collect();
        let expected: HashSet<Vec<u64>> =
            vec![vec![0, 0, 0], vec![0, 0, 1], vec![0, 1, 0], vec![0, 1, 1]]
                .into_iter()
                .collect();

        assert_eq!(coords, expected);
    }
}
