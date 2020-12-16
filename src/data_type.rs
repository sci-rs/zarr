use serde::{
    Deserialize,
    Serialize,
};

use crate::ChunkHeader;
use crate::VecDataChunk;

/// Data types representable in Zarr.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    INT8,
    INT16,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
}

/// Replace all RsType tokens with the provide type.
#[macro_export]
macro_rules! data_type_rstype_replace {
    // Open parenthesis.
    ($rstype:ty, @($($stack:tt)*) ($($first:tt)*) $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(() $($stack)*) $($first)* __paren $($rest)*)
    };

    // Open square bracket.
    ($rstype:ty, @($($stack:tt)*) [$($first:tt)*] $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(() $($stack)*) $($first)* __bracket $($rest)*)
    };

    // Open curly brace.
    ($rstype:ty, @($($stack:tt)*) {$($first:tt)*} $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(() $($stack)*) $($first)* __brace $($rest)*)
    };

    // Close parenthesis.
    ($rstype:ty, @(($($close:tt)*) ($($top:tt)*) $($stack:tt)*) __paren $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(($($top)* ($($close)*)) $($stack)*) $($rest)*)
    };

    // Close square bracket.
    ($rstype:ty, @(($($close:tt)*) ($($top:tt)*) $($stack:tt)*) __bracket $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(($($top)* [$($close)*]) $($stack)*) $($rest)*)
    };

    // Close curly brace.
    ($rstype:ty, @(($($close:tt)*) ($($top:tt)*) $($stack:tt)*) __brace $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(($($top)* {$($close)*}) $($stack)*) $($rest)*)
    };

    // Replace `RsType` token with $rstype.
    ($rstype:ty, @(($($top:tt)*) $($stack:tt)*) RsType $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(($($top)* $rstype) $($stack)*) $($rest)*)
    };

    // Munch a token that is not `RsType`.
    ($rstype:ty, @(($($top:tt)*) $($stack:tt)*) $first:tt $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(($($top)* $first) $($stack)*) $($rest)*)
    };

    // Terminal case.
    ($rstype:ty, @(($($top:tt)+))) => {
        $($top)+
    };

    // Initial case.
    ($rstype:ty, $($input:tt)+) => {
        data_type_rstype_replace!($rstype, @(()) $($input)*)
    };
}

/// Match a DataType-valued expression, and in each arm repeat the provided
/// code chunk with the token `RsType` replaced with the primitive type
/// appropriate for that arm.
#[macro_export]
macro_rules! data_type_match {
    ($match_expr:expr, $($expr:tt)*) => {
        {
            match $match_expr {
                $crate::DataType::UINT8 => $crate::data_type_rstype_replace!(u8, $($expr)*),
                $crate::DataType::UINT16 => $crate::data_type_rstype_replace!(u16, $($expr)*),
                $crate::DataType::UINT32 => $crate::data_type_rstype_replace!(u32, $($expr)*),
                $crate::DataType::UINT64 => $crate::data_type_rstype_replace!(u64, $($expr)*),
                $crate::DataType::INT8 => $crate::data_type_rstype_replace!(i8, $($expr)*),
                $crate::DataType::INT16 => $crate::data_type_rstype_replace!(i16, $($expr)*),
                $crate::DataType::INT32 => $crate::data_type_rstype_replace!(i32, $($expr)*),
                $crate::DataType::INT64 => $crate::data_type_rstype_replace!(i64, $($expr)*),
                $crate::DataType::FLOAT32 => $crate::data_type_rstype_replace!(f32, $($expr)*),
                $crate::DataType::FLOAT64 => $crate::data_type_rstype_replace!(f64, $($expr)*),
            }
        }
    };
}

impl DataType {
    /// Boilerplate method for reflection of primitive type sizes.
    pub fn size_of(self) -> usize {
        data_type_match!(self, { std::mem::size_of::<RsType>() })
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Trait implemented by primitive types that are reflected in Zarr.
///
/// The supertraits are not necessary for this trait, but are used to
/// remove redundant bounds elsewhere when operating generically over
/// data types.
pub trait ReflectedType: Send + Sync + Clone + Default + 'static {
    const VARIANT: DataType;

    fn create_data_chunk(header: ChunkHeader) -> VecDataChunk<Self> {
        VecDataChunk::<Self>::new(
            header.size,
            header.grid_position,
            vec![Self::default(); header.num_el],
        )
    }
}

macro_rules! reflected_type {
    ($d_name:ident, $d_type:ty) => {
        impl ReflectedType for $d_type {
            const VARIANT: DataType = DataType::$d_name;
        }
    };
}

reflected_type!(UINT8, u8);
reflected_type!(UINT16, u16);
reflected_type!(UINT32, u32);
reflected_type!(UINT64, u64);
reflected_type!(INT8, i8);
reflected_type!(INT16, i16);
reflected_type!(INT32, i32);
reflected_type!(INT64, i64);
reflected_type!(FLOAT32, f32);
reflected_type!(FLOAT64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data_type_reflection<T: ReflectedType>() {
        assert_eq!(std::mem::size_of::<T>(), T::VARIANT.size_of());
    }

    #[test]
    fn test_all_data_type_reflections() {
        test_data_type_reflection::<u8>();
        test_data_type_reflection::<u16>();
        test_data_type_reflection::<u32>();
        test_data_type_reflection::<u64>();
        test_data_type_reflection::<i8>();
        test_data_type_reflection::<i16>();
        test_data_type_reflection::<i32>();
        test_data_type_reflection::<i64>();
        test_data_type_reflection::<f32>();
        test_data_type_reflection::<f64>();
    }
}
