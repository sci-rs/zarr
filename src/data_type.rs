use half::f16;
use serde::{
    Deserialize,
    Serialize,
};

use crate::chunk::ChunkHeader;
use crate::VecDataChunk;

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum IntSize {
    B1,
    B2,
    B4,
    B8,
}

impl IntSize {
    fn deserial_char(c: char) -> Option<Self> {
        use IntSize::*;
        match c {
            '1' => Some(B1),
            '2' => Some(B2),
            '4' => Some(B4),
            '8' => Some(B8),
            _ => None,
        }
    }

    fn serial_char(&self) -> char {
        use IntSize::*;
        match self {
            B1 => '1',
            B2 => '2',
            B4 => '4',
            B8 => '8',
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum FloatSize {
    B2,
    B4,
    B8,
}

impl FloatSize {
    fn deserial_char(c: char) -> Option<Self> {
        use FloatSize::*;
        match c {
            '2' => Some(B2),
            '4' => Some(B4),
            '8' => Some(B8),
            _ => None,
        }
    }

    fn serial_char(&self) -> char {
        use FloatSize::*;
        match self {
            B2 => '2',
            B4 => '4',
            B8 => '8',
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Endian {
    Big,
    Little,
}

#[cfg(target_endian = "big")]
pub const NATIVE_ENDIAN: Endian = Endian::Big;
#[cfg(target_endian = "little")]
pub const NATIVE_ENDIAN: Endian = Endian::Little;
pub const NETWORK_ENDIAN: Endian = Endian::Big;

impl Endian {
    fn deserial_char(c: char) -> Option<Self> {
        match c {
            '>' => Some(Endian::Big),
            '<' => Some(Endian::Little),
            _ => None,
        }
    }

    fn serial_char(&self) -> char {
        match self {
            Endian::Big => '>',
            Endian::Little => '<',
        }
    }
}

// /// Data types representable in Zarr.
///
/// ```
/// use zarr::data_type::{Endian, IntSize, FloatSize, DataType};
/// use serde_json;
///
/// let d: DataType = serde_json::from_str("\"<f8\"").unwrap();
/// assert_eq!(d, DataType::Float {size: FloatSize::B8, endian: Endian::Little});
/// let d: DataType = serde_json::from_str("\">u4\"").unwrap();
/// assert_eq!(d, DataType::UInt {size: IntSize::B4, endian: Endian::Big});
/// ```
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum DataType {
    Bool,
    Int { size: IntSize, endian: Endian },
    UInt { size: IntSize, endian: Endian },
    Float { size: FloatSize, endian: Endian },
}

impl Serialize for DataType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use DataType::*;
        let mut buf = [0u8; 3];
        let s = match self {
            Bool => "bool",
            Int {
                size: IntSize::B1, ..
            } => "i1",
            UInt {
                size: IntSize::B1, ..
            } => "u1",
            Int { size, endian } => {
                endian.serial_char().encode_utf8(&mut buf[0..1]);
                'i'.encode_utf8(&mut buf[1..2]);
                size.serial_char().encode_utf8(&mut buf[2..3]);
                std::str::from_utf8(&buf[..]).unwrap()
            }
            UInt { size, endian } => {
                endian.serial_char().encode_utf8(&mut buf[0..1]);
                'u'.encode_utf8(&mut buf[1..2]);
                size.serial_char().encode_utf8(&mut buf[2..3]);
                std::str::from_utf8(&buf[..]).unwrap()
            }
            Float { size, endian } => {
                endian.serial_char().encode_utf8(&mut buf[0..1]);
                'f'.encode_utf8(&mut buf[1..2]);
                size.serial_char().encode_utf8(&mut buf[2..3]);
                std::str::from_utf8(&buf[..]).unwrap()
            }
        };
        serializer.serialize_str(s)
    }
}

struct DataTypeVisitor;

impl<'de> serde::de::Visitor<'de> for DataTypeVisitor {
    type Value = DataType;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a string of the format `bool|[<>]?[iuf][1248]`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        // type DataType = DataType;

        let dtype = match value {
            "bool" => DataType::Bool,
            "i1" => DataType::Int {
                size: IntSize::B1,
                endian: Endian::Little,
            },
            "u1" => DataType::UInt {
                size: IntSize::B1,
                endian: Endian::Little,
            },
            dtype @ _ if dtype.len() == 3 => {
                let endian = Endian::deserial_char(dtype.chars().nth(0).unwrap()).expect("TODO");
                match dtype.chars().nth(1).unwrap() {
                    'i' => {
                        let size = IntSize::deserial_char(dtype.chars().nth(2).unwrap()).unwrap();
                        DataType::Int { size, endian }
                    }
                    'u' => {
                        let size = IntSize::deserial_char(dtype.chars().nth(2).unwrap()).unwrap();
                        DataType::UInt { size, endian }
                    }
                    'f' => {
                        let size = FloatSize::deserial_char(dtype.chars().nth(2).unwrap()).unwrap();
                        DataType::Float { size, endian }
                    }
                    _ => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(value),
                            &self,
                        ))
                    }
                }
            }
            // [end @ '<' | '>', num @ 'i' | 'u' | 'f', byte @ '1' ... '8'] => {
            //     match
            // }
            // "<f8" => DataType::Float {
            //     size: FloatSize::B8,
            //     endian: Endian::Little,
            // },
            _ => {
                return Err(serde::de::Error::invalid_value(
                    serde::de::Unexpected::Str(value),
                    &self,
                ))
            }
        };

        Ok(dtype)
    }
}

impl<'de> Deserialize<'de> for DataType {
    fn deserialize<D>(deserializer: D) -> Result<DataType, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(DataTypeVisitor)
    }
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
                $crate::DataType::Bool => $crate::data_type_rstype_replace!(bool, $($expr)*),
                $crate::DataType::UInt {size: IntSize::B1, ..} => $crate::data_type_rstype_replace!(u8, $($expr)*),
                $crate::DataType::UInt {size: IntSize::B2, ..}=> $crate::data_type_rstype_replace!(u16, $($expr)*),
                $crate::DataType::UInt {size: IntSize::B4, ..} => $crate::data_type_rstype_replace!(u32, $($expr)*),
                $crate::DataType::UInt {size: IntSize::B8, ..} => $crate::data_type_rstype_replace!(u64, $($expr)*),
                $crate::DataType::Int {size: IntSize::B1, ..} => $crate::data_type_rstype_replace!(i8, $($expr)*),
                $crate::DataType::Int {size: IntSize::B2, ..}=> $crate::data_type_rstype_replace!(i16, $($expr)*),
                $crate::DataType::Int {size: IntSize::B4, ..} => $crate::data_type_rstype_replace!(i32, $($expr)*),
                $crate::DataType::Int {size: IntSize::B8, ..} => $crate::data_type_rstype_replace!(i64, $($expr)*),
                $crate::DataType::Float {size: FloatSize::B2, ..}=> $crate::data_type_rstype_replace!(f16, $($expr)*),
                $crate::DataType::Float {size: FloatSize::B4, ..} => $crate::data_type_rstype_replace!(f32, $($expr)*),
                $crate::DataType::Float {size: FloatSize::B8, ..} => $crate::data_type_rstype_replace!(f64, $($expr)*),
            }
        }
    };
}

impl DataType {
    /// Boilerplate method for reflection of primitive type sizes.
    pub fn size_of(self) -> usize {
        data_type_match!(self, { std::mem::size_of::<RsType>() })
    }

    pub fn endian(self) -> Endian {
        use DataType::*;
        match self {
            Int { endian, .. } | UInt { endian, .. } | Float { endian, .. } => endian,
            // These are single-byte types.
            _ => NATIVE_ENDIAN,
        }
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
    const ZARR_TYPE: DataType;

    fn create_data_chunk(header: ChunkHeader) -> VecDataChunk<Self> {
        VecDataChunk::<Self>::new(
            header.size,
            header.grid_position,
            vec![Self::default(); header.num_el],
        )
    }
}

macro_rules! reflected_type {
    ($d_name:expr, $d_type:ty) => {
        impl ReflectedType for $d_type {
            const ZARR_TYPE: DataType = $d_name;
        }
    };
}

#[rustfmt::skip] reflected_type!(DataType::Bool, bool);
#[rustfmt::skip] reflected_type!(DataType::UInt {size: IntSize::B1, endian: NATIVE_ENDIAN}, u8);
#[rustfmt::skip] reflected_type!(DataType::UInt {size: IntSize::B2, endian: NATIVE_ENDIAN}, u16);
#[rustfmt::skip] reflected_type!(DataType::UInt {size: IntSize::B4, endian: NATIVE_ENDIAN}, u32);
#[rustfmt::skip] reflected_type!(DataType::UInt {size: IntSize::B8, endian: NATIVE_ENDIAN}, u64);
#[rustfmt::skip] reflected_type!(DataType::Int {size: IntSize::B1, endian: NATIVE_ENDIAN}, i8);
#[rustfmt::skip] reflected_type!(DataType::Int {size: IntSize::B2, endian: NATIVE_ENDIAN}, i16);
#[rustfmt::skip] reflected_type!(DataType::Int {size: IntSize::B4, endian: NATIVE_ENDIAN}, i32);
#[rustfmt::skip] reflected_type!(DataType::Int {size: IntSize::B8, endian: NATIVE_ENDIAN}, i64);
#[rustfmt::skip] reflected_type!(DataType::Float {size: FloatSize::B2, endian: NATIVE_ENDIAN}, f16);
#[rustfmt::skip] reflected_type!(DataType::Float {size: FloatSize::B4, endian: NATIVE_ENDIAN}, f32);
#[rustfmt::skip] reflected_type!(DataType::Float {size: FloatSize::B8, endian: NATIVE_ENDIAN}, f64);

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data_type_reflection<T: ReflectedType>() {
        assert_eq!(std::mem::size_of::<T>(), T::ZARR_TYPE.size_of());
    }

    #[test]
    fn test_all_data_type_reflections() {
        test_data_type_reflection::<bool>();
        test_data_type_reflection::<u8>();
        test_data_type_reflection::<u16>();
        test_data_type_reflection::<u32>();
        test_data_type_reflection::<u64>();
        test_data_type_reflection::<i8>();
        test_data_type_reflection::<i16>();
        test_data_type_reflection::<i32>();
        test_data_type_reflection::<i64>();
        test_data_type_reflection::<f16>();
        test_data_type_reflection::<f32>();
        test_data_type_reflection::<f64>();
    }
}
