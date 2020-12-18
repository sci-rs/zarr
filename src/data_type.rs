use std::io::Write;

use half::f16;
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    GridCoord,
    MetadataError,
    VecDataChunk,
};

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

/// Data types representable in Zarr.
///
/// ```
/// use zarr::data_type::{Endian, IntSize, FloatSize, DataType};
/// use serde_json;
///
/// let d: DataType = serde_json::from_str("\"<f8\"").unwrap();
/// assert_eq!(d, DataType::Float {size: FloatSize::B8, endian: Endian::Little});
/// let d: DataType = serde_json::from_str("\">u4\"").unwrap();
/// assert_eq!(d, DataType::UInt {size: IntSize::B4, endian: Endian::Big});
/// let d: DataType = serde_json::from_str("\"r24\"").unwrap();
/// assert_eq!(d, DataType::Raw {size: 24});
/// ```
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum DataType {
    Bool,
    Int { size: IntSize, endian: Endian },
    UInt { size: IntSize, endian: Endian },
    Float { size: FloatSize, endian: Endian },
    Raw { size: usize },
}

impl Serialize for DataType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use DataType::*;
        let mut buf = [0u8; 32];
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
                std::str::from_utf8(&buf[..3]).unwrap()
            }
            UInt { size, endian } => {
                endian.serial_char().encode_utf8(&mut buf[0..1]);
                'u'.encode_utf8(&mut buf[1..2]);
                size.serial_char().encode_utf8(&mut buf[2..3]);
                std::str::from_utf8(&buf[..3]).unwrap()
            }
            Float { size, endian } => {
                endian.serial_char().encode_utf8(&mut buf[0..1]);
                'f'.encode_utf8(&mut buf[1..2]);
                size.serial_char().encode_utf8(&mut buf[2..3]);
                std::str::from_utf8(&buf[..3]).unwrap()
            }
            Raw { size } => {
                write!(&mut buf[..], "r{}", size).expect("TODO");
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
            dtype if dtype.starts_with('r') => {
                if let Ok(size) = dtype[1..].parse::<usize>() {
                    if size % 8 == 0 {
                        DataType::Raw { size }
                    } else {
                        // TODO: more specific error?
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(value),
                            &self,
                        ));
                    }
                } else {
                    return Err(serde::de::Error::invalid_value(
                        serde::de::Unexpected::Str(value),
                        &self,
                    ));
                }
            }
            dtype if dtype.len() == 3 => {
                let mut chars = dtype.chars();
                let endian = Endian::deserial_char(chars.next().unwrap()).expect("TODO");
                match chars.next().unwrap() {
                    'i' => {
                        let size = IntSize::deserial_char(chars.next().unwrap()).unwrap();
                        DataType::Int { size, endian }
                    }
                    'u' => {
                        let size = IntSize::deserial_char(chars.next().unwrap()).unwrap();
                        DataType::UInt { size, endian }
                    }
                    'f' => {
                        let size = FloatSize::deserial_char(chars.next().unwrap()).unwrap();
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

/// TODO
///
/// ```
/// use zarr::data_type::{
///     DataType,
///     IntSize,
///     Endian,
///     ExtensibleDataType,
///     FloatSize,
/// };
/// use serde_json;
///
/// let example_1: ExtensibleDataType = serde_json::from_str(r#""<f8""#).unwrap();
/// let ext_1 = ExtensibleDataType::Core(DataType::Float { size: FloatSize::B8, endian: Endian::Little });
/// assert_eq!(example_1, ext_1);
///
/// let example_2: ExtensibleDataType = serde_json::from_str(r#"
///     {
///        "extension": "https://purl.org/zarr/spec/protocol/extensions/datetime-dtypes/1.0",
///        "type": "<M8[ns]",
///        "fallback": "<i8"
///     }"#).unwrap();
/// let ext_2 = ExtensibleDataType::Extended {
///     extension: "https://purl.org/zarr/spec/protocol/extensions/datetime-dtypes/1.0".to_owned(),
///     type_string: "<M8[ns]".to_owned(),
///     fallback: Some(DataType::Int {size: IntSize::B8, endian: Endian::Little})
/// };
/// assert_eq!(example_2, ext_2);
/// ```
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExtensibleDataType {
    Core(DataType),
    Extended {
        extension: String,
        #[serde(rename = "type")]
        type_string: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        fallback: Option<DataType>,
    },
}

// TODO: needs to be a trait generalizing over core and extended data types
// providing sizing and other operations.

impl ExtensibleDataType {
    // TODO: kludge that will need to be replaced to support extended types.
    pub fn effective_type(&self) -> Result<DataType, MetadataError> {
        use ExtensibleDataType::*;
        match self {
            Core(d) => Ok(*d),
            Extended {
                fallback: Some(d), ..
            } => Ok(*d),
            _ => todo!(),
        }
    }
}

impl From<DataType> for ExtensibleDataType {
    fn from(d: DataType) -> Self {
        Self::Core(d)
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
    ($match_expr:expr, $raw_match:pat => $raw_expr:expr, $($expr:tt)*) => {
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
                $raw_match => $raw_expr,
            }
        }
    };
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
                $crate::DataType::Raw { .. } => $crate::data_type_rstype_replace!([u8], $($expr)*),
            }
        }
    };
}

impl DataType {
    /// Boilerplate method for reflection of primitive type sizes.
    pub fn size_of(self) -> usize {
        data_type_match!(self, DataType::Raw { size } => { std::mem::size_of::<u8>() * size / 8 }, {
            std::mem::size_of::<RsType>()
        })
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
// `DeserializedOwned` is necessary for deserialization of metadata `fill_value`.
pub trait ReflectedType:
    Send + Sync + Clone + Default + serde::de::DeserializeOwned + 'static
{
    const ZARR_TYPE: DataType;

    fn create_data_chunk(grid_position: &GridCoord, num_el: u32) -> VecDataChunk<Self> {
        VecDataChunk::<Self>::new(
            grid_position.clone(),
            vec![Self::default(); num_el as usize],
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

// TODO: As example
#[rustfmt::skip] reflected_type!(DataType::Raw {size: 8}, [u8; 1]);
#[rustfmt::skip] reflected_type!(DataType::Raw {size: 16}, [u8; 2]);
#[rustfmt::skip] reflected_type!(DataType::Raw {size: 24}, [u8; 3]);
#[rustfmt::skip] reflected_type!(DataType::Raw {size: 32}, [u8; 4]);

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
        test_data_type_reflection::<[u8; 1]>();
        test_data_type_reflection::<[u8; 2]>();
        test_data_type_reflection::<[u8; 3]>();
        test_data_type_reflection::<[u8; 4]>();
    }
}
