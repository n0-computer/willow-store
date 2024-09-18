use core::str;
use std::{
    borrow::Borrow,
    fmt::{Debug, Display, Formatter},
    str::FromStr,
    sync::Arc,
};

use rand::Rng;
use ref_cast::RefCast;
use zerocopy::{AsBytes, FromBytes, FromZeroes};

use crate::{
    FixedSize, KeyParams, LiftingCommutativeMonoid, LowerBound, NoQuotes, PointRef, RefFromSlice,
    TreeParams, VariableSize,
};

const ESCAPE: u8 = 1;
const SEPARATOR: u8 = 0;

fn escape_into<P, C>(path: P, result: &mut Vec<u8>)
where
    P: AsRef<[C]>,
    C: AsRef<[u8]>,
{
    result.push(0);
    for segment in path.as_ref() {
        for &byte in segment.as_ref() {
            match byte {
                ESCAPE => result.extend([ESCAPE, ESCAPE]),
                SEPARATOR => result.extend([ESCAPE, SEPARATOR]),
                _ => result.push(byte),
            }
        }
        result.push(SEPARATOR);
    }
}

fn escape<P, C>(path: P) -> Vec<u8>
where
    P: AsRef<[C]>,
    C: AsRef<[u8]>,
{
    let mut result = Vec::new();
    escape_into(path, &mut result);
    result
}

fn unescape(path: &[u8]) -> Vec<Vec<u8>> {
    let mut components = Vec::new();
    let mut segment = Vec::new();
    let mut escape = false;
    assert!(path.len() > 0 && path[0] == 0);
    let path = &path[1..];
    for &byte in path {
        if escape {
            segment.push(byte);
            escape = false;
        } else {
            match byte {
                ESCAPE => escape = true,
                SEPARATOR => {
                    components.push(segment);
                    segment = Vec::new();
                }
                _ => segment.push(byte),
            }
        }
    }
    components
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct Path {
    escaped: Arc<[u8]>,
    components: Vec<Vec<u8>>,
}

impl LowerBound for Path {
    fn min_value() -> Self {
        Self::from(vec![])
    }

    fn is_min_value(&self) -> bool {
        self.components.is_empty()
    }
}

/// Formats a single component.
/// - If the component is valid UTF-8, it is enclosed in quotes, and any quotes within
///   the string are escaped.
/// - If the component is not valid UTF-8, it is formatted as hexadecimal.
fn format_component(component: &[u8]) -> String {
    if let Ok(s) = str::from_utf8(component) {
        if s.chars()
            .all(|c| c.is_ascii_graphic() && c != '/' && c != '"')
        {
            // Valid UTF-8
            // Escape any quotes
            return format!("\"{s}\"");
        }
    }
    // Not valid UTF-8, or not printable, or contains component separator
    // Format as hex
    hex::encode(component)
}

fn parse_component(s: &str) -> anyhow::Result<Vec<u8>> {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') {
        // Remove quotes
        let s = &s[1..s.len() - 1];
        return Ok(s.as_bytes().to_vec());
    }
    // Parse as hex
    Ok(hex::decode(s)?)
}

impl Display for Path {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.components
                .iter()
                .map(|x| format_component(&x))
                .collect::<Vec<_>>()
                .join("/")
        )
    }
}

impl FromStr for Path {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let components: Vec<Vec<u8>> = s
            .split('/')
            .map(parse_component)
            .collect::<anyhow::Result<Vec<Vec<_>>>>()?;
        Ok(components.into())
    }
}

impl From<Vec<Vec<u8>>> for Path {
    fn from(components: Vec<Vec<u8>>) -> Self {
        let escaped = escape(&components).into();
        Path {
            components,
            escaped,
        }
    }
}

impl From<&[&[u8]]> for Path {
    fn from(components: &[&[u8]]) -> Self {
        let components: Vec<Vec<u8>> = components.iter().map(|&c| c.to_vec()).collect();
        components.into()
    }
}

impl From<&[&str]> for Path {
    fn from(components: &[&str]) -> Self {
        let components: Vec<Vec<u8>> = components.iter().map(|&c| c.as_bytes().to_vec()).collect();
        components.into()
    }
}

impl AsRef<[Vec<u8>]> for Path {
    fn as_ref(&self) -> &[Vec<u8>] {
        &self.components
    }
}

impl Borrow<PathRef> for Path {
    fn borrow(&self) -> &PathRef {
        PathRef::ref_cast(&self.escaped)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, FromBytes, FromZeroes, AsBytes, RefCast)]
#[repr(transparent)]
pub struct PathRef([u8]);

impl Debug for PathRef {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "PathRef({})", hex::encode(&self.0))
        } else {
            write!(f, "{}", self)
        }
    }
}

impl Display for PathRef {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let components = unescape(&self.0);
        write!(
            f,
            "{}",
            components
                .iter()
                .map(|x| format_component(&x))
                .collect::<Vec<_>>()
                .join("/")
        )
    }
}

impl ToOwned for PathRef {
    type Owned = Path;

    fn to_owned(&self) -> Self::Owned {
        let escaped = self.0.to_vec().into();
        Path {
            escaped,
            components: unescape(&self.0),
        }
    }
}

impl RefFromSlice for PathRef {
    fn ref_from_slice(slice: &[u8]) -> &Self {
        Self::ref_cast(slice)
    }
}

impl VariableSize for PathRef {
    fn size(&self) -> usize {
        self.0.len()
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, FromBytes, AsBytes, FromZeroes)]
#[repr(transparent)]
pub struct Subspace([u8; 32]);

impl LowerBound for Subspace {
    fn min_value() -> Self {
        Self::ZERO
    }

    fn is_min_value(&self) -> bool {
        self.0 == [0; 32]
    }
}

/// A way to create a subspace from a u64, just for testing
impl From<u64> for Subspace {
    fn from(value: u64) -> Self {
        let mut bytes = [0; 32];
        bytes[24..].copy_from_slice(&value.to_be_bytes());
        Self(bytes)
    }
}

impl Subspace {
    /// the lowest possible subspace
    pub const ZERO: Self = Self([0; 32]);

    pub fn random() -> Self {
        let mut bytes = [0; 32];
        rand::thread_rng().fill(&mut bytes);
        Self(bytes)
    }
}

impl FixedSize for Subspace {
    const SIZE: usize = 32;
}

impl Debug for Subspace {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "Subspace({})", hex::encode(self.0))
    }
}

impl Display for Subspace {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, FromBytes, AsBytes, FromZeroes)]
#[repr(transparent)]
pub struct Blake3Hash([u8; 32]);

impl Blake3Hash {
    /// the lowest possible subspace
    pub const ZERO: Self = Self([0; 32]);

    pub fn random() -> Self {
        let mut bytes = [0; 32];
        rand::thread_rng().fill(&mut bytes);
        Self(bytes)
    }
}

impl FixedSize for Blake3Hash {
    const SIZE: usize = 32;
}

impl Debug for Blake3Hash {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "Blake3Hash({})", hex::encode(self.0))
    }
}

impl Display for Blake3Hash {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, FromBytes, AsBytes, FromZeroes)]
#[repr(transparent)]
pub struct Timestamp([u8; 8]);

impl LowerBound for Timestamp {
    fn min_value() -> Self {
        Self::ZERO
    }

    fn is_min_value(&self) -> bool {
        self.0 == [0; 8]
    }
}

impl FixedSize for Timestamp {
    const SIZE: usize = 8;
}

impl From<u64> for Timestamp {
    fn from(value: u64) -> Self {
        Self(value.to_be_bytes())
    }
}

impl From<Timestamp> for u64 {
    fn from(value: Timestamp) -> Self {
        u64::from_be_bytes(value.0)
    }
}

impl Debug for Timestamp {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "Timestamp({})", u64::from(*self))
    }
}

impl Display for Timestamp {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let x = u64::from(*self);
        let s = x / 1000000;
        let us = x % 1000000;
        write!(f, "{}.{:06}", s, us)
    }
}

impl Timestamp {
    pub const ZERO: Self = Self([0; 8]);

    pub fn now() -> Self {
        let now = std::time::SystemTime::now();
        let since_epoch = now.duration_since(std::time::UNIX_EPOCH).unwrap();
        let secs = since_epoch.as_secs();
        let micros = since_epoch.subsec_micros();
        Self::from(secs * 1000000 + micros as u64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, AsBytes, FromZeroes)]
#[repr(packed)]
pub struct WillowValue {
    hash: Blake3Hash,
    size: u64,
}

impl WillowValue {
    pub fn hash(data: &[u8]) -> Self {
        let hash = Blake3Hash(blake3::hash(data).into());
        let size = data.len() as u64;
        Self { hash, size }
    }
}

impl FixedSize for WillowValue {
    const SIZE: usize = Blake3Hash::SIZE + 8;
}

#[derive(Clone, Copy, PartialEq, Eq, FromBytes, AsBytes, FromZeroes)]
#[repr(packed)]
pub struct Fingerprint([u8; 32]);

impl Debug for Fingerprint {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_tuple("")
            .field(&NoQuotes(&hex::encode(&self.0)))
            .finish()
    }
}

impl Fingerprint {
    pub const ZERO: Self = Self([0; 32]);
}

impl FixedSize for Fingerprint {
    const SIZE: usize = Blake3Hash::SIZE + 8;
}

impl LiftingCommutativeMonoid<PointRef<WillowTreeParams>, WillowValue> for Fingerprint {
    fn neutral() -> Self {
        Self::ZERO
    }

    fn lift(_key: &PointRef<WillowTreeParams>, value: &WillowValue) -> Self {
        Self(value.hash.0)
    }

    fn combine(&self, other: &Self) -> Self {
        let mut xor = [0; 32];
        for i in 0..32 {
            xor[i] = self.0[i] ^ other.0[i];
        }
        Self(xor)
    }
}

#[derive(Debug)]
pub struct WillowTreeParams;

impl KeyParams for WillowTreeParams {
    // timestamp
    type X = Subspace;
    // subspace
    type Y = Timestamp;
    // path
    type Z = PathRef;
    // owned path
    type ZOwned = Path;
}

impl TreeParams for WillowTreeParams {
    type V = WillowValue;
    type M = Fingerprint;
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, FromBytes, AsBytes, FromZeroes)]
#[repr(transparent)]
pub struct ValueSum(u64);

impl FixedSize for ValueSum {
    const SIZE: usize = 8;
}

impl LiftingCommutativeMonoid<PointRef<WillowTreeParams>, u64> for ValueSum {
    fn neutral() -> Self {
        Self(0)
    }

    fn lift(_key: &PointRef<WillowTreeParams>, value: &u64) -> Self {
        Self(*value)
    }

    fn combine(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use test_strategy::proptest;
    use testresult::TestResult;

    fn path_value_range() -> std::ops::Range<u8> {
        // having more than 3 or 4 values does not add much value to the test
        // 0 and 1 are special values (escape and separator)
        // 3 and 4 are normal values
        0..4
    }

    fn arb_path(max_components: usize, max_component_size: usize) -> impl Strategy<Value = Path> {
        prop::collection::vec(
            prop::collection::vec(path_value_range(), 0..max_component_size),
            0..max_components,
        )
        .prop_map(|components| Path::from(components))
    }

    impl Arbitrary for Path {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb_path(4, 12).boxed()
        }
    }

    #[proptest]
    fn test_escape_roundtrip(path: Path) {
        let path = path;
        let escaped = escape(&path);
        let unescaped = unescape(&escaped).into();
        assert_eq!(path, unescaped);
    }

    #[proptest]
    fn test_escape_preserves_order(a: Path, b: Path) {
        let ae = escape(&a);
        let be = escape(&b);
        assert_eq!(ae.cmp(&be), a.components.cmp(&b.components));
    }

    #[test]
    fn format_test() {
        let a = Path::from(["a", "b", "c"].as_ref());
        let b = Path::from([[01u8].as_ref(), &[02, 03], &[04, 05, 06]].as_ref());
        let c = Path::from_str(r#""a"/"b"/01020304"#).unwrap();
        println!("{}", a);
        println!("{}", b);
        println!("{}", c);
    }
}
