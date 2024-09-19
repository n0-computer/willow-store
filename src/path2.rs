use core::str;
use std::{
    borrow::Borrow,
    cmp::Ordering,
    fmt::{Debug, Display, Formatter},
    str::FromStr,
    sync::Arc,
    time::SystemTime,
};

use rand::Rng;
use ref_cast::RefCast;
use zerocopy::{AsBytes, FromBytes, FromZeroes};

use crate::{FixedSize, LowerBound, NoQuotes, RefFromSlice, VariableSize};

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

const ESCAPE: u8 = 1;
const SEPARATOR: u8 = 0;

fn escape_into<I, C>(components: I, result: &mut Vec<u8>)
where
    I: IntoIterator<Item = C>,
    C: AsRef<[u8]>,
{
    for segment in components.into_iter() {
        for &byte in segment.as_ref() {
            match byte {
                ESCAPE => result.extend([ESCAPE, ESCAPE]),
                SEPARATOR => result.extend([ESCAPE, SEPARATOR]),
                _ => result.push(byte),
            }
        }
        result.push(SEPARATOR);
    }
    // you might think that the trailing separator is unnecessary, but it is needed
    // to distinguish between the empty path and the path with one empty component
}

fn escape<I, C>(components: I) -> Vec<u8>
where
    I: IntoIterator<Item = C>,
    C: AsRef<[u8]>,
{
    let mut result = Vec::new();
    escape_into(components, &mut result);
    result
}

fn unescape(path: &[u8]) -> Vec<Vec<u8>> {
    let mut components = Vec::new();
    let mut segment = Vec::new();
    let mut escape = false;
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

pub struct Path2 {
    /// data, containing the escaped path concatenated with the unescaped components(*)
    ///
    /// data layout:
    ///   0..count * 4: start and end offsets of each component, as u16 big-endian
    ///   count * 4..escaped_start: unescaped components, if needed
    ///   escaped_start..: escaped path
    data: Arc<[u8]>,
    /// start of the escaped path in the data
    escaped_start: u16,
    /// number of components in the path
    count: u16,
}

impl From<Vec<Vec<u8>>> for Path2 {
    fn from(components: Vec<Vec<u8>>) -> Self {
        let escaped = escape(components.iter());
        Self::from_escaped(&escaped)
    }
}

impl From<&[&[u8]]> for Path2 {
    fn from(components: &[&[u8]]) -> Self {
        let components: Vec<Vec<u8>> = components.iter().map(|&c| c.to_vec()).collect();
        components.into()
    }
}

impl From<&[&str]> for Path2 {
    fn from(components: &[&str]) -> Self {
        let components: Vec<Vec<u8>> = components.iter().map(|&c| c.as_bytes().to_vec()).collect();
        components.into()
    }
}

impl Borrow<PathRef> for Path2 {
    fn borrow(&self) -> &PathRef {
        PathRef::ref_cast(self.escaped())
    }
}

impl PartialOrd for Path2 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.escaped().partial_cmp(other.escaped())
    }
}

impl Ord for Path2 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.escaped().cmp(other.escaped())
    }
}

impl PartialEq for Path2 {
    fn eq(&self, other: &Self) -> bool {
        self.escaped() == other.escaped()
    }
}

impl Eq for Path2 {}

impl Path2 {
    fn escaped(&self) -> &[u8] {
        let start = self.escaped_start as usize;
        &self.data[start..]
    }

    fn components(&self) -> impl Iterator<Item = &[u8]> {
        let data = &self.data;
        let count = self.count as usize;
        (0..count).map(move |i| {
            let base = i * 4;
            let start = u16::from_be_bytes(data[base..base + 2].try_into().unwrap()) as usize;
            let end = u16::from_be_bytes(data[base + 2..base + 4].try_into().unwrap()) as usize;
            &data[start..end]
        })
    }

    fn from_escaped(escaped: &[u8]) -> Self {
        let mut escape = false;
        let mut res = Vec::new();
        let mut sizes = smallvec::SmallVec::<[(u16, u16, bool); 8]>::new();
        let mut unescaped_start = res.len() as u16;
        let mut escaped_start = 0u16;
        for i in 0..escaped.len() {
            let byte = escaped[i];
            if escape {
                res.push(byte);
                escape = false;
            } else {
                match byte {
                    ESCAPE => escape = true,
                    SEPARATOR => {
                        let escaped_end = i as u16;
                        let unescaped_end = res.len() as u16;
                        if unescaped_end - unescaped_start == escaped_end - escaped_start {
                            sizes.push((escaped_start, escaped_end, false));
                            res.truncate(unescaped_start as usize);
                        } else {
                            sizes.push((unescaped_start, unescaped_end, true));
                        }
                        unescaped_start = res.len() as u16;
                        escaped_start = (i + 1) as u16;
                    }
                    _ => res.push(byte),
                }
            }
        }
        let count = sizes.len();
        let sizes_len = (count * 4) as u16;
        // make room for the sizes, we need 4 bytes per component
        res.splice(0..0, (0..sizes_len).map(|_| 0u8));
        let escaped_start = res.len() as u16;
        res.extend_from_slice(escaped);
        // adjust the offsets and store them in the allocated space
        for (i, (mut start, mut end, unescaped)) in sizes.into_iter().enumerate() {
            if unescaped {
                start += sizes_len;
                end += sizes_len;
            } else {
                start += escaped_start;
                end += escaped_start;
            }
            let base = 4 * i;
            res[base..base + 2].copy_from_slice(&(start as u16).to_be_bytes());
            res[base + 2..base + 4].copy_from_slice(&(end as u16).to_be_bytes());
        }
        let res = Self {
            data: res.into(),
            escaped_start,
            count: count as u16,
        };
        res
    }
}

impl Debug for Path2 {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "Path2({})", self)
    }
}

impl Display for Path2 {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.components()
                .map(|x| format_component(&x))
                .collect::<Vec<_>>()
                .join("/")
        )
    }
}

impl FromStr for Path2 {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let components: Vec<Vec<u8>> = s
            .split('/')
            .map(parse_component)
            .collect::<anyhow::Result<Vec<Vec<_>>>>()?;
        Ok(components.into())
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
    type Owned = Path2;

    fn to_owned(&self) -> Self::Owned {
        Path2::from_escaped(&self.0)
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

impl From<SystemTime> for Timestamp {
    fn from(time: SystemTime) -> Self {
        if let Ok(since_epoch) = time.duration_since(std::time::UNIX_EPOCH) {
            let secs = since_epoch.as_secs();
            let micros = since_epoch.subsec_micros();
            Self::from(secs * 1000000 + micros as u64)
        } else {
            Self::ZERO
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use test_strategy::proptest;

    fn component_value_range() -> std::ops::Range<u8> {
        // having more than 3 or 4 values does not add much value to the test
        // 0 and 1 are special values (escape and separator)
        // 3 and 4 are normal values
        0..4
    }

    fn arb_components(
        max_components: usize,
        max_component_size: usize,
    ) -> impl Strategy<Value = Components> {
        prop::collection::vec(
            prop::collection::vec(component_value_range(), 0..max_component_size),
            0..max_components,
        )
        .prop_map(|components| Components(components))
    }

    #[derive(Debug, Clone)]
    struct Components(Vec<Vec<u8>>);

    impl Arbitrary for Components {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb_components(4, 12).boxed()
        }
    }

    #[proptest]
    fn test_escape_roundtrip(comp: Components) {
        let comp = comp.0;
        let escaped = escape(&comp);
        let unescaped = unescape(&escaped);
        assert_eq!(comp, unescaped);
    }

    #[proptest]
    fn test_escape_preserves_order(a: Components, b: Components) {
        let ac = a.0;
        let bc = b.0;
        let ae = escape(&ac);
        let be = escape(&bc);
        assert_eq!(ae.cmp(&be), ac.cmp(&bc));
    }

    fn path_escape_roundtrip_impl(c: Components) {
        let c = c.0;
        let escaped = escape(&c);
        println!("{:?} {}", c, hex::encode(&escaped));
        let path = Path2::from_escaped(&escaped);
        let c2 = path.components().map(|x| x.to_vec()).collect::<Vec<_>>();
        assert_eq!(c, c2);
        println!("{:?} {:?}", c, path);
    }

    #[proptest]
    fn prop_path_escape_roundtrip(c: Components) {
        path_escape_roundtrip_impl(c);
    }

    #[test]
    fn test_path_escape_roundtrip() {
        let cases = vec![
            // vec![vec![0,0]],
            vec![vec![2]],
        ];
        for case in cases {
            path_escape_roundtrip_impl(Components(case));
        }
    }

    #[test]
    fn format_test() {
        let a = Path2::from(["a", "b", "c"].as_ref());
        let b = Path2::from([[01u8].as_ref(), &[02, 03], &[04, 05, 06]].as_ref());
        let c = Path2::from_str(r#""a"/"b"/01020304"#).unwrap();
        println!("{}", a);
        println!("{}", b);
        println!("{}", c);
    }
}
