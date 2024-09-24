//! A minimal self-contained implementation of the types of the [willow data model]
//!
//! This is used in tests and examples.
//!
//! [willow data model]: https://willowprotocol.org/specs/data-model/index.html
use std::{
    fmt::{Debug, Display, Formatter},
    time::SystemTime,
};

use zerocopy::{AsBytes, FromBytes, FromZeroes};

use crate::{
    fmt::NoQuotes, BlobSeq, BlobSeqRef, FixedSize, IsLowerBound, KeyParams,
    LiftingCommutativeMonoid, LowerBound, Node, Point, PointRef, TreeParams,
};

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, FromBytes, AsBytes, FromZeroes)]
#[repr(transparent)]
pub struct Subspace([u8; 32]);

impl IsLowerBound for Subspace {
    fn is_min_value(&self) -> bool {
        self.0 == [0; 32]
    }
}

impl LowerBound for Subspace {
    fn min_value() -> Self {
        Self::ZERO
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct WillowTreeParams;

pub type TNode = Node<WillowTreeParams>;

pub type TPoint = Point<WillowTreeParams>;

impl KeyParams for WillowTreeParams {
    // timestamp
    type X = Subspace;
    // subspace
    type Y = Timestamp;
    // path
    type Z = BlobSeqRef;
    // owned path
    type ZOwned = BlobSeq;
}

impl TreeParams for WillowTreeParams {
    type V = WillowValue;
    type M = Fingerprint;
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, FromBytes, AsBytes, FromZeroes)]
#[repr(transparent)]
pub struct Blake3Hash([u8; 32]);

impl Blake3Hash {
    /// the lowest possible subspace
    pub const ZERO: Self = Self([0; 32]);
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

impl IsLowerBound for Timestamp {
    fn is_min_value(&self) -> bool {
        self.0 == [0; 8]
    }
}

impl LowerBound for Timestamp {
    fn min_value() -> Self {
        Self::ZERO
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
