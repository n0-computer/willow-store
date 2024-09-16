use crate::{count_trailing_zeros, FixedSize, KeyParams, RefFromSlice, SortOrder, VariableSize};
use ref_cast::RefCast;
use std::{borrow::Borrow, cmp::Ordering, fmt::Debug, marker::PhantomData, ops::Deref, sync::Arc};
use zerocopy::{AsBytes, FromBytes};

/// An unsized type that represents a point in 3D space.
///
/// The first two fields are fixed size, while the third field can be of variable size.
///
/// Implements Ord and PartialOrd for convenience, however, if you want a specific order,
/// there are comparison functions that take a `SortOrder` enum.
#[repr(transparent)]
#[derive(RefCast)]
pub struct Point<P: KeyParams>(PhantomData<P>, [u8]);

impl<K: KeyParams> PartialEq for Point<K> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<K: KeyParams> Eq for Point<K> {}

impl<K: KeyParams> PartialOrd for Point<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<K: KeyParams> Ord for Point<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

pub struct OwnedPoint<P: KeyParams>(PhantomData<P>, Arc<[u8]>);

impl<P: KeyParams> Clone for OwnedPoint<P> {
    fn clone(&self) -> Self {
        OwnedPoint(PhantomData, self.1.clone())
    }
}

impl<P: KeyParams> Borrow<Point<P>> for OwnedPoint<P> {
    fn borrow(&self) -> &Point<P> {
        Point::ref_cast(&self.1)
    }
}

impl<P: KeyParams> OwnedPoint<P> {
    pub fn new(x: &P::X, y: &P::Y, z: &P::Z) -> Self {
        let mut buf = vec![0; P::X::SIZE + P::Y::SIZE + z.size()];
        x.write_to_prefix(&mut buf[0..]).unwrap();
        y.write_to_prefix(&mut buf[P::X::SIZE..]).unwrap();
        z.write_to_prefix(&mut buf[P::X::SIZE + P::Y::SIZE..])
            .unwrap();
        OwnedPoint(PhantomData, buf.into())
    }
}

impl<P: KeyParams> Debug for OwnedPoint<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}

impl<P: KeyParams> Deref for OwnedPoint<P> {
    type Target = Point<P>;

    fn deref(&self) -> &Self::Target {
        self.borrow()
    }
}

impl<P: KeyParams> PartialOrd for OwnedPoint<P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<P: KeyParams> Ord for OwnedPoint<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<P: KeyParams> PartialEq for OwnedPoint<P> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<P: KeyParams> Eq for OwnedPoint<P> {}

impl<P: KeyParams> ToOwned for Point<P> {
    type Owned = OwnedPoint<P>;

    fn to_owned(&self) -> Self::Owned {
        OwnedPoint(PhantomData, self.1.to_vec().into())
    }
}

impl<P: KeyParams> Point<P> {
    pub fn size(&self) -> usize {
        self.1.len()
    }

    pub fn rank(&self) -> u8 {
        let hash = blake3::hash(&self.1);
        count_trailing_zeros(hash.as_bytes())
    }

    pub fn x(&self) -> &P::X {
        P::X::ref_from_prefix(&self.1[0..]).unwrap()
    }

    pub fn y(&self) -> &P::Y {
        P::Y::ref_from_prefix(&self.1[P::X::SIZE..]).unwrap()
    }

    pub fn z(&self) -> &P::Z {
        P::Z::ref_from_slice(&self.1[P::X::SIZE + P::Y::SIZE..])
    }

    fn cmp_xyz(&self, other: &Self) -> std::cmp::Ordering {
        self.x()
            .cmp(other.x())
            .then(self.y().cmp(other.y()))
            .then(self.z().cmp(other.z()))
    }

    fn cmp_yzx(&self, other: &Self) -> std::cmp::Ordering {
        self.y()
            .cmp(other.y())
            .then(self.z().cmp(other.z()))
            .then(self.x().cmp(other.x()))
    }

    fn cmp_zxy(&self, other: &Self) -> std::cmp::Ordering {
        self.z()
            .cmp(other.z())
            .then(self.x().cmp(other.x()))
            .then(self.y().cmp(other.y()))
    }

    pub fn cmp_at_rank(&self, other: &Self, rank: u8) -> std::cmp::Ordering {
        self.cmp_with_order(other, SortOrder::from(rank))
    }

    pub fn cmp_with_order(&self, other: &Self, order: SortOrder) -> std::cmp::Ordering {
        match order {
            SortOrder::ZXY => self.cmp_zxy(other),
            SortOrder::YZX => self.cmp_yzx(other),
            SortOrder::XYZ => self.cmp_xyz(other),
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.1
    }
}

impl<P: KeyParams> std::fmt::Debug for Point<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("")
            .field(&self.x())
            .field(&self.y())
            .field(&self.z())
            .finish()
    }
}
