use crate::{count_trailing_zeros, FixedSize, KeyParams, SortOrder, TreeParams, VariableSize};
use ref_cast::RefCast;
use std::{borrow::Borrow, cmp::Ordering, fmt::Debug, marker::PhantomData, ops::Deref, sync::Arc};
use zerocopy::{AsBytes, FromBytes};

/// An unsized type that represents a point in 3D space.
///
/// The first two fields are fixed size, while the third field can be of variable size.
///
/// Implements Ord and PartialOrd for convenience, however, if you want a specific order,
/// use the xyz, yzx, or zxy methods to get a typed wrapper that implements the desired order.
#[repr(transparent)]
#[derive(RefCast)]
pub struct Point2<P: KeyParams>(PhantomData<P>, [u8]);

impl<K: KeyParams> PartialEq for Point2<K> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<K: KeyParams> Eq for Point2<K> {}

impl<K: KeyParams> PartialOrd for Point2<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<K: KeyParams> Ord for Point2<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

pub struct OwnedPoint2<P: KeyParams>(PhantomData<P>, Arc<[u8]>);

impl<P: TreeParams> OwnedPoint2<P> {
    pub fn new(x: &P::X, y: &P::Y, z: &P::Z) -> Self {
        let mut buf = vec![0; P::X::SIZE + P::Y::SIZE + z.size()];
        x.write_to_prefix(&mut buf[0..]).unwrap();
        y.write_to_prefix(&mut buf[P::X::SIZE..]).unwrap();
        z.write_to_prefix(&mut buf[P::X::SIZE + P::Y::SIZE..])
            .unwrap();
        OwnedPoint2(PhantomData, buf.into())
    }
}

impl<P: KeyParams> Debug for OwnedPoint2<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}

impl<P: KeyParams> Borrow<Point2<P>> for OwnedPoint2<P> {
    fn borrow(&self) -> &Point2<P> {
        Point2::ref_cast(&self.1)
    }
}

impl<P: KeyParams> Deref for OwnedPoint2<P> {
    type Target = Point2<P>;

    fn deref(&self) -> &Self::Target {
        self.borrow()
    }
}

impl<P: KeyParams> ToOwned for Point2<P> {
    type Owned = OwnedPoint2<P>;

    fn to_owned(&self) -> Self::Owned {
        OwnedPoint2(PhantomData, self.1.to_vec().into())
    }
}

impl<P: KeyParams> Point2<P> {
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
        P::Z::ref_from_prefix(&self.1[P::X::SIZE + P::Y::SIZE..]).unwrap()
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

    pub fn xyz(&self) -> &XYZ2<P> {
        XYZ2::ref_cast(self)
    }

    pub fn yzx(&self) -> &YZX2<P> {
        YZX2::ref_cast(self)
    }

    pub fn zxy(&self) -> &ZXY2<P> {
        ZXY2::ref_cast(self)
    }
}

impl<P: KeyParams> std::fmt::Debug for Point2<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("")
            .field(&self.x())
            .field(&self.y())
            .field(&self.z())
            .finish()
    }
}

#[repr(transparent)]
#[derive(RefCast)]
pub struct XYZ2<K: KeyParams>(Point2<K>);

impl<K: KeyParams> Debug for XYZ2<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("XYZ").field(&&self.0).finish()
    }
}

impl<K: KeyParams> Ord for XYZ2<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp_xyz(&other.0)
    }
}

impl<K: KeyParams> PartialOrd for XYZ2<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: KeyParams> PartialEq for XYZ2<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: KeyParams> Eq for XYZ2<K> {}

#[repr(transparent)]
#[derive(RefCast)]
pub struct YZX2<K: KeyParams>(Point2<K>);

#[repr(transparent)]
#[derive(RefCast)]
pub struct ZXY2<K: KeyParams>(Point2<K>);

/// A point in 3D space, serving as a key for a tree.
///
/// This does not implement `Ord` or `PartialOrd` because there are multiple
/// orderings that are equally valid.
pub struct Point<K: KeyParams> {
    pub x: K::X, // fixed size
    pub y: K::Y, // fixed size
    pub z: K::Z, // variable size
}

impl<K: KeyParams> std::fmt::Debug for Point<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("")
            .field(&self.x)
            .field(&self.y)
            .field(&self.z)
            .finish()
    }
}

pub struct XYZ<K: KeyParams>(Point<K>);

impl<K: KeyParams> Debug for XYZ<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("XYZ").field(&self.0).finish()
    }
}

impl<K: KeyParams> Clone for XYZ<K> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<K: KeyParams> From<Point<K>> for XYZ<K> {
    fn from(p: Point<K>) -> Self {
        XYZ(p)
    }
}

impl<K: KeyParams> From<XYZ<K>> for Point<K> {
    fn from(p: XYZ<K>) -> Self {
        p.0
    }
}

impl<K: KeyParams> PartialEq for XYZ<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: KeyParams> Eq for XYZ<K> {}

impl<K: KeyParams> PartialOrd for XYZ<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: KeyParams> Ord for XYZ<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp_xyz(&other.0)
    }
}

pub struct YZX<K: KeyParams>(Point<K>);

impl<K: KeyParams> Debug for YZX<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("YZX").field(&self.0).finish()
    }
}

impl<K: KeyParams> Clone for YZX<K> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<K: KeyParams> From<Point<K>> for YZX<K> {
    fn from(p: Point<K>) -> Self {
        Self(p)
    }
}

impl<K: KeyParams> PartialEq for YZX<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: KeyParams> Eq for YZX<K> {}

impl<K: KeyParams> PartialOrd for YZX<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: KeyParams> Ord for YZX<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp_yzx(&other.0)
    }
}

pub struct ZXY<K: KeyParams>(Point<K>);

impl<K: KeyParams> Debug for ZXY<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ZXY").field(&self.0).finish()
    }
}

impl<K: KeyParams> Clone for ZXY<K> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<K: KeyParams> From<Point<K>> for ZXY<K> {
    fn from(p: Point<K>) -> Self {
        Self(p)
    }
}

impl<K: KeyParams> PartialEq for ZXY<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: KeyParams> Eq for ZXY<K> {}

impl<K: KeyParams> PartialOrd for ZXY<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: KeyParams> Ord for ZXY<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp_zxy(&other.0)
    }
}

impl<K: KeyParams> PartialEq for Point<K> {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

impl<K: KeyParams> Eq for Point<K> {}

impl<K: KeyParams> PartialOrd for Point<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp_xyz(other))
    }
}

impl<K: KeyParams> Clone for Point<K> {
    fn clone(&self) -> Self {
        Point {
            x: self.x.clone(),
            y: self.y.clone(),
            z: self.z.clone(),
        }
    }
}

impl<K: KeyParams> From<(K::X, K::Y, K::Z)> for Point<K> {
    fn from((x, y, z): (K::X, K::Y, K::Z)) -> Self {
        Point::new(x, y, z)
    }
}

impl<K: KeyParams> VariableSize for Point<K> {
    fn size(&self) -> usize {
        K::X::SIZE + K::Y::SIZE + self.z.size()
    }

    fn write(&self, buf: &mut [u8]) {
        let x_start = 0;
        let y_start: usize = K::X::SIZE;
        let z_start: usize = K::X::SIZE + K::Y::SIZE;
        self.x.write_to_prefix(&mut buf[x_start..]).unwrap();
        self.y.write_to_prefix(&mut buf[y_start..]).unwrap();
        self.z.write_to_prefix(&mut buf[z_start..]).unwrap();
    }

    fn read(buf: &[u8]) -> Self {
        let x_start = 0;
        let y_start: usize = K::X::SIZE;
        let z_start: usize = K::X::SIZE + K::Y::SIZE;
        let x = K::X::read_from_prefix(&buf[x_start..]).unwrap();
        let y = K::Y::read_from_prefix(&buf[y_start..]).unwrap();
        let z = K::Z::read_from_prefix(&buf[z_start..]).unwrap();
        Point::new(x, y, z)
    }
}

impl<K: KeyParams> Point<K> {
    pub fn new(x: K::X, y: K::Y, z: K::Z) -> Self {
        Point { x, y, z }
    }

    pub fn xyz(self) -> XYZ<K> {
        self.into()
    }

    pub fn yzx(self) -> YZX<K> {
        self.into()
    }

    pub fn zxy(self) -> ZXY<K> {
        self.into()
    }

    fn cmp_xyz(&self, other: &Self) -> std::cmp::Ordering {
        self.x
            .cmp(&other.x)
            .then(self.y.cmp(&other.y))
            .then(self.z.cmp(&other.z))
    }

    fn cmp_yzx(&self, other: &Self) -> std::cmp::Ordering {
        self.y
            .cmp(&other.y)
            .then(self.z.cmp(&other.z))
            .then(self.x.cmp(&other.x))
    }

    fn cmp_zxy(&self, other: &Self) -> std::cmp::Ordering {
        self.z
            .cmp(&other.z)
            .then(self.x.cmp(&other.x))
            .then(self.y.cmp(&other.y))
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
}
