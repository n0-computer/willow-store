use crate::{FixedSize, KeyParams, SortOrder, VariableSize};
use serde::Serialize;
use std::{cmp::Ordering, fmt::Debug};
use zerocopy::{AsBytes, FromBytes};

/// A point in 3D space, serving as a key for a tree.
///
/// This does not implement `Ord` or `PartialOrd` because there are multiple
/// orderings that are equally valid.
#[derive(Serialize)]
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
