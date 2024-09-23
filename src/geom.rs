//! Geometric types and operations.
use crate::{
    count_trailing_zeros, FixedSize, IsLowerBound, KeyParams, LowerBound, RefFromSlice, SortOrder,
    VariableSize,
};
use ref_cast::RefCast;
use std::{
    borrow::Borrow,
    cmp::Ordering,
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::Deref,
    sync::Arc,
};
use zerocopy::{AsBytes, FromBytes};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryRange<T> {
    // min is inclusive
    min: T,
    // max is exclusive, None means unbounded
    max: Option<T>,
}

impl<T> From<std::ops::Range<T>> for QueryRange<T> {
    fn from(range: std::ops::Range<T>) -> Self {
        QueryRange {
            min: range.start,
            max: Some(range.end),
        }
    }
}

impl<T> From<std::ops::RangeFrom<T>> for QueryRange<T> {
    fn from(range: std::ops::RangeFrom<T>) -> Self {
        QueryRange {
            min: range.start,
            max: None,
        }
    }
}

impl<T: Display + IsLowerBound> Display for QueryRange<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.min.is_min_value() {
            match &self.max {
                Some(max) => write!(f, "..{}", max),
                None => write!(f, ".."),
            }
        } else {
            match &self.max {
                Some(max) => write!(f, "{}..{}", self.min, max),
                None => write!(f, "{}..", self.min),
            }
        }
    }
}

impl<T: Ord> QueryRange<T> {
    pub fn new(min: T, max: Option<T>) -> Self {
        Self { min, max }
    }

    pub fn all() -> Self
    where
        T: LowerBound,
    {
        Self {
            min: T::min_value(),
            max: None,
        }
    }

    #[inline(always)]
    pub fn contains<U: Ord + ?Sized>(&self, value: &U) -> bool
    where
        T: Borrow<U>,
    {
        if value < self.min.borrow() {
            return false;
        }
        if let Some(max) = &self.max {
            if value >= max.borrow() {
                return false;
            }
        }
        true
    }

    pub fn contains_range_inclusive_opt(&self, range: &RangeInclusiveOpt<T>) -> bool {
        match range.min {
            Some(ref range_min) => {
                if range_min < &self.min {
                    return false;
                }
            }
            None => return false,
        }
        match range.max {
            Some(ref range_max) => {
                if let Some(max) = &self.max {
                    // max is exclusive, so if range_max == max, it's not contained
                    if range_max >= max {
                        return false;
                    }
                }
            }
            None => {
                if self.max.is_some() {
                    return false;
                }
            }
        }
        true
    }

    pub fn intersects_range_inclusive_opt(&self, range: &RangeInclusiveOpt<T>) -> bool {
        match range.min {
            Some(ref range_min) => {
                if let Some(max) = &self.max {
                    if range_min > max {
                        return false;
                    }
                }
            }
            None => {}
        }
        match range.max {
            Some(ref range_max) => {
                if range_max < &self.min {
                    return false;
                }
            }
            None => {}
        }
        true
    }

    pub fn left(&self, key: &T) -> QueryRange<T>
    where
        T: Clone,
    {
        if &self.min < key {
            let min = self.min.clone();
            if let Some(max) = &self.max {
                let max = std::cmp::min(max, key).clone();
                QueryRange::new(min, Some(max))
            } else {
                QueryRange::new(min, Some(key.clone()))
            }
        } else {
            // empty range
            QueryRange::new(key.clone(), Some(key.clone()))
        }
    }

    pub fn right(&self, key: &T) -> QueryRange<T>
    where
        T: Clone,
    {
        if let Some(max) = &self.max {
            if max > key {
                let min = std::cmp::max(&self.min, key).clone();
                QueryRange::new(min, Some(max.clone()))
            } else {
                // empty range
                QueryRange::new(key.clone(), Some(key.clone()))
            }
        } else {
            let min = std::cmp::max(&self.min, key).clone();
            QueryRange::new(min, None)
        }
    }
}

pub struct QueryRange3d<P: KeyParams> {
    pub x: QueryRange<P::X>,
    pub y: QueryRange<P::Y>,
    pub z: QueryRange<P::ZOwned>,
}

impl<P: KeyParams> Display for QueryRange3d<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {:?} {:?}", self.x, self.y, self.z)
    }
}

impl<P: KeyParams> Debug for QueryRange3d<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryRange3d")
            .field("x", &DD(&self.x))
            .field("y", &DD(&self.y))
            .field("z", &DD(&self.z))
            .finish()
    }
}

impl<P: KeyParams> Clone for QueryRange3d<P> {
    fn clone(&self) -> Self {
        QueryRange3d {
            x: self.x.clone(),
            y: self.y.clone(),
            z: self.z.clone(),
        }
    }
}

impl<P: KeyParams> QueryRange3d<P> {
    pub fn new(x: QueryRange<P::X>, y: QueryRange<P::Y>, z: QueryRange<P::ZOwned>) -> Self {
        Self { x, y, z }
    }

    pub fn borrrowed_x(&self) -> QueryRange<&P::X> {
        QueryRange::new(self.x.min.borrow(), self.x.max.as_ref().map(|x| x.borrow()))
    }

    pub fn borrowed_y(&self) -> QueryRange<&P::Y> {
        QueryRange::new(self.y.min.borrow(), self.y.max.as_ref().map(|y| y.borrow()))
    }

    pub fn borrowed_z(&self) -> QueryRange<&P::Z> {
        QueryRange::new(self.z.min.borrow(), self.z.max.as_ref().map(|z| z.borrow()))
    }

    pub fn all() -> Self
    where
        P::X: LowerBound,
        P::Y: LowerBound,
        P::ZOwned: LowerBound,
    {
        Self {
            x: QueryRange::all(),
            y: QueryRange::all(),
            z: QueryRange::all(),
        }
    }

    #[inline(always)]
    pub fn contains(&self, point: &PointRef<P>) -> bool {
        self.x.contains(point.x()) && self.y.contains(point.y()) && self.z.contains(point.z())
    }

    #[inline(always)]
    pub fn overlaps_left(&self, key: &PointRef<P>, order: SortOrder) -> bool {
        match order {
            SortOrder::XYZ => &self.x.min <= key.x(),
            SortOrder::YZX => &self.y.min <= key.y(),
            SortOrder::ZXY => &self.z.min.borrow() <= &key.z(),
        }
    }

    #[inline(always)]
    pub fn overlaps_right(&self, key: &PointRef<P>, order: SortOrder) -> bool {
        match order {
            SortOrder::XYZ => !self
                .x
                .max
                .as_ref()
                .map(|x_max| x_max < &key.x())
                .unwrap_or_default(),
            SortOrder::YZX => !self
                .y
                .max
                .as_ref()
                .map(|y_max| y_max < &key.y())
                .unwrap_or_default(),
            SortOrder::ZXY => !self
                .z
                .max
                .as_ref()
                .map(|z_max| z_max.borrow() < &key.z())
                .unwrap_or_default(),
        }
    }

    pub fn left(&self, key: &PointRef<P>, order: SortOrder) -> Self {
        match order {
            SortOrder::XYZ => Self {
                x: self.x.left(key.x()),
                y: self.y.clone(),
                z: self.z.clone(),
            },
            SortOrder::YZX => Self {
                x: self.x.clone(),
                y: self.y.left(key.y()),
                z: self.z.clone(),
            },
            SortOrder::ZXY => Self {
                x: self.x.clone(),
                y: self.y.clone(),
                z: self.z.left(&key.z().to_owned()),
            },
        }
    }

    pub fn right(&self, key: &PointRef<P>, order: SortOrder) -> Self {
        match order {
            SortOrder::XYZ => Self {
                x: self.x.right(key.x()),
                y: self.y.clone(),
                z: self.z.clone(),
            },
            SortOrder::YZX => Self {
                x: self.x.clone(),
                y: self.y.right(key.y()),
                z: self.z.clone(),
            },
            SortOrder::ZXY => Self {
                x: self.x.clone(),
                y: self.y.clone(),
                z: self.z.right(&key.z().to_owned()),
            },
        }
    }

    pub fn pretty(&self) -> String
    where
        P::X: Display,
        P::Y: Display,
        P::ZOwned: Display,
    {
        format!("(x:{}, y:{}, z:{})", self.x, self.y, self.z)
    }
}

/// A range with inclusive start and end bounds.
// pub struct RangeInclusive<T: Ord> {
//     min: T,
//     max: T,
// }

// impl<T: Ord> RangeInclusive<T> {
//     pub fn new(min: T, max: T) -> Self {
//         RangeInclusive { min, max }
//     }

//     pub fn contains(&self, value: &T) -> bool {
//         value >= &self.min && value <= &self.max
//     }

//     pub fn union(&self, that: &RangeInclusive<T>) -> Self
//     where
//         T: Clone,
//     {
//         let min = std::cmp::min(&self.min, &that.min);
//         let max = std::cmp::max(&self.max, &that.max);
//         Self {
//             min: min.clone(),
//             max: max.clone(),
//         }
//     }
// }

/// A range with either inclusive or open start and end bounds.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RangeInclusiveOpt<T> {
    min: Option<T>,
    max: Option<T>,
}

impl<T: Debug> Display for RangeInclusiveOpt<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.min, &self.max) {
            (Some(min), Some(max)) => write!(f, "[{:?}, {:?}]", min, max),
            (Some(min), None) => write!(f, "[{:?}, ∞)", min),
            (None, Some(max)) => write!(f, "(-∞, {:?}]", max),
            (None, None) => write!(f, "(-∞, ∞)"),
        }
    }
}

impl<T: Debug> Debug for RangeInclusiveOpt<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RangeInclusiveOpt")
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

impl<T: Clone> RangeInclusiveOpt<T> {
    pub fn new(min: Option<T>, max: Option<T>) -> Self {
        RangeInclusiveOpt { min, max }
    }

    pub fn single(value: T) -> Self {
        RangeInclusiveOpt {
            min: Some(value.clone()),
            max: Some(value),
        }
    }

    pub fn from(min: T) -> Self {
        RangeInclusiveOpt {
            min: Some(min),
            max: None,
        }
    }

    pub fn to(max: T) -> Self {
        RangeInclusiveOpt {
            min: None,
            max: Some(max),
        }
    }

    pub fn all() -> Self {
        RangeInclusiveOpt {
            min: None,
            max: None,
        }
    }

    pub fn union(&self, that: RangeInclusiveOpt<T>) -> Self
    where
        T: Ord,
    {
        let min = match (&self.min, &that.min) {
            (Some(a), Some(b)) => Some(std::cmp::min(a, b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        let max = match (&self.max, &that.max) {
            (Some(a), Some(b)) => Some(std::cmp::max(a, b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        Self {
            min: min.cloned(),
            max: max.cloned(),
        }
    }

    pub fn contains<U: ?Sized>(&self, value: &U) -> bool
    where
        U: Ord,
        T: Borrow<U>,
    {
        if let Some(min) = &self.min {
            if value < min.borrow() {
                return false;
            }
        }
        if let Some(max) = &self.max {
            if value > max.borrow() {
                return false;
            }
        }
        true
    }
}

/// A bounding box in 3D space.
///
/// Bounds are either omitted (None) or inclusive.
pub struct BBoxRef<'a, P: KeyParams> {
    x: RangeInclusiveOpt<&'a P::X>,
    y: RangeInclusiveOpt<&'a P::Y>,
    z: RangeInclusiveOpt<&'a P::Z>,
}

impl<'a, P: KeyParams> BBoxRef<'a, P> {
    pub fn new(
        x: RangeInclusiveOpt<&'a P::X>,
        y: RangeInclusiveOpt<&'a P::Y>,
        z: RangeInclusiveOpt<&'a P::Z>,
    ) -> Self {
        Self { x, y, z }
    }

    pub fn all() -> Self {
        Self {
            x: RangeInclusiveOpt::all(),
            y: RangeInclusiveOpt::all(),
            z: RangeInclusiveOpt::all(),
        }
    }

    pub fn contains(&self, point: &PointRef<P>) -> bool {
        self.x.contains(point.x()) && self.y.contains(point.y()) && self.z.contains(point.z())
    }

    pub fn contained_in(&self, query: &QueryRange3d<P>) -> bool {
        query.borrrowed_x().contains_range_inclusive_opt(&self.x)
            && query.borrowed_y().contains_range_inclusive_opt(&self.y)
            && query.borrowed_z().contains_range_inclusive_opt(&self.z)
    }

    pub fn intersects(&self, query: &QueryRange3d<P>) -> bool {
        query.borrrowed_x().intersects_range_inclusive_opt(&self.x)
            && query.borrowed_y().intersects_range_inclusive_opt(&self.y)
            && query.borrowed_z().intersects_range_inclusive_opt(&self.z)
    }

    pub fn split_left(&self, key: &'a PointRef<P>, order: SortOrder) -> Self {
        match order {
            SortOrder::XYZ => Self {
                x: RangeInclusiveOpt::new(self.x.min, Some(key.x())),
                y: self.y,
                z: self.z,
            },
            SortOrder::YZX => Self {
                x: self.x,
                y: RangeInclusiveOpt::new(self.y.min, Some(key.y())),
                z: self.z,
            },
            SortOrder::ZXY => Self {
                x: self.x,
                y: self.y,
                z: RangeInclusiveOpt::new(self.z.min, Some(key.z())),
            },
        }
    }

    pub fn split_right(&self, key: &'a PointRef<P>, order: SortOrder) -> Self {
        match order {
            SortOrder::XYZ => Self {
                x: RangeInclusiveOpt::new(Some(key.x()), self.x.max),
                y: self.y,
                z: self.z,
            },
            SortOrder::YZX => Self {
                x: self.x,
                y: RangeInclusiveOpt::new(Some(key.y()), self.y.max),
                z: self.z,
            },
            SortOrder::ZXY => Self {
                x: self.x,
                y: self.y,
                z: RangeInclusiveOpt::new(Some(key.z()), self.z.max),
            },
        }
    }
}

// /// A bounding box in 3D space.
// ///
// /// Bounds are either omitted (None) or inclusive.
// pub struct BBox<P: KeyParams> {
//     x: RangeInclusiveOpt<P::X>,
//     y: RangeInclusiveOpt<P::Y>,
//     z: RangeInclusiveOpt<P::ZOwned>,
// }

// impl<P: KeyParams> Clone for BBox<P> {
//     fn clone(&self) -> Self {
//         BBox {
//             x: self.x.clone(),
//             y: self.y.clone(),
//             z: self.z.clone(),
//         }
//     }
// }

// impl<P: KeyParams> Display for BBox<P> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{} {} {}", self.x, self.y, self.z)
//     }
// }

// impl<P: KeyParams> Debug for BBox<P> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("BBox")
//             .field("x", &self.x)
//             .field("y", &self.y)
//             .field("z", &self.z)
//             .finish()
//     }
// }

// impl<P: KeyParams> BBox<P> {
//     pub fn new(
//         x: RangeInclusiveOpt<P::X>,
//         y: RangeInclusiveOpt<P::Y>,
//         z: RangeInclusiveOpt<P::ZOwned>,
//     ) -> Self {
//         BBox { x, y, z }
//     }

//     pub fn all() -> Self {
//         BBox {
//             x: RangeInclusiveOpt::all(),
//             y: RangeInclusiveOpt::all(),
//             z: RangeInclusiveOpt::all(),
//         }
//     }

//     pub fn contains(&self, point: &PointRef<P>) -> bool {
//         self.x.contains(point.x()) && self.y.contains(point.y()) && self.z.contains(point.z())
//     }

//     pub fn contained_in(&self, query: &QueryRange3d<P>) -> bool {
//         query.x.contains_range_inclusive_opt(&self.x)
//             && query.y.contains_range_inclusive_opt(&self.y)
//             && query.z.contains_range_inclusive_opt(&self.z)
//     }

//     pub fn split_left(&self, key: &PointRef<P>, order: SortOrder) -> BBox<P> {
//         match order {
//             SortOrder::XYZ => BBox {
//                 x: RangeInclusiveOpt::new(self.x.min.clone(), Some(key.x().clone())),
//                 y: self.y.clone(),
//                 z: self.z.clone(),
//             },
//             SortOrder::YZX => BBox {
//                 x: self.x.clone(),
//                 y: RangeInclusiveOpt::new(self.y.min.clone(), Some(key.y().clone())),
//                 z: self.z.clone(),
//             },
//             SortOrder::ZXY => BBox {
//                 x: self.x.clone(),
//                 y: self.y.clone(),
//                 z: RangeInclusiveOpt::new(self.z.min.clone(), Some(key.z().to_owned())),
//             },
//         }
//     }

//     pub fn split_right(&self, key: &PointRef<P>, order: SortOrder) -> BBox<P> {
//         match order {
//             SortOrder::XYZ => BBox {
//                 x: RangeInclusiveOpt::new(Some(key.x().clone()), self.x.max.clone()),
//                 y: self.y.clone(),
//                 z: self.z.clone(),
//             },
//             SortOrder::YZX => BBox {
//                 x: self.x.clone(),
//                 y: RangeInclusiveOpt::new(Some(key.y().clone()), self.y.max.clone()),
//                 z: self.z.clone(),
//             },
//             SortOrder::ZXY => BBox {
//                 x: self.x.clone(),
//                 y: self.y.clone(),
//                 z: RangeInclusiveOpt::new(Some(key.z().to_owned()), self.z.max.clone()),
//             },
//         }
//     }
// }

/// An unsized type that represents a point in 3D space.
///
/// The first two fields are fixed size, while the third field can be of variable size.
///
/// Implements Ord and PartialOrd for convenience, however, if you want a specific order,
/// there are comparison functions that take a `SortOrder` enum.
#[derive(RefCast, Hash)]
#[repr(transparent)]
pub struct PointRef<P: KeyParams>(PhantomData<P>, [u8]);

impl<K: KeyParams> PartialEq for PointRef<K> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<K: KeyParams> Eq for PointRef<K> {}

impl<K: KeyParams> PartialOrd for PointRef<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<K: KeyParams> Ord for PointRef<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Point<P: KeyParams>(PhantomData<P>, Arc<[u8]>);

impl<P: KeyParams> Borrow<PointRef<P>> for Point<P> {
    fn borrow(&self) -> &PointRef<P> {
        PointRef::ref_cast(&self.1)
    }
}

impl<P: KeyParams> Point<P> {
    pub fn new(x: &P::X, y: &P::Y, z: &P::Z) -> Self {
        let mut buf = vec![0; P::X::SIZE + P::Y::SIZE + z.size()];
        x.write_to_prefix(&mut buf[0..]).unwrap();
        y.write_to_prefix(&mut buf[P::X::SIZE..]).unwrap();
        z.write_to_prefix(&mut buf[P::X::SIZE + P::Y::SIZE..])
            .unwrap();
        Point(PhantomData, buf.into())
    }
}

impl<P: KeyParams> Debug for Point<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.deref(), f)
    }
}

impl<P: KeyParams> Deref for Point<P> {
    type Target = PointRef<P>;

    fn deref(&self) -> &Self::Target {
        self.borrow()
    }
}

impl<P: KeyParams> ToOwned for PointRef<P> {
    type Owned = Point<P>;

    fn to_owned(&self) -> Self::Owned {
        Point(PhantomData, self.1.to_vec().into())
    }
}

impl<P: KeyParams> PointRef<P> {
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

impl<P: KeyParams> std::fmt::Debug for PointRef<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("")
            .field(&self.x())
            .field(&self.y())
            .field(&self.z())
            .finish()
    }
}

impl<P: KeyParams> std::fmt::Display for PointRef<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("")
            .field(&DD(self.x()))
            .field(&DD(self.y()))
            .field(&DD(self.z()))
            .finish()
    }
}

struct DD<T>(T);

impl<T: std::fmt::Display> Debug for DD<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}
