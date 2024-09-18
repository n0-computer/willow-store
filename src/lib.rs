//! A db backed kd tree for k=3.
//!
//! Original idea from [Aljoscha Meyer] in [kv_3d_storage].
//!
//! This is intended as persistence for [willow].
//!
//! This data structure combines the idea of a [kd tree] with a [zip tree].
//! Each node is assigned an u8 rank based on the key. The rank should be
//! drawn from a geometric distribution with p = 0.5. We use the number
//! of trailing zeros in the blake3 hash of the key to determine the rank.
//!
//! To turn the zip tree into a kd tree, the partitioning for each node is
//! alternated based on the rank (xyz, yzx, zxy). See the [`SortOrder`] enum
//! for the exact mapping.
//!
//! Due to the random distribution of the ranks, this will provide partitioning
//! in all three dimensions. Very frequently, you should get alternating
//! dimensions so that every 3 levels you partition all 3 dimensions. However,
//! unlike in a traditional kd tree, this can not be relied on.
//!
//! # Summaries
//!
//! In addition to key and value, each node also stores a summary. The summary
//! can be computed (lifted) from an individual element, and summaries can be
//! combined.
//!
//! The combine operation together with a zero element must form a commutative
//! monoid. The rules for computing summaries are currently encoded in the
//! [`LiftingCommutativeMonoid`] trait.
//!
//! For a leaf node, the summary is just the lifted value. For a branch node
//! the summary is the summary of the optional left and right child and the
//! value.
//!
//! # Operations
//!
//! ## Iteratiion over elements in a box
//!
//! A kd tree supports efficient queries for *boxes* or 3d ranges in 3D space.
//! This is one of the main motivations for using a 3d tree. Point queries would
//! be possible with a simpler data structure.
//!
//! ## Sorted iteration over elements in a box
//!
//! Elements in the tree are alternately partitioned by the three sort orders.
//! While the elements are not strictly sorted along any of the three sort
//! orders, the partitioning allows for an efficient merge sort like
//! implementation of sorted iteration by any of the three sort orders.
//!
//! ## Summary of elements in a box
//!
//! The second main motivation for using a 3d tree is the ability to quickly
//! compute summaries of elements in a box. Each node has a bounding box for
//! which it is responsible. If the bounding box is fully contained in the query
//! box, the summary of the node can be used without further recursing into the
//! children.
//!
//! ## Insertion
//!
//! Insertion works like in a zip tree, except that the ordering to be used is
//! alternated based on the rank.
//!
//! ### Insertion algorithm
//!
//! TLDR for the insertion algorithm: For a new node to be inserted, You first
//! perform a normal binary search until you find the insertion location.
//! Everything above the insertion point can remain the same. The tree below
//! the insertion point needs to be unzipped into two separate trees that become
//! the left and right child of the new node.
//!
//! The insertion algorithm is as follows:
//!
//! Find the insertion point by binary search.
//!
//! Below the insertion point, a node to the *left* of the new value needs to be
//! fixed if its *right* child is to the right of the new value. To find a
//! node to be fixed, just follow right children until you either reach a leaf
//! node or a node that is to the right of the new value. The predecessor of
//! that node is a node to be fixed later, and the node itself is the new right
//! child of a node to be fixed on the right side.
//!
//! For nodes to the *right* of the new value, the same is done in reverse.
//!
//! The fix operation will alternate between fixing left and right of the
//! insertion point until it reaches a leaf.
//!
//! # Persistence
//!
//! The data structure is backed by a database, but not a persistent data
//! structure in the functional programming sense, since nodes are modified in
//! place for efficiency during mutation ops. For persistent snapshots, we rely
//! on the underlying database.
//!
//! [zip tree]: https://arxiv.org/pdf/1806.06726
//! [kd tree]: https://dl.acm.org/doi/pdf/10.1145/361002.361007
//! [Aljoscha Meyer]: https://aljoscha-meyer.de/
//! [kv_3d_storage]: https://github.com/AljoschaMeyer/kv_3d_storage/blob/d311cdee31ce7f5b5f50f9798507b958fe0f887b/src/lib.rs
//! [willow]: https://willowprotocol.org/
use std::{
    borrow::Borrow,
    cmp::Ordering,
    collections::BTreeSet,
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use anyhow::Result;
use genawaiter::sync::{Co, Gen};
pub use point::PointRef;

mod point;
pub use point::Point;
mod store;
use ref_cast::RefCast;
mod layout;
mod path;
use layout::*;
pub use store::BlobStore;
pub use store::MemStore;
use zerocopy::{AsBytes, FromBytes, FromZeroes};

pub trait FixedSize {
    const SIZE: usize;
}

pub trait VariableSize {
    fn size(&self) -> usize;
}

pub trait SerDe: VariableSize {
    fn write(&self, buf: &mut [u8]);
    fn read(buf: &[u8]) -> Self;

    fn to_vec(&self) -> Vec<u8> {
        let mut buf = vec![0; self.size()];
        self.write(&mut buf);
        buf
    }
}

impl FixedSize for u64 {
    const SIZE: usize = 8;
}

impl VariableSize for u64 {
    fn size(&self) -> usize {
        8
    }
}

impl VariableSize for [u8] {
    fn size(&self) -> usize {
        8
    }
}

impl SerDe for u64 {
    fn write(&self, buf: &mut [u8]) {
        buf.copy_from_slice(&self.to_be_bytes());
    }
    fn read(buf: &[u8]) -> Self {
        u64::from_be_bytes(buf.try_into().unwrap())
    }
}

impl FixedSize for store::NodeId {
    const SIZE: usize = 8;
}

impl FixedSize for u8 {
    const SIZE: usize = 1;
}

pub trait RefFromSlice {
    fn ref_from_slice(slice: &[u8]) -> &Self;
}

impl RefFromSlice for [u8] {
    fn ref_from_slice(slice: &[u8]) -> &Self {
        slice
    }
}

impl RefFromSlice for u64 {
    fn ref_from_slice(slice: &[u8]) -> &Self {
        u64::ref_from(slice).unwrap()
    }
}

pub trait KeyParams {
    type X: Ord + Debug + Display + AsBytes + FromBytes + FixedSize + Clone;
    type Y: Ord + Debug + Display + AsBytes + FromBytes + FixedSize + Clone;
    type Z: Ord
        + Debug
        + Display
        + AsBytes
        + VariableSize
        + ToOwned<Owned = Self::ZOwned>
        + RefFromSlice
        + ?Sized;
    type ZOwned: Debug + Display + Borrow<Self::Z> + Ord + Clone;
}

pub trait LiftingCommutativeMonoid<K: ?Sized, V> {
    /// The neutral element of the monoid.
    ///
    /// It isn't called zero to avoid name collison with the FromZeroes trait.
    fn neutral() -> Self;
    /// Lift a key value pair into the monoid.
    ///
    /// This is taking both key and value by reference to avoid cloning.
    fn lift(key: &K, value: &V) -> Self;
    /// Combine op. This must be commutative, and `neutral` must be the identity.
    fn combine(&self, other: &Self) -> Self;
}

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

impl<T: Display> Display for QueryRange<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.max {
            Some(max) => write!(f, "{}..{}", self.min, max),
            None => write!(f, "{}..", self.min),
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
    x: QueryRange<P::X>,
    y: QueryRange<P::Y>,
    z: QueryRange<P::ZOwned>,
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

struct NoQuotes<'a>(&'a str);

impl<'a> Debug for NoQuotes<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

struct DD<T>(T);

impl<T: Display> Debug for DD<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
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

    pub fn contains(&self, point: &PointRef<P>) -> bool {
        self.x.contains(point.x()) && self.y.contains(point.y()) && self.z.contains(point.z())
    }

    pub fn overlaps_left(&self, key: &PointRef<P>, rank: u8) -> bool {
        match SortOrder::from(rank) {
            SortOrder::XYZ => &self.x.min <= key.x(),
            SortOrder::YZX => &self.y.min <= key.y(),
            SortOrder::ZXY => &self.z.min.borrow() <= &key.z(),
        }
    }

    pub fn overlaps_right(&self, key: &PointRef<P>, rank: u8) -> bool {
        match SortOrder::from(rank) {
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
pub struct RangeInclusive<T: Ord> {
    min: T,
    max: T,
}

impl<T: Ord> RangeInclusive<T> {
    pub fn new(min: T, max: T) -> Self {
        RangeInclusive { min, max }
    }

    pub fn contains(&self, value: &T) -> bool {
        value >= &self.min && value <= &self.max
    }

    pub fn union(&self, that: &RangeInclusive<T>) -> Self
    where
        T: Clone,
    {
        let min = std::cmp::min(&self.min, &that.min);
        let max = std::cmp::max(&self.max, &that.max);
        Self {
            min: min.clone(),
            max: max.clone(),
        }
    }
}

/// A range with either inclusive or open start and end bounds.
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

impl<T: Clone> Clone for RangeInclusiveOpt<T> {
    fn clone(&self) -> Self {
        Self {
            min: self.min.clone(),
            max: self.max.clone(),
        }
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
pub struct BBox<P: KeyParams> {
    x: RangeInclusiveOpt<P::X>,
    y: RangeInclusiveOpt<P::Y>,
    z: RangeInclusiveOpt<P::ZOwned>,
}

impl<P: KeyParams> Clone for BBox<P> {
    fn clone(&self) -> Self {
        BBox {
            x: self.x.clone(),
            y: self.y.clone(),
            z: self.z.clone(),
        }
    }
}

impl<P: KeyParams> Display for BBox<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}

impl<P: KeyParams> Debug for BBox<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BBox")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .finish()
    }
}

/// Result of asserting invariants for a node.
pub struct AssertInvariantsRes<S: TreeParams> {
    /// min and max in xyz order
    xyz_min: Point<S>,
    xyz_max: Point<S>,
    /// min and max in yzx order
    yzx_min: Point<S>,
    yzx_max: Point<S>,
    /// min and max in zxy order
    zxy_min: Point<S>,
    zxy_max: Point<S>,
    /// summary of the node
    summary: S::M,
    /// rank of the node
    rank: u8,
    /// Count of the node
    count: u64,
}

impl<T: TreeParams> AssertInvariantsRes<T> {
    pub fn single(point: Point<T>, rank: u8, value: T::V) -> Self {
        AssertInvariantsRes {
            xyz_min: point.clone(),
            xyz_max: point.clone(),
            yzx_min: point.clone(),
            yzx_max: point.clone(),
            zxy_min: point.clone(),
            zxy_max: point.clone(),
            summary: T::M::lift(&point, &value),
            rank,
            count: 1,
        }
    }

    pub fn combine(&self, that: &Self) -> Self {
        let xyz_min =
            if self.xyz_min.cmp_with_order(&that.xyz_min, SortOrder::XYZ) == Ordering::Less {
                self.xyz_min.clone()
            } else {
                that.xyz_min.clone()
            };
        let xyz_max =
            if self.xyz_max.cmp_with_order(&that.xyz_max, SortOrder::XYZ) == Ordering::Greater {
                self.xyz_max.clone()
            } else {
                that.xyz_max.clone()
            };
        let yzx_min =
            if self.yzx_min.cmp_with_order(&that.yzx_min, SortOrder::YZX) == Ordering::Less {
                self.yzx_min.clone()
            } else {
                that.yzx_min.clone()
            };
        let yzx_max =
            if self.yzx_max.cmp_with_order(&that.yzx_max, SortOrder::YZX) == Ordering::Greater {
                self.yzx_max.clone()
            } else {
                that.yzx_max.clone()
            };
        let zxy_min =
            if self.zxy_min.cmp_with_order(&that.zxy_min, SortOrder::ZXY) == Ordering::Less {
                self.zxy_min.clone()
            } else {
                that.zxy_min.clone()
            };
        let zxy_max =
            if self.zxy_max.cmp_with_order(&that.zxy_max, SortOrder::ZXY) == Ordering::Greater {
                self.zxy_max.clone()
            } else {
                that.zxy_max.clone()
            };
        let rank = self.rank.max(that.rank);
        let summary = self.summary.combine(&that.summary);
        let count = self.count + that.count;
        AssertInvariantsRes {
            xyz_min,
            xyz_max,
            yzx_min,
            yzx_max,
            zxy_min,
            zxy_max,
            rank,
            summary,
            count,
        }
    }
}

impl<P: KeyParams> BBox<P> {
    pub fn new(
        x: RangeInclusiveOpt<P::X>,
        y: RangeInclusiveOpt<P::Y>,
        z: RangeInclusiveOpt<P::ZOwned>,
    ) -> Self {
        BBox { x, y, z }
    }

    pub fn all() -> Self {
        BBox {
            x: RangeInclusiveOpt::all(),
            y: RangeInclusiveOpt::all(),
            z: RangeInclusiveOpt::all(),
        }
    }

    pub fn contains(&self, point: &PointRef<P>) -> bool {
        self.x.contains(point.x()) && self.y.contains(point.y()) && self.z.contains(point.z())
    }

    pub fn contained_in(&self, query: &QueryRange3d<P>) -> bool {
        query.x.contains_range_inclusive_opt(&self.x)
            && query.y.contains_range_inclusive_opt(&self.y)
            && query.z.contains_range_inclusive_opt(&self.z)
    }

    pub fn split_left(&self, key: &PointRef<P>, rank: u8) -> BBox<P> {
        match SortOrder::from(rank) {
            SortOrder::XYZ => BBox {
                x: RangeInclusiveOpt::new(self.x.min.clone(), Some(key.x().clone())),
                y: self.y.clone(),
                z: self.z.clone(),
            },
            SortOrder::YZX => BBox {
                x: self.x.clone(),
                y: RangeInclusiveOpt::new(self.y.min.clone(), Some(key.y().clone())),
                z: self.z.clone(),
            },
            SortOrder::ZXY => BBox {
                x: self.x.clone(),
                y: self.y.clone(),
                z: RangeInclusiveOpt::new(self.z.min.clone(), Some(key.z().to_owned())),
            },
        }
    }

    pub fn split_right(&self, key: &PointRef<P>, rank: u8) -> BBox<P> {
        match SortOrder::from(rank) {
            SortOrder::XYZ => BBox {
                x: RangeInclusiveOpt::new(Some(key.x().clone()), self.x.max.clone()),
                y: self.y.clone(),
                z: self.z.clone(),
            },
            SortOrder::YZX => BBox {
                x: self.x.clone(),
                y: RangeInclusiveOpt::new(Some(key.y().clone()), self.y.max.clone()),
                z: self.z.clone(),
            },
            SortOrder::ZXY => BBox {
                x: self.x.clone(),
                y: self.y.clone(),
                z: RangeInclusiveOpt::new(Some(key.z().to_owned()), self.z.max.clone()),
            },
        }
    }
}

pub trait LowerBound {
    fn min_value() -> Self;

    fn is_min_value(&self) -> bool;
}

impl LowerBound for u64 {
    fn min_value() -> Self {
        0
    }

    fn is_min_value(&self) -> bool {
        *self == 0
    }
}

///
pub trait ValueParams: PartialEq + Eq + Clone + Debug + FixedSize + AsBytes + FromBytes {}

impl<T: PartialEq + Eq + Clone + Debug + FixedSize + AsBytes + FromBytes> ValueParams for T {}

/// Tree params for a 3D tree. This extends `KeyParams` with a value and
/// summary type.
pub trait TreeParams: KeyParams + Sized {
    type V: ValueParams;
    type M: LiftingCommutativeMonoid<PointRef<Self>, Self::V>
        + Clone
        + Debug
        + Eq
        + AsBytes
        + FixedSize
        + FromBytes;
}

pub trait NodeStore<P: TreeParams>: store::BlobStore {
    fn create_node(&mut self, data: &NodeData<P>) -> Result<Node<P>> {
        Ok(Node(self.create(data.as_slice())?, PhantomData))
    }

    fn update_node(&mut self, id: Node<P>, data: &NodeData<P>) -> Result<()> {
        self.update(id.0, data.as_slice())
    }

    fn put_node(&mut self, data: OwnedNodeData<P>) -> Result<IdAndData<P>> {
        let id = Node(self.create(data.as_slice())?, PhantomData);
        Ok(IdAndData::new(id, data))
    }

    fn data(&self, id: Node<P>) -> Result<OwnedNodeData<P>> {
        let data = store::BlobStore::read(self, id.0)?;
        Ok(OwnedNodeData::new(data))
    }

    /// Get a node by id, returning None if the id is None.
    ///
    /// This is just a convenience method for the common case where you have an
    /// optional id and want to get the node if it exists.
    fn data_opt(&self, id: Node<P>) -> Result<Option<OwnedNodeData<P>>> {
        Ok(if id.is_empty() {
            None
        } else {
            Some(self.data(id)?)
        })
    }

    /// Get a node by id, returning None if the id is None.
    /// Also return the id along with the node.
    fn get_node(&self, id: Node<P>) -> Result<Option<IdAndData<P>>> {
        Ok(if id.is_empty() {
            None
        } else {
            IdAndData::new(id, self.data(id)?).into()
        })
    }

    /// Get a node by id, returning None if the id is None.
    /// Also return the id along with the node.
    fn get_non_empty(&self, id: Node<P>) -> Result<IdAndData<P>> {
        if id.is_empty() {
            panic!("Empty id");
        }
        Ok(IdAndData::new(id, self.data(id)?))
    }
}

impl<T: store::BlobStore, P: TreeParams> NodeStore<P> for T {}

#[derive(Debug, Clone, Copy)]
pub struct SplitOpts {
    /// Up to how many values to send immediately, before sending only a fingerprint.
    pub max_set_size: usize,
    /// `k` in the protocol, how many splits to generate. at least 2
    pub split_factor: usize,
}

impl Default for SplitOpts {
    fn default() -> Self {
        SplitOpts {
            max_set_size: 1,
            split_factor: 2,
        }
    }
}

#[derive(Debug)]
pub enum SplitAction<P: TreeParams> {
    SendFingerprint(P::M),
    SendEntries(u64),
}

pub type RangeSplit<P> = (QueryRange3d<P>, SplitAction<P>);

#[repr(transparent)]
#[derive(AsBytes, FromZeroes, FromBytes)]
pub struct Node<P: TreeParams>(store::NodeId, PhantomData<P>);

impl<P: TreeParams> From<store::NodeId> for Node<P> {
    fn from(id: store::NodeId) -> Self {
        Node(id, PhantomData)
    }
}

impl<P: TreeParams> Clone for Node<P> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<P: TreeParams> Copy for Node<P> {}

impl<P: TreeParams> Debug for Node<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            write!(f, "Node(EMPTY)")
        } else {
            write!(f, "Node({})", self.0)
        }
    }
}

impl<P: TreeParams> Display for Node<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            write!(f, "EMPTY")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl<P: TreeParams> PartialEq for Node<P> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<P: TreeParams> Eq for Node<P> {}

impl<P: TreeParams> Node<P> {
    pub const EMPTY: Self = Self(store::NodeId::EMPTY, PhantomData);

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn count(&self, store: &impl NodeStore<P>) -> Result<u64> {
        Ok(if let Some(data) = store.data_opt(*self)? {
            1 + data.left().count(store)? + data.right().count(store)?
        } else {
            0
        })
    }

    pub fn id(&self) -> store::NodeId {
        self.0
    }

    pub fn insert(
        &mut self,
        key: &PointRef<P>,
        value: &P::V,
        store: &mut impl NodeStore<P>,
    ) -> Result<Option<P::V>> {
        self.insert_rec(OwnedNodeData::leaf(key, value), store)
    }

    fn insert_rec(
        &mut self,
        x: OwnedNodeData<P>,
        store: &mut impl NodeStore<P>,
    ) -> Result<Option<P::V>> {
        if let Some(mut this) = store.get_node(*self)? {
            let x_cmp_cur = x.key().cmp_at_rank(this.key(), this.rank());
            if x_cmp_cur == Ordering::Equal {
                // just replace the value
                let mut res = x.value().clone();
                std::mem::swap(&mut res, this.value_mut());
                let res = Some(res);
                this.recalculate_summary(store)?;
                this.persist(store)?;
                Ok(res)
            } else if x.rank() < this.rank()
                || (x.rank() == this.rank() && x_cmp_cur == Ordering::Greater)
            {
                // cur is above x, just go down
                let value = x.value().clone();
                let res = match x_cmp_cur {
                    Ordering::Less => this.left_mut().insert_rec(x, store)?,
                    Ordering::Greater => this.right_mut().insert_rec(x, store)?,
                    Ordering::Equal => unreachable!(),
                };
                if res.is_some() {
                    this.recalculate_summary(store)?;
                } else {
                    this.add_summary(value);
                }
                this.persist(store)?;
                Ok(res)
            } else {
                let x = store.put_node(x)?;
                let mut parts = vec![x];
                self.split_all(store, &mut parts)?;
                *self = Node::from_unique_nodes(store, parts)?;
                Ok(None)
            }
        } else {
            // self is empty
            *self = store.create_node(&x)?.into();
            Ok(None)
        }
    }

    pub fn delete(
        &mut self,
        key: &PointRef<P>,
        store: &mut impl NodeStore<P>,
    ) -> Result<Option<P::V>> {
        self.delete_rec(key, store)
    }

    fn delete_rec(
        &mut self,
        key: &PointRef<P>,
        store: &mut impl NodeStore<P>,
    ) -> Result<Option<P::V>> {
        if let Some(mut this) = store.get_node(*self)? {
            let key_cmp_cur = key.cmp_at_rank(this.key(), this.rank());
            if key_cmp_cur == Ordering::Equal {
                let removed = this.value().clone();
                let mut res = Vec::new();
                this.left().split_all(store, &mut res)?;
                this.right().split_all(store, &mut res)?;
                let merged = Node::from_unique_nodes(store, res)?;
                store.delete(self.0)?;
                *self = merged;
                Ok(Some(removed))
            } else {
                let res = match key_cmp_cur {
                    Ordering::Less => this.left_mut().delete_rec(key, store)?,
                    Ordering::Greater => this.right_mut().delete_rec(key, store)?,
                    Ordering::Equal => unreachable!(),
                };
                if res.is_some() {
                    this.recalculate_summary(store)?;
                    this.persist(store)?;
                }
                // update summary
                Ok(res)
            }
        } else {
            Ok(None)
        }
    }

    pub fn assert_invariants(
        &self,
        store: &impl NodeStore<P>,
        include_summary: bool,
    ) -> Result<()> {
        if !self.is_empty() {
            let data = store.get_non_empty(*self)?;
            data.assert_invariants(store, include_summary)?;
        }
        Ok(())
    }

    pub fn get(&self, key: &PointRef<P>, store: &impl NodeStore<P>) -> Result<Option<P::V>> {
        Ok(self.get0(key, store)?.map(|x| x.value().clone()))
    }

    fn get0(
        &self,
        key: &PointRef<P>,
        store: &impl NodeStore<P>,
    ) -> Result<Option<OwnedNodeData<P>>> {
        if let Some(data) = store.data_opt(*self)? {
            match key.cmp_at_rank(data.key(), data.rank()) {
                Ordering::Less => {
                    if !data.left().is_empty() {
                        data.left().get0(key, store)
                    } else {
                        Ok(None)
                    }
                }
                Ordering::Greater => {
                    if !data.right().is_empty() {
                        data.right().get0(key, store)
                    } else {
                        Ok(None)
                    }
                }
                Ordering::Equal => Ok(Some(data)),
            }
        } else {
            Ok(None)
        }
    }

    pub fn dump(&self, store: &impl NodeStore<P>) -> Result<()> {
        self.dump0("".into(), store)
    }

    fn dump0(&self, prefix: String, store: &impl NodeStore<P>) -> Result<()> {
        if let Some(data) = store.data_opt(*self)? {
            data.left().dump0(format!("{}  ", prefix), store)?;
            println!(
                "{}{} rank={} order={:?} value={:?} summary={:?}",
                prefix,
                data.key(),
                data.rank(),
                SortOrder::from(data.rank()),
                data.value(),
                data.summary(),
            );
            data.right().dump0(format!("{}  ", prefix), store)?;
        } else {
            println!("{}Empty", prefix);
        }
        Ok(())
    }

    pub fn from_iter<I: IntoIterator<Item = (Point<P>, P::V)>>(
        iter: I,
        store: &mut impl NodeStore<P>,
    ) -> Result<Node<P>> {
        let mut nodes: Vec<_> = iter
            .into_iter()
            .map(|(key, value)| OwnedNodeData::leaf(&key, &value))
            .collect();
        // Before we sort, remove all but the first occurence of each point.
        let mut uniques = BTreeSet::new();
        nodes.retain(|node| uniques.insert(node.key().to_owned()));
        let nodes = nodes
            .into_iter()
            .map(|data| store.put_node(data))
            .collect::<Result<_>>()?;
        let node = Node::from_unique_nodes(store, nodes)?;
        Ok(node)
    }

    pub fn from_unique_nodes(
        store: &mut impl NodeStore<P>,
        mut nodes: Vec<IdAndData<P>>,
    ) -> Result<Node<P>> {
        // if rank is equal, compare keys at rank
        nodes.sort_by(|p1, p2| {
            p2.rank()
                .cmp(&p1.rank())
                .then(p1.key().cmp_at_rank(p2.key(), p1.rank()))
        });
        let mut tree = Node::EMPTY;
        for node in nodes {
            node.persist(store)?;
            tree.insert_no_balance(&node, store)?;
        }
        Ok(tree)
    }

    fn insert_no_balance(
        &mut self,
        node: &IdAndData<P>,
        store: &mut impl NodeStore<P>,
    ) -> Result<()> {
        if self.is_empty() {
            *self = node.id
        } else {
            store.get_non_empty(*self)?.insert_no_balance(node, store)?;
        }
        Ok(())
    }

    /// Iterate over the entire tree in its natural order.
    ///
    /// The order is implementation dependent and should not be relied on.
    pub fn iter<'a>(
        &'a self,
        store: &'a impl NodeStore<P>,
    ) -> impl Iterator<Item = Result<(Point<P>, P::V)>> + 'a {
        Gen::new(|co| async move {
            if let Err(cause) = self.iter0(store, &co).await {
                co.yield_(Err(cause)).await;
            }
        })
        .into_iter()
    }

    async fn iter0(
        &self,
        store: &impl NodeStore<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        if let Some(data) = store.data_opt(*self)? {
            Box::pin(data.left().iter0(store, co)).await?;
            co.yield_(Ok((data.key().to_owned(), data.value().to_owned())))
                .await;
            Box::pin(data.right().iter0(store, co)).await?;
        }
        Ok(())
    }

    /// Get a summary of the elements in a 3d range.
    ///
    /// The result is identical to iterating over the elements in the 3d range
    /// and combining the summaries of each element, but will be much more
    /// efficient for large trees.
    pub fn summary(&self, query: &QueryRange3d<P>, store: &impl NodeStore<P>) -> Result<P::M> {
        if let Some(node) = store.get_node(*self)? {
            let bbox = BBox::all();
            node.summary0(query, &bbox, store)
        } else {
            Ok(P::M::neutral())
        }
    }

    /// Count the number of elements in a 3d range.
    pub fn count_range(&self, query: &QueryRange3d<P>, store: &impl NodeStore<P>) -> Result<u64> {
        let bbox = BBox::all();
        self.count_range_rec(query, &bbox, store)
    }

    fn count_range_rec(
        &self,
        query: &QueryRange3d<P>,
        bbox: &BBox<P>,
        store: &impl NodeStore<P>,
    ) -> Result<u64> {
        let (lc, sc, rc) = self.count_range_parts(query, bbox, store)?;
        Ok(lc + sc + rc)
    }

    fn count_range_parts(
        &self,
        query: &QueryRange3d<P>,
        bbox: &BBox<P>,
        store: &impl NodeStore<P>,
    ) -> Result<(u64, u64, u64)> {
        Ok(if let Some(data) = store.data_opt(*self)? {
            let sc = if query.contains(data.key()) { 1 } else { 0 };
            let lc = if !data.left().is_empty() && query.overlaps_left(data.key(), data.rank()) {
                data.left().count_range_rec(
                    query,
                    &bbox.split_left(data.key(), data.rank()),
                    store,
                )?
            } else {
                0
            };
            let rc = if !data.right().is_empty() && query.overlaps_right(data.key(), data.rank()) {
                data.right().count_range_rec(
                    query,
                    &bbox.split_right(data.key(), data.rank()),
                    store,
                )?
            } else {
                0
            };
            (lc, sc, rc)
        } else {
            (0, 0, 0)
        })
    }

    pub fn split_range<'a>(
        &'a self,
        query: QueryRange3d<P>,
        split_factor: u64,
        store: &'a impl NodeStore<P>,
    ) -> impl Iterator<Item = Result<(QueryRange3d<P>, u64)>> + 'a
    where
        P::X: Display,
        P::Y: Display,
        P::ZOwned: Display,
    {
        Gen::new(|co| async move {
            let Ok(total_count) = self.count_range(&query, store) else {
                return;
            };
            if let Err(cause) = self
                .split_range2(query, total_count, split_factor, &co, store)
                .await
            {
                co.yield_(Err(cause)).await;
            }
        })
        .into_iter()
    }

    pub fn find_split_plane(
        &self,
        query: &QueryRange3d<P>,
        store: &impl NodeStore<P>,
    ) -> Result<Option<(QueryRange3d<P>, u64, QueryRange3d<P>, u64)>> {
        let total_count = self.count_range(query, store)?;
        self.find_split_plane0(*self, query, total_count, store)
    }

    fn find_split_plane0(
        &self,
        root: Node<P>,
        query: &QueryRange3d<P>,
        total_count: u64,
        store: &impl NodeStore<P>,
    ) -> Result<Option<(QueryRange3d<P>, u64, QueryRange3d<P>, u64)>> {
        if total_count <= 1 {
            return Ok(None);
        }
        if let Some(data) = store.data_opt(*self)? {
            let o = data.sort_order();
            for sort_order in [o, o.inc(), o.inc().inc()] {
                let left = query.left(data.key(), sort_order);
                let left_count = root.count_range(&left, store)?;
                if left_count < total_count && left_count > 0 {
                    let right_count = total_count - left_count;
                    let right = query.right(data.key(), sort_order);
                    return Ok(Some((left, left_count, right, right_count)));
                }
            }
            if let Some(res) = data
                .left()
                .find_split_plane0(root, query, total_count, store)?
            {
                return Ok(Some(res));
            }
            if let Some(res) = data
                .right()
                .find_split_plane0(root, query, total_count, store)?
            {
                return Ok(Some(res));
            }
            return Ok(None);
        } else {
            return Ok(None);
        }
    }

    async fn split_range2<'a>(
        &'a self,
        query: QueryRange3d<P>,
        total_count: u64,
        split_factor: u64,
        co: &'a Co<Result<(QueryRange3d<P>, u64)>>,
        store: &'a impl NodeStore<P>,
    ) -> Result<()> {
        if total_count == 0 {
            // nothing to split
            return Ok(());
        }
        if split_factor == 1 || total_count == 1 {
            // just send the whole thing
            co.yield_(Ok((query, total_count))).await;
            return Ok(());
        }
        let (left, left_count, right, right_count) = self
            .find_split_plane(&query, store)?
            .expect("must find split plane");
        let left_factor = ((left_count * split_factor) / total_count).max(1);
        let right_factor = split_factor - left_factor;
        Box::pin(self.split_range2(left, left_count, left_factor, co, store)).await?;
        Box::pin(self.split_range2(right, right_count, right_factor, co, store)).await?;
        Ok(())
    }

    async fn split_range0<'a>(
        &'a self,
        root: Node<P>,
        query: QueryRange3d<P>,
        total_count: u64,
        split_factor: u64,
        co: &'a Co<Result<(QueryRange3d<P>, u64)>>,
        store: &'a impl NodeStore<P>,
    ) -> Result<()>
    where
        P::X: Display,
        P::Y: Display,
        P::ZOwned: Display,
    {
        if total_count == 0 {
            return Ok(());
        }
        if total_count == 1 || split_factor == 1 {
            co.yield_(Ok((query.clone(), total_count))).await;
            return Ok(());
        }
        if let Some(data) = store.data_opt(*self)? {
            // let ((n1, q1, c1), (n2, q2, c2)) = root.find_split(total_count, &data, &query, store)?;
            // let (f1, f2) = if c1 == 0 {
            //     // just go right
            //     (0, split_factor)
            // } else if c2 == 0 {
            //     // just go left
            //     (split_factor, 0)
            // } else {
            //     // split in proportion to the number of elements, making sure that
            //     // both sides get at least one element
            //     assert!(c1 + c2 == total_count);
            //     assert!(total_count >= 2);
            //     assert!(split_factor >= 2);
            //     let f1 = ((c1 * split_factor) / total_count).max(1);
            //     let f2 = split_factor - f1;
            //     (f1, f2)
            // };
            // if c1 > 0 {
            //     Box::pin(n1.split_range0(root, q1, c1, f1, co, store)).await?;
            // }
            // if c2 > 0 {
            //     Box::pin(n2.split_range0(root, q2, c2, f2, co, store)).await?;
            // }
            // return Ok(());
            let sort_order = data.sort_order();
            let left = query.left(data.key(), sort_order);
            let right = query.right(data.key(), sort_order);
            // println!("splitting\n{} into\n{}\nand\n{}\nin\n{:?}", query.pretty(), left.pretty(), right.pretty(), data.sort_order());
            let left_count = root.count_range(&left, store)?;
            let right_count = total_count - left_count;
            if split_factor == 1 {
                co.yield_(Ok((query.clone(), total_count))).await;
                return Ok(());
            }
            let (left_factor, right_factor) = if left_count == 0 {
                // just go right
                (0, split_factor)
            } else if right_count == 0 {
                // just go left
                (split_factor, 0)
            } else {
                // split in proportion to the number of elements, making sure that
                // both sides get at least one element
                let left_factor = ((left_count * split_factor) / total_count).max(1);
                let right_factor = split_factor - left_factor;
                (left_factor, right_factor)
            };
            if left_factor > 0 {
                Box::pin(
                    data.left()
                        .split_range0(root, left, left_count, left_factor, co, store),
                )
                .await?;
            }
            if right_factor > 0 {
                Box::pin(data.right().split_range0(
                    root,
                    right,
                    right_count,
                    right_factor,
                    co,
                    store,
                ))
                .await?;
            }
        } else {
            // this is not necessarily 1, since we could have points from "above" that
            // are inside the query range
            let total_count = root.count_range(&query, store)?;
            if total_count > 0 {
                co.yield_(Ok((query.clone(), total_count))).await;
            }
        }
        Ok(())
    }

    /// Query a 3d range in the tree in its natural order.
    ///
    /// The order is implementation dependent and should not be relied on.
    pub fn query<'a>(
        &'a self,
        query: &'a QueryRange3d<P>,
        store: &'a impl NodeStore<P>,
    ) -> impl Iterator<Item = Result<(Point<P>, P::V)>> + 'a {
        Gen::new(|co| async move {
            if let Err(cause) = self.query0(query, store, &co).await {
                co.yield_(Err(cause)).await;
            }
        })
        .into_iter()
    }

    async fn query0(
        &self,
        query: &QueryRange3d<P>,
        store: &impl NodeStore<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        if let Some(data) = store.data_opt(*self)? {
            if query.overlaps_left(data.key(), data.rank()) {
                Box::pin(data.left().query0(query, store, co)).await?;
            }
            if query.contains(data.key()) {
                co.yield_(Ok((data.key().to_owned(), data.value().to_owned())))
                    .await;
            }
            if query.overlaps_right(data.key(), data.rank()) {
                Box::pin(data.right().query0(query, store, co)).await?;
            }
        }
        Ok(())
    }

    /// Query a 3d range in the tree in a defined order.
    ///
    /// The order is defined by the `ordering` parameter. All orderings use all
    /// three dimensions, so the result is fully deteministic.
    pub fn query_ordered<'a>(
        self,
        query: &'a QueryRange3d<P>,
        ordering: SortOrder,
        store: &'a impl NodeStore<P>,
    ) -> impl Iterator<Item = Result<(Point<P>, P::V)>> + 'a {
        Gen::new(|co| async move {
            if let Err(cause) = self.query_ordered0(query, ordering, store, &co).await {
                co.yield_(Err(cause)).await;
            }
        })
        .into_iter()
    }

    async fn query_ordered0(
        &self,
        query: &QueryRange3d<P>,
        ordering: SortOrder,
        store: &impl NodeStore<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        let Some(data) = store.data_opt(*self)? else {
            return Ok(());
        };
        // we know that the node partitions the space in the correct way, so
        // we can just concatenate the results from the left, self and right
        if data.sort_order() == ordering {
            if query.overlaps_left(data.key(), data.rank()) {
                Box::pin(data.left().query_ordered0(query, ordering, store, co)).await?;
            }
            if query.contains(data.key()) {
                co.yield_(Ok((data.key().to_owned(), data.value().to_owned())))
                    .await;
            }
            if query.overlaps_right(data.key(), data.rank()) {
                Box::pin(data.right().query_ordered0(query, ordering, store, co)).await?;
            }
        } else {
            // the node does not partition the space in the correct way, so we
            // need to merge the results from the left, self and right
            // still better than a full sort!
            let iter1 = if query.overlaps_left(data.key(), data.rank()) {
                itertools::Either::Left(
                    data.left()
                        .query_ordered(query, ordering, store)
                        .into_iter(),
                )
            } else {
                itertools::Either::Right(std::iter::empty())
            };
            let iter2 = if query.contains(data.key()) {
                itertools::Either::Left(std::iter::once(Ok((
                    data.key().to_owned(),
                    data.value().to_owned(),
                ))))
            } else {
                itertools::Either::Right(std::iter::empty())
            };
            let iter3 = if query.overlaps_right(data.key(), data.rank()) {
                itertools::Either::Left(
                    data.right()
                        .query_ordered(query, ordering, store)
                        .into_iter(),
                )
            } else {
                itertools::Either::Right(std::iter::empty())
            };
            merge3(iter1, iter2, iter3, ordering, co).await?;
        }
        Ok(())
    }

    /// Split an entire tree into leafs. The nodes are modified in place.
    ///
    /// Caution: the nodes are not persisted, so the returned vec is inconsistent
    /// with the store. You need to persist the nodes yourself.
    fn split_all(&self, store: &mut impl NodeStore<P>, res: &mut Vec<IdAndData<P>>) -> Result<()> {
        if let Some(mut data) = store.get_node(*self)? {
            data.left().split_all(store, res)?;
            data.right().split_all(store, res)?;
            // turn the node into a leaf
            data.deref_mut().make_leaf();
            // persist (should we do this here or later?)
            // data.persist(store)?;
            res.push(data);
        }
        Ok(())
    }
}

#[repr(transparent)]
#[derive(RefCast)]
pub struct NodeData<P: TreeParams>(PhantomData<P>, [u8]);

pub struct OwnedNodeData<P: TreeParams>(PhantomData<P>, Vec<u8>);

impl<P: TreeParams> Clone for OwnedNodeData<P> {
    fn clone(&self) -> Self {
        Self(PhantomData, self.1.clone())
    }
}

impl<P: TreeParams> PartialEq for OwnedNodeData<P> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<P: TreeParams> Eq for OwnedNodeData<P> {}

impl<P: TreeParams> OwnedNodeData<P> {
    pub fn new(data: Vec<u8>) -> Self {
        Self(PhantomData, data)
    }

    /// Creates a leaf data from the given key and value.
    pub fn leaf(key: &PointRef<P>, value: &P::V) -> Self {
        let data = vec![0; key_offset::<P>() + key.size()];
        let mut res = Self(PhantomData, data);
        *res.left_mut() = Node::EMPTY; // not strictly necessary, since it's already 0
        *res.right_mut() = Node::EMPTY; // not strictly necessary, since it's already 0
        *res.rank_mut() = key.rank();
        *res.count_mut() = 1.into();
        *res.value_mut() = value.clone();
        *res.summary_mut() = P::M::lift(key, value);
        res.1[key_offset::<P>()..].copy_from_slice(key.as_slice());
        res
    }
}

impl<P: TreeParams> Debug for OwnedNodeData<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.deref())
    }
}

impl<P: TreeParams> ToOwned for NodeData<P> {
    type Owned = OwnedNodeData<P>;

    fn to_owned(&self) -> Self::Owned {
        OwnedNodeData(PhantomData, self.1.to_vec())
    }
}

impl<P: TreeParams> Borrow<NodeData<P>> for OwnedNodeData<P> {
    fn borrow(&self) -> &NodeData<P> {
        NodeData::ref_cast(&self.1)
    }
}

impl<P: TreeParams> Deref for OwnedNodeData<P> {
    type Target = NodeData<P>;

    fn deref(&self) -> &Self::Target {
        NodeData::ref_cast(&self.1)
    }
}

impl<P: TreeParams> DerefMut for OwnedNodeData<P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        NodeData::ref_cast_mut(&mut self.1)
    }
}

impl<P: TreeParams> Debug for NodeData<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeData")
            .field("left", &self.left())
            .field("right", &self.right())
            .field("value", &self.value())
            .field("summary", &self.summary())
            .field("rank", &self.rank())
            .field("key", &self.key())
            .finish()
    }
}

impl<P: TreeParams> NodeData<P> {
    // pub fn new(data: T) -> Self {
    //     debug_assert!(data.as_ref().len() >= min_data_size::<P>());
    //     Self(data, PhantomData)
    // }

    pub fn as_slice(&self) -> &[u8] {
        &self.1
    }

    pub fn hex(&self) -> String {
        hex::encode(self.as_slice())
    }

    pub fn is_leaf(&self) -> bool {
        self.left().is_empty() && self.right().is_empty()
    }

    pub fn left(&self) -> Node<P> {
        Node::read_from_prefix(&self.1[LEFT_OFFSET..]).unwrap()
    }

    pub fn right(&self) -> Node<P> {
        Node::read_from_prefix(&self.1[RIGHT_OFFSET..]).unwrap()
    }

    pub fn value(&self) -> &P::V {
        P::V::ref_from_prefix(&self.1[VALUE_OFFSET..]).unwrap()
    }

    pub fn summary(&self) -> &P::M {
        P::M::ref_from_prefix(&self.1[summary_offset::<P>()..]).unwrap()
    }

    pub fn rank(&self) -> u8 {
        self.1.as_ref()[RANK_OFFSET]
    }

    pub fn count(&self) -> u64 {
        zerocopy::big_endian::U64::read_from_prefix(&self.1[COUNT_OFFSET..])
            .unwrap()
            .into()
    }

    pub fn sort_order(&self) -> SortOrder {
        SortOrder::from(self.rank())
    }

    pub fn key(&self) -> &PointRef<P> {
        let slice = &self.1[key_offset::<P>()..];
        PointRef::ref_cast(slice)
    }

    pub fn make_leaf(&mut self) {
        *self.left_mut() = Node::EMPTY;
        *self.right_mut() = Node::EMPTY;
        *self.summary_mut() = P::M::lift(self.key(), self.value());
        *self.count_mut() = 1.into();
    }

    fn summary0(
        &self,
        query: &QueryRange3d<P>,
        bbox: &BBox<P>,
        store: &impl NodeStore<P>,
    ) -> Result<P::M> {
        if self.is_leaf() {
            if query.contains(self.key()) {
                return Ok(self.summary().clone());
            } else {
                return Ok(P::M::neutral());
            }
        }
        if bbox.contained_in(&query) {
            return Ok(self.summary().clone());
        }
        let mut summary = P::M::neutral();
        if query.overlaps_left(self.key(), self.rank()) {
            if let Some(left) = store.data_opt(self.left())? {
                let left_bbox = bbox.split_left(self.key(), self.rank());
                summary = summary.combine(&left.summary0(query, &left_bbox, store)?);
            }
        }
        if query.contains(self.key()) {
            summary = summary.combine(&P::M::lift(self.key(), self.value()));
        }
        if query.overlaps_right(self.key(), self.rank()) {
            if let Some(right) = store.data_opt(self.right())? {
                let right_bbox = bbox.split_right(self.key(), self.rank());
                summary = summary.combine(&right.summary0(query, &right_bbox, store)?);
            }
        }
        Ok(summary)
    }

    fn recalculate_summary(&mut self, store: &impl NodeStore<P>) -> Result<()> {
        let mut res = P::M::neutral();
        let mut count = 1;
        if let Some(left) = store.data_opt(self.left())? {
            res = res.combine(left.summary());
            count += left.count();
        }
        res = res.combine(&P::M::lift(self.key(), self.value()));
        if let Some(right) = store.data_opt(self.right())? {
            res = res.combine(right.summary());
            count += right.count();
        }
        *self.summary_mut() = res;
        *self.count_mut() = count.into();
        Ok(())
    }

    fn add_summary(&mut self, value: P::V) {
        *self.summary_mut() = self.summary().combine(&P::M::lift(self.key(), &value));
        *self.count_mut() = (self.count() + 1).into();
    }

    pub fn assert_invariants(
        &self,
        store: &impl NodeStore<P>,
        include_summary: bool,
    ) -> Result<AssertInvariantsRes<P>> {
        let left = store.data_opt(self.left())?;
        let right = store.data_opt(self.right())?;
        let left_res = left
            .as_ref()
            .map(|node| node.assert_invariants(store, include_summary))
            .transpose()?;
        let right_res = right
            .as_ref()
            .map(|node| node.assert_invariants(store, include_summary))
            .transpose()?;
        if let Some(ref left) = left_res {
            assert!(left.rank < self.rank());
        }
        if let Some(ref right) = right_res {
            assert!(right.rank <= self.rank());
        }
        match SortOrder::from(self.rank()) {
            SortOrder::XYZ => {
                if let Some(ref left) = left_res {
                    assert!(
                        left.xyz_max
                            .deref()
                            .cmp_with_order(self.key(), SortOrder::XYZ)
                            == Ordering::Less
                    );
                }
                if let Some(ref right) = right_res {
                    assert!(
                        right
                            .xyz_min
                            .deref()
                            .cmp_with_order(self.key(), SortOrder::XYZ)
                            == Ordering::Greater
                    );
                }
            }
            SortOrder::YZX => {
                if let Some(ref left) = left_res {
                    assert!(
                        left.yzx_max
                            .deref()
                            .cmp_with_order(self.key(), SortOrder::YZX)
                            == Ordering::Less
                    );
                }
                if let Some(ref right) = right_res {
                    assert!(
                        right
                            .yzx_min
                            .deref()
                            .cmp_with_order(self.key(), SortOrder::YZX)
                            == Ordering::Greater
                    );
                }
            }
            SortOrder::ZXY => {
                if let Some(ref left) = left_res {
                    assert!(
                        left.zxy_max
                            .deref()
                            .cmp_with_order(self.key(), SortOrder::ZXY)
                            == Ordering::Less
                    );
                }
                if let Some(ref right) = right_res {
                    assert!(
                        right
                            .zxy_min
                            .deref()
                            .cmp_with_order(self.key(), SortOrder::ZXY)
                            == Ordering::Greater
                    );
                }
            }
        }
        let mut count = 1;
        let mut res =
            AssertInvariantsRes::single(self.key().to_owned(), self.rank(), self.value().clone());
        if let Some(left) = left_res {
            res = res.combine(&left);
            count += left.count;
        }
        if let Some(right) = right_res {
            res = res.combine(&right);
            count += right.count;
        }
        if include_summary {
            assert_eq!(&res.summary, self.summary());
            assert_eq!(res.count, count);
        }
        Ok(res)
    }

    // fn summary0(
    //     &self,
    //     query: &QueryRange3d<P>,
    //     bbox: &BBox<P>,
    //     store: &impl StoreExt<P>,
    // ) -> Result<P::M> {
    //     if self.is_leaf() {
    //         if query.contains(self.key()) {
    //             return Ok(self.summary().clone());
    //         } else {
    //             return Ok(P::M::neutral());
    //         }
    //     }
    //     if bbox.contained_in(&query) {
    //         return Ok(self.summary().clone());
    //     }
    //     let mut summary = P::M::neutral();
    //     if query.overlaps_left(self.key(), self.rank()) {
    //         if let Some(left) = store.data_opt(self.left())? {
    //             let left_bbox = bbox.split_left(self.key(), self.rank());
    //             summary = summary.combine(&left.summary0(query, &left_bbox, store)?);
    //         }
    //     }
    //     if query.contains(self.key()) {
    //         summary = summary.combine(&P::M::lift((self.key().clone(), self.value().clone())));
    //     }
    //     if query.overlaps_right(self.key(), self.rank()) {
    //         if let Some(right) = store.data_opt(self.right())? {
    //             let right_bbox = bbox.split_right(self.key(), self.rank());
    //             summary = summary.combine(&right.summary0(query, &right_bbox, store)?);
    //         }
    //     }
    //     Ok(summary)
    // }
}

impl<P: TreeParams> NodeData<P> {
    pub fn left_mut(&mut self) -> &mut Node<P> {
        Node::mut_from_prefix(&mut self.1[LEFT_OFFSET..]).unwrap()
    }

    pub fn right_mut(&mut self) -> &mut Node<P> {
        Node::mut_from_prefix(&mut self.1[RIGHT_OFFSET..]).unwrap()
    }

    pub fn summary_mut(&mut self) -> &mut P::M {
        P::M::mut_from_prefix(&mut self.1[summary_offset::<P>()..]).unwrap()
    }

    pub fn count_mut(&mut self) -> &mut zerocopy::big_endian::U64 {
        zerocopy::big_endian::U64::mut_from_prefix(&mut self.1[COUNT_OFFSET..]).unwrap()
    }

    pub fn value_mut(&mut self) -> &mut P::V {
        P::V::mut_from_prefix(&mut self.1[VALUE_OFFSET..]).unwrap()
    }

    pub fn rank_mut(&mut self) -> &mut u8 {
        &mut self.1.as_mut()[RANK_OFFSET]
    }

    // fn recalculate_summary(&mut self, store: &impl StoreExt<P>) -> Result<()>
    //     where T: AsRef<[u8]>
    // {
    //     let mut res = P::M::neutral();
    //     if let Some(left) = store.data_opt(self.left())? {
    //         res = res.combine(left.summary());
    //     }
    //     res = res.combine(&P::M::lift((self.key().clone(), self.value().clone())));
    //     if let Some(right) = store.data_opt(self.right())? {
    //         res = res.combine(right.summary());
    //     }
    //     *self.summary_mut() = res;
    //     Ok(())
    // }

    // fn add_summary(&mut self, value: P::V)
    //     where T: AsRef<[u8]>
    // {
    //     *self.summary_mut() = self
    //         .summary()
    //         .combine(&P::M::lift((self.key().clone(), value)));
    // }
}

/// A node combines node data with a node id.
///
/// When modifying a node, the id should not be modified. After a modification,
/// the node should be updated in the store, otherwise there will be horrible
/// inconsistencies.
pub struct IdAndData<P: TreeParams> {
    id: Node<P>,
    data: OwnedNodeData<P>,
}

impl<P: TreeParams> Debug for IdAndData<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdAndData")
            .field("id", &self.id)
            .field("data", &self.data)
            .finish()
    }
}

impl<P: TreeParams> Clone for IdAndData<P> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            data: self.data.clone(),
        }
    }
}

impl<P: TreeParams> Deref for IdAndData<P> {
    type Target = NodeData<P>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<P: TreeParams> DerefMut for IdAndData<P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<P: TreeParams> IdAndData<P> {
    fn new(id: Node<P>, data: OwnedNodeData<P>) -> Self {
        assert!(!id.is_empty());
        IdAndData { id, data }
    }

    fn persist(&self, store: &mut impl NodeStore<P>) -> Result<()> {
        if !self.left().is_empty() {
            let left = store.data(self.left())?;
            assert!(self.key().cmp_at_rank(left.key(), self.rank()) == Ordering::Greater);
            assert!(left.rank() < self.rank());
        }
        if !self.right().is_empty() {
            let right = store.data(self.right())?;
            assert!(self.key().cmp_at_rank(right.key(), self.rank()) == Ordering::Less);
            assert!(right.rank() <= self.rank());
        }
        store.update_node(self.id, &self.data)
    }

    // Insert a new node into the tree without balancing.
    fn insert_no_balance(&mut self, node: &Self, store: &mut impl NodeStore<P>) -> Result<()> {
        assert!(node.is_leaf());
        match node.key().cmp_at_rank(self.key(), self.rank()) {
            Ordering::Less => {
                self.left_mut().insert_no_balance(node, store)?;
            }
            Ordering::Greater => {
                self.right_mut().insert_no_balance(node, store)?;
            }
            Ordering::Equal => {
                panic!("Duplicate keys not supported in insert_no_balance");
            }
        }
        *self.summary_mut() = self.summary().combine(node.summary());
        *self.count_mut() = (self.count() + node.count()).into();
        self.persist(store)?;
        Ok(())
    }
}

pub fn count_trailing_zeros(hash: &[u8; 32]) -> u8 {
    let mut rank = 0;
    for byte in hash.iter().rev() {
        if *byte == 0 {
            rank += 8;
        } else {
            rank += byte.trailing_zeros() as u8;
            break;
        }
    }
    rank
}

async fn merge3<P: TreeParams>(
    a: impl Iterator<Item = Result<(Point<P>, P::V)>>,
    b: impl Iterator<Item = Result<(Point<P>, P::V)>>,
    c: impl Iterator<Item = Result<(Point<P>, P::V)>>,
    ordering: SortOrder,
    co: &Co<Result<(Point<P>, P::V)>>,
) -> Result<()> {
    enum Smallest {
        A,
        B,
        C,
    }
    use Smallest::*;
    let mut a = a.peekable();
    let mut b = b.peekable();
    let mut c = c.peekable();
    let cmp = |a: &Result<(Point<P>, P::V)>, b: &Result<(Point<P>, P::V)>| match (a, b) {
        (Ok((ak, _)), Ok((bk, _))) => ak.cmp_with_order(&bk, ordering),
        (Err(_), Ok(_)) => Ordering::Less,
        (Ok(_), Err(_)) => Ordering::Greater,
        (Err(_), Err(_)) => Ordering::Equal,
    };
    loop {
        let min = match (a.peek(), b.peek(), c.peek()) {
            (Some(a), Some(b), Some(c)) => {
                if cmp(a, b) == Ordering::Less {
                    if cmp(a, c) == Ordering::Less {
                        A
                    } else {
                        C
                    }
                } else {
                    if cmp(b, c) == Ordering::Less {
                        B
                    } else {
                        C
                    }
                }
            }
            (Some(a), Some(b), None) => {
                if cmp(a, b) == Ordering::Less {
                    A
                } else {
                    B
                }
            }
            (Some(a), None, Some(c)) => {
                if cmp(a, c) == Ordering::Less {
                    A
                } else {
                    C
                }
            }
            (None, Some(b), Some(c)) => {
                if cmp(b, c) == Ordering::Less {
                    B
                } else {
                    C
                }
            }
            (Some(_), None, None) => A,
            (None, Some(_), None) => B,
            (None, None, Some(_)) => C,
            (None, None, None) => break,
        };
        match min {
            Smallest::A => co.yield_(a.next().unwrap()).await,
            Smallest::B => co.yield_(b.next().unwrap()).await,
            Smallest::C => co.yield_(c.next().unwrap()).await,
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum SortOrder {
    XYZ,
    YZX,
    ZXY,
}

impl From<u8> for SortOrder {
    fn from(rank: u8) -> Self {
        match rank % 3 {
            2 => SortOrder::XYZ,
            1 => SortOrder::YZX,
            0 => SortOrder::ZXY,
            _ => unreachable!(),
        }
    }
}

impl SortOrder {
    fn inc(self) -> Self {
        match self {
            SortOrder::XYZ => SortOrder::YZX,
            SortOrder::YZX => SortOrder::ZXY,
            SortOrder::ZXY => SortOrder::XYZ,
        }
    }
}
