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
use core::panic;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use anyhow::Result;
use genawaiter::sync::{Co, Gen};
use point::{XYZ, YZX, ZXY};
use serde::Serialize;

mod point;
pub use point::Point;
use zerocopy::{AsBytes, FromBytes, FromZeroes};

macro_rules! assert_lt {
    ($left:expr, $right:expr) => {
        assert!($left < $right, "{:?} < {:?}", $left, $right)
    };
}

macro_rules! assert_gt {
    ($left:expr, $right:expr) => {
        assert!($left > $right, "{:?} > {:?}", $left, $right)
    };
}

///
pub trait CoordParams:
    Ord + PartialEq + Eq + Serialize + Clone + Debug + Display + FromBytes + AsBytes
{
}

impl<
        T: Ord
            + PartialEq
            + Eq
            + Serialize
            + Clone
            + Debug
            + Display
            + FromBytes
            + FromZeroes
            + AsBytes,
    > CoordParams for T
{
}

pub trait FixedSize {
    const SIZE: usize;
}

pub trait VariableSize {
    fn size(&self) -> usize;
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
    fn write(&self, buf: &mut [u8]) {
        buf.copy_from_slice(&self.to_be_bytes());
    }
    fn read(buf: &[u8]) -> Self {
        u64::from_be_bytes(buf.try_into().unwrap())
    }
}

impl FixedSize for NodeId {
    const SIZE: usize = 8;
}

impl FixedSize for u8 {
    const SIZE: usize = 1;
}

pub trait KeyParams {
    type X: CoordParams + FixedSize;
    type Y: CoordParams + FixedSize;
    type Z: CoordParams + VariableSize;
}

pub trait LiftingCommutativeMonoid<T> {
    fn neutral() -> Self;
    fn lift(value: T) -> Self;
    fn combine(&self, other: &Self) -> Self;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryRange<T> {
    // min is inclusive
    min: T,
    // max is exclusive, None means unbounded
    max: Option<T>,
}

impl<T: Display> Display for QueryRange<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.max {
            Some(max) => write!(f, "[{}, {})", self.min, max),
            None => write!(f, "[{}, ∞)", self.min),
        }
    }
}

impl<T: Ord> QueryRange<T> {
    pub fn new(min: T, max: Option<T>) -> Self {
        Self { min, max }
    }

    pub fn contains(&self, value: &T) -> bool {
        if value < &self.min {
            return false;
        }
        if let Some(max) = &self.max {
            if value >= max {
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
}

pub struct QueryRange3d<P: KeyParams> {
    x: QueryRange<P::X>,
    y: QueryRange<P::Y>,
    z: QueryRange<P::Z>,
}

impl<P: KeyParams> Display for QueryRange3d<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}

impl<P: KeyParams> Debug for QueryRange3d<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryRange3d")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .finish()
    }
}

impl<P: KeyParams> QueryRange3d<P> {
    pub fn new(x: QueryRange<P::X>, y: QueryRange<P::Y>, z: QueryRange<P::Z>) -> Self {
        Self { x, y, z }
    }

    pub fn contains(&self, point: &Point<P>) -> bool {
        self.x.contains(&point.x) && self.y.contains(&point.y) && self.z.contains(&point.z)
    }

    pub fn overlaps_left(&self, key: &Point<P>, rank: u8) -> bool {
        match SortOrder::from(rank) {
            SortOrder::XYZ => self.x.min <= key.x,
            SortOrder::YZX => self.y.min <= key.y,
            SortOrder::ZXY => self.z.min <= key.z,
        }
    }

    pub fn overlaps_right(&self, key: &Point<P>, rank: u8) -> bool {
        match SortOrder::from(rank) {
            SortOrder::XYZ => !self
                .x
                .max
                .as_ref()
                .map(|x_max| x_max < &key.x)
                .unwrap_or_default(),
            SortOrder::YZX => !self
                .y
                .max
                .as_ref()
                .map(|y_max| y_max < &key.y)
                .unwrap_or_default(),
            SortOrder::ZXY => !self
                .z
                .max
                .as_ref()
                .map(|z_max| z_max < &key.z)
                .unwrap_or_default(),
        }
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

impl<T: CoordParams> Display for RangeInclusiveOpt<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.min, &self.max) {
            (Some(min), Some(max)) => write!(f, "[{}, {}]", min, max),
            (Some(min), None) => write!(f, "[{}, ∞)", min),
            (None, Some(max)) => write!(f, "(-∞, {}]", max),
            (None, None) => write!(f, "(-∞, ∞)"),
        }
    }
}

impl<T: CoordParams> Debug for RangeInclusiveOpt<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RangeInclusiveOpt")
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

impl<T: CoordParams> Clone for RangeInclusiveOpt<T> {
    fn clone(&self) -> Self {
        Self {
            min: self.min.clone(),
            max: self.max.clone(),
        }
    }
}

impl<T: CoordParams> RangeInclusiveOpt<T> {
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

    pub fn union(&self, that: RangeInclusiveOpt<T>) -> Self {
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

    pub fn contains(&self, value: &T) -> bool {
        if let Some(min) = &self.min {
            if value < &min {
                return false;
            }
        }
        if let Some(max) = &self.max {
            if value > &max {
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
    z: RangeInclusiveOpt<P::Z>,
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
    xyz: RangeInclusive<XYZ<S>>,
    /// min and max in yzx order
    yzx: RangeInclusive<YZX<S>>,
    /// min and max in zxy order
    zxy: RangeInclusive<ZXY<S>>,
    /// summary of the node
    summary: S::M,
    /// rank of the node
    rank: u8,
}

impl<T: TreeParams> AssertInvariantsRes<T> {
    pub fn single(point: Point<T>, rank: u8, value: T::V) -> Self {
        let xyz = point.clone().xyz();
        let yzx = point.clone().yzx();
        let zxy = point.clone().zxy();
        AssertInvariantsRes {
            xyz: RangeInclusive {
                min: xyz.clone(),
                max: xyz,
            },
            yzx: RangeInclusive {
                min: yzx.clone(),
                max: yzx,
            },
            zxy: RangeInclusive {
                min: zxy.clone(),
                max: zxy,
            },
            summary: T::M::lift((point, value)),
            rank,
        }
    }

    pub fn combine(&self, other: &Self) -> Self {
        let xyz = self.xyz.union(&other.xyz);
        let yzx = self.yzx.union(&other.yzx);
        let zxy = self.zxy.union(&other.zxy);
        let rank = self.rank.max(other.rank);
        let summary = self.summary.combine(&other.summary);
        AssertInvariantsRes {
            xyz,
            yzx,
            zxy,
            rank,
            summary,
        }
    }
}

impl<P: KeyParams> BBox<P> {
    pub fn new(
        x: RangeInclusiveOpt<P::X>,
        y: RangeInclusiveOpt<P::Y>,
        z: RangeInclusiveOpt<P::Z>,
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

    pub fn contains(&self, point: &Point<P>) -> bool {
        self.x.contains(&point.x) && self.y.contains(&point.y) && self.z.contains(&point.z)
    }

    pub fn contained_in(&self, query: &QueryRange3d<P>) -> bool {
        query.x.contains_range_inclusive_opt(&self.x)
            && query.y.contains_range_inclusive_opt(&self.y)
            && query.z.contains_range_inclusive_opt(&self.z)
    }

    pub fn split_left(&self, key: &Point<P>, rank: u8) -> BBox<P> {
        match SortOrder::from(rank) {
            SortOrder::XYZ => BBox {
                x: RangeInclusiveOpt::new(self.x.min.clone(), Some(key.x.clone())),
                y: self.y.clone(),
                z: self.z.clone(),
            },
            SortOrder::YZX => BBox {
                x: self.x.clone(),
                y: RangeInclusiveOpt::new(self.y.min.clone(), Some(key.y.clone())),
                z: self.z.clone(),
            },
            SortOrder::ZXY => BBox {
                x: self.x.clone(),
                y: self.y.clone(),
                z: RangeInclusiveOpt::new(self.z.min.clone(), Some(key.z.clone())),
            },
        }
    }

    pub fn split_right(&self, key: &Point<P>, rank: u8) -> BBox<P> {
        match SortOrder::from(rank) {
            SortOrder::XYZ => BBox {
                x: RangeInclusiveOpt::new(Some(key.x.clone()), self.x.max.clone()),
                y: self.y.clone(),
                z: self.z.clone(),
            },
            SortOrder::YZX => BBox {
                x: self.x.clone(),
                y: RangeInclusiveOpt::new(Some(key.y.clone()), self.y.max.clone()),
                z: self.z.clone(),
            },
            SortOrder::ZXY => BBox {
                x: self.x.clone(),
                y: self.y.clone(),
                z: RangeInclusiveOpt::new(Some(key.z.clone()), self.z.max.clone()),
            },
        }
    }
}

///
pub trait ValueParams:
    PartialEq + Eq + Serialize + Clone + Debug + FixedSize + AsBytes + FromBytes
{
}

impl<T: PartialEq + Eq + Serialize + Clone + Debug + FixedSize + AsBytes + FromBytes> ValueParams
    for T
{
}

/// Tree params for a 3D tree. This extends `KeyParams` with a value and
/// summary type.
pub trait TreeParams: KeyParams + Sized {
    type V: ValueParams;
    type M: LiftingCommutativeMonoid<(Point<Self>, Self::V)>
        + Clone
        + Debug
        + Eq
        + AsBytes
        + FixedSize
        + FromBytes;
}

#[inline(always)]
fn min_key_size<P: KeyParams>() -> usize {
    P::X::SIZE + P::Y::SIZE
}

#[inline(always)]
fn min_data_size<P: TreeParams>() -> usize {
    8 + 8 + 1 + P::V::SIZE + P::M::SIZE + min_key_size::<P>()
}

#[inline(always)]
fn left_offset<P: TreeParams>() -> usize {
    0
}

#[inline(always)]
fn right_offset<P: TreeParams>() -> usize {
    8
}

#[inline(always)]
fn value_offset<P: TreeParams>() -> usize {
    16
}

#[inline(always)]
fn summary_offset<P: TreeParams>() -> usize {
    16 + P::V::SIZE
}

#[inline(always)]
fn rank_offset<P: TreeParams>() -> usize {
    16 + P::V::SIZE + P::M::SIZE
}

#[inline(always)]
fn key_offset<P: TreeParams>() -> usize {
    16 + P::V::SIZE + P::M::SIZE + 1
}

/// A simple store trait for storing blobs.
pub trait Store<T: VariableSize> {
    fn create(&mut self, node: &T) -> Result<NodeId>;
    fn read(&self, id: NodeId) -> Result<T>;
    fn update(&mut self, id: NodeId, node: &T) -> Result<()>;
    fn delete(&mut self, id: NodeId) -> Result<()>;
}

impl<T: VariableSize> Store<T> for Box<dyn Store<T>> {
    fn create(&mut self, node: &T) -> Result<NodeId> {
        self.as_mut().create(node)
    }

    fn read(&self, id: NodeId) -> Result<T> {
        self.as_ref().read(id)
    }

    fn update(&mut self, id: NodeId, node: &T) -> Result<()> {
        self.as_mut().update(id, node)
    }

    fn delete(&mut self, id: NodeId) -> Result<()> {
        self.as_mut().delete(id)
    }
}

pub trait StoreExt<P: TreeParams>: Store<NodeData<P>> {
    fn put_node(&mut self, data: NodeData<P>) -> Result<IdAndData<P>> {
        let id = Node(self.create(&data)?, PhantomData);
        Ok(IdAndData::new(id, data))
    }

    fn data(&self, id: Node<P>) -> Result<NodeData<P>> {
        Store::read(self, id.0)
    }

    /// Get a node by id, returning None if the id is None.
    ///
    /// This is just a convenience method for the common case where you have an
    /// optional id and want to get the node if it exists.
    fn data_opt(&self, id: Node<P>) -> Result<Option<NodeData<P>>> {
        Ok(if id.is_empty() {
            None
        } else {
            Some(self.read(id.0)?)
        })
    }

    /// Get a node by id, returning None if the id is None.
    /// Also return the id along with the node.
    fn get_node(&self, id: Node<P>) -> Result<Option<IdAndData<P>>> {
        Ok(if id.is_empty() {
            None
        } else {
            IdAndData::new(id, self.read(id.0)?).into()
        })
    }

    /// Get a node by id, returning None if the id is None.
    /// Also return the id along with the node.
    fn get_non_empty(&self, id: Node<P>) -> Result<IdAndData<P>> {
        if id.is_empty() {
            panic!("Empty id");
        }
        Ok(IdAndData::new(id, self.read(id.0)?))
    }
}

impl<T: Store<NodeData<P>>, P: TreeParams> StoreExt<P> for T {}

pub struct MemStore {
    nodes: BTreeMap<NodeId, Vec<u8>>,
}

impl MemStore {
    pub fn new() -> Self {
        MemStore {
            nodes: BTreeMap::new(),
        }
    }
}

impl<T: VariableSize> Store<T> for MemStore {
    fn create(&mut self, node: &T) -> Result<NodeId> {
        let id = NodeId::from((self.nodes.len() as u64) + 1);
        assert!(!id.is_empty());
        self.nodes.insert(id, node.to_vec());
        Ok(id)
    }

    fn update(&mut self, id: NodeId, node: &T) -> Result<()> {
        assert!(!id.is_empty());
        self.nodes.insert(id, node.to_vec());
        Ok(())
    }

    fn read(&self, id: NodeId) -> Result<T> {
        assert!(!id.is_empty());
        match self.nodes.get(&id) {
            Some(data) => Ok(T::read(data)),
            None => Err(anyhow::anyhow!("Node not found")),
        }
    }

    fn delete(&mut self, id: NodeId) -> Result<()> {
        assert!(!id.is_empty());
        self.nodes.remove(&id);
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, AsBytes, FromZeroes, FromBytes)]
#[repr(transparent)]
pub struct NodeId([u8; 8]);

impl Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id = u64::from_be_bytes(self.0);
        write!(f, "NodeId({})", id)
    }
}

impl Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id = u64::from_be_bytes(self.0);
        write!(f, "{}", id)
    }
}

impl From<u64> for NodeId {
    fn from(id: u64) -> Self {
        NodeId(id.to_be_bytes())
    }
}

impl NodeId {
    pub const EMPTY: Self = NodeId([0; 8]);

    pub fn is_empty(&self) -> bool {
        self == &Self::EMPTY
    }
}

#[repr(transparent)]
#[derive(AsBytes, FromZeroes, FromBytes)]
pub struct Node<P: TreeParams>(NodeId, PhantomData<P>);

impl<P: TreeParams> From<NodeId> for Node<P> {
    fn from(id: NodeId) -> Self {
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
    pub const EMPTY: Self = Self(NodeId::EMPTY, PhantomData);

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn count(&self, store: &impl StoreExt<P>) -> Result<u64> {
        Ok(if let Some(data) = store.data_opt(*self)? {
            1 + data.left().count(store)? + data.right().count(store)?
        } else {
            0
        })
    }

    pub fn id(&self) -> NodeId {
        self.0
    }

    pub fn insert(
        &mut self,
        key: Point<P>,
        value: P::V,
        store: &mut impl StoreExt<P>,
    ) -> Result<Option<P::V>> {
        self.insert_rec(NodeData::single(key, value), store)
    }

    fn insert_rec(&mut self, x: NodeData<P>, store: &mut impl StoreExt<P>) -> Result<Option<P::V>> {
        if let Some(mut this) = store.get_node(*self)? {
            let x_cmp_cur = x.key().cmp_at_rank(this.key(), this.rank());
            if x_cmp_cur == Ordering::Equal {
                // just replace the value
                let mut res = x.value;
                std::mem::swap(&mut res, &mut this.value);
                let res = Some(res);
                this.recalculate_summary(store)?;
                this.persist(store)?;
                Ok(res)
            } else if x.rank < this.rank || (x.rank == this.rank && x_cmp_cur == Ordering::Greater)
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
            *self = store.create(&x)?.into();
            Ok(None)
        }
    }

    pub fn delete(&mut self, key: Point<P>, store: &mut impl StoreExt<P>) -> Result<Option<P::V>> {
        self.delete_rec(key, store)
    }

    fn delete_rec(&mut self, key: Point<P>, store: &mut impl StoreExt<P>) -> Result<Option<P::V>> {
        if let Some(mut this) = store.get_node(*self)? {
            let key_cmp_cur = key.cmp_at_rank(this.key(), this.rank);
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

    pub fn assert_invariants(&self, store: &impl StoreExt<P>, include_summary: bool) -> Result<()> {
        if !self.is_empty() {
            let data = store.get_non_empty(*self)?;
            data.assert_invariants(store, include_summary)?;
        }
        Ok(())
    }

    pub fn get(&self, key: Point<P>, store: &impl StoreExt<P>) -> Result<Option<P::V>> {
        Ok(self.get0(key, store)?.map(|x| x.value().clone()))
    }

    fn get0(&self, key: Point<P>, store: &impl StoreExt<P>) -> Result<Option<NodeData<P>>> {
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

    pub fn dump(&self, store: &impl StoreExt<P>) -> Result<()> {
        self.dump0("".into(), store)
    }

    fn dump0(&self, prefix: String, store: &impl StoreExt<P>) -> Result<()> {
        if let Some(data) = store.data_opt(*self)? {
            println!("{} left:", prefix);
            data.left().dump0(format!("{}  ", prefix), store)?;
            println!(
                "{}{:?} rank={} order={:?} value={:?} summary={:?}",
                prefix,
                data.key(),
                data.rank(),
                SortOrder::from(data.rank()),
                data.value(),
                data.summary(),
            );
            println!("{} right:", prefix);
            data.right().dump0(format!("{}  ", prefix), store)?;
        } else {
            println!("{}Empty", prefix);
        }
        Ok(())
    }

    pub fn from_iter<I: IntoIterator<Item = (Point<P>, P::V)>>(
        iter: I,
    ) -> Result<(impl StoreExt<P>, Node<P>)> {
        let mut nodes: Vec<_> = iter
            .into_iter()
            .map(|(key, value)| NodeData::single(key, value))
            .collect();
        // Before we sort, remove all but the first occurence of each point.
        let mut uniques = BTreeSet::new();
        nodes.retain(|node| uniques.insert(node.key().clone().xyz()));
        let mut store = MemStore::new();
        let nodes = nodes
            .into_iter()
            .map(|data| store.put_node(data))
            .collect::<Result<_>>()?;
        let node = Node::from_unique_nodes(&mut store, nodes)?;
        // if rank is equal, compare keys at rank
        Ok((store, node))
    }

    pub fn from_unique_nodes(
        store: &mut impl StoreExt<P>,
        mut nodes: Vec<IdAndData<P>>,
    ) -> Result<Node<P>> {
        // if rank is equal, compare keys at rank
        nodes.sort_by(|p1, p2| {
            p2.rank()
                .cmp(&p1.rank())
                .then(p1.key().cmp_at_rank(p2.key(), p1.rank()))
        });
        let mut tree = Node::EMPTY;
        // println!("merging {} nodes", nodes.len());
        // for node in &nodes {
        //     println!("{:?} {}", node.key, node.rank);
        // }
        for node in nodes {
            // let node = store.put_node(node.data)?;
            // println!("{} {}", i, tree.is_empty());
            // tree.dump(store)?;
            node.persist(store)?;
            tree.insert_no_balance(&node, store)?;
        }
        Ok(tree)
    }

    fn insert_no_balance(
        &mut self,
        node: &IdAndData<P>,
        store: &mut impl StoreExt<P>,
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
        store: &'a impl StoreExt<P>,
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
        store: &impl StoreExt<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        if let Some(data) = store.data_opt(*self)? {
            Box::pin(data.left().iter0(store, co)).await?;
            co.yield_(Ok((data.key().clone(), data.value().clone())))
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
    pub fn summary(&self, query: &QueryRange3d<P>, store: &impl StoreExt<P>) -> Result<P::M> {
        if let Some(node) = store.get_node(*self)? {
            let bbox = BBox::all();
            node.summary0(query, &bbox, store)
        } else {
            Ok(P::M::neutral())
        }
    }

    /// Query a 3d range in the tree in its natural order.
    ///
    /// The order is implementation dependent and should not be relied on.
    pub fn query<'a>(
        &'a self,
        query: &'a QueryRange3d<P>,
        store: &'a impl StoreExt<P>,
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
        store: &impl StoreExt<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        if let Some(data) = store.data_opt(*self)? {
            if query.overlaps_left(data.key(), data.rank()) {
                Box::pin(data.left().query0(query, store, co)).await?;
            }
            if query.contains(data.key()) {
                co.yield_(Ok((data.key().clone(), data.value().clone())))
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
        store: &'a impl StoreExt<P>,
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
        store: &impl StoreExt<P>,
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
                co.yield_(Ok((data.key().clone(), data.value().clone())))
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
                    data.key().clone(),
                    data.value().clone(),
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
    fn split_all(&self, store: &mut impl StoreExt<P>, res: &mut Vec<IdAndData<P>>) -> Result<()> {
        if let Some(mut data) = store.get_node(*self)? {
            data.left().split_all(store, res)?;
            data.right().split_all(store, res)?;
            // turn the node into a leaf
            *data.left_mut() = Node::EMPTY;
            *data.right_mut() = Node::EMPTY;
            data.summary = P::M::lift((data.key().clone(), data.value().clone()));
            // persist (should we do this here or later?)
            data.persist(store)?;
            res.push(data);
        }
        Ok(())
    }
}

pub struct NodeData2<P: TreeParams, T = Vec<u8>>(T, PhantomData<P>);

impl<P: TreeParams, T: AsRef<[u8]>> NodeData2<P, T> {
    pub fn new(data: T) -> Self {
        debug_assert!(data.as_ref().len() >= min_data_size::<P>());
        Self(data, PhantomData)
    }

    pub fn left(&self) -> Node<P> {
        Node::read_from_prefix(&&self.0.as_ref()[left_offset::<P>()..]).unwrap()
    }

    pub fn right(&self) -> Node<P> {
        Node::read_from_prefix(&&self.0.as_ref()[right_offset::<P>()..]).unwrap()
    }

    pub fn value(&self) -> &P::V {
        P::V::ref_from_prefix(&self.0.as_ref()[value_offset::<P>()..]).unwrap()
    }

    pub fn summary(&self) -> &P::M {
        P::M::ref_from_prefix(&self.0.as_ref()[summary_offset::<P>()..]).unwrap()
    }

    pub fn rank(&self) -> u8 {
        self.0.as_ref()[rank_offset::<P>()]
    }

    pub fn key(&self) -> Point2<P, &[u8]> {
        Point2::new(&self.0.as_ref()[key_offset::<P>()..])
    }
}

impl<P: TreeParams, T: AsMut<[u8]>> NodeData2<P, T> {
    pub fn left_mut(&mut self) -> &mut Node<P> {
        Node::mut_from_prefix(&mut self.0.as_mut()[left_offset::<P>()..]).unwrap()
    }

    pub fn right_mut(&mut self) -> &mut Node<P> {
        Node::mut_from_prefix(&mut self.0.as_mut()[right_offset::<P>()..]).unwrap()
    }

    pub fn summary_mut(&mut self) -> &mut P::M {
        P::M::mut_from_prefix(&mut self.0.as_mut()[summary_offset::<P>()..]).unwrap()
    }
}

pub struct Point2<P: KeyParams, T: AsRef<[u8]> = Vec<u8>>(T, PhantomData<P>);

impl<P: KeyParams, T: AsRef<[u8]>> Point2<P, T> {
    pub fn new(data: T) -> Self {
        debug_assert!(data.as_ref().len() >= min_key_size::<P>());
        Self(data, PhantomData)
    }

    pub fn x(&self) -> P::X {
        P::X::read_from_prefix(&self.0.as_ref()[0..]).unwrap()
    }

    pub fn y(&self) -> P::Y {
        P::Y::read_from_prefix(&self.0.as_ref()[P::X::SIZE..]).unwrap()
    }

    pub fn z(&self) -> P::Z {
        P::Z::read_from_prefix(&self.0.as_ref()[P::X::SIZE + P::Y::SIZE..]).unwrap()
    }
}

/// Data for a node in the tree.
/// This gets persisted as a single &[u8] in the database, using a [`NodeId`]
/// as key. However, NodeData can exist in memory without a NodeId.
///
/// TODO:
/// - Use some zero copy stuff to make this more efficient.
/// - Have dedicated types for owned and borrowed NodeData (?).
pub struct NodeData<P: TreeParams> {
    left: Node<P>,  // 8 bytes, 0 if empty
    right: Node<P>, // 8 bytes, 0 if empty
    value: P::V,    // fixed size
    summary: P::M,  // fixed size
    rank: u8,       // 1 byte
    key: Point<P>,  // variable size due to path
}

impl<P: TreeParams> Debug for NodeData<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeData")
            .field("key", &self.key())
            .field("rank", &self.rank())
            .field("value", self.value())
            .field("summary", self.summary())
            .field("left", &self.left())
            .field("right", &self.right())
            .finish()
    }
}

impl<P: TreeParams> Clone for NodeData<P> {
    fn clone(&self) -> Self {
        Self {
            key: self.key().clone(),
            rank: self.rank(),
            value: self.value().clone(),
            summary: self.summary().clone(),
            left: self.left(),
            right: self.right(),
        }
    }
}

impl<P: TreeParams> PartialEq for NodeData<P> {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
            && self.rank() == other.rank()
            && self.value() == other.value()
            && self.summary() == other.summary()
            && self.left() == other.left()
            && self.right() == other.right()
    }
}

impl<P: TreeParams> Eq for NodeData<P> {}

impl<P: TreeParams> NodeData<P> {
    fn key(&self) -> &Point<P> {
        &self.key
    }

    fn left(&self) -> Node<P> {
        self.left
    }

    fn right(&self) -> Node<P> {
        self.right
    }

    fn left_mut(&mut self) -> &mut Node<P> {
        &mut self.left
    }

    fn right_mut(&mut self) -> &mut Node<P> {
        &mut self.right
    }

    fn value(&self) -> &P::V {
        &self.value
    }

    fn summary(&self) -> &P::M {
        &self.summary
    }

    fn summary0(
        &self,
        query: &QueryRange3d<P>,
        bbox: &BBox<P>,
        store: &impl StoreExt<P>,
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
            summary = summary.combine(&P::M::lift((self.key().clone(), self.value().clone())));
        }
        if query.overlaps_right(self.key(), self.rank()) {
            if let Some(right) = store.data_opt(self.right())? {
                let right_bbox = bbox.split_right(self.key(), self.rank());
                summary = summary.combine(&right.summary0(query, &right_bbox, store)?);
            }
        }
        Ok(summary)
    }

    fn recalculate_summary(&mut self, store: &impl StoreExt<P>) -> Result<()> {
        let mut res = P::M::neutral();
        if let Some(left) = store.data_opt(self.left())? {
            res = res.combine(left.summary());
        }
        res = res.combine(&P::M::lift((self.key().clone(), self.value().clone())));
        if let Some(right) = store.data_opt(self.right())? {
            res = res.combine(right.summary());
        }
        self.summary = res;
        Ok(())
    }

    fn add_summary(&mut self, value: P::V) {
        self.summary = self
            .summary()
            .combine(&P::M::lift((self.key().clone(), value)));
    }
}

/// A node combines node data with a node id.
///
/// When modifying a node, the id should not be modified. After a modification,
/// the node should be updated in the store, otherwise there will be horrible
/// inconsistencies.
pub struct IdAndData<P: TreeParams> {
    id: Node<P>,
    data: NodeData<P>,
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

impl<P: TreeParams> VariableSize for NodeData<P> {
    fn size(&self) -> usize {
        8 + // left
        8 + // right
        P::V::SIZE + // value
        P::M::SIZE + // summary
        1 + // rank
        P::X::SIZE + // x
        P::Y::SIZE + // y
        self.key().z.size() // z
    }

    fn write(&self, buf: &mut [u8]) {
        const L_START: usize = 0;
        const R_START: usize = 8;
        const V_START: usize = 16;
        let m_start: usize = 16 + P::V::SIZE;
        let rank_start: usize = 16 + P::V::SIZE + P::M::SIZE;
        let key_start: usize = 16 + P::V::SIZE + P::M::SIZE + 1;
        assert_eq!(buf.len(), self.size());
        self.left().0.write_to(&mut buf[L_START..R_START]).unwrap();
        self.right().0.write_to(&mut buf[R_START..V_START]).unwrap();
        self.value().write_to(&mut buf[V_START..m_start]).unwrap();
        self.summary()
            .write_to(&mut buf[m_start..rank_start])
            .unwrap();
        self.rank()
            .write_to(&mut buf[rank_start..key_start])
            .unwrap();
        self.key().write(&mut buf[key_start..]);
    }

    fn read(buf: &[u8]) -> Self {
        const L_START: usize = 0;
        const R_START: usize = 8;
        const V_START: usize = 16;
        let m_start: usize = 16 + P::V::SIZE;
        let rank_start: usize = 16 + P::V::SIZE + P::M::SIZE;
        let x_start: usize = 16 + P::V::SIZE + P::M::SIZE + 1;
        let y_start: usize = 16 + P::V::SIZE + P::M::SIZE + 1 + P::X::SIZE;
        let z_start: usize = 16 + P::V::SIZE + P::M::SIZE + 1 + P::X::SIZE + P::Y::SIZE;
        let left = Node::read_from_prefix(&buf[L_START..]).unwrap();
        let right = Node::read_from_prefix(&buf[R_START..]).unwrap();
        let value = P::V::read_from_prefix(&buf[V_START..]).unwrap();
        let summary = P::M::read_from_prefix(&buf[m_start..]).unwrap();
        let rank = u8::read_from_prefix(&buf[rank_start..]).unwrap();
        let x = P::X::read_from_prefix(&buf[x_start..]).unwrap();
        let y = P::Y::read_from_prefix(&buf[y_start..]).unwrap();
        let z = P::Z::read_from_prefix(&buf[z_start..]).unwrap();
        Self {
            key: Point { x, y, z },
            rank,
            value,
            summary,
            left,
            right,
        }
    }
}

impl<P: TreeParams> NodeData<P> {
    /// True if the node is a leaf.
    fn is_leaf(&self) -> bool {
        self.left().is_empty() && self.right().is_empty()
    }

    pub fn rank(&self) -> u8 {
        self.rank
    }

    /// Get the sort order for the node, based on the rank.
    pub fn sort_order(&self) -> SortOrder {
        self.rank.into()
    }

    /// Create a new node data with the given key and value.
    pub fn single(key: Point<P>, value: P::V) -> Self {
        let summary = P::M::lift((key.clone(), value.clone()));
        let mut key_bytes = vec![0; key.size()];
        key.write(&mut key_bytes);
        let key_hash: [u8; 32] = blake3::hash(&key_bytes).into();
        let rank = count_trailing_zeros(&key_hash);
        NodeData {
            key,
            rank,
            value,
            summary,
            left: Node::EMPTY,
            right: Node::EMPTY,
        }
    }

    pub fn assert_invariants(
        &self,
        store: &impl StoreExt<P>,
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
                    assert_lt!(left.xyz.max, self.key().clone().xyz());
                }
                if let Some(ref right) = right_res {
                    assert_gt!(right.xyz.min, self.key().clone().xyz());
                }
            }
            SortOrder::YZX => {
                if let Some(ref left) = left_res {
                    assert_lt!(left.yzx.max, self.key().clone().yzx());
                }
                if let Some(ref right) = right_res {
                    assert_gt!(right.yzx.min, self.key().clone().yzx());
                }
            }
            SortOrder::ZXY => {
                if let Some(ref left) = left_res {
                    assert_lt!(left.zxy.max, self.key().clone().zxy());
                }
                if let Some(ref right) = right_res {
                    assert_gt!(right.zxy.min, self.key().clone().zxy());
                }
            }
        }
        let mut res =
            AssertInvariantsRes::single(self.key().clone(), self.rank(), self.value.clone());
        if let Some(left) = left_res {
            res = res.combine(&left);
        }
        if let Some(right) = right_res {
            res = res.combine(&right);
        }
        if include_summary {
            assert_eq!(&res.summary, self.summary());
        }
        Ok(res)
    }
}

impl<P: TreeParams> IdAndData<P> {
    fn new(id: Node<P>, data: NodeData<P>) -> Self {
        assert!(!id.is_empty());
        IdAndData { id, data }
    }

    fn persist(&self, store: &mut impl StoreExt<P>) -> Result<()> {
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
        store.update(self.id.0, &self.data)
    }

    // Insert a new node into the tree without balancing.
    fn insert_no_balance(&mut self, node: &Self, store: &mut impl StoreExt<P>) -> Result<()> {
        assert!(node.is_leaf());
        let IdAndData {
            data:
                NodeData {
                    key: parent_key,
                    rank: parent_rank,
                    summary: parent_summary,
                    left,
                    right,
                    value: _,
                },
            ..
        } = self;
        match node.key().cmp_at_rank(parent_key, *parent_rank) {
            Ordering::Less => {
                left.insert_no_balance(node, store)?;
            }
            Ordering::Greater => {
                right.insert_no_balance(node, store)?;
            }
            Ordering::Equal => {
                panic!("Duplicate keys not supported in insert_no_balance");
            }
        }
        *parent_summary = parent_summary.combine(&node.summary);
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
