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
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use anyhow::Result;
use genawaiter::sync::{Co, Gen};
use point::{XYZ, YZX, ZXY};
use serde::Serialize;

mod point;
pub use point::Point;

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
pub trait CoordParams: Ord + PartialEq + Eq + Serialize + Clone + Debug + Display {}

impl<T: Ord + PartialEq + Eq + Serialize + Clone + Debug + Display> CoordParams for T {}

pub trait FixedSize {
    const SIZE: usize;
    fn write(&self, buf: &mut [u8]);
    fn read(buf: &[u8]) -> Self;
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
    fn write(&self, buf: &mut [u8]) {
        buf.copy_from_slice(&self.to_be_bytes());
    }
    fn read(buf: &[u8]) -> Self {
        u64::from_be_bytes(buf.try_into().unwrap())
    }
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
    fn write(&self, buf: &mut [u8]) {
        buf.copy_from_slice(&self.0);
    }
    fn read(buf: &[u8]) -> Self {
        NodeId(buf.try_into().unwrap())
    }
}

impl FixedSize for u8 {
    const SIZE: usize = 1;
    fn write(&self, buf: &mut [u8]) {
        buf[0] = *self;
    }
    fn read(buf: &[u8]) -> Self {
        buf[0]
    }
}

pub trait KeyParams {
    type X: CoordParams + FixedSize;
    type Y: CoordParams + FixedSize;
    type Z: CoordParams + VariableSize;
}

pub trait LiftingCommutativeMonoid<T> {
    fn zero() -> Self;
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
pub trait ValueParams: PartialEq + Eq + Serialize + Clone + Debug + FixedSize {}

impl<T: PartialEq + Eq + Serialize + Clone + Debug + FixedSize> ValueParams for T {}

/// Tree params for a 3D tree. This extends `KeyParams` with a value and
/// summary type.
pub trait TreeParams: KeyParams + Sized {
    type V: ValueParams;
    type M: LiftingCommutativeMonoid<(Point<Self>, Self::V)> + Clone + Debug + Eq + FixedSize;
}

/// A simple store trait for storing nodes by id.
pub trait Store<T: VariableSize> {
    fn put(&mut self, node: &T) -> Result<NodeId>;
    fn data(&self, id: &NodeId) -> Result<T>;
    fn update(&mut self, id: &NodeId, node: &T) -> Result<()>;
}

pub trait StoreExt<P: TreeParams>: Store<NodeData<P>> {
    fn put_node(&mut self, data: NodeData<P>) -> Result<NonEmptyNode<P>> {
        let id = self.put(&data)?;
        Ok(NonEmptyNode::new(id, data))
    }

    /// Get a node by id, returning None if the id is None.
    ///
    /// This is just a convenience method for the common case where you have an
    /// optional id and want to get the node if it exists.
    fn data_opt(&self, id: &NodeId) -> Result<Option<NodeData<P>>> {
        Ok(if id.is_empty() {
            None
        } else {
            Some(self.data(id)?)
        })
    }

    /// Get a node by id, returning None if the id is None.
    /// Also return the id along with the node.
    fn get_node(&self, id: &NodeId) -> Result<Node<P>> {
        Ok(if id.is_empty() {
            Node::Empty
        } else {
            NonEmptyNode::new(*id, self.data(id)?).into()
        })
    }
}

impl<P: TreeParams> StoreExt<P> for MemStore {}

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
    fn put(&mut self, node: &T) -> Result<NodeId> {
        let id = NodeId::from((self.nodes.len() as u64) + 1);
        assert!(!id.is_empty());
        self.nodes.insert(id, node.to_vec());
        Ok(id)
    }

    fn update(&mut self, id: &NodeId, node: &T) -> Result<()> {
        assert!(!id.is_empty());
        self.nodes.insert(*id, node.to_vec());
        Ok(())
    }

    fn data(&self, id: &NodeId) -> Result<T> {
        assert!(!id.is_empty());
        match self.nodes.get(id) {
            Some(data) => Ok(T::read(data)),
            None => Err(anyhow::anyhow!("Node not found")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeId([u8; 8]);

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

/// Data for a node in the tree.
/// This gets persisted as a single &[u8] in the database, using a [`NodeId`]
/// as key. However, NodeData can exist in memory without a NodeId.
///
/// TODO:
/// - Use some zero copy stuff to make this more efficient.
/// - Have dedicated types for owned and borrowed NodeData (?).
pub struct NodeData<P: TreeParams> {
    left: NodeId,  // 8 bytes, 0 if empty
    right: NodeId, // 8 bytes, 0 if empty
    value: P::V,   // fixed size
    summary: P::M, // fixed size
    rank: u8,      // 1 byte
    key: Point<P>, // variable size due to path
}

impl<P: TreeParams> Debug for NodeData<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeData")
            .field("key", &self.key)
            .field("rank", &self.rank)
            .field("value", &self.value)
            .field("summary", &self.summary)
            .field("left", &self.left)
            .field("right", &self.right)
            .finish()
    }
}

impl<P: TreeParams> Clone for NodeData<P> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            rank: self.rank,
            value: self.value.clone(),
            summary: self.summary.clone(),
            left: self.left.clone(),
            right: self.right.clone(),
        }
    }
}

impl<P: TreeParams> PartialEq for NodeData<P> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
            && self.rank == other.rank
            && self.value == other.value
            && self.summary == other.summary
            && self.left == other.left
            && self.right == other.right
    }
}

impl<P: TreeParams> Eq for NodeData<P> {}

/// A node combines node data with a node id.
///
/// When modifying a node, the id should not be modified. After a modification,
/// the node should be updated in the store, otherwise there will be horrible
/// inconsistencies.
pub struct NonEmptyNode<P: TreeParams> {
    id: NodeId,
    data: NodeData<P>,
}

pub enum Node<P: TreeParams> {
    Empty,
    NonEmpty(NonEmptyNode<P>),
}

impl<P: TreeParams> From<NonEmptyNode<P>> for Node<P> {
    fn from(node: NonEmptyNode<P>) -> Self {
        Node::NonEmpty(node)
    }
}

impl<P: TreeParams> Node<P> {
    pub fn id(&self) -> NodeId {
        match self {
            Node::Empty => NodeId::EMPTY,
            Node::NonEmpty(node) => node.id,
        }
    }

    /// Get the key of the node. Panics if the node is empty.
    pub fn key(&self) -> &Point<P> {
        if let Node::NonEmpty(node) = self {
            &node.data.key
        } else {
            panic!("Empty node")
        }
    }

    pub fn left(&self) -> NodeId {
        if let Node::NonEmpty(node) = self {
            node.data.left
        } else {
            panic!("Empty node")
        }
    }

    pub fn right(&self) -> NodeId {
        if let Node::NonEmpty(node) = self {
            node.data.right
        } else {
            panic!("Empty node")
        }
    }

    pub fn is_empty(&self) -> bool {
        self.id().is_empty()
    }

    pub fn non_empty(&self) -> Option<&NonEmptyNode<P>> {
        match self {
            Node::Empty => None,
            Node::NonEmpty(node) => Some(node),
        }
    }

    pub fn non_empty_mut(&mut self) -> Option<&mut NonEmptyNode<P>> {
        match self {
            Node::Empty => None,
            Node::NonEmpty(node) => Some(node),
        }
    }

    pub fn assert_invariants(&self, store: &impl StoreExt<P>) -> Result<()> {
        if let Node::NonEmpty(node) = self {
            node.assert_invariants(store)?;
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
            if let Node::NonEmpty(node) = self {
                if let Err(cause) = node.iter_unordered0(store, &co).await {
                    co.yield_(Err(cause)).await;
                }
            }
        })
        .into_iter()
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
            if let Node::NonEmpty(node) = self {
                if let Err(cause) = node.query0(query, store, &co).await {
                    co.yield_(Err(cause)).await;
                }
            }
        })
        .into_iter()
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
            if let Node::NonEmpty(node) = self {
                if let Err(cause) = node.query_ordered0(query, ordering, store, &co).await {
                    co.yield_(Err(cause)).await;
                }
            }
        })
        .into_iter()
    }

    /// Get a summary of the elements in a 3d range.
    ///
    /// The result is identical to iterating over the elements in the 3d range
    /// and combining the summaries of each element, but will be much more
    /// efficient for large trees.
    pub fn summary(&self, query: &QueryRange3d<P>, store: &impl StoreExt<P>) -> Result<P::M> {
        if let Node::NonEmpty(node) = self {
            let bbox = BBox::all();
            node.summary0(query, &bbox, store)
        } else {
            Ok(P::M::zero())
        }
    }

    pub fn get(&self, key: Point<P>, store: &impl StoreExt<P>) -> Result<Option<P::V>> {
        Ok(if let Node::NonEmpty(node) = self {
            node.get0(key, store)?.map(|node| node.value)
        } else {
            None
        })
    }

    /// Replace the value of a node in the tree and update the summary.
    ///
    /// Returns the old value. If the key is not found, returns None and does
    /// not modify the tree.
    pub fn update(
        &mut self,
        key: Point<P>,
        value: P::V,
        store: &mut impl StoreExt<P>,
    ) -> Result<Option<P::V>> {
        Ok(if let Node::NonEmpty(node) = self {
            let data = NodeData::single(key, value);
            let old = node.update0(&data, store)?;
            old
        } else {
            None
        })
    }

    pub fn insert(
        &mut self,
        key: Point<P>,
        value: P::V,
        store: &mut impl StoreExt<P>,
    ) -> Result<Option<P::V>> {
        let data = NodeData::single(key, value);
        let Node::NonEmpty(node) = self else {
            *self = store.put_node(data)?.into();
            return Ok(None);
        };
        if let Some(old) = node.update0(&data, store)? {
            return Ok(Some(old));
        }
        let id = NonEmptyNode::insert0(node.id, data, store)?;
        *self = store.get_node(&id)?;
        Ok(None)
    }

    pub fn dump(&self, store: &impl StoreExt<P>) -> Result<()> {
        if let Node::NonEmpty(node) = self {
            node.dump0("".into(), store)
        } else {
            println!("Empty");
            Ok(())
        }
    }
}

impl<P: TreeParams> Clone for Node<P> {
    fn clone(&self) -> Self {
        match self {
            Node::Empty => Node::Empty,
            Node::NonEmpty(node) => Node::NonEmpty(node.clone()),
        }
    }
}

impl<P: TreeParams> Clone for NonEmptyNode<P> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            data: self.data.clone(),
        }
    }
}

impl<P: TreeParams> Deref for NonEmptyNode<P> {
    type Target = NodeData<P>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<P: TreeParams> DerefMut for NonEmptyNode<P> {
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
        self.key.z.size() // z
    }

    fn write(&self, buf: &mut [u8]) {
        const L_START: usize = 0;
        const R_START: usize = 8;
        const V_START: usize = 16;
        let m_start: usize = 16 + P::V::SIZE;
        let rank_start: usize = 16 + P::V::SIZE + P::M::SIZE;
        let key_start: usize = 16 + P::V::SIZE + P::M::SIZE + 1;
        assert_eq!(buf.len(), self.size());
        self.left.write(&mut buf[L_START..R_START]);
        self.right.write(&mut buf[R_START..V_START]);
        self.value.write(&mut buf[V_START..m_start]);
        self.summary.write(&mut buf[m_start..rank_start]);
        self.rank.write(&mut buf[rank_start..key_start]);
        self.key.write(&mut buf[key_start..]);
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
        let left = NodeId::read(&buf[L_START..R_START]);
        let right = NodeId::read(&buf[R_START..V_START]);
        let value = P::V::read(&buf[V_START..m_start]);
        let summary = P::M::read(&buf[m_start..rank_start]);
        let rank = u8::read(&buf[rank_start..x_start]);
        let x = P::X::read(&buf[x_start..y_start]);
        let y = P::Y::read(&buf[y_start..z_start]);
        let z = P::Z::read(&buf[z_start..]);
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
        self.left.is_empty() && self.right.is_empty()
    }

    /// Get the sort order for the node, based on the rank.
    pub fn sort_order(&self) -> SortOrder {
        self.rank.into()
    }

    async fn iter_unordered0(
        &self,
        store: &impl StoreExt<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        if let Some(left) = store.data_opt(&self.left)? {
            Box::pin(left.iter_unordered0(store, co)).await?;
        }
        co.yield_(Ok((self.key.clone(), self.value.clone()))).await;
        if let Some(right) = store.data_opt(&self.right)? {
            Box::pin(right.iter_unordered0(store, co)).await?;
        }
        Ok(())
    }

    fn summary0(
        &self,
        query: &QueryRange3d<P>,
        bbox: &BBox<P>,
        store: &impl StoreExt<P>,
    ) -> Result<P::M> {
        if self.is_leaf() {
            if query.contains(&self.key) {
                return Ok(self.summary.clone());
            } else {
                return Ok(P::M::zero());
            }
        }
        if bbox.contained_in(&query) {
            return Ok(self.summary.clone());
        }
        let mut summary = P::M::zero();
        if query.overlaps_left(&self.key, self.rank) {
            if let Some(left) = store.data_opt(&self.left)? {
                let left_bbox = bbox.split_left(&self.key, self.rank);
                summary = summary.combine(&left.summary0(query, &left_bbox, store)?);
            }
        }
        if query.contains(&self.key) {
            summary = summary.combine(&P::M::lift((self.key.clone(), self.value.clone())));
        }
        if query.overlaps_right(&self.key, self.rank) {
            if let Some(right) = store.data_opt(&self.right)? {
                let right_bbox = bbox.split_right(&self.key, self.rank);
                summary = summary.combine(&right.summary0(query, &right_bbox, store)?);
            }
        }
        Ok(summary)
    }

    async fn query0(
        &self,
        query: &QueryRange3d<P>,
        store: &impl StoreExt<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        if query.overlaps_left(&self.key, self.rank) {
            if let Some(left) = store.data_opt(&self.left)? {
                Box::pin(left.query0(query, store, co)).await?;
            }
        }
        if query.contains(&self.key) {
            co.yield_(Ok((self.key.clone(), self.value.clone()))).await;
        }
        if query.overlaps_right(&self.key, self.rank) {
            if let Some(right) = store.data_opt(&self.right)? {
                Box::pin(right.query0(query, store, co)).await?;
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
        // we know that the node partitions the space in the correct way, so
        // we can just concatenate the results from the left, self and right
        if self.sort_order() == ordering {
            if query.overlaps_left(&self.key, self.rank) {
                if let Some(left) = store.data_opt(&self.left)? {
                    Box::pin(left.query_ordered0(query, ordering, store, co)).await?;
                }
            }
            if query.contains(&self.key) {
                co.yield_(Ok((self.key.clone(), self.value.clone()))).await;
            }
            if query.overlaps_right(&self.key, self.rank) {
                if let Some(right) = store.data_opt(&self.right)? {
                    Box::pin(right.query_ordered0(query, ordering, store, co)).await?;
                }
            }
        } else {
            // the node does not partition the space in the correct way, so we
            // need to merge the results from the left, self and right
            // still better than a full sort!
            let iter1 = if query.overlaps_left(&self.key, self.rank) {
                if let Some(left) = store.data_opt(&self.left)? {
                    itertools::Either::Left(left.query_ordered(query, ordering, store).into_iter())
                } else {
                    itertools::Either::Right(std::iter::empty())
                }
            } else {
                itertools::Either::Right(std::iter::empty())
            };
            let iter2 = if query.contains(&self.key) {
                itertools::Either::Left(std::iter::once(Ok((self.key.clone(), self.value.clone()))))
            } else {
                itertools::Either::Right(std::iter::empty())
            };
            let iter3 = if query.overlaps_right(&self.key, self.rank) {
                if let Some(right) = store.data_opt(&self.right)? {
                    itertools::Either::Left(right.query_ordered(query, ordering, store).into_iter())
                } else {
                    itertools::Either::Right(std::iter::empty())
                }
            } else {
                itertools::Either::Right(std::iter::empty())
            };
            merge3(iter1, iter2, iter3, ordering, co).await?;
        }
        Ok(())
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
            left: NodeId::EMPTY,
            right: NodeId::EMPTY,
        }
    }

    fn get0(&self, key: Point<P>, store: &impl StoreExt<P>) -> Result<Option<NodeData<P>>> {
        match key.cmp_at_rank(&self.key, self.rank) {
            Ordering::Less => {
                if !self.left.is_empty() {
                    let left = store.data(&self.left)?;
                    left.get0(key, store)
                } else {
                    Ok(None)
                }
            }
            Ordering::Greater => {
                if !self.right.is_empty() {
                    let right = store.data(&self.right)?;
                    right.get0(key, store)
                } else {
                    Ok(None)
                }
            }
            Ordering::Equal => Ok(Some(self.clone())),
        }
    }

    pub fn from_iter<I: IntoIterator<Item = (Point<P>, P::V)>>(
        iter: I,
    ) -> Result<(impl StoreExt<P>, NodeId)> {
        let mut nodes: Vec<_> = iter
            .into_iter()
            .map(|(key, value)| NodeData::single(key, value))
            .collect();
        // Before we sort, remove all but the first occurence of each point.
        let mut uniques = BTreeSet::new();
        nodes.retain(|node| uniques.insert(node.key.clone().xyz()));
        // if rank is equal, compare keys at rank
        nodes.sort_by(|p1, p2| {
            p2.rank
                .cmp(&p1.rank)
                .then(p1.key.cmp_at_rank(&p2.key, p1.rank))
        });
        let mut store = MemStore::new();
        let root_data = nodes.remove(0);
        let mut root = store.put_node(root_data)?;
        for node in nodes {
            let node = store.put_node(node.clone())?;
            root.insert_no_balance(&node, &mut store)?;
        }
        Ok((store, root.id))
    }

    fn dump0(&self, prefix: String, store: &impl StoreExt<P>) -> Result<()> {
        println!(
            "{}{:?} rank={} order={:?} value={:?}",
            prefix,
            self.key,
            self.rank,
            SortOrder::from(self.rank),
            self.value
        );
        if !self.left.is_empty() {
            println!("{} left:", prefix);
            let left = store.data(&self.left)?;
            left.dump0(format!("{}  ", prefix), store)?;
        }
        if !self.right.is_empty() {
            println!("{} right:", prefix);
            let right = store.data(&self.right)?;
            right.dump0(format!("{}  ", prefix), store)?;
        }
        Ok(())
    }

    pub fn assert_invariants(&self, store: &impl StoreExt<P>) -> Result<AssertInvariantsRes<P>> {
        let NodeData {
            key,
            rank,
            value: _,
            summary,
            left,
            right,
        } = self;
        let left = store.data_opt(left)?;
        let right = store.data_opt(right)?;
        let left_res = left
            .as_ref()
            .map(|node| node.assert_invariants(store))
            .transpose()?;
        let right_res = right
            .as_ref()
            .map(|node| node.assert_invariants(store))
            .transpose()?;
        if let Some(ref left) = left_res {
            assert!(left.rank < *rank);
        }
        if let Some(ref right) = right_res {
            assert!(right.rank <= *rank);
        }
        match SortOrder::from(*rank) {
            SortOrder::XYZ => {
                if let Some(ref left) = left_res {
                    assert_lt!(left.xyz.max, key.clone().xyz());
                }
                if let Some(ref right) = right_res {
                    assert_gt!(right.xyz.min, key.clone().xyz());
                }
            }
            SortOrder::YZX => {
                if let Some(ref left) = left_res {
                    assert_lt!(left.yzx.max, key.clone().yzx());
                }
                if let Some(ref right) = right_res {
                    assert_gt!(right.yzx.min, key.clone().yzx());
                }
            }
            SortOrder::ZXY => {
                if let Some(ref left) = left_res {
                    assert_lt!(left.zxy.max, key.clone().zxy());
                }
                if let Some(ref right) = right_res {
                    assert_gt!(right.zxy.min, key.clone().zxy());
                }
            }
        }
        let mut res = AssertInvariantsRes::single(key.clone(), *rank, self.value.clone());
        if let Some(left) = left_res {
            res = res.combine(&left);
        }
        if let Some(right) = right_res {
            res = res.combine(&right);
        }
        assert_eq!(res.summary, *summary);
        Ok(res)
    }
}

impl<P: TreeParams> NonEmptyNode<P> {
    fn new(id: NodeId, data: NodeData<P>) -> Self {
        assert!(!id.is_empty());
        NonEmptyNode { id, data }
    }

    fn id(&self) -> NodeId {
        self.id
    }

    fn persist(&self, store: &mut impl StoreExt<P>) -> Result<()> {
        store.update(&self.id, &self.data)
    }

    // Insert a new node into the tree without balancing.
    fn insert_no_balance(&mut self, node: &Self, store: &mut impl StoreExt<P>) -> Result<()> {
        assert!(node.is_leaf());
        let NonEmptyNode {
            id,
            data:
                NodeData {
                    key: parent_key,
                    rank: parent_rank,
                    summary: parent_summary,
                    left,
                    right,
                    value: _,
                },
        } = self;
        match node.key.cmp_at_rank(parent_key, *parent_rank) {
            Ordering::Less => {
                if let Node::NonEmpty(mut left) = store.get_node(&left)? {
                    left.insert_no_balance(node, store)?;
                } else {
                    *left = store.put(node)?;
                }
            }
            Ordering::Greater => {
                if let Node::NonEmpty(mut right) = store.get_node(&right)? {
                    right.insert_no_balance(node, store)?;
                } else {
                    *right = store.put(node)?;
                }
            }
            Ordering::Equal => {
                panic!("Duplicate keys not supported in insert_no_balance");
            }
        }
        *parent_summary = parent_summary.combine(&node.summary);
        store.update(id, &self.data)?;
        Ok(())
    }

    fn update0(
        &mut self,
        data: &NodeData<P>,
        store: &mut impl StoreExt<P>,
    ) -> Result<Option<P::V>> {
        assert!(data.is_leaf());
        let key_cmp = data.key.cmp_at_rank(&self.key, self.rank);
        let old_value = match key_cmp {
            Ordering::Equal => {
                // just replace the value and update the summary
                let old_value = self.value.clone();
                self.value = data.value.clone();
                Some(old_value)
            }
            Ordering::Less => {
                if let Node::NonEmpty(mut left) = store.get_node(&self.left)? {
                    left.update0(data, store)?
                } else {
                    None
                }
            }
            Ordering::Greater => {
                if let Node::NonEmpty(mut right) = store.get_node(&self.right)? {
                    right.update0(data, store)?
                } else {
                    None
                }
            }
        };
        if old_value.is_some() {
            // recalculate the summary from scratch
            let mut summary = P::M::lift((self.key.clone(), self.value.clone()));
            if let Some(left) = store.data_opt(&self.left)? {
                summary = summary.combine(&left.summary);
            }
            if let Some(right) = store.data_opt(&self.right)? {
                summary = summary.combine(&right.summary);
            }
            self.summary = summary;
            self.persist(store)?;
        };
        Ok(old_value)
    }

    fn insert0(
        mut root_id: NodeId,
        x: NodeData<P>,
        store: &mut impl StoreExt<P>,
    ) -> Result<NodeId> {
        use Rel::*;
        let root = store.get_node(&root_id)?;
        let mut x = store.put_node(x)?;
        let key = x.key.clone();
        let rank = x.rank;
        // Compare with x.key at x.rank
        let x_cmp = |x: &Point<P>| {
            let x_cmp = x.cmp_at_rank(&key, rank);
            match x_cmp {
                Ordering::Less => Left,
                Ordering::Greater => Right,
                Ordering::Equal => {
                    println!("x={:?} key={:?}", x, key);
                    unreachable!("Duplicate keys not supported in insert0");
                }
            }
        };
        let mut prev = Node::Empty;
        let mut cur = root;
        // while cur != null and (rank < cur.rank or (rank = cur.rank and key > cur.key)) do
        //   prev ← cur
        //   cur ← if key < cur.key then cur.left else cur.right
        while let Node::NonEmpty(t) = cur {
            let rel = x_cmp(&t.key);
            if rank < t.rank || rank == t.rank && rel == Left {
                // cur is above x, just go down
                cur = match rel {
                    Left => store.get_node(&t.left)?,
                    Right => store.get_node(&t.right)?,
                };
                prev = Node::NonEmpty(t);
            } else {
                // cur is below x
                cur = Node::NonEmpty(t);
                break;
            }
        }
        // if cur = root then root ← x
        //   else if key < prev.key then prev.left ← x
        //   else prev.right ← x
        if prev.is_empty() {
            root_id = x.id;
        } else {
            let prev = prev.non_empty_mut().expect("prev is empty");
            match x_cmp(&prev.key) {
                Left => prev.left = x.id,
                Right => prev.right = x.id,
            }
            prev.persist(store)?;
        }
        // if cur = null then {x.left ← x.right ← null; return}
        if cur.is_empty() {
            return Ok(x.id);
        }
        {
            let cur = cur.non_empty().expect("cur is empty");
            // if key < cur.key then x.right ← cur else x.left ← cur
            match x_cmp(&cur.key) {
                Left => x.right = cur.id,
                Right => x.left = cur.id,
            }
        }
        x.persist(store)?;
        // prev ← x
        prev = Node::NonEmpty(x.clone());
        // while cur 6= null do
        //   fix ← prev
        //   if cur .key < key then
        //     repeat {prev ← cur ; cur ← cur .right}
        //     until cur = null or cur .key > key
        //   else
        //     repeat {prev ← cur ; cur ← cur .left}
        //     until cur = null or cur .key < key
        //   if fix.key > key or (fix = x and prev.key > key) then
        //     fix.left ← cur
        //   else
        //     fix.right ← cur

        // while cur 6= null do
        while !cur.is_empty() {
            //   fix ← prev
            let mut fix = prev;
            match x_cmp(&cur.key()) {
                //     repeat {prev ← cur ; cur ← cur.right}
                //     until cur = null or cur.key > key
                Left => loop {
                    prev = cur.clone();
                    cur = store.get_node(&prev.right())?;
                    let Node::NonEmpty(cur_ne) = &cur else {
                        break;
                    };
                    if x_cmp(&cur_ne.key) == Right {
                        break;
                    }
                },
                //     repeat {prev ← cur ; cur ← cur.left}
                //     until cur = null or cur.key < key
                Right => loop {
                    prev = cur.clone();
                    cur = store.get_node(&prev.left())?;
                    let Node::NonEmpty(cur_ne) = &cur else {
                        break;
                    };
                    if x_cmp(&cur_ne.key) == Left {
                        break;
                    }
                },
            }
            //   if fix.key > key or (fix = x and prev.key > key) then
            //     fix.left ← cur
            //   else
            //     fix.right ← cur
            println!("{} {}", x.id, fix.id());
            let fix = fix.non_empty_mut().expect("fix is empty");
            let prev = prev.non_empty().expect("prev is empty");
            if ((fix.id == x.id) && x_cmp(&prev.key) == Right)
                || ((fix.id != x.id) && x_cmp(&fix.key) == Right)
            {
                fix.left = cur.id();
            } else {
                fix.right = cur.id();
            }
            fix.persist(store)?;
        }
        Ok(root_id)
    }
}

fn count_trailing_zeros(hash: &[u8; 32]) -> u8 {
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

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
enum Rel {
    Left,
    Right,
}
