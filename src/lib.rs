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
    num::NonZeroU64, ops::Deref,
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

pub trait KeyParams {
    type X: CoordParams;
    type Y: CoordParams;
    type Z: CoordParams;
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

pub struct AssertInvariantsRes<S: TreeParams> {
    xyz: RangeInclusive<XYZ<S>>,
    yzx: RangeInclusive<YZX<S>>,
    zxy: RangeInclusive<ZXY<S>>,
    summary: S::M,
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
pub trait ValueParams: PartialEq + Eq + Serialize + Clone + Debug {}

impl<T: PartialEq + Eq + Serialize + Clone + Debug> ValueParams for T {}

/// Tree params for a 3D tree. This extends `KeyParams` with a value and
/// summary type.
pub trait TreeParams: KeyParams + Sized {
    type V: ValueParams;
    type M: LiftingCommutativeMonoid<(Point<Self>, Self::V)> + Clone + Debug + Eq;
}

/// A simple store trait for storing nodes by id.
pub trait Store<P: TreeParams> {
    fn put(&mut self, node: &NodeData<P>) -> Result<NodeId>;
    fn update(&mut self, id: &NodeId, node: &NodeData<P>) -> Result<()>;
    fn get(&self, id: &NodeId) -> Result<NodeData<P>>;

    /// Get a node by id, returning None if the id is None.
    ///
    /// This is just a convenience method for the common case where you have an
    /// optional id and want to get the node if it exists.
    fn get_opt(&self, id: &Option<NodeId>) -> Result<Option<NodeData<P>>> {
        match id {
            Some(id) => Ok(Some(self.get(id)?)),
            None => Ok(None),
        }
    }

    /// Get a node by id, returning None if the id is None.
    /// Also return the id along with the node.
    fn get_opt_with_id(&self, id: &Option<NodeId>) -> Result<Option<(NodeId, NodeData<P>)>> {
        match id {
            Some(id) => {
                let node = self.get(id)?;
                Ok(Some((*id, node)))
            }
            None => Ok(None),
        }
    }
}

pub struct MemStore<P: TreeParams> {
    nodes: BTreeMap<NodeId, NodeData<P>>,
}

impl<P: TreeParams> MemStore<P> {
    fn new() -> Self {
        MemStore {
            nodes: BTreeMap::new(),
        }
    }
}

impl<P: TreeParams> Store<P> for MemStore<P> {
    fn put(&mut self, node: &NodeData<P>) -> Result<NodeId> {
        let id = NodeId::new((self.nodes.len() as u64) + 1).unwrap();
        self.nodes.insert(id, node.clone());
        Ok(id)
    }

    fn update(&mut self, id: &NodeId, node: &NodeData<P>) -> Result<()> {
        self.nodes.insert(*id, node.clone());
        Ok(())
    }

    fn get(&self, id: &NodeId) -> Result<NodeData<P>> {
        self.nodes
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Node not found"))
    }
}

pub type NodeId = NonZeroU64;

/// Data for a node in the tree.
/// This gets persisted as a single &[u8] in the database.
///
/// TODO:
/// - Use some zero copy stuff to make this more efficient.
/// - Have dedicated types for owned and borrowed NodeData (?).
pub struct NodeData<P: TreeParams> {
    key: Point<P>,         // variable size due to path
    rank: u8,              // 1 byte
    value: P::V,           // fixed size
    summary: P::M,         // fixed size
    left: Option<NodeId>,  // 8 bytes
    right: Option<NodeId>, // 8 bytes
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

/// A node combines node data with a node id.
pub struct Node<P: TreeParams> {
    id: NodeId,
    data: NodeData<P>,
}

impl<P: TreeParams> Clone for Node<P> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            data: self.data.clone(),
        }
    }
}

impl<P: TreeParams> Deref for Node<P> {
    type Target = NodeData<P>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<P: TreeParams> NodeData<P> {
    /// True if the node is a leaf.
    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// Get the sort order for the node, based on the rank.
    pub fn sort_order(&self) -> SortOrder {
        self.rank.into()
    }

    /// Iterate over the entire tree in its natural order.
    ///
    /// The order is implementation dependent and should not be relied on.
    pub fn iter_unordered<'a>(
        &'a self,
        store: &'a impl Store<P>,
    ) -> impl IntoIterator<Item = Result<(Point<P>, P::V)>> + 'a {
        Gen::new(|co| async move {
            if let Err(cause) = self.iter_unordered0(store, &co).await {
                co.yield_(Err(cause)).await;
            }
        })
    }

    /// Query a 3d range in the tree in its natural order.
    ///
    /// The order is implementation dependent and should not be relied on.
    pub fn query_unordered<'a>(
        &'a self,
        query: &'a QueryRange3d<P>,
        store: &'a impl Store<P>,
    ) -> impl IntoIterator<Item = Result<(Point<P>, P::V)>> + 'a {
        Gen::new(|co| async move {
            if let Err(cause) = self.query_unordered0(query, store, &co).await {
                co.yield_(Err(cause)).await;
            }
        })
    }

    /// Query a 3d range in the tree in a defined order.
    ///
    /// The order is defined by the `ordering` parameter. All orderings use all
    /// three dimensions, so the result is fully deteministic.
    pub fn query_ordered<'a>(
        self,
        query: &'a QueryRange3d<P>,
        ordering: SortOrder,
        store: &'a impl Store<P>,
    ) -> impl IntoIterator<Item = Result<(Point<P>, P::V)>> + 'a {
        Gen::new(|co| async move {
            if let Err(cause) = self.query_ordered0(query, ordering, store, &co).await {
                co.yield_(Err(cause)).await;
            }
        })
    }

    /// Get a summary of the elements in a 3d range.
    ///
    /// The result is identical to iterating over the elements in the 3d range
    /// and combining the summaries of each element, but will be much more
    /// efficient for large trees.
    pub fn summary(&self, query: &QueryRange3d<P>, store: &impl Store<P>) -> Result<P::M> {
        let bbox = BBox::all();
        self.summary0(query, &bbox, store)
    }

    async fn iter_unordered0(
        &self,
        store: &impl Store<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        if let Some(left) = store.get_opt(&self.left)? {
            Box::pin(left.iter_unordered0(store, co)).await?;
        }
        co.yield_(Ok((self.key.clone(), self.value.clone()))).await;
        if let Some(right) = store.get_opt(&self.right)? {
            Box::pin(right.iter_unordered0(store, co)).await?;
        }
        Ok(())
    }

    fn summary0(
        &self,
        query: &QueryRange3d<P>,
        bbox: &BBox<P>,
        store: &impl Store<P>,
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
            if let Some(left) = store.get_opt(&self.left)? {
                let left_bbox = bbox.split_left(&self.key, self.rank);
                summary = summary.combine(&left.summary0(query, &left_bbox, store)?);
            }
        }
        if query.contains(&self.key) {
            summary = summary.combine(&P::M::lift((self.key.clone(), self.value.clone())));
        }
        if query.overlaps_right(&self.key, self.rank) {
            if let Some(right) = store.get_opt(&self.right)? {
                let right_bbox = bbox.split_right(&self.key, self.rank);
                summary = summary.combine(&right.summary0(query, &right_bbox, store)?);
            }
        }
        Ok(summary)
    }

    async fn query_unordered0(
        &self,
        query: &QueryRange3d<P>,
        store: &impl Store<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        if query.overlaps_left(&self.key, self.rank) {
            if let Some(left) = store.get_opt(&self.left)? {
                Box::pin(left.query_unordered0(query, store, co)).await?;
            }
        }
        if query.contains(&self.key) {
            co.yield_(Ok((self.key.clone(), self.value.clone()))).await;
        }
        if query.overlaps_right(&self.key, self.rank) {
            if let Some(right) = store.get_opt(&self.right)? {
                Box::pin(right.query_unordered0(query, store, co)).await?;
            }
        }
        Ok(())
    }

    async fn query_ordered0(
        &self,
        query: &QueryRange3d<P>,
        ordering: SortOrder,
        store: &impl Store<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        // we know that the node partitions the space in the correct way, so
        // we can just concatenate the results from the left, self and right
        if self.sort_order() == ordering {
            if query.overlaps_left(&self.key, self.rank) {
                if let Some(left) = store.get_opt(&self.left)? {
                    Box::pin(left.query_ordered0(query, ordering, store, co)).await?;
                }
            }
            if query.contains(&self.key) {
                co.yield_(Ok((self.key.clone(), self.value.clone()))).await;
            }
            if query.overlaps_right(&self.key, self.rank) {
                if let Some(right) = store.get_opt(&self.right)? {
                    Box::pin(right.query_ordered0(query, ordering, store, co)).await?;
                }
            }
        } else {
            // the node does not partition the space in the correct way, so we
            // need to merge the results from the left, self and right
            // still better than a full sort!
            let iter1 = if query.overlaps_left(&self.key, self.rank) {
                if let Some(left) = store.get_opt(&self.left)? {
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
                if let Some(right) = store.get_opt(&self.right)? {
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
        let key_bytes = postcard::to_allocvec(&key).expect("Failed to serialize key");
        let key_hash: [u8; 32] = blake3::hash(&key_bytes).into();
        let rank = count_trailing_zeros(&key_hash);
        NodeData {
            key,
            rank,
            value,
            summary,
            left: None,
            right: None,
        }
    }

    pub fn insert(
        &mut self,
        key: Point<P>,
        value: P::V,
        store: &mut impl Store<P>,
    ) -> Result<Option<P::V>> {
        let node = NodeData::single(key, value);
        self.insert0(&store.put(self)?, &node, store)
    }

    fn insert_iterative(
        mut root_id: Option<NodeId>,
        mut x: NodeData<P>,
        x_id: NodeId,
        store: &mut impl Store<P>,
    ) -> Result<NodeId> {
        #[derive(Debug, Clone, PartialEq, Eq, Copy)]
        enum Rel {
            Left,
            Right,
        }
        use Rel::*;
        let key = x.key.clone();
        let rank = x.rank;
        let cmp_new = |x: &Point<P>| {
            let key_cmp = x.cmp_at_rank(&key, rank);
            match key_cmp {
                Ordering::Less => Left,
                Ordering::Greater => Right,
                Ordering::Equal => unreachable!(),
            }
        };
        let mut curr_id = root_id;
        let mut prev_id = None;
        let mut prev = None;
        // find the correct place to insert the new node
        let mut rel = Right;
        // while cur != null and (rank < cur.rank or (rank = cur.rank and key > cur.key)) do
        //   prev ← cur
        //   cur ← if key < cur.key then cur.left else cur.right
        while let Some(curr) = store.get_opt(&curr_id)? {
            rel = cmp_new(&curr.key);
            if rank < curr.rank || rank == curr.rank && rel == Left {
                prev_id = curr_id;
                curr_id = match rel {
                    Left => curr.left,
                    Right => curr.right,
                };
                prev = Some(curr);
            } else {
                break;
            }
        }
        // if cur = root then root ← x
        //   else if key < prev.key then prev.left ← x
        //   else prev.right ← x
        if curr_id == root_id {
            root_id = Some(x_id);
        } else {
            let prev_id = prev_id.expect("prev_id is None");
            let mut prev = prev.expect("prev is None");
            match rel {
                Left => prev.left = Some(x_id),
                Right => prev.right = Some(x_id),
            }
            store.update(&prev_id, &prev)?;
        }
        // if cur = null then {x.left ← x.right ← null; return}
        if curr_id.is_none() {
            return Ok(x_id);
        }
        // if key < cur.key then x.right ← cur else x.left ← cur
        match rel {
            Left => x.right = curr_id,
            Right => x.left = curr_id,
        }
        store.update(&x_id, &x)?;
        // prev ← x
        prev_id = Some(x_id);
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
        while curr_id.is_some() {
            let fix_id = prev_id.unwrap();
            let mut fix = store.get(&fix_id)?;
            let mut curr = store.get(&curr_id.unwrap())?;
            let mut prev;
            match cmp_new(&curr.key) {
                Left => loop {
                    prev_id = curr_id;
                    curr_id = curr.right;
                    prev = curr;
                    match store.get_opt(&curr_id)? {
                        Some(x) => {
                            curr = x;
                            if cmp_new(&curr.key) == Right {
                                break;
                            }
                        }
                        None => break,
                    }
                },
                Right => loop {
                    prev_id = curr_id;
                    curr_id = curr.left;
                    prev = curr;
                    match store.get_opt(&curr_id)? {
                        Some(x) => {
                            curr = x;
                            if cmp_new(&curr.key) == Left {
                                break;
                            }
                        }
                        None => break,
                    }
                },
            }
            let cmp = cmp_new(&fix.key);
            if cmp == Right || ((fix_id == x_id) && cmp_new(&prev.key) == Right) {
                fix.left = curr_id;
            } else {
                fix.right = curr_id;
            }
            store.update(&fix_id, &fix)?;
        }
        Ok(root_id.unwrap())
    }

    fn insert0(
        &mut self,
        self_id: &NodeId,
        node: &NodeData<P>,
        store: &mut impl Store<P>,
    ) -> Result<Option<P::V>> {
        assert!(node.is_leaf());
        let key_cmp = node.key.cmp_at_rank(&self.key, self.rank);
        let rank_cmp = node.rank.cmp(&self.rank);
        let old_value = match key_cmp {
            Ordering::Equal => {
                // equal ordering means equal rank, since the rank is deterministically derived from the key
                assert!(rank_cmp == Ordering::Equal);
                let mut summary = node.summary.clone();
                if let Some(left) = store.get_opt(&self.left)? {
                    summary = summary.combine(&left.summary);
                }
                if let Some(right) = store.get_opt(&self.right)? {
                    summary = summary.combine(&right.summary);
                }
                // just replace the value and update the summary
                let old_value = self.value.clone();
                self.value = node.value.clone();
                self.summary = summary;
                store.update(self_id, self)?;
                return Ok(Some(old_value));
            }
            Ordering::Less => {
                if let Some((id, mut child)) = store.get_opt_with_id(&self.left)? {
                    if rank_cmp == Ordering::Less {
                        child.insert0(&id, node, store)?
                    } else {
                        // node will be the new parent
                        // my right child will be the right child of the node
                        // my left child will be the left child of the node
                        todo!()
                    }
                } else {
                    self.left = Some(store.put(&node)?);
                    None
                }
            }
            Ordering::Greater => {
                if let Some((id, mut child)) = store.get_opt_with_id(&self.right)? {
                    if rank_cmp == Ordering::Greater || rank_cmp == Ordering::Equal {
                        child.insert0(&id, node, store)?
                    } else {
                        todo!()
                    }
                } else {
                    self.right = Some(store.put(&node)?);
                    None
                }
            }
        };
        self.summary = if old_value.is_none() {
            // just add to the summary
            self.summary.combine(&node.summary)
        } else {
            // recalculate the summary from scratch
            let mut summary = P::M::lift((self.key.clone(), self.value.clone()));
            if let Some(left) = store.get_opt(&self.left)? {
                summary = summary.combine(&left.summary);
            }
            if let Some(right) = store.get_opt(&self.right)? {
                summary = summary.combine(&right.summary);
            }
            summary
        };
        store.update(self_id, self)?;
        Ok(old_value)
    }

    pub fn get(&self, key: Point<P>, store: &impl Store<P>) -> Result<Option<P::V>> {
        match self.get0(key, store)? {
            Some(node) => Ok(Some(node.value)),
            None => Ok(None),
        }
    }

    fn get0(&self, key: Point<P>, store: &impl Store<P>) -> Result<Option<NodeData<P>>> {
        match key.cmp_at_rank(&self.key, self.rank) {
            Ordering::Less => {
                if let Some(left) = self.left {
                    let left = store.get(&left)?;
                    left.get0(key, store)
                } else {
                    Ok(None)
                }
            }
            Ordering::Greater => {
                if let Some(right) = self.right {
                    let right = store.get(&right)?;
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
    ) -> Result<(MemStore<P>, NodeId)> {
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
        let mut root = nodes.remove(0);
        let root_id = store.put(&root)?;
        for node in nodes {
            root.insert_no_balance(&root_id, &node, &mut store)?;
        }
        Ok((store, root_id))
    }

    pub fn dump(&self, store: &impl Store<P>) -> Result<()> {
        self.dump0("".into(), store)
    }

    fn dump0(&self, prefix: String, store: &impl Store<P>) -> Result<()> {
        println!(
            "{}{:?} rank={} order={:?} value={:?}",
            prefix,
            self.key,
            self.rank,
            SortOrder::from(self.rank),
            self.value
        );
        if let Some(left) = self.left {
            println!("{} left:", prefix);
            let left = store.get(&left)?;
            left.dump0(format!("{}  ", prefix), store)?;
        }
        if let Some(right) = self.right {
            println!("{} right:", prefix);
            let right = store.get(&right)?;
            right.dump0(format!("{}  ", prefix), store)?;
        }
        Ok(())
    }

    pub fn assert_invariants(&self, store: &impl Store<P>) -> Result<AssertInvariantsRes<P>> {
        let NodeData {
            key,
            rank,
            value: _,
            summary,
            left,
            right,
        } = self;
        let left = left.map(|id| store.get(&id)).transpose()?;
        let right = right.map(|id| store.get(&id)).transpose()?;
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

    // Insert a new node into the tree without balancing.
    fn insert_no_balance(
        &mut self,
        self_id: &NodeId,
        node: &Self,
        store: &mut impl Store<P>,
    ) -> Result<()> {
        assert!(node.is_leaf());
        let NodeData {
            key: parent_key,
            rank: parent_rank,
            summary: parent_summary,
            left,
            right,
            value: _,
        } = self;
        match node.key.cmp_at_rank(parent_key, *parent_rank) {
            Ordering::Less => {
                if let Some((left_id, mut left)) = store.get_opt_with_id(&left)? {
                    left.insert_no_balance(&left_id, node, store)?;
                } else {
                    *left = Some(store.put(node)?);
                }
            }
            Ordering::Greater => {
                if let Some((right_id, mut right)) = store.get_opt_with_id(&right)? {
                    right.insert_no_balance(&right_id, node, store)?;
                } else {
                    *right = Some(store.put(node)?);
                }
            }
            Ordering::Equal => {
                panic!("Duplicate keys not supported in insert_no_balance");
            }
        }
        *parent_summary = parent_summary.combine(&node.summary);
        store.update(self_id, self)?;
        Ok(())
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
