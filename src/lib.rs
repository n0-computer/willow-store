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
//! # OpLog
//!
//! Currently, node ids are reused when a value is modified. If we would use
//! new node ids on change and keep the old ones around, we could implement
//! an oplog that you can use to get the entire history of the database.
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

mod geom;
use geom::BBoxRef;
pub use geom::{Point, PointRef, QueryRange, QueryRange3d};
mod store;
use ref_cast::RefCast;
mod layout;
// mod path;
mod blob_seq;
mod fmt;
use layout::*;
pub use store::{mem::MemStore, BlobStore, NodeId};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

pub use blob_seq::{BlobSeq, BlobSeqRef};
pub use store::redb::{RedbBlobStore, Snapshot, Tables};

#[cfg(any(test, feature = "mock-willow"))]
pub mod mock_willow;

/// Provide the size of a fixed size type as a constant.
pub trait FixedSize {
    const SIZE: usize;
}

/// Provide the size of a variable size type.
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

/// Type parameters for keys.
///
/// Keys have three components, x, y, and z. X and y must be fixed size, z can
/// be variable size. For Z we need a pair of owned and borrowed representation.
pub trait KeyParams: Eq + PartialEq + Ord + PartialOrd + Clone + Debug {
    type X: Ord + Debug + Display + AsBytes + FromBytes + FixedSize + Clone + LowerBound;
    type Y: Ord + Debug + Display + AsBytes + FromBytes + FixedSize + Clone + LowerBound;
    type ZOwned: Debug + Display + Borrow<Self::Z> + Ord + Clone + LowerBound;
    type Z: Ord
        + Debug
        + Display
        + AsBytes
        + VariableSize
        + ToOwned<Owned = Self::ZOwned>
        + RefFromSlice
        + IsLowerBound
        + ?Sized;
}

/// Trait for the summary operation of the tree.
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

/// This is a separate trait because we can implement [IsLowerBound] for an
/// unsized type, but not [LowerBound].
pub trait IsLowerBound {
    fn is_min_value(&self) -> bool;
}

pub trait LowerBound: IsLowerBound {
    fn min_value() -> Self;
}

impl IsLowerBound for u64 {
    fn is_min_value(&self) -> bool {
        *self == 0
    }
}

impl LowerBound for u64 {
    fn min_value() -> Self {
        0
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
}

impl<T: store::BlobStore, P: TreeParams> NodeStore<P> for T {}

pub trait NodeStoreRead<P: TreeParams>: store::BlobStoreRead {
    fn data(&self, id: Node<P>) -> Result<OwnedNodeData<P>> {
        let data = store::BlobStoreRead::read(self, id.0)?;
        Ok(OwnedNodeData::new(data))
    }

    fn peek_data<T>(&self, id: Node<P>, f: impl Fn(&NodeData<P>) -> T) -> Result<T> {
        store::BlobStoreRead::peek(self, id.0, |data| f(NodeData::ref_cast(data)))
    }

    fn peek_data_opt<T>(&self, id: Node<P>, f: impl Fn(&NodeData<P>) -> T) -> Result<Option<T>> {
        Ok(if id.is_empty() {
            None
        } else {
            Some(self.peek_data(id, f)?)
        })
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

impl<T: store::BlobStoreRead, P: TreeParams> NodeStoreRead<P> for T {}

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

impl<P: TreeParams> From<Node<P>> for store::NodeId {
    fn from(id: Node<P>) -> Self {
        store::NodeId::from(id.0)
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

    pub fn non_empty(&self) -> Option<Self> {
        if self.is_empty() {
            None
        } else {
            Some(*self)
        }
    }

    pub fn count(&self, store: &impl NodeStoreRead<P>) -> Result<u64> {
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
                // this would be what we would do instead to create a new node
                // (you could then use the nodes ordered by id as a log)
                // store.delete(self.0)?;
                // *self = store.create_node(&this)?;
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
        store: &impl NodeStoreRead<P>,
        include_summary: bool,
    ) -> Result<()> {
        if !self.is_empty() {
            let data = store.get_non_empty(*self)?;
            data.assert_invariants(store, include_summary)?;
        }
        Ok(())
    }

    pub fn get(&self, key: &PointRef<P>, store: &impl NodeStoreRead<P>) -> Result<Option<P::V>> {
        Ok(self.get0(key, store)?.map(|x| x.value().clone()))
    }

    fn get0(
        &self,
        key: &PointRef<P>,
        store: &impl NodeStoreRead<P>,
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

    pub fn dump(&self, store: &impl NodeStoreRead<P>) -> Result<()> {
        self.dump0("".into(), store)
    }

    fn dump0(&self, prefix: String, store: &impl NodeStoreRead<P>) -> Result<()> {
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

    pub fn iter<'a>(
        &'a self,
        store: &'a impl NodeStoreRead<P>,
    ) -> impl Iterator<Item = Result<(Point<P>, P::V)>> + 'a {
        self.iter_impl(
            &|data, _| (data.key().to_owned(), data.value().clone()),
            store,
        )
    }

    pub fn values<'a>(
        &'a self,
        store: &'a impl NodeStoreRead<P>,
    ) -> impl Iterator<Item = Result<P::V>> + 'a {
        self.iter_impl(&|data, _| data.value().clone(), store)
    }

    /// Computes the average depth of the nodes in the tree, as a measure of the tree's balance.
    pub fn average_node_depth(&self, store: &impl NodeStoreRead<P>) -> Result<(u64, u64)> {
        let mut sum = 0;
        for item in self.iter_impl(&|_, depth| depth, store) {
            sum += item?;
        }
        Ok((sum, self.count(store)?))
    }

    /// Iterate over the entire tree in its natural order.
    ///
    /// The order is implementation dependent and should not be relied on.
    ///
    /// Provides a projection function to project out the desired data
    fn iter_impl<'a, T: 'a>(
        &'a self,
        project: &'a impl Fn(&NodeData<P>, u64) -> T,
        store: &'a impl NodeStoreRead<P>,
    ) -> impl Iterator<Item = Result<T>> + 'a {
        Gen::new(|co| async move {
            if let Err(cause) = self.iter_rec(0, &project, store, &co).await {
                co.yield_(Err(cause)).await;
            }
        })
        .into_iter()
    }

    async fn iter_rec<T>(
        &self,
        depth: u64,
        project: &impl Fn(&NodeData<P>, u64) -> T,
        store: &impl NodeStoreRead<P>,
        co: &Co<Result<T>>,
    ) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        let (left, right, item) = store.peek_data(*self, |data| {
            let res = project(data, depth);
            (data.left(), data.right(), res)
        })?;
        Box::pin(left.iter_rec(depth + 1, project, store, co)).await?;
        co.yield_(Ok(item)).await;
        Box::pin(right.iter_rec(depth + 1, project, store, co)).await?;
        Ok(())
    }

    /// Get a summary of the elements in a 3d range.
    ///
    /// The result is identical to iterating over the elements in the 3d range
    /// and combining the summaries of each element, but will be much more
    /// efficient for large trees.
    pub fn range_summary(
        &self,
        query: &QueryRange3d<P>,
        store: &impl NodeStoreRead<P>,
    ) -> Result<P::M> {
        let bbox = BBoxRef::all();
        self.range_summary_rec(query, &bbox, store)
    }

    fn range_summary_rec(
        &self,
        query: &QueryRange3d<P>,
        bbox: &BBoxRef<P>,
        store: &impl NodeStoreRead<P>,
    ) -> Result<P::M> {
        if self.is_empty() {
            return Ok(P::M::neutral());
        }
        store.peek_data(*self, |data| -> Result<P::M> {
            let key = data.key();
            let order = data.sort_order();
            if data.is_leaf() {
                return Ok(if query.contains(key) {
                    data.summary().clone()
                } else {
                    P::M::neutral()
                });
            }
            if bbox.contained_in(&query) {
                return Ok(data.summary().clone());
            }
            let mut summary = P::M::neutral();
            let left = data.left().filter(|| query.overlaps_left(key, order));
            let right = data.right().filter(|| query.overlaps_right(key, order));
            if !left.is_empty() {
                let left_bbox = bbox.split_left(key, order);
                summary = summary.combine(&left.range_summary_rec(query, &left_bbox, store)?);
            }
            if query.contains(key) {
                summary = summary.combine(&P::M::lift(key, data.value()));
            }
            if !right.is_empty() {
                let right_bbox = bbox.split_right(key, order);
                summary = summary.combine(&right.range_summary_rec(query, &right_bbox, store)?);
            }
            Ok(summary)
        })?
    }

    /// Count the number of elements in a 3d range.
    pub fn range_count(
        &self,
        query: &QueryRange3d<P>,
        store: &impl NodeStoreRead<P>,
    ) -> Result<u64> {
        let bbox = BBoxRef::all();
        self.range_count_rec(query, &bbox, store)
    }

    fn range_count_rec(
        &self,
        query: &QueryRange3d<P>,
        bbox: &BBoxRef<P>,
        store: &impl NodeStoreRead<P>,
    ) -> Result<u64> {
        if self.is_empty() {
            return Ok(0);
        }
        store.peek_data(*self, |data| range_count(data, bbox, query, store))?
    }

    pub fn split_range<'a>(
        &'a self,
        query: QueryRange3d<P>,
        split_factor: u64,
        store: &'a impl NodeStoreRead<P>,
    ) -> impl Iterator<Item = Result<(QueryRange3d<P>, u64)>> + 'a
    where
        P::X: Display,
        P::Y: Display,
        P::ZOwned: Display,
    {
        Gen::new(|co| async move {
            let count = match self.range_count(&query, store) {
                Ok(count) => count,
                Err(cause) => {
                    co.yield_(Err(cause)).await;
                    return;
                }
            };
            if let Err(cause) = self
                .split_range_rec(query, count, split_factor, &co, store)
                .await
            {
                co.yield_(Err(cause)).await;
            }
        })
        .into_iter()
    }

    pub fn find_split_plane_supernew(
        &self,
        query: &QueryRange3d<P>,
        store: &impl NodeStoreRead<P>,
    ) -> Result<Option<(QueryRange3d<P>, Node<P>, QueryRange3d<P>, Node<P>)>> {
        if self.is_empty() {
            return Ok(None);
        }
        store.peek_data(*self, |data| {
            let key = data.key();
            let order = data.sort_order();
            let left = data.left();
            let right = data.right();
            if !left.is_empty() {
                // this is just wrong. left and right only work if the two keys are different in the first dimension of the order.
                let lq = query.left(key, order);
                let rq = query.right(key, order);
                // todo: if the split is in the same direction as order, we can
                // return left and right instead of self and self.
                Ok(Some((lq, *self, rq, *self)))
            } else if !data.right().is_empty() {
                let (lq, rq) = store.peek_data(right, |right| {
                    let right_key = right.key();
                    (query.left(key, order), query.right(right_key, order))
                })?;
                // todo: if the split is in the same direction as order, we can
                // return left and right instead of self and self.
                Ok(Some((lq, *self, rq, right)))
            } else {
                Ok(None)
            }
        })?
    }

    pub fn find_split_plane_old(
        &self,
        query: &QueryRange3d<P>,
        count: u64,
        store: &impl NodeStoreRead<P>,
    ) -> Result<Option<(QueryRange3d<P>, u64, QueryRange3d<P>, u64)>> {
        let bbox = BBoxRef::all();
        self.find_split_plane_rec(*self, query, count, &bbox, store)
    }

    pub fn find_split_plane(
        &self,
        query: &QueryRange3d<P>,
        count: u64,
        store: &impl NodeStoreRead<P>,
    ) -> Result<Option<(QueryRange3d<P>, u64, QueryRange3d<P>, u64)>> {
        let Some((point, order)) = self.find_split_plane_2(query, store)? else {
            return Ok(None);
        };
        let left = query.left(&point, order);
        let right = query.right(&point, order);
        let left_count = self.range_count(&left, store)?;
        let right_count = count - left_count;
        Ok(Some((left, left_count, right, right_count)))
    }

    pub fn find_split_plane_2(
        &self,
        query: &QueryRange3d<P>,
        store: &impl NodeStoreRead<P>,
    ) -> Result<Option<(Point<P>, SortOrder)>> {
        let mut points = self.query_interleaved(query, &|_, data| data.key().to_owned(), store);
        let Some(a) = points.next() else {
            return Ok(None);
        };
        let a = a?;
        let Some(b) = points.next() else {
            return Ok(None);
        };
        let b = b?;
        match a.x().cmp(b.x()) {
            Ordering::Less => {
                return Ok(Some((b, SortOrder::XYZ)));
            }
            Ordering::Greater => {
                return Ok(Some((a, SortOrder::XYZ)));
            }
            Ordering::Equal => {}
        }
        match a.y().cmp(b.y()) {
            Ordering::Less => {
                return Ok(Some((b, SortOrder::YZX)));
            }
            Ordering::Greater => {
                return Ok(Some((a, SortOrder::YZX)));
            }
            Ordering::Equal => {}
        }
        match a.z().cmp(b.z()) {
            Ordering::Less => {
                return Ok(Some((b, SortOrder::ZXY)));
            }
            Ordering::Greater => {
                return Ok(Some((a, SortOrder::ZXY)));
            }
            Ordering::Equal => {
                unreachable!("two points are equal");
            }
        }
    }

    fn find_split_plane_rec(
        &self,
        root: Node<P>,
        query: &QueryRange3d<P>,
        count: u64,
        bbox: &BBoxRef<P>,
        store: &impl NodeStoreRead<P>,
    ) -> Result<Option<(QueryRange3d<P>, u64, QueryRange3d<P>, u64)>> {
        if self.is_empty() {
            return Ok(None);
        }
        if count <= 1 {
            return Ok(None);
        }
        if !bbox.intersects(query) {
            return Ok(None);
        }
        let data = store.data(*self)?;
        let key = data.key();
        let order = data.sort_order();
        if query.contains(key) {
            for order in [order, order.inc(), order.inc().inc()] {
                let left = query.left(key, order);
                let left_count = root.range_count(&left, store)?;
                let right_count = count - left_count;
                if left_count < count && left_count > 0 {
                    let right = query.right(key, order);
                    return Ok(Some((left, left_count, right, right_count)));
                }
            }
        }
        if let Some(res) = data.left().find_split_plane_rec(
            root,
            query,
            count,
            &bbox.split_left(key, order),
            store,
        )? {
            return Ok(Some(res));
        }
        if let Some(res) = data.right().find_split_plane_rec(
            root,
            query,
            count,
            &bbox.split_right(key, order),
            store,
        )? {
            return Ok(Some(res));
        }
        Ok(None)
    }

    async fn split_range_rec(
        &self,
        query: QueryRange3d<P>,
        count: u64,
        split_factor: u64,
        co: &Co<Result<(QueryRange3d<P>, u64)>>,
        store: &impl NodeStoreRead<P>,
    ) -> Result<()> {
        // we split the query, but not the tree
        let root = self;
        if count == 0 {
            // nothing to split
            return Ok(());
        }
        if split_factor == 1 || count == 1 {
            // just send the whole thing
            co.yield_(Ok((query, count))).await;
            return Ok(());
        }
        let (left, left_count, right, right_count) = root
            .find_split_plane(&query, count, store)?
            .expect("must find split plane");
        let left_factor = ((left_count * split_factor) / count).max(1);
        let right_factor = split_factor - left_factor;
        Box::pin(root.split_range_rec(left, left_count, left_factor, co, store)).await?;
        Box::pin(root.split_range_rec(right, right_count, right_factor, co, store)).await?;
        Ok(())
    }

    /// Query a 3d range in the tree in its natural order.
    ///
    /// The order is implementation dependent and should not be relied on.
    pub fn query<'a>(
        &'a self,
        query: &'a QueryRange3d<P>,
        store: &'a impl NodeStoreRead<P>,
    ) -> impl Iterator<Item = Result<(Point<P>, P::V)>> + 'a {
        let project = |_, data: &NodeData<P>| (data.key().to_owned(), data.value().clone());
        Gen::new(|co| async move {
            if let Err(cause) = self.query_rec(query, &project, store, &co).await {
                co.yield_(Err(cause)).await;
            }
        })
        .into_iter()
    }

    /// internal convenience function to filter nodes by some
    fn filter(&self, f: impl Fn() -> bool) -> Self {
        if !self.is_empty() && f() {
            *self
        } else {
            Node::EMPTY
        }
    }

    async fn query_rec<T>(
        &self,
        query: &QueryRange3d<P>,
        project: &impl Fn(Node<P>, &NodeData<P>) -> T,
        store: &impl NodeStoreRead<P>,
        co: &Co<Result<T>>,
    ) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        // project out the data we need, left, right and kv
        let (left, kv, right) = store.peek_data(*self, |data| {
            let key = data.key();
            let order = data.sort_order();
            let left = data.left().filter(|| query.overlaps_left(key, order));
            let right = data.right().filter(|| query.overlaps_right(key, order));
            let kv = if query.contains(key) {
                Some(project(*self, data))
            } else {
                None
            };
            (left, kv, right)
        })?;
        // do the recursion after the projection
        if !left.is_empty() {
            Box::pin(left.query_rec(query, project, store, co)).await?;
        }
        if let Some(kv) = kv {
            co.yield_(Ok(kv)).await;
        }
        if !right.is_empty() {
            Box::pin(right.query_rec(query, project, store, co)).await?;
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
        store: &'a impl NodeStoreRead<P>,
    ) -> impl Iterator<Item = Result<(Point<P>, P::V)>> + 'a {
        Gen::new(|co| async move {
            if let Err(cause) = self.query_ordered_rec(query, ordering, store, &co).await {
                co.yield_(Err(cause)).await;
            }
        })
        .into_iter()
    }

    async fn query_ordered_rec(
        &self,
        query: &QueryRange3d<P>,
        ordering: SortOrder,
        store: &impl NodeStoreRead<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        let (left, right, kv, node_order) = store.peek_data(*self, |data| {
            let order = data.sort_order();
            let key = data.key();
            let left = data.left().filter(|| query.overlaps_left(key, order));
            let right = data.right().filter(|| query.overlaps_right(key, order));
            let kv = if query.contains(key) {
                Some((key.to_owned(), data.value().to_owned()))
            } else {
                None
            };
            (left, right, kv, order)
        })?;
        if node_order == ordering {
            if !left.is_empty() {
                Box::pin(left.query_ordered_rec(query, ordering, store, co)).await?;
            }
            if let Some(kv) = kv {
                co.yield_(Ok(kv)).await;
            }
            if !right.is_empty() {
                Box::pin(right.query_ordered_rec(query, ordering, store, co)).await?;
            }
        } else {
            // the node does not partition the space in the correct way, so we
            // need to merge the results from the left, self and right
            // still better than a full sort!
            let iter1 = if !left.is_empty() {
                Some(left.query_ordered(query, ordering, store).into_iter())
            } else {
                None
            }
            .into_iter()
            .flatten();
            let iter2 = if let Some(kv) = kv {
                Some(std::iter::once(Ok(kv)))
            } else {
                None
            }
            .into_iter()
            .flatten();
            let iter3 = if !right.is_empty() {
                Some(right.query_ordered(query, ordering, store).into_iter())
            } else {
                None
            }
            .into_iter()
            .flatten();
            merge3(iter1, iter2, iter3, ordering, co).await?;
        }
        Ok(())
    }

    /// iterate over all elements in the query range,
    /// yielding first the node itself then the left and right subtrees
    /// interleaved. This is roughly like a breadth-first traversal.
    /// allow projecting out a value from each node
    pub fn query_interleaved<'a, T: 'a>(
        self,
        query: &'a QueryRange3d<P>,
        project: &'a impl Fn(Node<P>, &NodeData<P>) -> T,
        store: &'a impl NodeStoreRead<P>,
    ) -> Box<dyn Iterator<Item = Result<T>> + 'a> {
        match self.query_interleaved_rec(query, project, store) {
            Ok(iter) => Box::new(iter),
            Err(cause) => Box::new(std::iter::once(Err(cause))),
        }
    }

    fn query_interleaved_rec<'a, T: 'a>(
        self,
        query: &'a QueryRange3d<P>,
        project: &'a impl Fn(Node<P>, &NodeData<P>) -> T,
        store: &'a impl NodeStoreRead<P>,
    ) -> Result<impl Iterator<Item = Result<T>> + 'a> {
        let res = if self.is_empty() {
            None
        } else {
            let (left, right, kv) = store.peek_data(self, |data| {
                let order = data.sort_order();
                let key = data.key();
                let left = data.left().filter(|| query.overlaps_left(key, order));
                let right = data.right().filter(|| query.overlaps_right(key, order));
                let kv = if query.contains(key) {
                    Some(project(self, data))
                } else {
                    None
                };
                (left, right, kv)
            })?;
            let left = left
                .non_empty()
                .map(|left| left.query_interleaved(query, project, store).into_iter())
                .into_iter()
                .flatten();
            let right = right
                .non_empty()
                .map(|right| right.query_interleaved(query, project, store).into_iter())
                .into_iter()
                .flatten();
            let me = kv.map(anyhow::Ok).into_iter();
            Some(me.chain(itertools::interleave(left, right)))
        };
        Ok(res.into_iter().flatten())
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

/// the logic for range_count, put into a function so it is visible in flamegraph
fn range_count<P: TreeParams>(
    data: &NodeData<P>,
    bbox: &BBoxRef<P>,
    query: &QueryRange3d<P>,
    store: &impl NodeStoreRead<P>,
) -> Result<u64> {
    if bbox.contained_in(&query) {
        Ok(data.count())
    } else {
        let key = data.key();
        let order = data.sort_order();
        let left = data.left().filter(|| query.overlaps_left(key, order));
        let right = data.right().filter(|| query.overlaps_right(key, order));
        let mut count = 0;
        if query.contains(key) {
            count += 1;
        }
        if !left.is_empty() {
            let left_bbox = bbox.split_left(key, order);
            count += left.range_count_rec(query, &left_bbox, store)?;
        }
        if !right.is_empty() {
            let right_bbox = bbox.split_right(key, order);
            count += right.range_count_rec(query, &right_bbox, store)?;
        }
        Ok(count)
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
        *res.count_mut() = 1;
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
        // Use native byte order for count
        u64::read_from_prefix(&self.1[COUNT_OFFSET..]).unwrap()
    }

    pub fn count_mut(&mut self) -> &mut u64 {
        // Use native byte order for count
        u64::mut_from_prefix(&mut self.1[COUNT_OFFSET..]).unwrap()
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
        *self.count_mut() = 1;
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
        store: &impl NodeStoreRead<P>,
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

    pub fn left_mut(&mut self) -> &mut Node<P> {
        Node::mut_from_prefix(&mut self.1[LEFT_OFFSET..]).unwrap()
    }

    pub fn right_mut(&mut self) -> &mut Node<P> {
        Node::mut_from_prefix(&mut self.1[RIGHT_OFFSET..]).unwrap()
    }

    pub fn summary_mut(&mut self) -> &mut P::M {
        P::M::mut_from_prefix(&mut self.1[summary_offset::<P>()..]).unwrap()
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
