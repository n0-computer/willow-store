use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Display},
    future::Future,
    ops::RangeBounds,
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

///
pub trait ValueParams: PartialEq + Eq + Serialize + Clone + Debug {}

impl<T: PartialEq + Eq + Serialize + Clone + Debug> ValueParams for T {}

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

/// Tree params for a 3D tree. This extends `KeyParams` with a value and
/// summary type.
pub trait TreeParams: KeyParams + Sized {
    type V: ValueParams;
    type M: LiftingCommutativeMonoid<(Point<Self>, Self::V)> + Clone + Debug + Eq;
}

/// A simple store trait for storing nodes by id.
pub trait Store<P: TreeParams> {
    fn put(&mut self, node: &Node<P>) -> Result<NodeId>;
    fn update(&mut self, id: &NodeId, node: &Node<P>) -> Result<()>;
    fn get(&self, id: &NodeId) -> Result<Node<P>>;
}

pub struct MemStore<P: TreeParams> {
    nodes: BTreeMap<NodeId, Node<P>>,
}

impl<P: TreeParams> MemStore<P> {
    fn new() -> Self {
        MemStore {
            nodes: BTreeMap::new(),
        }
    }
}

impl<P: TreeParams> Store<P> for MemStore<P> {
    fn put(&mut self, node: &Node<P>) -> Result<NodeId> {
        let id = self.nodes.len() as NodeId;
        self.nodes.insert(id, node.clone());
        Ok(id)
    }

    fn update(&mut self, id: &NodeId, node: &Node<P>) -> Result<()> {
        self.nodes.insert(*id, node.clone());
        Ok(())
    }

    fn get(&self, id: &NodeId) -> Result<Node<P>> {
        self.nodes
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Node not found"))
    }
}

pub type NodeId = u64;

pub struct Node<P: TreeParams> {
    key: Point<P>,
    rank: u8,
    value: P::V,
    summary: P::M,
    left: Option<NodeId>,
    right: Option<NodeId>,
}

impl<P: TreeParams> Clone for Node<P> {
    fn clone(&self) -> Self {
        Node {
            key: self.key.clone(),
            rank: self.rank,
            value: self.value.clone(),
            summary: self.summary.clone(),
            left: self.left.clone(),
            right: self.right.clone(),
        }
    }
}

impl<P: TreeParams> Node<P> {
    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    pub fn sort_order(&self) -> SortOrder {
        self.rank.into()
    }

    /// Iterate over the tree in its own natural order.
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

    async fn iter_unordered0(
        &self,
        store: &impl Store<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        if let Some(left) = self.left.map(|id| store.get(&id)).transpose()? {
            Box::pin(left.iter_unordered0(store, co)).await?;
        }
        co.yield_(Ok((self.key.clone(), self.value.clone()))).await;
        if let Some(right) = self.right.map(|id| store.get(&id)).transpose()? {
            Box::pin(right.iter_unordered0(store, co)).await?;
        }
        Ok(())
    }

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

    pub fn summary(&self, query: &QueryRange3d<P>, store: &impl Store<P>) -> Result<P::M> {
        let bbox = BBox::all();
        self.summary0(query, &bbox, store)
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
            println!("bbox contained in query");
            println!("bbox: {}", bbox);
            println!("query: {}", query);
            return Ok(self.summary.clone());
        }
        let mut summary = P::M::zero();
        if let Some(left) = self.left.map(|id| store.get(&id)).transpose()? {
            let left_bbox = bbox.split_left(&self.key, self.rank);
            summary = summary.combine(&left.summary0(query, &left_bbox, store)?);
        }
        if query.contains(&self.key) {
            summary = summary.combine(&P::M::lift((self.key.clone(), self.value.clone())));
        }
        if let Some(right) = self.right.map(|id| store.get(&id)).transpose()? {
            let right_bbox = bbox.split_right(&self.key, self.rank);
            summary = summary.combine(&right.summary0(query, &right_bbox, store)?);
        }
        Ok(summary)
    }

    async fn query_unordered0(
        &self,
        query: &QueryRange3d<P>,
        store: &impl Store<P>,
        co: &Co<Result<(Point<P>, P::V)>>,
    ) -> Result<()> {
        if let Some(left) = self.left.map(|id| store.get(&id)).transpose()? {
            let overlap = match self.sort_order() {
                SortOrder::XYZ => query.x.min <= self.key.x,
                SortOrder::YZX => query.y.min <= self.key.y,
                SortOrder::ZXY => query.z.min <= self.key.z,
            };
            if overlap {
                Box::pin(left.query_unordered0(query, store, co)).await?;
            }
        }
        if query.contains(&self.key) {
            co.yield_(Ok((self.key.clone(), self.value.clone()))).await;
        }
        if let Some(right) = self.right.map(|id| store.get(&id)).transpose()? {
            let overlap = match self.sort_order() {
                SortOrder::XYZ => !query
                    .x
                    .max
                    .as_ref()
                    .map(|x_max| x_max < &self.key.x)
                    .unwrap_or_default(),
                SortOrder::YZX => !query
                    .y
                    .max
                    .as_ref()
                    .map(|y_max| y_max < &self.key.y)
                    .unwrap_or_default(),
                SortOrder::ZXY => !query
                    .z
                    .max
                    .as_ref()
                    .map(|z_max| z_max < &self.key.z)
                    .unwrap_or_default(),
            };
            if overlap {
                Box::pin(right.query_unordered0(query, store, co)).await?;
            }
        }
        Ok(())
    }

    /// Create a new node with the given key and value.
    pub fn single(key: Point<P>, value: P::V) -> Self {
        let summary = P::M::lift((key.clone(), value.clone()));
        let key_bytes = postcard::to_allocvec(&key).expect("Failed to serialize key");
        let key_hash: [u8; 32] = blake3::hash(&key_bytes).into();
        let rank = count_trailing_zeros(&key_hash);
        Node {
            key,
            rank,
            value,
            summary,
            left: None,
            right: None,
        }
    }

    pub fn from_iter<I: IntoIterator<Item = (Point<P>, P::V)>>(
        iter: I,
    ) -> Result<(MemStore<P>, NodeId)> {
        let mut nodes: Vec<_> = iter
            .into_iter()
            .map(|(key, value)| Node::single(key, value))
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
            root.insert_no_balance(
                &root_id,
                node.key,
                node.value,
                node.rank,
                &node.summary,
                &mut store,
            )?;
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
        let Node {
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
        point: Point<P>,
        value: P::V,
        rank: u8,
        summary: &P::M,
        store: &mut impl Store<P>,
    ) -> Result<()> {
        let Node {
            key: parent_key,
            rank: parent_rank,
            summary: parent_summary,
            left,
            right,
            value: _,
        } = self;
        match point.cmp_at_rank(parent_key, *parent_rank) {
            Ordering::Less => {
                if let Some(left_id) = left {
                    let mut left = store.get(left_id)?;
                    left.insert_no_balance(left_id, point, value, rank, summary, store)?;
                } else {
                    *left = Some(store.put(&Node::single(point, value))?);
                }
            }
            Ordering::Greater => {
                if let Some(right_id) = right {
                    let mut right = store.get(right_id)?;
                    right.insert_no_balance(right_id, point, value, rank, summary, store)?;
                } else {
                    *right = Some(store.put(&Node::single(point, value))?);
                }
            }
            Ordering::Equal => {
                panic!("Duplicate keys not supported in insert_no_balance");
            }
        }
        *parent_summary = parent_summary.combine(summary);
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

#[derive(Debug)]
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
