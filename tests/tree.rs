use std::{
    collections::BTreeSet,
    fmt::{Debug, Display},
    ops::Deref,
};

use prop::sample::SizeRange;
use proptest::prelude::*;
use test_strategy::proptest;
use testresult::TestResult;
use willow_store::{
    FixedSize, KeyParams, LiftingCommutativeMonoid, MemStore, NodeStore, OwnedNodeData, OwnedPoint,
    Point, QueryRange, QueryRange3d, SortOrder, TreeParams, VariableSize,
};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

type TPoint = OwnedPoint<TestParams>;
type TQuery = QueryRange3d<TestParams>;

#[derive(Debug)]
struct TestParams;

impl KeyParams for TestParams {
    type X = u64;
    type Y = u64;
    type Z = u64;
}

#[derive(PartialEq, Eq, PartialOrd, Ord, AsBytes, FromZeroes, FromBytes, Clone, Copy)]
#[repr(transparent)]
struct TestCoord([u8; 8]);

impl From<u64> for TestCoord {
    fn from(x: u64) -> Self {
        TestCoord(x.to_be_bytes())
    }
}

impl From<TestCoord> for u64 {
    fn from(x: TestCoord) -> Self {
        u64::from_be_bytes(x.0)
    }
}

impl Debug for TestCoord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let x: u64 = (*self).into();
        write!(f, "{}", x)
    }
}

impl Display for TestCoord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let x: u64 = (*self).into();
        write!(f, "{}", x)
    }
}

impl FixedSize for TestCoord {
    const SIZE: usize = 8;
}

// impl VariableSize for TestCoord {
//     fn size(&self) -> usize {
//         8
//     }
// }

impl TreeParams for TestParams {
    type V = u64;
    type M = ValueSum;
}

#[derive(Clone, PartialEq, Eq, Debug, AsBytes, FromZeroes, FromBytes)]
#[repr(C)]
struct ValueSum(u64);

impl FixedSize for ValueSum {
    const SIZE: usize = 8;
}

impl LiftingCommutativeMonoid<Point<TestParams>, u64> for ValueSum {
    fn neutral() -> Self {
        ValueSum(0)
    }

    fn lift(_k: &Point<TestParams>, v: &u64) -> Self {
        ValueSum(*v)
    }

    fn combine(&self, other: &Self) -> Self {
        ValueSum(self.0 + other.0)
    }
}

fn point() -> impl Strategy<Value = TPoint> {
    (0..100u64, 0..100u64, 0..100u64).prop_map(|(x, y, z)| TPoint::new(&x, &y, &z))
}

/// A point with just x coordinate set to non-zero.
///
/// Values of this kind compare independent of rank.
fn xpoint() -> impl Strategy<Value = TPoint> {
    (0..100u64).prop_map(|x| TPoint::new(&x, &0, &0))
}

fn treecontents_with_opts(
    p: impl Strategy<Value = TPoint>,
    v: impl Strategy<Value = u64>,
    s: impl Into<SizeRange>,
) -> impl Strategy<Value = Vec<(TPoint, u64)>> {
    prop::collection::vec((p, v), s).prop_map(|mut m| {
        let mut keys = BTreeSet::new();
        m.retain(|(p, _)| keys.insert(p.clone()));
        m
    })
}

fn treecontents() -> impl Strategy<Value = Vec<(TPoint, u64)>> {
    treecontents_with_opts(point(), 0..100u64, 1..100usize)
}

fn sortorder() -> impl Strategy<Value = SortOrder> {
    prop_oneof![
        Just(SortOrder::XYZ),
        Just(SortOrder::YZX),
        Just(SortOrder::ZXY)
    ]
}

fn query() -> impl Strategy<Value = TQuery> {
    (
        0..100u64,
        1..=100u64,
        0..100u64,
        1..=100u64,
        0..100u64,
        1..=100u64,
    )
        .prop_map(|(x0, dx, y0, dy, z0, dz)| {
            fn end(a: u64, b: u64) -> Option<u64> {
                let r = a + b;
                if r > 100 {
                    None
                } else {
                    Some(r)
                }
            }
            TQuery::new(
                QueryRange::new(x0, end(x0, dx)),
                QueryRange::new(y0, end(y0, dy)),
                QueryRange::new(z0, end(z0, dz)),
            )
        })
}

fn tpoint(x: u64, y: u64, z: u64) -> TPoint {
    TPoint::new(&x, &y, &z)
}

/// Tests that creating a tree from unique points works.
///
/// The tree should conform to the invariants of the tree structure,
/// and contain all the points that were inserted.
fn tree_creation_impl(items: Vec<(TPoint, u64)>) -> TestResult<()> {
    let mut store = MemStore::new();
    let node = willow_store::Node::from_iter(items.clone(), &mut store)?;
    // tree.dump(&store)?;
    node.assert_invariants(&store, true)?;
    let mut actual = node.iter(&store).map(|x| x.unwrap()).collect::<Vec<_>>();
    let mut expected = items;
    actual.sort_by_key(|(p, _)| p.clone());
    expected.sort_by_key(|(p, _)| p.clone());
    assert_eq!(actual, expected);
    Ok(())
}

#[proptest]
fn prop_tree_creation(#[strategy(treecontents())] items: Vec<(TPoint, u64)>) {
    tree_creation_impl(items).unwrap();
}

#[test]
fn test_tree_creation() -> TestResult<()> {
    let items = vec![(tpoint(83, 0, 0), 0), (tpoint(0, 0, 1), 2)];
    tree_creation_impl(items)?;

    Ok(())
}

/// Test the `query_unordered` method of the tree.
///
/// The method should return all the points that are contained in the query.
/// This is tested by comparing with a brute-force implementation.
fn tree_query_unordered_impl(items: Vec<(TPoint, u64)>, query: TQuery) -> TestResult<()> {
    let mut store = MemStore::new();
    let tree = willow_store::Node::from_iter(items.clone(), &mut store)?;
    let mut actual = tree
        .query(&query, &store)
        .map(|x| x.unwrap())
        .collect::<Vec<_>>();
    let mut expected = items
        .iter()
        .filter(|(p, _)| query.contains(p))
        .cloned()
        .collect::<Vec<_>>();

    actual.sort_by_key(|(p, _)| p.clone());
    expected.sort_by_key(|(p, _)| p.clone());
    println!("{} {} {}", items.len(), actual.len(), expected.len());
    assert_eq!(actual, expected);
    Ok(())
}

/// Test the `query_ordered` method of the tree.
///
/// The method should return all the points that are contained in the query,
/// ordered according to the given ordering.
///
/// This is tested by comparing with a brute-force implementation.
fn tree_query_ordered_impl(
    items: Vec<(TPoint, u64)>,
    query: TQuery,
    ordering: SortOrder,
) -> TestResult<()> {
    let mut store = MemStore::new();
    let tree = willow_store::Node::from_iter(items.clone(), &mut store)?;
    let actual = tree
        .query_ordered(&query, ordering, &store)
        .map(|x| x.unwrap())
        .collect::<Vec<_>>();
    let mut expected = items
        .iter()
        .filter(|(p, _)| query.contains(p))
        .cloned()
        .collect::<Vec<_>>();

    expected.sort_by(|(ak, _), (bk, _)| ak.cmp_with_order(bk, ordering));
    println!("{} {} {}", items.len(), actual.len(), expected.len());
    assert_eq!(actual, expected);
    Ok(())
}

/// Test the `summary` method of the tree.
///
/// The method should return the summary of the values of all the points in the query.
/// This is tested by comparing with a brute-force implementation.
fn tree_summary_impl(items: Vec<(TPoint, u64)>, query: TQuery) -> TestResult<()> {
    let mut store = MemStore::new();
    let tree = willow_store::Node::from_iter(items.clone(), &mut store)?;
    let actual = tree.summary(&query, &store)?;
    let mut expected = ValueSum::neutral();
    for (key, value) in &items {
        if query.contains(key) {
            expected = expected.combine(&ValueSum::lift(key, value));
        }
    }
    assert_eq!(actual, expected);
    Ok(())
}

fn tree_get_impl(items: Vec<(TPoint, u64)>) -> TestResult<()> {
    let mut store = MemStore::new();
    let tree = willow_store::Node::from_iter(items.clone(), &mut store)?;
    let key_set = items
        .iter()
        .map(|(k, _)| k.clone())
        .collect::<BTreeSet<_>>();
    // Compute a set of interesting points that are not in the tree.
    let mut non_keys = BTreeSet::new();
    let mut insert = |p: &TPoint| {
        let p = p.clone();
        if !key_set.contains(&p) {
            non_keys.insert(p);
        }
    };
    for (k, _) in &items {
        let t = OwnedPoint::new(&(*k.x() + 1), k.y(), k.z());
        insert(&t);
        let t = OwnedPoint::new(k.x(), &(*k.y() + 1), k.z());
        insert(&t);
        let t = OwnedPoint::new(k.x(), k.y(), &(*k.z() + 1));
        insert(&t);
    }
    for (key, value) in items {
        let actual = tree.get(&key, &store)?;
        assert_eq!(actual, Some(value));
    }
    for key in non_keys {
        let actual = tree.get(&key, &store)?;
        assert_eq!(actual, None);
    }
    Ok(())
}

fn tree_update_impl(items: Vec<(TPoint, u64)>) -> TestResult<()> {
    let mut store = MemStore::new();
    let mut tree = willow_store::Node::from_iter(items.clone(), &mut store)?;
    for (k, v) in &items {
        let new_v = v + 1;
        let old = tree.insert(&k, &new_v, &mut store)?;
        assert_eq!(old, Some(*v));
    }
    tree.assert_invariants(&store, true)?;
    Ok(())
}

fn tree_insert_impl(items: Vec<(TPoint, u64)>) -> TestResult<()> {
    let mut store = MemStore::new();
    let mut items2 = items.clone();
    let (key, value) = items2.pop().unwrap();
    let mut node = willow_store::Node::from_iter(items2.clone(), &mut store)?;
    let x: willow_store::Node<TestParams> = store
        .create_node(&OwnedNodeData::leaf(&key, &value))?
        .into();
    println!("before:");
    node.dump(&store)?;
    println!("");
    println!("insert");
    x.dump(&store)?;
    println!("");
    node.insert(&key, &value, &mut store)?;
    println!("after:");
    node.dump(&store)?;
    println!("");
    node.assert_invariants(&store, true)?;
    for (k, v) in &items {
        let actual = node.get(&k, &store)?;
        assert_eq!(actual, Some(*v));
    }
    Ok(())
}

fn tree_delete_impl(items: Vec<(TPoint, u64)>) -> TestResult<()> {
    if items.is_empty() {
        return Ok(());
    }
    let mut items2 = items.clone();
    let (key, _) = items2.pop().unwrap();
    let mut store = MemStore::new();
    let mut node = willow_store::Node::from_iter(items.clone(), &mut store)?;
    assert!(node.get(&key, &store)?.is_some());
    println!("Deleting {:?}", key);
    node.delete(&key, &mut store)?;
    node.assert_invariants(&store, true)?;
    for (key, expected) in items2 {
        let actual = node.get(&key, &store)?;
        assert_eq!(actual, Some(expected));
    }
    Ok(())
}

#[proptest]
fn prop_tree_query_unordered(
    #[strategy(treecontents())] items: Vec<(TPoint, u64)>,
    #[strategy(query())] query: TQuery,
) {
    tree_query_unordered_impl(items, query).unwrap();
}

#[proptest]
fn prop_tree_query_ordered(
    #[strategy(treecontents())] items: Vec<(TPoint, u64)>,
    #[strategy(query())] query: TQuery,
    #[strategy(sortorder())] ordering: SortOrder,
) {
    tree_query_ordered_impl(items, query, ordering).unwrap();
}

#[test]
fn test_tree_summary() -> TestResult<()> {
    let cases = vec![
        (
            vec![(tpoint(0, 0, 0), 0), (tpoint(0, 0, 1), 1)],
            TQuery::new(
                QueryRange::new(0, Some(1)),
                QueryRange::new(0, Some(1)),
                QueryRange::new(0, Some(1)),
            ),
        ),
        (
            vec![
                (tpoint(3, 43, 0), 0),
                (tpoint(54, 20, 0), 0),
                (tpoint(45, 20, 13), 0),
                (tpoint(40, 8, 6), 0),
                (tpoint(92, 86, 54), 0),
                (tpoint(71, 33, 42), 1),
            ],
            TQuery::new(
                QueryRange::new(7, None),
                QueryRange::new(66, None),
                QueryRange::new(3, Some(55)),
            ),
        ),
    ];
    for (items, query) in cases {
        tree_summary_impl(items, query)?;
    }

    Ok(())
}

#[proptest]
fn prop_tree_summary(
    #[strategy(treecontents())] items: Vec<(TPoint, u64)>,
    #[strategy(query())] query: TQuery,
) {
    tree_summary_impl(items, query).unwrap();
}

#[proptest]
fn prop_tree_get(#[strategy(treecontents())] items: Vec<(TPoint, u64)>) {
    tree_get_impl(items).unwrap();
}

#[proptest]
fn prop_tree_replace(#[strategy(treecontents())] items: Vec<(TPoint, u64)>) {
    tree_update_impl(items).unwrap();
}

#[proptest]
fn prop_tree_insert(
    #[strategy(treecontents_with_opts(xpoint(), 0..100u64, 1..100usize))] items: Vec<(TPoint, u64)>,
) {
    tree_insert_impl(items).unwrap();
}

#[proptest]
fn prop_tree_delete(
    #[strategy(treecontents_with_opts(xpoint(), 0..100u64, 1..100usize))] items: Vec<(TPoint, u64)>,
) {
    tree_delete_impl(items).unwrap();
}

fn parse_case(case: Vec<((u64, u64, u64), u64)>) -> Vec<(TPoint, u64)> {
    case.into_iter()
        .map(|((x, y, z), v)| (tpoint(x, y, z), v))
        .collect()
}

#[test]
fn test_tree_insert() -> TestResult<()> {
    tracing_subscriber::fmt::try_init().ok();
    let cases = vec![
        // vec![((0, 0, 1), 0), ((0, 0, 0), 0)],
        // vec![((0, 0, 0), 0), ((0, 0, 1), 0)],
        // vec![((1, 0, 6), 0), ((0, 3, 0), 0)],
        // vec![((18, 9, 28), 0), ((0, 0, 0), 0)],
        // vec![((0, 0, 0), 1), ((1, 0, 0), 0)],
        // vec![((2, 0, 0), 0), ((0, 0, 0), 0), ((1, 0, 0), 0)],
        // vec![((18, 9, 28), 0), ((0, 0, 0), 0), ((0, 2, 0), 0)],
        // vec![((0, 0, 0), 0), ((1, 0, 0), 0), ((2, 0, 0), 0)],
        // vec![((0, 0, 0), 0)],
        vec![((32, 0, 0), 0), ((54, 0, 0), 1), ((0, 0, 0), 0)],
        // vec![((0, 0, 0), 0), ((54, 0, 0), 1)],
    ];
    for items in cases {
        let items = parse_case(items);
        tree_insert_impl(items)?;
    }

    Ok(())
}

#[test]
#[ignore]
fn test_tree_ins2() -> TestResult<()> {
    tracing_subscriber::fmt::try_init().ok();
    let cases = vec![vec![((0, 0, 0), 0), ((19, 0, 0), 0), ((21, 0, 0), 0)]];
    for items in cases {
        let items = parse_case(items);
        tree_insert_impl(items)?;
    }

    Ok(())
}

#[test]
fn test_tree_replace() -> TestResult<()> {
    let items = vec![(tpoint(31, 28, 33), 0), (tpoint(43, 10, 34), 0)];
    tree_update_impl(items)?;
    Ok(())
}

#[proptest]
fn prop_nodedata_create(#[strategy(point())] p: TPoint, v: u64) {
    let node = OwnedNodeData::leaf(&p, &v);
    let summary = ValueSum::lift(&p, &v);
    assert_eq!(node.key(), p.deref());
    assert_eq!(node.value(), &v);
    assert_eq!(node.summary(), &summary);
    assert!(node.is_leaf());
}

fn node_bytes_roundtrip_impl(p: TPoint, v: u64) -> TestResult<()> {
    let node = OwnedNodeData::leaf(&p, &v);
    let bytes = node.as_slice().to_vec();
    let node2 = OwnedNodeData::<TestParams>::new(bytes);
    assert_eq!(node, node2);
    Ok(())
}

#[proptest]
fn prop_node_bytes_roundtrip(#[strategy(point())] p: TPoint, v: u64) {
    node_bytes_roundtrip_impl(p, v).unwrap();
}

#[test]
fn test_node_bytes_roundtrip() -> TestResult<()> {
    let p = tpoint(0, 0, 0);
    let v = 0;
    node_bytes_roundtrip_impl(p, v)?;
    Ok(())
}

fn point_with_rank(rank: u8) -> impl Strategy<Value = TPoint> {
    point().prop_filter("rank", move |x| x.rank() == rank)
}

fn points_with_rank(rank: u8) -> impl Strategy<Value = Vec<(TPoint, u64)>> {
    proptest::collection::vec((point_with_rank(rank), 0..100u64), 1..10)
}

#[proptest]
fn prop_same_rank(#[strategy(points_with_rank(0))] items: Vec<(TPoint, u64)>) {
    let mut store = MemStore::new();
    let tree = willow_store::Node::from_iter(items.clone(), &mut store).unwrap();
    tree.dump(&store).unwrap();
}

#[test]
fn test_tree_delete() -> TestResult<()> {
    let cases = vec![
        // vec![((4, 0, 0), 0), ((8, 0, 0), 0)]
        vec![((67, 0, 0), 0), ((54, 0, 0), 0), ((0, 0, 0), 0)],
    ];
    for items in cases {
        let items = parse_case(items);
        tree_delete_impl(items)?;
    }
    Ok(())
}
