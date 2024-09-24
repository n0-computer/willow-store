use std::{collections::BTreeSet, fmt::Debug, marker::PhantomData, ops::Deref};

use prop::sample::SizeRange;
use proptest::prelude::*;
use test_strategy::proptest;
use testresult::TestResult;
use willow_store::{
    FixedSize, KeyParams, LiftingCommutativeMonoid, MemStore, NodeStore, OwnedNodeData, Point,
    PointRef, QueryRange, QueryRange3d, SortOrder, TreeParams,
};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

type TPoint = Point<TestParams>;
type TQuery = QueryRange3d<TestParams>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TestParams;

impl KeyParams for TestParams {
    type X = u64;
    type Y = u64;
    type Z = u64;
    type ZOwned = u64;
}

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

impl LiftingCommutativeMonoid<PointRef<TestParams>, u64> for ValueSum {
    fn neutral() -> Self {
        ValueSum(0)
    }

    fn lift(_k: &PointRef<TestParams>, v: &u64) -> Self {
        ValueSum(*v)
    }

    fn combine(&self, other: &Self) -> Self {
        ValueSum(self.0 + other.0)
    }
}

fn point() -> impl Strategy<Value = TPoint> {
    (0..100u64, 0..100u64, 0..100u64).prop_map(|(x, y, z)| TPoint::new(&x, &y, &z))
}

fn flatpoint() -> impl Strategy<Value = TPoint> {
    (0..10u64, 0..10u64, 0..1u64).prop_map(|(x, y, z)| TPoint::new(&x, &y, &z))
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

fn flattreecontents() -> impl Strategy<Value = Vec<(TPoint, u64)>> {
    treecontents_with_opts(flatpoint(), 0..100u64, 1..100usize)
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

fn flatquery() -> impl Strategy<Value = TQuery> {
    (0..10u64, 1..=10u64, 0..10u64, 1..=10u64, 0..1u64, 1..2u64).prop_map(
        |(x0, dx, y0, dy, z0, dz)| {
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
        },
    )
}

fn tpoint(x: u64, y: u64, z: u64) -> TPoint {
    TPoint::new(&x, &y, &z)
}

struct TreeTests<P: TreeParams>(PhantomData<P>);

impl<P: TreeParams> TreeTests<P> {
    /// Tests that creating a tree from unique points works.
    ///
    /// The tree should conform to the invariants of the tree structure,
    /// and contain all the points that were inserted.
    fn creation(items: Vec<(Point<P>, P::V)>) -> TestResult<()> {
        let mut store = MemStore::new();
        let node = willow_store::Node::from_iter(items.clone(), &mut store)?;
        // tree.dump(&store)?;
        node.assert_invariants(&store, true)?;
        let mut actual = node.iter(&store).map(|x| x.unwrap()).collect::<Vec<_>>();
        let mut expected = items.clone();
        actual.sort_by_key(|(p, _)| p.clone());
        expected.sort_by_key(|(p, _)| p.clone());
        assert_eq!(store.size(), items.len());
        assert_eq!(store.max_id(), items.len() as u64);
        assert_eq!(actual, expected);
        Ok(())
    }

    /// Test the `query_unordered` method of the tree.
    ///
    /// The method should return all the points that are contained in the query.
    /// This is tested by comparing with a brute-force implementation.
    fn query_unordered(items: Vec<(TPoint, u64)>, query: TQuery) -> TestResult<()> {
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
    fn query_ordered(
        items: Vec<(Point<P>, P::V)>,
        query: QueryRange3d<P>,
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
    fn summary(items: Vec<(Point<P>, P::V)>, query: QueryRange3d<P>) -> TestResult<()> {
        let mut store = MemStore::new();
        let tree = willow_store::Node::from_iter(items.clone(), &mut store)?;
        let actual = tree.range_summary(&query, &store)?;
        let mut expected = P::M::neutral();
        for (key, value) in &items {
            if query.contains(key) {
                expected = expected.combine(&P::M::lift(key, value));
            }
        }
        assert_eq!(actual, expected);
        Ok(())
    }

    /// Test the `count_range` method of the tree.
    ///
    /// The method should return the number of elements in the query.
    /// This is tested by comparing with a brute-force implementation.
    fn count_range(items: Vec<(Point<P>, P::V)>, query: QueryRange3d<P>) -> TestResult<()> {
        let mut store = MemStore::new();
        let tree = willow_store::Node::from_iter(items.clone(), &mut store)?;
        let actual = tree.range_count(&query, &store)?;
        let mut expected = 0;
        for (key, _value) in &items {
            if query.contains(key) {
                expected += 1;
            }
        }
        assert_eq!(actual, expected);
        Ok(())
    }

    fn get(items: Vec<(Point<P>, P::V)>) -> TestResult<()>
    where
        P: TreeParams<X = u64, Y = u64, Z = u64>,
    {
        let mut store = MemStore::new();
        let tree = willow_store::Node::from_iter(items.clone(), &mut store)?;
        let key_set = items
            .iter()
            .map(|(k, _)| k.clone())
            .collect::<BTreeSet<_>>();
        // Compute a set of interesting points that are not in the tree.
        let mut non_keys = BTreeSet::new();
        let mut insert = |p: &Point<P>| {
            let p = p.clone();
            if !key_set.contains(&p) {
                non_keys.insert(p);
            }
        };
        for (k, _) in &items {
            let t = Point::new(&(*k.x() + 1), k.y(), k.z());
            insert(&t);
            let t = Point::new(k.x(), &(*k.y() + 1), k.z());
            insert(&t);
            let t = Point::new(k.x(), k.y(), &(*k.z() + 1));
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

    fn update(items: Vec<(Point<P>, P::V)>) -> TestResult<()>
    where
        P: TreeParams<V = u64>,
    {
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

    fn insert(items: Vec<(Point<P>, P::V)>) -> TestResult<()> {
        let mut store = MemStore::new();
        let mut items2 = items.clone();
        let (key, value) = items2.pop().unwrap();
        let mut node = willow_store::Node::from_iter(items2.clone(), &mut store)?;
        let x: willow_store::Node<P> = store
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
            assert_eq!(actual, Some(v.clone()));
        }
        Ok(())
    }

    fn delete(items: Vec<(Point<P>, P::V)>) -> TestResult<()> {
        if items.is_empty() {
            return Ok(());
        }
        let mut items2 = items.clone();
        let (key, _) = items2.pop().unwrap();
        let mut store = MemStore::new();
        let mut node = willow_store::Node::from_iter(items.clone(), &mut store)?;
        assert!(node.get(&key, &store)?.is_some());
        let n = store.size();
        assert!(n == items.len());
        println!("Deleting {:?}", key);
        node.delete(&key, &mut store)?;
        node.assert_invariants(&store, true)?;
        let n2 = store.size();
        assert!(n2 == items.len() - 1);
        for (key, expected) in items2 {
            let actual = node.get(&key, &store)?;
            assert_eq!(actual, Some(expected));
        }
        Ok(())
    }

    fn node_bytes_roundtrip(p: Point<P>, v: P::V) -> TestResult<()> {
        let node = OwnedNodeData::leaf(&p, &v);
        let bytes = node.as_slice().to_vec();
        let node2 = OwnedNodeData::<P>::new(bytes);
        assert_eq!(node, node2);
        Ok(())
    }
}

#[proptest]
fn prop_tree_creation(#[strategy(treecontents())] items: Vec<(TPoint, u64)>) {
    TreeTests::<TestParams>::creation(items).unwrap();
}

#[test]
fn test_tree_creation() -> TestResult<()> {
    let items = vec![(tpoint(83, 0, 0), 0), (tpoint(0, 0, 1), 2)];
    TreeTests::<TestParams>::creation(items)?;
    Ok(())
}

fn tree_find_split_plane_impl(items: Vec<(TPoint, u64)>, query: TQuery) -> TestResult<()> {
    let mut store = MemStore::new();
    let tree = willow_store::Node::from_iter(items.clone(), &mut store)?;
    let count = tree.range_count(&query, &store)?;
    let Some((left, left_count, right, right_count)) =
        tree.find_split_plane(&query, count, &store)?
    else {
        assert!(count <= 1);
        return Ok(());
    };
    assert_eq!(left_count + right_count, count);
    assert_eq!(left_count, tree.range_count(&left, &store)?);
    assert_eq!(right_count, tree.range_count(&right, &store)?);
    let Some((p, order)) = tree.find_split_plane_2(&query, &store)? else {
        assert!(count <= 1);
        return Ok(());
    };
    Ok(())
}

fn tree_split_impl(items: Vec<(TPoint, u64)>, query: TQuery) -> TestResult<()> {
    // println!("Query: {}", query.pretty());
    let mut store = MemStore::new();
    let tree = willow_store::Node::from_iter(items.clone(), &mut store)?;
    let total_count = tree.range_count(&query, &store)?;
    // println!("total count: {}", total_count);
    let ranges = tree
        .split_range(query.clone(), 3, &store)
        .collect::<Vec<_>>();
    let ranges = ranges.into_iter().map(|x| x.unwrap()).collect::<Vec<_>>();
    // tree.dump(&store)?;
    if total_count == 0 {
        assert_eq!(ranges.len(), 0);
        return Ok(());
    }
    if total_count > 1 {
        assert!(ranges.len() > 1);
    }
    println!("Ranges:");
    for (range, count) in ranges.iter() {
        println!("{} {}", range.pretty(), count);
        assert!(*count > 0);
    }
    let total_count_from_ranges = ranges.iter().map(|(_, count)| count).sum::<u64>();
    assert_eq!(total_count_from_ranges, total_count);
    for (range, count) in ranges {
        let expected = items
            .iter()
            .filter(|(p, _)| range.contains(p) && query.contains(p))
            .count();
        // println!("range={} query={} expected={}", range.pretty(), query.pretty(), expected);
        assert_eq!(count, expected as u64);
    }
    Ok(())
}

#[proptest]
fn prop_tree_query_unordered(
    #[strategy(treecontents())] items: Vec<(TPoint, u64)>,
    #[strategy(query())] query: TQuery,
) {
    TreeTests::<TestParams>::query_unordered(items, query).unwrap();
}

#[proptest]
fn prop_tree_query_ordered(
    #[strategy(treecontents())] items: Vec<(TPoint, u64)>,
    #[strategy(query())] query: TQuery,
    #[strategy(sortorder())] ordering: SortOrder,
) {
    TreeTests::<TestParams>::query_ordered(items, query, ordering).unwrap();
}

#[proptest]
fn prop_tree_split_range(
    #[strategy(flattreecontents())] items: Vec<(TPoint, u64)>,
    #[strategy(flatquery())] query: TQuery,
) {
    tree_split_impl(items, query).unwrap();
}

#[proptest]
fn prop_tree_find_split_plane(
    #[strategy(flattreecontents())] items: Vec<(TPoint, u64)>,
    #[strategy(flatquery())] query: TQuery,
) {
    tree_find_split_plane_impl(items, query).unwrap();
}

#[test]
fn test_tree_split_range() -> TestResult<()> {
    /*
    minimal failing input: input = _PropTreeSplitRangeArgs {
        items: [
            (
                (
                    3,
                    0,
                    0,
                ),
                0,
            ),
            (
                (
                    0,
                    6,
                    0,
                ),
                0,
            ),
            (
                (
                    0,
                    7,
                    0,
                ),
                0,
            ),
            (
                (
                    5,
                    0,
                    0,
                ),
                0,
            ),
        ],
        query: QueryRange3d {
            x: QueryRange {
                min: 0,
                max: Some(
                    1,
                ),
            },
            y: QueryRange {
                min: 5,
                max: Some(
                    8,
                ),
            },
            z: QueryRange {
                min: 0,
                max: Some(
                    1,
                ),
            },
        },
    }

         */
    let cases = vec![
        (
            vec![
                (tpoint(7, 6, 0), 0),
                (tpoint(2, 2, 0), 0),
                (tpoint(2, 1, 0), 0),
            ],
            TQuery::new(
                QueryRange::new(0, Some(8)),
                QueryRange::new(0, Some(7)),
                QueryRange::new(0, Some(1)),
            ),
        ),
        (
            vec![
                (tpoint(4, 4, 0), 0),
                (tpoint(7, 1, 0), 0),
                (tpoint(4, 3, 0), 0),
            ],
            TQuery::new(
                QueryRange::new(0, Some(5)),
                QueryRange::new(4, Some(5)),
                QueryRange::new(0, Some(1)),
            ),
        ),
        (
            vec![
                (tpoint(2, 2, 0), 0),
                (tpoint(6, 0, 0), 0),
                (tpoint(5, 4, 0), 0),
            ],
            TQuery::new(
                QueryRange::new(0, Some(6)),
                QueryRange::new(0, Some(5)),
                QueryRange::new(0, Some(1)),
            ),
        ),
        (
            vec![(tpoint(8, 4, 0), 0), (tpoint(7, 4, 0), 0)],
            TQuery::new(
                QueryRange::new(5, Some(9)),
                QueryRange::new(0, Some(5)),
                QueryRange::new(0, Some(1)),
            ),
        ),
        (
            vec![(tpoint(4, 3, 0), 0), (tpoint(5, 1, 0), 0)],
            TQuery::new(
                QueryRange::new(0, Some(5)),
                QueryRange::new(4, Some(5)),
                QueryRange::new(0, Some(1)),
            ),
        ),
        (
            vec![(tpoint(0, 0, 0), 0), (tpoint(9, 1, 0), 0)],
            TQuery::new(
                QueryRange::new(7, Some(10)),
                QueryRange::new(0, Some(2)),
                QueryRange::new(0, Some(1)),
            ),
        ),
        (
            vec![
                (tpoint(3, 0, 0), 0),
                (tpoint(0, 6, 0), 0),
                (tpoint(0, 7, 0), 0),
                (tpoint(5, 0, 0), 0),
            ],
            TQuery::new(
                QueryRange::new(0, Some(1)),
                QueryRange::new(5, Some(8)),
                QueryRange::new(0, Some(1)),
            ),
        ),
    ];
    for (items, query) in cases {
        tree_split_impl(items, query)?;
    }

    Ok(())
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
        TreeTests::<TestParams>::summary(items, query)?;
    }

    Ok(())
}

#[proptest]
fn prop_tree_summary(
    #[strategy(treecontents())] items: Vec<(TPoint, u64)>,
    #[strategy(query())] query: TQuery,
) {
    TreeTests::<TestParams>::summary(items, query).unwrap();
}

#[proptest]
fn prop_tree_count_range(
    #[strategy(treecontents())] items: Vec<(TPoint, u64)>,
    #[strategy(query())] query: TQuery,
) {
    TreeTests::<TestParams>::count_range(items, query).unwrap();
}

#[proptest]
fn prop_tree_get(#[strategy(treecontents())] items: Vec<(TPoint, u64)>) {
    TreeTests::<TestParams>::get(items).unwrap();
}

#[proptest]
fn prop_tree_replace(#[strategy(treecontents())] items: Vec<(TPoint, u64)>) {
    TreeTests::<TestParams>::update(items).unwrap();
}

#[proptest]
fn prop_tree_insert(
    #[strategy(treecontents_with_opts(xpoint(), 0..100u64, 1..100usize))] items: Vec<(TPoint, u64)>,
) {
    TreeTests::<TestParams>::insert(items).unwrap();
}

#[proptest]
fn prop_tree_delete(
    #[strategy(treecontents_with_opts(xpoint(), 0..100u64, 1..100usize))] items: Vec<(TPoint, u64)>,
) {
    TreeTests::<TestParams>::delete(items).unwrap();
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
        vec![((0, 0, 1), 0), ((0, 0, 0), 0)],
        vec![((0, 0, 0), 0), ((0, 0, 1), 0)],
        vec![((1, 0, 6), 0), ((0, 3, 0), 0)],
        vec![((18, 9, 28), 0), ((0, 0, 0), 0)],
        vec![((0, 0, 0), 1), ((1, 0, 0), 0)],
        vec![((2, 0, 0), 0), ((0, 0, 0), 0), ((1, 0, 0), 0)],
        vec![((18, 9, 28), 0), ((0, 0, 0), 0), ((0, 2, 0), 0)],
        vec![((0, 0, 0), 0), ((1, 0, 0), 0), ((2, 0, 0), 0)],
        vec![((0, 0, 0), 0)],
        vec![((32, 0, 0), 0), ((54, 0, 0), 1), ((0, 0, 0), 0)],
        vec![((0, 0, 0), 0), ((54, 0, 0), 1)],
    ];
    for items in cases {
        let items = parse_case(items);
        TreeTests::<TestParams>::insert(items)?;
    }

    Ok(())
}

#[test]
fn test_tree_replace() -> TestResult<()> {
    let items = vec![(tpoint(31, 28, 33), 0), (tpoint(43, 10, 34), 0)];
    TreeTests::<TestParams>::update(items)?;
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

#[proptest]
fn prop_node_bytes_roundtrip(#[strategy(point())] p: TPoint, v: u64) {
    TreeTests::<TestParams>::node_bytes_roundtrip(p, v).unwrap();
}

#[test]
fn test_node_bytes_roundtrip() -> TestResult<()> {
    let p = tpoint(0, 0, 0);
    let v = 0;
    TreeTests::<TestParams>::node_bytes_roundtrip(p, v)?;
    Ok(())
}

fn point_with_rank(rank: u8) -> impl Strategy<Value = TPoint> {
    point().prop_filter("rank", move |x| x.rank() == rank)
}

fn points_with_rank(rank: u8) -> impl Strategy<Value = Vec<(TPoint, u64)>> {
    proptest::collection::vec((point_with_rank(rank), 0..100u64), 1..10)
}

#[test]
fn test_tree_delete() -> TestResult<()> {
    let cases = vec![
        vec![((4, 0, 0), 0), ((8, 0, 0), 0)],
        vec![((67, 0, 0), 0), ((54, 0, 0), 0), ((0, 0, 0), 0)],
    ];
    for items in cases {
        let items = parse_case(items);
        TreeTests::<TestParams>::delete(items)?;
    }
    Ok(())
}

#[test]
fn align_test() {
    println!("{}", std::mem::align_of::<zerocopy::big_endian::U64>());
    println!("{}", std::mem::align_of::<zerocopy::native_endian::U64>());
    println!("{}", std::mem::align_of::<u64>());
}
