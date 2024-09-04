use std::collections::BTreeSet;

use prop::strategy;
use proptest::prelude::*;
use test_strategy::proptest;
use testresult::TestResult;
use willow_store::{
    KeyParams, LiftingCommutativeMonoid, Node, Point, QueryRange, QueryRange3d, SortOrder, Store,
    TreeParams,
};

type TPoint = Point<TestParams>;
type TQuery = QueryRange3d<TestParams>;

#[derive(Debug)]
struct TestParams;

impl KeyParams for TestParams {
    type X = u64;
    type Y = u64;
    type Z = u64;
}

impl TreeParams for TestParams {
    type V = u64;
    type M = ValueSum;
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct ValueSum(u64);

impl LiftingCommutativeMonoid<(Point<TestParams>, u64)> for ValueSum {
    fn zero() -> Self {
        ValueSum(0)
    }

    fn lift(v: (Point<TestParams>, u64)) -> Self {
        ValueSum(v.1)
    }

    fn combine(&self, other: &Self) -> Self {
        ValueSum(self.0 + other.0)
    }
}

fn point() -> impl Strategy<Value = TPoint> {
    (0..100u64, 0..100u64, 0..100u64).prop_map(|(x, y, z)| TPoint::new(x, y, z))
}

fn treecontents() -> impl Strategy<Value = Vec<(TPoint, u64)>> {
    prop::collection::vec((point(), 0..100u64), 1..1000).prop_map(|mut m| {
        let mut keys = BTreeSet::new();
        m.retain(|(p, _)| keys.insert(p.clone().xyz()));
        m
    })
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

/// Tests that creating a tree from unique points works.
///
/// The tree should conform to the invariants of the tree structure,
/// and contain all the points that were inserted.
fn tree_creation_impl(items: Vec<(TPoint, u64)>) -> TestResult<()> {
    let (store, id) = willow_store::NodeData::from_iter(items.clone())?;
    let tree = store.get(&id)?;
    // tree.dump(&store)?;
    tree.assert_invariants(&store)?;
    let mut actual = tree
        .iter_unordered(&store)
        .into_iter()
        .map(|x| x.unwrap())
        .collect::<Vec<_>>();
    let mut expected = items;
    actual.sort_by_key(|(p, _)| p.clone().xyz());
    expected.sort_by_key(|(p, _)| p.clone().xyz());
    assert_eq!(actual, expected);
    Ok(())
}

#[proptest]
fn prop_tree_creation(#[strategy(treecontents())] items: Vec<(TPoint, u64)>) {
    tree_creation_impl(items).unwrap();
}

#[test]
fn test_tree_creation() -> TestResult<()> {
    let items = vec![(TPoint::new(83, 0, 0), 0), (TPoint::new(0, 0, 1), 2)];
    tree_creation_impl(items)?;

    Ok(())
}

/// Test the `query_unordered` method of the tree.
///
/// The method should return all the points that are contained in the query.
/// This is tested by comparing with a brute-force implementation.
fn tree_query_unordered_impl(items: Vec<(TPoint, u64)>, query: TQuery) -> TestResult<()> {
    let (store, id) = willow_store::NodeData::from_iter(items.clone())?;
    let tree = store.get(&id)?;
    let mut actual = tree
        .query_unordered(&query, &store)
        .into_iter()
        .map(|x| x.unwrap())
        .collect::<Vec<_>>();
    let mut expected = items
        .iter()
        .filter(|(p, _)| query.contains(p))
        .cloned()
        .collect::<Vec<_>>();

    actual.sort_by_key(|(p, _)| p.clone().xyz());
    expected.sort_by_key(|(p, _)| p.clone().xyz());
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
    let (store, id) = willow_store::NodeData::from_iter(items.clone())?;
    let tree = store.get(&id)?;
    let actual = tree
        .query_ordered(&query, ordering, &store)
        .into_iter()
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
    let (store, id) = willow_store::NodeData::from_iter(items.clone())?;
    let tree = store.get(&id)?;
    let actual = tree.summary(&query, &store)?;
    let mut expected = ValueSum::zero();
    for (key, value) in &items {
        if query.contains(key) {
            expected = expected.combine(&ValueSum::lift((key.clone(), *value)));
        }
    }
    assert_eq!(actual, expected);
    Ok(())
}

fn tree_get_impl(items: Vec<(TPoint, u64)>) -> TestResult<()> {
    let (store, id) = willow_store::NodeData::from_iter(items.clone())?;
    let key_set = items
        .iter()
        .map(|(k, _)| k.clone().xyz())
        .collect::<BTreeSet<_>>();
    let tree = store.get(&id)?;
    // Compute a set of interesting points that are not in the tree.
    let mut non_keys = BTreeSet::new();
    let mut insert = |p: &TPoint| {
        let p = p.clone().xyz();
        if !key_set.contains(&p) {
            non_keys.insert(p);
        }
    };
    for (k, _) in &items {
        let mut t = k.clone();
        t.x += 1;
        insert(&t);
        let mut t = k.clone();
        t.y += 1;
        insert(&t);
        let mut t = k.clone();
        t.z += 1;
        insert(&t);
    }
    for (key, value) in items {
        let actual = tree.get(key, &store)?;
        assert_eq!(actual, Some(value));
    }
    for key in non_keys {
        let actual = tree.get(key.into(), &store)?;
        assert_eq!(actual, None);
    }
    Ok(())
}

fn tree_insert_replace_impl(items: Vec<(TPoint, u64)>) -> TestResult<()> {
    let (mut store, id) = willow_store::NodeData::from_iter(items.clone())?;
    let mut tree = store.get_node_opt(&Some(id))?.unwrap();
    for (k, v) in &items {
        let new_v = v + 1;
        tree.replace(k.clone(), new_v, &mut store)?;
    }
    tree.assert_invariants(&store)?;
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
            vec![(TPoint::new(0, 0, 0), 0), (TPoint::new(0, 0, 1), 1)],
            TQuery::new(
                QueryRange::new(0, Some(1)),
                QueryRange::new(0, Some(1)),
                QueryRange::new(0, Some(1)),
            ),
        ),
        (
            vec![
                (TPoint::new(3, 43, 0), 0),
                (TPoint::new(54, 20, 0), 0),
                (TPoint::new(45, 20, 13), 0),
                (TPoint::new(40, 8, 6), 0),
                (TPoint::new(92, 86, 54), 0),
                (TPoint::new(71, 33, 42), 1),
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
fn prop_tree_insert_replace(#[strategy(treecontents())] items: Vec<(TPoint, u64)>) {
    tree_insert_replace_impl(items).unwrap();
}

#[test]
fn test_tree_insert_replace() -> TestResult<()> {
    let items = vec![(TPoint::new(31, 28, 33), 0), (TPoint::new(43, 10, 34), 0)];
    tree_insert_replace_impl(items)?;

    Ok(())
}
