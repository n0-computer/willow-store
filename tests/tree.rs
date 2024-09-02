use proptest::prelude::*;
use test_strategy::proptest;
use testresult::TestResult;
use willow_store::{KeyParams, LiftingCommutativeMonoid, Point, Store, TreeParams};

type TPoint = Point<TestParams>;

#[derive(Debug)]
struct TestParams;

impl KeyParams for TestParams {
    type X = u64;
    type Y = u64;
    type Z = u64;
}

impl TreeParams for TestParams {
    type V = u64;
    type S = ValueSum;
}

#[derive(Clone)]
struct ValueSum(u64);

impl LiftingCommutativeMonoid<u64> for ValueSum {
    fn zero() -> Self {
        ValueSum(0)
    }

    fn lift(v: u64) -> Self {
        ValueSum(v)
    }

    fn combine(&self, other: &Self) -> Self {
        ValueSum(self.0 + other.0)
    }
}

fn point() -> impl Strategy<Value = TPoint> {
    (0..100u64, 0..100u64, 0..100u64).prop_map(|(x, y, z)| TPoint::new(x, y, z))
}

fn treecontents() -> impl Strategy<Value = Vec<(TPoint, u64)>> {
    prop::collection::vec((point(), 0..100u64), 1..100)
}

fn tree_creation_impl(items: Vec<(TPoint, u64)>) -> TestResult<()> {
    let (store, id) = willow_store::Node::from_iter(items)?;
    let tree = store.get(&id)?;
    tree.dump(&store)?;
    tree.assert_invariants(&store)?;

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
