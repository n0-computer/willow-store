//! Non-persisten reference implementation of a zip tree, closely following the
//! description in the paper.
//!
//! To literally follow the paper, we implement semantics of an OO language
//! with freely shareable pointers that can possibly be null (empty).
use std::{cell::RefCell, cmp::Ordering, collections::BTreeSet, rc::Rc};

use proptest::prelude::{any, Strategy};
use rand::{seq::SliceRandom, SeedableRng};
use test_strategy::proptest;
use willow_store::count_trailing_zeros;

struct NodeData {
    key: (u64, u64, u64),
    rank: u8,
    left: Node,
    right: Node,
}

fn cmp_at_rank((xa, ya, za): (u64, u64, u64), (xb, yb, zb): (u64, u64, u64), rank: u8) -> Ordering {
    match rank % 3 {
        0 => xa.cmp(&xb).then(ya.cmp(&yb)).then(za.cmp(&zb)),
        1 => ya.cmp(&yb).then(za.cmp(&zb)).then(xa.cmp(&xb)),
        2 => za.cmp(&zb).then(xa.cmp(&xb)).then(ya.cmp(&yb)),
        _ => unreachable!(),
    }
}

#[derive(Clone)]
struct Node(Option<Rc<RefCell<NodeData>>>);

impl Node {
    const EMPTY: Self = Self(None);

    fn leaf(key: (u64, u64, u64), rank: u8) -> Self {
        Self(Some(Rc::new(RefCell::new(NodeData {
            key,
            rank,
            left: Self::EMPTY,
            right: Self::EMPTY,
        }))))
    }

    fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    fn rank(&self) -> u8 {
        self.0.as_ref().unwrap().borrow().rank
    }

    fn key(&self) -> (u64, u64, u64) {
        self.0.as_ref().unwrap().borrow().key
    }

    fn left(&self) -> Node {
        self.0.as_ref().unwrap().borrow().left.clone()
    }

    fn right(&self) -> Node {
        self.0.as_ref().unwrap().borrow().right.clone()
    }

    fn set_left(&self, left: Node) {
        self.0.as_ref().unwrap().borrow_mut().left = left;
    }

    fn set_right(&self, right: Node) {
        self.0.as_ref().unwrap().borrow_mut().right = right;
    }

    fn same(lhs: &Self, rhs: &Self) -> bool {
        Rc::ptr_eq(&lhs.0.as_ref().unwrap(), &rhs.0.as_ref().unwrap())
    }

    fn print(&self) {
        self.print0("".to_string());
    }

    fn print0(&self, indent: String) {
        if self.is_empty() {
            println!("{indent}EMPTY");
        } else {
            self.left().print0(indent.clone() + "  ");
            println!("{indent}key={:?} rank={}", self.key(), self.rank());
            self.right().print0(indent + "  ");
        }
    }

    fn check_invariants(&self) {
        if self.is_empty() {
            return;
        }
        let rank = self.rank();
        self.left().check_invariants();
        self.right().check_invariants();
        if !self.left().is_empty() {
            assert!(self.left().rank() < rank);
            assert!(self.left().key() < self.key());
        }
        if !self.right().is_empty() {
            assert!(self.right().rank() <= rank);
            assert!(self.right().key() > self.key());
        }
    }

    fn from_iter(iter: impl IntoIterator<Item = ((u64, u64, u64), u8)>) -> Self {
        let mut root = Self::EMPTY;
        for (key, rank) in iter {
            let x = Self::leaf(key, rank);
            root = insert_rec(x, root);
        }
        root
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        if self.is_empty() {
            other.is_empty()
        } else {
            Node::same(self, other)
        }
    }
}

// contains is trivial and not given in the paper.
fn contains_rec(root: Node, key: (u64, u64, u64)) -> bool {
    if root.is_empty() {
        false
    } else if key == root.key() {
        true
    } else if cmp_at_rank(key, root.key(), root.rank()) == Ordering::Less {
        contains_rec(root.left(), key)
    } else {
        contains_rec(root.right(), key)
    }
}

// insert(x, root):
// if root = null then {x.left ← x.right ← null; x.rank ← RandomRank; return x}
// if x.key < root.key then
//   if insert(x, root.left) = x then
//     if x.rank < root.rank then root.left ← x
//     else {root.left ← x.right; x.right ← root; return x}
// else
//   if insert(x, root.right) = x then
//     if x.rank ≤ root.rank then root.right ← x
//     else {root.right ← x.left; x.left ← root; return x}
// return root
fn insert_rec<'a>(x: Node, root: Node) -> Node {
    if root.is_empty() {
        return x;
    }
    if cmp_at_rank(x.key(), root.key(), root.rank()) == Ordering::Less {
        if insert_rec(x.clone(), root.left()) == x {
            if x.rank() < root.rank() {
                root.set_left(x);
            } else {
                root.set_left(x.right());
                x.set_right(root);
                return x;
            }
        }
    } else {
        if insert_rec(x.clone(), root.right()) == x {
            if x.rank() <= root.rank() {
                root.set_right(x);
            } else {
                root.set_right(x.left());
                x.set_left(root);
                return x;
            }
        }
    }
    return root;
}

// zip(x, y):
// if x = null then return y
// if y = null then return x
// if x.rank < y.rank then {y.left ← zip(x, y.left); return y}
// else {x.right ← zip(x.right , y); return x}
fn zip(x: Node, y: Node) -> Node {
    if x.is_empty() {
        return y;
    }
    if y.is_empty() {
        return x;
    }
    if x.rank() < y.rank() {
        let y = y;
        y.set_left(zip(x, y.left()));
        return y;
    } else {
        let x = x;
        x.set_right(zip(x.right(), y));
        return x;
    }
}

// delete(x, root):
// if x.key = root.key then return zip(root.left , root.right)
// if x.key < root.key then
//   if x.key = root.left.key then
//     root.left ← zip(root.left.left , root.left.right)
//   else delete(x, root.left)
// else
//   if x.key = root.right.key then
//     root.right ← zip(root.right.left , root.right.right)
//   else delete(x, root.right)
// return root
fn delete_rec(x: Node, root: Node) -> Node {
    if x.key() == root.key() {
        return zip(root.left(), root.right());
    }
    if cmp_at_rank(x.key(), root.key(), root.rank()) == Ordering::Less {
        if x.key() == root.left().key() {
            root.set_left(zip(root.left().left(), root.left().right()));
        } else {
            root.set_left(delete_rec(x, root.left()));
        }
    } else {
        if x.key() == root.right().key() {
            root.set_right(zip(root.right().left(), root.right().right()));
        } else {
            root.set_right(delete_rec(x, root.right()));
        }
    }
    return root;
}

// insert(x)
//   rank ← x.rank ← RandomRank
//   key ← x.key
//   cur ← root
//   while cur 6= null and (rank < cur .rank or (rank = cur .rank and key > cur .key)) do
//     prev ← cur
//     cur ← if key < cur .key then cur .left else cur .right
//   if cur = root then root ← x
//   else if key < prev.key then prev.left ← x
//   else prev.right ← x
//   if cur = null then {x.left ← x.right ← null; return}
//   if key < cur .key then x.right ← cur else x.left ← cur
//   prev ← x
//   while cur 6= null do
//     fix ← prev
//   if cur .key < key then
//     repeat {prev ← cur ; cur ← cur .right}
//     until cur = null or cur .key > key
//   else
//     repeat {prev ← cur ; cur ← cur .left}
//     until cur = null or cur .key < key
//   if fix .key > key or (fix = x and prev.key > key) then
//     fix .left ← cur
//   else
//     fix .right ← cur
fn insert(root: &mut Node, x: Node) {
    let rank = x.rank();
    let key = x.key();
    let mut cur = root.clone();
    let mut prev = Node::EMPTY;
    while !cur.is_empty() && (rank < cur.rank() || (rank == cur.rank() && key > cur.key())) {
        prev = cur.clone();
        cur = if key < cur.key() {
            cur.left()
        } else {
            cur.right()
        };
    }
    if &cur == root {
        *root = x.clone();
    } else if key < prev.key() {
        prev.set_left(x.clone());
    } else {
        prev.set_right(x.clone());
    }
    if cur.is_empty() {
        x.set_left(Node::EMPTY);
        x.set_right(Node::EMPTY);
        return;
    }
    if key < cur.key() {
        x.set_right(cur.clone());
    } else {
        x.set_left(cur.clone());
    }
    prev = x.clone();
    while !cur.is_empty() {
        let fix = prev.clone();
        if cur.key() < key {
            loop {
                prev = cur.clone();
                cur = cur.right();
                if cur.is_empty() || cur.key() > key {
                    break;
                }
            }
        } else {
            loop {
                prev = cur.clone();
                cur = cur.left();
                if cur.is_empty() || cur.key() < key {
                    break;
                }
            }
        }
        if fix.key() > key || (fix == x && prev.key() > key) {
            fix.set_left(cur.clone());
        } else {
            fix.set_right(cur.clone());
        }
    }
}

// delete(x)
//   key ← x.key
//   cur ← root
//   while key != cur.key do
//     prev ← cur
//     cur ← if key < cur.key then cur.left else cur.right
//   left ← cur.left; right ← cur.right
//   if left = null then cur ← right
//   else if right = null then cur ← left
//   else if left.rank ≥ right.rank then cur ← left
//   else cur ← right
//   if root = x then root ← cur
//   else if key < prev.key then prev.left ← cur
//   else prev.right ← cur
//   while left 6= null and right 6= null do
//     if left.rank ≥ right.rank then
//       repeat {prev ← left; left ← left.right}
//       until left = null or left.rank < right.rank
//       prev.right ← right
//     else
//       repeat {prev ← right; right ← right.left}
//       until right = null or left.rank ≥ right.rank
//       prev.left ← left
fn delete(root: &mut Node, x: Node) {
    let key = x.key();
    let mut cur = root.clone();
    let mut prev = Node::EMPTY;
    while key != cur.key() {
        prev = cur.clone();
        cur = if key < cur.key() {
            cur.left()
        } else {
            cur.right()
        };
    }
    let mut left = cur.left();
    let mut right = cur.right();
    if left.is_empty() {
        cur = right.clone();
    } else if right.is_empty() {
        cur = left.clone();
    } else if left.rank() >= right.rank() {
        cur = left.clone();
    } else {
        cur = right.clone();
    }
    // in the paper this is root = x, but in this case they mean value equality,
    // not reference equality as usual!
    if root.key() == x.key() {
        *root = cur.clone();
    } else if key < prev.key() {
        prev.set_left(cur.clone());
    } else {
        prev.set_right(cur.clone());
    }
    while !left.is_empty() && !right.is_empty() {
        if left.rank() >= right.rank() {
            loop {
                prev = left.clone();
                left = left.right();
                if left.is_empty() || left.rank() < right.rank() {
                    break;
                }
            }
            prev.set_right(right.clone());
        } else {
            loop {
                prev = right.clone();
                right = right.left();
                if right.is_empty() || left.rank() >= right.rank() {
                    break;
                }
            }
            prev.set_left(left.clone());
        }
    }
}

fn add_rank(
    keys: impl IntoIterator<Item = (u64, u64, u64)>,
) -> impl Iterator<Item = ((u64, u64, u64), u8)> {
    keys.into_iter().map(|i| {
        let ser = postcard::to_allocvec(&i).unwrap();
        let hash: [u8; 32] = blake3::hash(&ser).into();
        let rank = count_trailing_zeros(&hash);
        (i, rank)
    })
}

fn insert_rec_impl(keys: TestSet) {
    let items = keys.0;
    let mut root = Node::EMPTY;
    for (key, rank) in items.clone() {
        let x = Node::leaf(key, rank);
        root = insert_rec(x, root);
    }
    // root.print();
    root.check_invariants();
    for item in items {
        assert!(contains_rec(root.clone(), item.0));
    }
}

fn insert_impl(keys: TestSet) {
    let items = keys.0;
    let mut root = Node::EMPTY;
    for (key, rank) in items.clone() {
        let x = Node::leaf(key, rank);
        insert(&mut root, x);
    }
    // root.print();
    root.check_invariants();
    for item in items {
        assert!(contains_rec(root.clone(), item.0));
    }
}

fn delete_rec_impl(keys: TestSet) {
    let items = keys.0;
    if items.is_empty() {
        return;
    }
    let (k, r) = items[0];
    let tree = Node::from_iter(items.iter().cloned());
    let tree = delete_rec(Node::leaf(k, r), tree);
    tree.check_invariants();
    for (key, _) in items.into_iter().skip(1) {
        assert!(contains_rec(tree.clone(), key));
    }
    assert!(!contains_rec(tree.clone(), k));
}

fn delete_impl(keys: TestSet) {
    let items = keys.0;
    if items.is_empty() {
        return;
    }
    let (k, r) = items[0];
    let mut tree = Node::from_iter(items.iter().cloned());
    delete(&mut tree, Node::leaf(k, r));
    tree.check_invariants();
    for (key, _) in items.into_iter().skip(1) {
        assert!(contains_rec(tree.clone(), key));
    }
    assert!(!contains_rec(tree.clone(), k));
}

#[derive(Debug)]
struct TestSet(Vec<((u64, u64, u64), u8)>);

fn random_test_set() -> impl Strategy<Value = TestSet> {
    (any::<BTreeSet<(u64, u64, u64)>>(), any::<u64>()).prop_map(|(items, seed)| {
        let seed = blake3::hash(&seed.to_be_bytes()).into();
        let mut rng = rand::rngs::SmallRng::from_seed(seed);
        let mut items = add_rank(items).collect::<Vec<_>>();
        items.shuffle(&mut rng);
        TestSet(items)
    })
}

#[proptest]
fn prop_kd_insert_rec(#[strategy(random_test_set())] values: TestSet) {
    insert_rec_impl(values);
}

#[proptest]
fn prop_kd_insert(#[strategy(random_test_set())] values: TestSet) {
    insert_impl(values);
}

#[proptest]
fn prop_kd_delete_rec(#[strategy(random_test_set())] values: TestSet) {
    delete_rec_impl(values);
}

#[proptest]
fn prop_kd_delete(#[strategy(random_test_set())] values: TestSet) {
    delete_impl(values);
}
