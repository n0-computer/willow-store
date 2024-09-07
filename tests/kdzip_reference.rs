//! Non-persisten reference implementation of a zip tree, closely following the
//! description in the paper.
//!
//! To literally follow the paper, we implement semantics of an OO language
//! with freely shareable pointers that can possibly be null (empty).
use std::{cell::RefCell, cmp::Ordering, collections::BTreeSet, fmt::Debug, rc::Rc};

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

fn order_at_rank(rank: u8) -> &'static str {
    match rank % 3 {
        0 => "xyz",
        1 => "yzx",
        2 => "zyz",
        _ => unreachable!(),
    }
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

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            write!(f, "EMPTY")
        } else {
            write!(f, "Node({:?}, {})", self.key(), self.rank())
        }
    }
}

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

    fn is_leaf(&self) -> bool {
        !self.is_empty() && self.left().is_empty() && self.right().is_empty()
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
            println!(
                "{indent}key={:?} rank={} order={}",
                self.key(),
                self.rank(),
                order_at_rank(self.rank())
            );
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
            assert!(cmp_at_rank(self.left().key(), self.key(), self.rank()) == Ordering::Less);
        }
        if !self.right().is_empty() {
            assert!(self.right().rank() <= rank);
            assert!(cmp_at_rank(self.right().key(), self.key(), self.rank()) == Ordering::Greater);
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

    fn from_iter_reference(iter: impl IntoIterator<Item = ((u64, u64, u64), u8)>) -> Self {
        let mut nodes = iter
            .into_iter()
            .map(|(key, rank)| Node::leaf(key, rank))
            .collect::<Vec<_>>();
        // Before we sort, remove all but the first occurence of each point.
        let mut uniques = BTreeSet::new();
        nodes.retain(|node| uniques.insert(node.key()));
        nodes.sort_by(|p1, p2| {
            p2.rank()
                .cmp(&p1.rank())
                .then(cmp_at_rank(p1.key(), p2.key(), p1.rank()))
        });
        let mut root = Self::EMPTY;
        for node in nodes {
            insert_no_balance(&mut root, node);
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
fn contains_rec(node: Node, key: (u64, u64, u64)) -> bool {
    if node.is_empty() {
        false
    } else if key == node.key() {
        true
    } else if cmp_at_rank(key, node.key(), node.rank()) == Ordering::Less {
        contains_rec(node.left(), key)
    } else {
        contains_rec(node.right(), key)
    }
}

fn split_to_vec(node: &Node, key: (u64, u64, u64), rank: u8) -> Vec<Node> {
    let mut res = vec![];
    split_rec(node, key, rank, &mut res);
    if res.is_empty() {
        res.push(node.clone());
    }
    res
}

/// Convert a node into a list of nodes, non-recursively.
fn flatten(node: &Node, res: &mut Vec<Node>) {
    if node.is_empty() {
        return;
    }
    if !node.left().is_empty() {
        res.push(node.left().clone());
        node.set_left(Node::EMPTY);
    }
    res.push(node.clone());
    if !node.right().is_empty() {
        res.push(node.right().clone());
        node.set_right(Node::EMPTY);
    }
}

// splits a tree into parts that don't overlap with the key
fn split_rec(node: &Node, key: (u64, u64, u64), rank: u8, res: &mut Vec<Node>) -> (bool, bool) {
    if node.is_empty() {
        return (false, false);
    }
    let same_sort = node.rank() % 3 == rank % 3;
    let cmp = cmp_at_rank(node.key(), key, rank);
    if same_sort {
        match cmp {
            Ordering::Less => {
                // left does not need to be split
                // self does not need to be split
                // right might need to be split
                let (has_left, has_right) = split_rec(&node.right(), key, rank, res);
                if has_left && has_right {
                    assert!(has_right);
                    node.set_right(Node::EMPTY);
                }
                if has_right {
                    flatten(node, res);
                }
                (true, has_right)
            }
            Ordering::Greater => {
                // left might need to be split
                // self does not need to be split
                // right does not need to be split
                let (has_left, has_right) = split_rec(&node.left(), key, rank, res);
                if has_left && has_right {
                    node.set_left(Node::EMPTY);
                }
                if has_left {
                    flatten(node, res);
                }
                (has_left, true)
            }
            Ordering::Equal => {
                panic!("duplicate key");
            }
        }
    } else {
        assert!(cmp != Ordering::Equal, "duplicate key");
        let (left_left, left_right) = split_rec(&node.left(), key, rank, res);
        if left_left && left_right {
            node.set_left(Node::EMPTY);
        }
        let (right_left, right_right) = split_rec(&node.right(), key, rank, res);
        if right_left && right_right {
            node.set_right(Node::EMPTY);
        }
        let self_left = cmp == Ordering::Less;
        let self_right = cmp == Ordering::Greater;
        let res_left = left_left || self_left || right_left;
        let res_right = left_right || self_right || right_right;
        if res_left && res_right {
            flatten(node, res);
        }
        (res_left, res_right)
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
    println!("insert root={:?} x={:?}", root, x);
    let rank = x.rank();
    let key = x.key();
    let mut cur = root.clone();
    let mut prev = Node::EMPTY;
    while !cur.is_empty()
        && (rank < cur.rank()
            || (rank == cur.rank() && cmp_at_rank(key, cur.key(), cur.rank()) == Ordering::Greater))
    {
        prev = cur.clone();
        cur = if cmp_at_rank(key, cur.key(), cur.rank()) == Ordering::Less {
            cur.left()
        } else {
            cur.right()
        };
    }
    if &cur == root {
        println!("x becomes new root");
        *root = x.clone();
    } else if cmp_at_rank(key, prev.key(), prev.rank()) == Ordering::Less {
        prev.set_left(x.clone());
    } else {
        prev.set_right(x.clone());
    }
    if cur.is_empty() {
        assert!(x.is_leaf());
        return;
    }
    if cmp_at_rank(key, cur.key(), rank) == Ordering::Less {
        println!("cur {:?} becomes right child of x {:?}", cur, x);
        x.set_right(cur.clone());
    } else {
        println!("cur {:?} becomes left child of x {:?}", cur, x);
        x.set_left(cur.clone());
    }
    prev = x.clone();
    while !cur.is_empty() {
        let fix = prev.clone();
        println!("fix key={:?} rank={}", fix.key(), fix.rank());
        if cmp_at_rank(cur.key(), key, rank) == Ordering::Less {
            loop {
                prev = cur.clone();
                cur = cur.right();
                if cur.is_empty() || cmp_at_rank(cur.key(), key, rank) == Ordering::Greater {
                    break;
                }
            }
        } else {
            loop {
                prev = cur.clone();
                cur = cur.left();
                if cur.is_empty() || cmp_at_rank(cur.key(), key, rank) == Ordering::Less {
                    break;
                }
            }
        }
        if cmp_at_rank(fix.key(), key, rank) == Ordering::Greater
            || (fix == x && cmp_at_rank(prev.key(), key, rank) == Ordering::Greater)
        {
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

fn insert_no_balance(root: &mut Node, x: Node) {
    if root.is_empty() {
        *root = x;
        return;
    }
    match cmp_at_rank(root.key(), x.key(), root.rank()) {
        Ordering::Less => {
            if root.right().is_empty() {
                root.set_right(x);
            } else {
                insert_no_balance(&mut root.right(), x);
            }
        }
        Ordering::Greater => {
            if root.left().is_empty() {
                root.set_left(x);
            } else {
                insert_no_balance(&mut root.left(), x);
            }
        }
        Ordering::Equal => {
            panic!("duplicate key");
        }
    }
}

struct Relation {
    left: bool,
    right: bool,
}

impl Relation {
    fn empty() -> Self {
        Self {
            left: false,
            right: false,
        }
    }
    fn new(left: bool, right: bool) -> Self {
        Self { left, right }
    }
    fn combine(lhs: Self, rhs: Self) -> Self {
        Self::new(lhs.left || rhs.left, lhs.right || rhs.right)
    }
    fn is_both(&self) -> bool {
        self.left && self.right
    }
}

mod tests {
    use super::*;
    /// Compute rank from key, like in real use
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

    /// Assign a random rank with the right distribution to each key
    fn random_rank<'a>(
        keys: impl IntoIterator<Item = (u64, u64, u64)> + 'a,
        rng: &'a mut impl rand::Rng,
    ) -> impl Iterator<Item = ((u64, u64, u64), u8)> + 'a {
        keys.into_iter().map(move |i| {
            let x = rng.gen::<u64>();
            let hash: [u8; 32] = blake3::hash(&x.to_be_bytes()).into();
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
        let reference = keys.0;
        if reference.is_empty() {
            return;
        }
        let mut items = reference.clone();
        let (key, rank) = items.pop().unwrap();
        let mut root = Node::from_iter_reference(items.clone());
        {
            println!("BEFORE");
            println!("actual:");
            root.print();
            // root.print();
            root.check_invariants();
            let reference_node = Node::from_iter_reference(items.clone());
            println!("reference:");
            reference_node.print();
            println!("");
        }
        insert(&mut root, Node::leaf(key, rank));
        {
            println!("AFTER");
            println!("actual:");
            root.print();
            // root.print();
            root.check_invariants();
            let reference_node = Node::from_iter_reference(reference.clone());
            println!("reference:");
            reference_node.print();
        }
        for item in reference {
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

    fn split_impl(keys: TestSet) {
        let reference = keys.0;
        if reference.is_empty() {
            return;
        }
        let mut items = reference.clone();
        let (key, rank) = items.pop().unwrap();
        let tree = Node::from_iter_reference(items.iter().cloned());
        for (key, _rank) in items.iter().cloned() {
            if !contains_rec(tree.clone(), key) {
                println!("broken tree:");
                tree.print();
                println!("missing key={:?}", key);
                assert!(false);
            }
        }
        println!("full:");
        tree.print();
        let parts = split_to_vec(&tree, key, rank);
        println!("items.len()={} parts.len()={}", items.len(), parts.len());

        for (key, _rank) in items {
            let count = parts
                .iter()
                .filter(|x| contains_rec((*x).clone(), key))
                .count();
            if count != 1 {
                for part in &parts {
                    println!("part:");
                    part.print();
                }
                println!("missing key={:?}", key);
            }
            assert_eq!(count, 1);
        }
    }

    /// A set of points with ranks.
    ///
    /// Must not contain duplicates.
    ///
    /// Rank is usually computed from the key, but can be provided explicitly for testing.
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

    fn random_test_set_2() -> impl Strategy<Value = TestSet> {
        (any::<BTreeSet<(u64, u64, u64)>>(), any::<u64>()).prop_map(|(items, seed)| {
            let seed = blake3::hash(&seed.to_be_bytes()).into();
            let mut rng = rand::rngs::SmallRng::from_seed(seed);
            let mut items = random_rank(items, &mut rng).collect::<Vec<_>>();
            items.shuffle(&mut rng);
            TestSet(items)
        })
    }

    fn uniform_test_set() -> impl Strategy<Value = TestSet> {
        (any::<BTreeSet<u64>>(), any::<u64>()).prop_map(|(items, seed)| {
            let seed = blake3::hash(&seed.to_be_bytes()).into();
            let mut rng = rand::rngs::SmallRng::from_seed(seed);
            let items = items
                .into_iter()
                .map(|x| (x, x, x))
                .collect::<BTreeSet<_>>();
            let mut items = add_rank(items).collect::<Vec<_>>();
            items.shuffle(&mut rng);
            TestSet(items)
        })
    }

    fn uniform_test_set_2() -> impl Strategy<Value = TestSet> {
        (any::<BTreeSet<u64>>(), any::<u64>()).prop_map(|(items, seed)| {
            let seed = blake3::hash(&seed.to_be_bytes()).into();
            let mut rng = rand::rngs::SmallRng::from_seed(seed);
            let items = items
                .into_iter()
                .map(|x| (x, x, x))
                .collect::<BTreeSet<_>>();
            let mut items = random_rank(items, &mut rng).collect::<Vec<_>>();
            items.shuffle(&mut rng);
            TestSet(items)
        })
    }

    #[proptest]
    #[ignore]
    fn prop_kd_insert_rec(#[strategy(random_test_set())] values: TestSet) {
        insert_rec_impl(values);
    }

    #[proptest]
    #[ignore]
    fn prop_kd_insert(#[strategy(random_test_set_2())] values: TestSet) {
        insert_impl(values);
    }

    #[test]
    fn test_kd_insert() {
        let cases = vec![
            // TestSet(vec![
            //     ((1, 0, 2), 0),
            //     ((1, 0, 0), 0),
            //     ((0, 0, 3), 0),
            //     ((1, 0, 1), 1),
            // ]),
            TestSet(vec![
                ((0, 0, 0), 3),
                ((0, 0, 1), 1),
                ((0, 4741757182755845721, 0), 3),
                ((7769037433229280596, 0, 0), 4),
            ]),
        ];
        for case in cases {
            let node = Node::from_iter_reference(case.0.iter().cloned());
            node.print();
            for (key, rank) in &case.0 {
                assert!(contains_rec(node.clone(), *key));
            }
        }
    }

    #[proptest]
    #[ignore]
    fn prop_kd_delete_rec(#[strategy(random_test_set())] values: TestSet) {
        delete_rec_impl(values);
    }

    #[proptest]
    #[ignore]
    fn prop_kd_delete(#[strategy(random_test_set())] values: TestSet) {
        delete_impl(values);
    }

    #[proptest]
    fn prop_kd_insert_rec_uniform(#[strategy(uniform_test_set())] values: TestSet) {
        insert_rec_impl(values);
    }

    #[proptest]
    fn prop_kd_insert_uniform(#[strategy(uniform_test_set())] values: TestSet) {
        insert_impl(values);
    }

    #[proptest]
    fn prop_kd_delete_rec_uniform(#[strategy(uniform_test_set())] values: TestSet) {
        delete_rec_impl(values);
    }

    #[proptest]
    fn prop_kd_delete_uniform(#[strategy(uniform_test_set())] values: TestSet) {
        delete_impl(values);
    }

    #[proptest]
    fn prop_kd_split_uniform(#[strategy(random_test_set_2())] values: TestSet) {
        split_impl(values);
    }
}
