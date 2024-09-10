//! Non-persisten reference implementation of a zip tree, closely following the
//! description in the paper.
//!
//! To literally follow the paper, we implement semantics of an OO language
//! with freely shareable pointers that can possibly be null (empty).
use std::{cell::RefCell, cmp::Ordering, collections::BTreeSet, fmt::Debug, rc::Rc, u8};

use proptest::prelude::{any, Strategy};
use rand::{seq::SliceRandom, SeedableRng};
use test_strategy::proptest;
use willow_store::count_trailing_zeros;

struct NodeData {
    key: (u64, u64),
    rank: u8,
    left: Node,
    right: Node,
}

fn order_at_rank(rank: u8) -> &'static str {
    match rank % 2 {
        0 => "xy",
        1 => "yx",
        _ => unreachable!(),
    }
}

fn cmp_at_rank((xa, ya): (u64, u64), (xb, yb): (u64, u64), rank: u8) -> Ordering {
    match rank % 2 {
        0 => xa.cmp(&xb).then(ya.cmp(&yb)),
        1 => ya.cmp(&yb).then(xa.cmp(&xb)),
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

    fn leaf(key: (u64, u64), rank: u8) -> Self {
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

    fn key(&self) -> (u64, u64) {
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

    fn from_iter(iter: impl IntoIterator<Item = ((u64, u64), u8)>) -> Self {
        let mut root = Self::EMPTY;
        for (key, rank) in iter {
            let x = Self::leaf(key, rank);
            root = insert_rec(x, root);
        }
        root
    }

    fn from_unique_nodes(mut nodes: Vec<Node>) -> Self {
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

    fn from_iter_reference(iter: impl IntoIterator<Item = ((u64, u64), u8)>) -> Self {
        let mut nodes = iter
            .into_iter()
            .map(|(key, rank)| Node::leaf(key, rank))
            .collect::<Vec<_>>();
        // Before we sort, remove all but the first occurence of each point.
        let mut uniques = BTreeSet::new();
        nodes.retain(|node| uniques.insert(node.key()));
        Self::from_unique_nodes(nodes)
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
fn contains_rec(node: Node, key: (u64, u64)) -> bool {
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

fn structurally_equal(a: Node, b: Node) -> bool {
    if a.is_empty() {
        b.is_empty()
    } else if b.is_empty() {
        false
    } else {
        structurally_equal(a.left(), b.left())
            && structurally_equal(a.right(), b.right())
            && a.rank() == b.rank()
            && a.key() == b.key()
    }
}

fn split_all_to_vec(node: &Node) -> Vec<Node> {
    let mut res = vec![];
    split_all(node.clone(), &mut res);
    res
}

fn split_to_vec(node: &Node, key: (u64, u64), rank: u8) -> Vec<Node> {
    let mut res = vec![];
    split_rec(node, key, rank, rank, &mut res);
    // split_all(node.clone(), &mut res);
    if res.is_empty() && !node.is_empty() {
        res.push(node.clone());
    }
    res
}

fn split_all(node: Node, res: &mut Vec<Node>) {
    if node.is_empty() {
        return;
    }
    split_all(node.left(), res);
    split_all(node.right(), res);
    node.set_left(Node::EMPTY);
    node.set_right(Node::EMPTY);
    res.push(node);
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

fn foreach(node: &Node, f: &mut impl FnMut(((u64, u64), u8))) {
    if node.is_empty() {
        return;
    }
    foreach(&node.left(), f);
    f((node.key(), node.rank()));
    foreach(&node.right(), f);
}

fn range(node: &Node, rank: u8) -> Option<((u64, u64), (u64, u64))> {
    let mut res: Option<((u64, u64), (u64, u64))> = None;
    foreach(node, &mut |((x, y), _)| {
        if let Some((min, max)) = &mut res {
            if cmp_at_rank((x, y), max.clone(), rank) == Ordering::Greater {
                *max = (x, y);
            }
            if cmp_at_rank((x, y), min.clone(), rank) == Ordering::Less {
                *min = (x, y);
            }
        } else {
            res = Some(((x, y), (x, y)));
        }
    });
    res
}

fn lr_rec(node: &Node, key: (u64, u64), rank: u8) -> (bool, bool) {
    if node.is_empty() {
        return (false, false);
    }
    let same_sort = node.rank() % 2 == rank % 2;
    let cmp = cmp_at_rank(node.key(), key, rank);
    let (l, r) = if same_sort {
        match cmp {
            Ordering::Less => {
                let (_left, right) = lr_rec(&node.right(), key, rank);
                (true, right)
            }
            Ordering::Greater => {
                let (left, _right) = lr_rec(&node.left(), key, rank);
                (left, true)
            }
            Ordering::Equal => (true, true),
        }
    } else {
        assert!(cmp != Ordering::Equal, "duplicate key");
        let (ll, lr) = lr_rec(&node.left(), key, rank);
        let (sl, sr) = match cmp {
            Ordering::Less => (true, false),
            Ordering::Equal => (true, true),
            Ordering::Greater => (false, true),
        };
        let (rl, rr) = lr_rec(&node.right(), key, rank);
        (ll || sl || rl, lr || sr || rr)
    };
    (l, r)
}

fn split_rec_2(
    node: &Node,
    key: (u64, u64),
    rank: u8,
    parent_rank: u8,
    res: &mut Vec<Node>,
) -> (bool, bool) {
    if node.is_empty() {
        return (false, false);
    }
    tracing::info!("{:?}", node);
    let same_sort = node.rank() % 2 == rank % 2;
    let cmp = cmp_at_rank(node.key(), key, rank);
    let n0 = res.len();
    let (l, r) = if same_sort {
        match cmp {
            Ordering::Less => {
                let n = res.len();
                let (left, right) = split_rec_2(&node.right(), key, rank, node.rank(), res);
                if n != res.len() {
                    node.set_right(Node::EMPTY);
                }
                (true, right)
            }
            Ordering::Greater => {
                let n = res.len();
                let (left, right) = split_rec_2(&node.left(), key, rank, node.rank(), res);
                if n != res.len() {
                    node.set_left(Node::EMPTY);
                }
                (left, true)
            }
            Ordering::Equal => (true, true),
        }
    } else {
        let n = res.len();
        let (ll, lr) = split_rec_2(&node.left(), key, rank, node.rank(), res);
        if n != res.len() {
            node.set_left(Node::EMPTY);
        }
        let (sl, sr) = match cmp {
            Ordering::Less => (true, false),
            Ordering::Equal => (true, true),
            Ordering::Greater => (false, true),
        };
        let n = res.len();
        let (rl, rr) = split_rec_2(&node.right(), key, rank, node.rank(), res);
        if n != res.len() {
            node.set_right(Node::EMPTY);
        }
        (ll || sl || rl, lr || sr || rr)
    };
    if l != r || rank == parent_rank || n0 != res.len() {
        flatten(node, res);
    }
    (l, r)
}

// splits a tree into parts that don't overlap with the key
fn split_rec(node: &Node, key: (u64, u64), rank: u8, parent_rank: u8, res: &mut Vec<Node>) -> bool {
    if node.is_empty() {
        return false;
    }
    if node.rank() < parent_rank {
        // if let Some((min, max)) = range(node, rank) {
        //     if cmp_at_rank(min, key, rank) == Ordering::Less
        //         || cmp_at_rank(max, key, rank) == Ordering::Greater
        //     {
        //         return false;
        //     }
        // }
        // let (l, r) = lr_rec(node, key, rank);
        // if !l || !r {
        //     return false;
        // }
    }
    let same_sort = node.rank() % 2 == rank % 2;
    let cmp = cmp_at_rank(node.key(), key, rank);
    if same_sort {
        match cmp {
            Ordering::Less => {
                if node.right().is_empty() {
                    println!("retaining");
                    node.print();
                    return false;
                }
                // if split_rec(&node.left(), key, rank, node.rank(), res) {
                //     node.set_left(Node::EMPTY);
                // }
                if split_rec(&node.right(), key, rank, node.rank(), res) {
                    node.set_right(Node::EMPTY);
                }
                flatten(node, res);
                true
            }
            Ordering::Greater => {
                // left might need to be split
                // self does not need to be split
                // right does not need to be split
                // if node.right().is_empty() || node.right().rank() == node.rank() {
                // if split_rec(&node.right(), key, rank, node.rank(), res) {
                //     node.set_right(Node::EMPTY);
                // }
                // }
                if split_rec(&node.left(), key, rank, node.rank(), res) {
                    node.set_left(Node::EMPTY);
                }
                flatten(node, res);
                true
            }
            Ordering::Equal => {
                panic!("duplicate key");
            }
        }
    } else {
        assert!(cmp != Ordering::Equal, "duplicate key");
        if split_rec(&node.left(), key, rank, node.rank(), res) {
            node.set_left(Node::EMPTY);
        }
        if split_rec(&node.right(), key, rank, node.rank(), res) {
            node.set_right(Node::EMPTY);
        }
        flatten(node, res);
        true
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

fn count(node: &Node) -> usize {
    if node.is_empty() {
        0
    } else {
        1 + count(&node.left()) + count(&node.right())
    }
}

fn insert_brute_force(root: &mut Node, x: Node) {
    let initial_count = count(root);
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
    // cur is either empty or below x, see exit condition of while loop
    let mut parts = Vec::new();
    split_all(cur.clone(), &mut parts);
    parts.push(x.clone());
    // print stats
    // println!("{} {}", initial_count, parts.len());
    // just flatten cur, add x, and build a new tree
    let merged = Node::from_unique_nodes(parts);
    if &cur == root {
        *root = merged;
    } else if cmp_at_rank(key, prev.key(), prev.rank()) == Ordering::Less {
        prev.set_left(merged);
    } else {
        prev.set_right(merged);
    }
}

fn delete_brute_force(root: &mut Node, key: (u64, u64)) {
    assert!(contains_rec(root.clone(), key));
    let mut cur = root.clone();
    let mut prev = Node::EMPTY;
    while key != cur.key() {
        prev = cur.clone();
        cur = if cmp_at_rank(key, cur.key(), cur.rank()) == Ordering::Less {
            cur.left()
        } else {
            cur.right()
        };
        if cur.is_empty() {
            return;
        }
    }
    let mut res = Vec::new();
    split_all(cur.left(), &mut res);
    split_all(cur.right(), &mut res);
    let merged = Node::from_unique_nodes(res);
    if prev.is_empty() {
        *root = merged;
    } else if prev.left() == cur {
        prev.set_left(merged);
    } else {
        prev.set_right(merged);
    }
    // delete(cur);
}

mod tests {
    use super::*;
    /// Compute rank from key, like in real use
    fn add_rank(
        keys: impl IntoIterator<Item = (u64, u64)>,
    ) -> impl Iterator<Item = ((u64, u64), u8)> {
        keys.into_iter().map(|i| {
            let ser = postcard::to_allocvec(&i).unwrap();
            let hash: [u8; 32] = blake3::hash(&ser).into();
            let rank = count_trailing_zeros(&hash);
            (i, rank)
        })
    }

    /// Assign a random rank with the right distribution to each key
    fn random_rank<'a>(
        keys: impl IntoIterator<Item = (u64, u64)> + 'a,
        rng: &'a mut impl rand::Rng,
    ) -> impl Iterator<Item = ((u64, u64), u8)> + 'a {
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

    fn insert_impl(keys: TestSet, op: fn(&mut Node, Node)) {
        let reference = keys.0;
        if reference.is_empty() {
            return;
        }
        let mut items = reference.clone();
        let (key, rank) = items.pop().unwrap();
        let mut root = Node::from_iter_reference(items.clone());
        {
            // println!("BEFORE");
            // println!("actual:");
            // root.print();
            // // root.print();
            root.check_invariants();
            // let reference_node = Node::from_iter_reference(items.clone());
            // println!("reference:");
            // reference_node.print();
            // println!("");
        }
        op(&mut root, Node::leaf(key, rank));
        {
            // println!("AFTER");
            // println!("actual:");
            // root.print();
            // root.print();
            root.check_invariants();
            // let reference_node = Node::from_iter_reference(reference.clone());
            // println!("reference:");
            // reference_node.print();
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
        let tree = Node::from_iter_reference(items.iter().cloned());
        let tree = delete_rec(Node::leaf(k, r), tree);
        tree.check_invariants();
        for (key, _) in items.into_iter().skip(1) {
            assert!(contains_rec(tree.clone(), key));
        }
        assert!(!contains_rec(tree.clone(), k));
    }

    fn delete_impl(keys: TestSet, op: fn(&mut Node, Node)) {
        let items = keys.0;
        if items.is_empty() {
            return;
        }
        let (k, r) = items[0];
        let mut tree = Node::from_iter_reference(items.iter().cloned());
        // println!("before");
        // tree.print();
        // println!();
        // println!("removing");
        // Node::leaf(k, r).print();
        // println!();
        op(&mut tree, Node::leaf(k, r));
        // println!("after");
        // tree.print();
        // println!();
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
        let parts = split_to_vec(&tree, key, rank);
        for (key, _rank) in items {
            let count = parts
                .iter()
                .filter(|x| contains_rec((*x).clone(), key))
                .count();
            assert_eq!(count, 1);
        }
    }

    fn insert_split_impl(keys: TestSet) {
        let reference = keys.0;
        if reference.is_empty() {
            return;
        }
        let items = reference.clone();
        // build the tree using the reference implementation
        let reference_tree = Node::from_iter_reference(items.iter().cloned());
        // filter out the root
        let root_key = reference_tree.key();
        let root_rank = reference_tree.rank();
        let items: Vec<_> = items
            .into_iter()
            .filter(|(key, _rank)| key != &root_key)
            .collect();
        assert!(items.len() == reference.len() - 1);
        let tree_without_root = Node::from_iter_reference(items.iter().cloned());
        println!("split:");
        Node::leaf(root_key, root_rank).print();
        println!("tree without root:");
        tree_without_root.print();
        let mut parts = split_to_vec(&tree_without_root, root_key, root_rank);
        for (key, _rank) in items {
            let count = parts
                .iter()
                .filter(|x| contains_rec((*x).clone(), key))
                .count();
            assert_eq!(count, 1);
        }
        parts.push(Node::leaf(root_key, root_rank));
        println!("{}/{}", parts.len(), reference.len());
        let root = Node::from_unique_nodes(parts);
        // assert!(structurally_equal(root.clone(), reference_tree.clone()));
        let mut fail = false;
        for (key, _rank) in reference.iter().cloned() {
            fail |= !contains_rec(root.clone(), key);
        }
        if fail {
            println!("----------");
            println!("actual:");
            root.print();
            let expected = Node::from_iter_reference(reference.iter().cloned());
            println!("expected:");
            expected.print();
            println!("----------");
            assert!(false);
        }
    }

    /// A set of points with ranks.
    ///
    /// Must not contain duplicates.
    ///
    /// Rank is usually computed from the key, but can be provided explicitly for testing.
    #[derive(Debug)]
    struct TestSet(Vec<((u64, u64), u8)>);

    fn small_key() -> impl Strategy<Value = (u64, u64)> {
        (0..100u64, 0..100u64)
    }

    fn small_key_set() -> impl Strategy<Value = BTreeSet<(u64, u64)>> {
        proptest::collection::btree_set(small_key(), 0..1000)
    }

    fn random_test_set() -> impl Strategy<Value = TestSet> {
        (small_key_set(), any::<u64>()).prop_map(|(items, seed)| {
            let seed = blake3::hash(&seed.to_be_bytes()).into();
            let mut rng = rand::rngs::SmallRng::from_seed(seed);
            let mut items = add_rank(items).collect::<Vec<_>>();
            items.shuffle(&mut rng);
            TestSet(items)
        })
    }

    fn random_test_set_2() -> impl Strategy<Value = TestSet> {
        (small_key_set(), any::<u64>()).prop_map(|(items, seed)| {
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
            let items = items.into_iter().map(|x| (x, x)).collect::<BTreeSet<_>>();
            let mut items = add_rank(items).collect::<Vec<_>>();
            items.shuffle(&mut rng);
            TestSet(items)
        })
    }

    fn uniform_test_set_2() -> impl Strategy<Value = TestSet> {
        (any::<BTreeSet<u64>>(), any::<u64>()).prop_map(|(items, seed)| {
            let seed = blake3::hash(&seed.to_be_bytes()).into();
            let mut rng = rand::rngs::SmallRng::from_seed(seed);
            let items = items.into_iter().map(|x| (x, x)).collect::<BTreeSet<_>>();
            let mut items = random_rank(items, &mut rng).collect::<Vec<_>>();
            items.shuffle(&mut rng);
            TestSet(items)
        })
    }

    #[proptest]
    #[ignore = "broken for kd"]
    fn prop_kd_insert_rec(#[strategy(random_test_set())] values: TestSet) {
        insert_rec_impl(values);
    }

    #[proptest]
    #[ignore = "broken for kd"]
    fn prop_kd_insert(#[strategy(random_test_set_2())] values: TestSet) {
        insert_impl(values, insert);
    }

    #[proptest]
    fn prop_kd_insert_brute(#[strategy(random_test_set_2())] values: TestSet) {
        insert_impl(values, insert_brute_force);
    }

    #[test]
    fn test_kd_insert() {
        let cases: Vec<TestSet> = vec![
            // TestSet(vec![
            //     ((1, 0, 2), 0),
            //     ((1, 0, 0), 0),
            //     ((0, 0, 3), 0),
            //     ((1, 0, 1), 1),
            // ]),
            // TestSet(vec![
            //     ((0, 0, 0), 3),
            //     ((0, 0, 1), 1),
            //     ((0, 4741757182755845721, 0), 3),
            //     ((7769037433229280596, 0, 0), 4),
            // ]),
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
    #[ignore = "broken for kd"]
    fn prop_kd_delete_rec(#[strategy(random_test_set())] values: TestSet) {
        delete_rec_impl(values);
    }

    #[proptest]
    #[ignore = "broken for kd"]
    fn prop_kd_delete(#[strategy(random_test_set())] values: TestSet) {
        delete_impl(values, delete);
    }

    #[proptest]
    fn prop_kd_delete_brute(#[strategy(random_test_set())] values: TestSet) {
        delete_impl(values, |a, b| delete_brute_force(a, b.key()));
    }

    #[proptest]
    fn prop_kd_insert_rec_uniform(#[strategy(uniform_test_set())] values: TestSet) {
        insert_rec_impl(values);
    }

    #[proptest]
    fn prop_kd_insert_uniform(#[strategy(uniform_test_set())] values: TestSet) {
        insert_impl(values, insert);
    }

    #[proptest]
    fn prop_kd_delete_rec_uniform(#[strategy(uniform_test_set())] values: TestSet) {
        delete_rec_impl(values);
    }

    #[proptest]
    fn prop_kd_delete_uniform(#[strategy(uniform_test_set())] values: TestSet) {
        delete_impl(values, delete);
    }

    #[proptest]
    fn prop_kd_split(#[strategy(random_test_set_2())] values: TestSet) {
        split_impl(values);
    }

    #[proptest]
    #[ignore = "broken for kd"]
    fn prop_kd_insert_split(#[strategy(random_test_set())] values: TestSet) {
        insert_split_impl(values);
    }

    #[test]
    fn test_kd_delete_brute_force() {
        let cases = vec![
            // TestSet(
            //     vec![
            //         (
            //             (
            //                 1,
            //                 3,
            //             ),
            //             4,
            //         ),
            //         (
            //             (
            //                 3,
            //                 0,
            //             ),
            //             3,
            //         ),
            //         (
            //             (
            //                 0,
            //                 0,
            //             ),
            //             1,
            //         ),
            //     ],
            //     ),
            TestSet(vec![
                ((0, 3), 1),
                ((1, 0), 1),
                ((0, 7), 4),
                ((3, 0), 3),
                ((2, 0), 0),
                ((0, 0), 1),
                ((0, 1), 0),
            ]),
        ];
        for case in cases {
            delete_impl(case, |a, b| delete_brute_force(a, b.key()));
        }
    }

    #[test]
    #[ignore = "broken for kd"]
    fn test_kd_insert_split() {
        let cases = vec![TestSet(vec![
            ((0, 0), 1),
            ((0, 2), 3),
            ((1, 3), 4),
            ((2, 0), 0),
        ])];
        for case in cases {
            insert_split_impl(case);
        }
    }
}
