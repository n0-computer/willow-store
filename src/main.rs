use std::{
    cmp::Ordering, fmt::{Debug, Display}, ops::{Bound, RangeBounds}, sync::Arc
};

use bytes::{Bytes, BytesMut};
use genawaiter::rc::{Co, Gen};
use redb::{ReadableTable, WriteTransaction};
use serde::{Deserialize, Serialize};

trait Config {
    const VALUE_SIZE: usize;
    const FINGERPRINT_SIZE: usize;
}

type INode = u64;
type Value = [u8; 32];
type BlobKey<'a> = (INode, &'a [u8]);
type FingerprintKey<'a> = (INode, &'a [u8]);
const BLOBS_TABLE: redb::TableDefinition<BlobKey<'static>, BlobValue> =
    redb::TableDefinition::new("blobs-v0");
const FINGERPRINTS_TABLE: redb::TableDefinition<FingerprintKey<'static>, Fingerprint> =
    redb::TableDefinition::new("fingerprints-v0");

type Path<'a> = &'a [&'a [u8]];

#[derive(Clone, Copy, Default)]
struct Fingerprint([u8; 32]);

impl From<Value> for Fingerprint {
    fn from(value: Value) -> Self {
        Self(value)
    }
}

impl Debug for Fingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fingerprint({})", hex::encode(&self.0))
    }
}

impl Display for Fingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(&self.0))
    }
}

impl std::ops::BitXorAssign for Fingerprint {
    fn bitxor_assign(&mut self, rhs: Self) {
        for i in 0..32 {
            self.0[i] ^= rhs.0[i];
        }
    }
}

impl redb::Value for Fingerprint {
    type SelfType<'a> = Self;

    type AsBytes<'a> = [u8; 32];

    fn fixed_width() -> Option<usize> {
        Some(32)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a {
        Self(data.try_into().unwrap())
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'a,
        Self: 'b {
        value.0
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("Fingerprint")
    }
}

#[derive(Clone, Default)]
struct OwnedPath(Arc<Vec<Bytes>>);

impl Debug for OwnedPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.as_ref().iter().map(|x| hex::encode(x)))
            .finish()
    }
}

impl AsRef<[Bytes]> for OwnedPath {
    fn as_ref(&self) -> &[Bytes] {
        &self.0
    }
}

impl OwnedPath {
    fn get_mut(&mut self) -> &mut Vec<Bytes> {
        Arc::make_mut(&mut self.0)
    }
}

#[derive(Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
struct BlobValue {
    value: Option<Value>,
    dir: Option<INode>,
}

impl Debug for BlobValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlobValue")
            .field("value", &self.value.map(hex::encode))
            .field("dir", &self.dir)
            .finish()
    }

}

impl<'x> redb::Value for BlobValue {
    type SelfType<'s> = BlobValue
    where
        Self: 's;

    type AsBytes<'s> = Vec<u8>;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        postcard::from_bytes(data).unwrap()
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'a,
        Self: 'b,
    {
        postcard::to_stdvec(value).unwrap()
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("BlobValue")
    }
}

struct PathGenerator {
    ringbuffer: BytesMut,
    current_path: OwnedPath,
}

impl PathGenerator {
    fn new() -> Self {
        Self {
            ringbuffer: BytesMut::with_capacity(1024),
            current_path: OwnedPath::default(),
        }
    }

    fn pop(&mut self) {
        self.current_path.get_mut().pop();
    }

    fn set_last(&mut self, component: &[u8]) {
        self.ringbuffer.clear();
        self.ringbuffer.extend_from_slice(component);
        *self.current_path.get_mut().last_mut().unwrap() = self.ringbuffer.split().freeze();
    }

    fn push(&mut self, component: &[u8]) {
        self.ringbuffer.clear();
        self.ringbuffer.extend_from_slice(component);
        let component = self.ringbuffer.split().freeze();
        self.push_owned(component);
    }

    fn push_owned(&mut self, component: Bytes) {
        self.current_path.get_mut().push(component);
    }

    fn path(&self) -> &OwnedPath {
        &self.current_path
    }
}

struct Tables<'txn> {
    blobs: redb::Table<'txn, BlobKey<'static>, BlobValue>,
    fingerprints: redb::Table<'txn, FingerprintKey<'static>, Fingerprint>,
}

impl<'a> Tables<'a> {
    fn new(txn: &'a WriteTransaction) -> std::result::Result<Self, redb::TableError> {
        Ok(Self {
            blobs: txn.open_table(BLOBS_TABLE)?,
            fingerprints: txn.open_table(FINGERPRINTS_TABLE)?,
        })
    }
}

fn complete_range(inode: INode) -> std::ops::Range<BlobKey<'static>> {
    let start = (inode, EMPTY_PATH);
    let end = (inode + 1, EMPTY_PATH);
    start..end
}

self_cell::self_cell! {
    struct TransactionAndTablesInner {
        owner: WriteTransaction,
        #[covariant]
        dependent: Tables,
    }
}

impl std::fmt::Debug for TransactionAndTables {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransactionAndTables").finish()
    }
}

struct TransactionAndTables {
    inner: TransactionAndTablesInner,
}

impl TransactionAndTables {
    pub fn new(tx: WriteTransaction) -> std::result::Result<Self, redb::TableError> {
        Ok(Self {
            inner: TransactionAndTablesInner::try_new(tx, |tx| Tables::new(tx))?,
        })
    }

    pub fn tables(&self) -> &Tables {
        self.inner.borrow_dependent()
    }

    pub fn with_tables_mut<T>(&mut self, f: impl FnOnce(&mut Tables) -> T) -> T {
        self.inner.with_dependent_mut(|_, t| f(t))
    }

    pub fn commit(self) -> std::result::Result<(), redb::CommitError> {
        self.inner.into_owner().commit()
    }
}

#[derive(Debug, Default)]
enum CurrentTransaction {
    #[default]
    None,
    Write(TransactionAndTables),
}

struct TreeStore {
    db: redb::Database,
    current_transaction: CurrentTransaction,
}

const ROOT_INODE: INode = 0;
const EMPTY_PATH: &'static [u8] = &[];
const EMPTY_FULL_PATH: Path<'static> = &[];

trait RangeController<'a> {
    // go one level down in the directory tree
    fn down(&mut self);
    // set the current last component of the path.
    fn set_last(&mut self, component: &[u8]);
    // get the range when iterating over the directory below the current path
    fn range(&self, inode: INode) -> RcRange<BlobKey<'a>>;
    // go one level up in the directory tree
    fn up(&mut self);
}

trait IterController<'a> {
    type Item;
    // go one level down in the directory tree
    fn down(&mut self);
    // set the current last component of the path.
    fn set_last(&mut self, component: &[u8]);
    // get the range when iterating over the directory below the current path
    fn range(&self, inode: INode) -> RcRange<BlobKey<'a>>;
    // produce an item, given a value
    fn item(&self, value: Option<&Value>) -> Option<Self::Item>;
    // go one level up in the directory tree
    fn up(&mut self);
}

struct AllPathsIterController {
    path_gen: PathGenerator,
}

struct RangeFromIterController<'a> {
    path_gen: PathGenerator,
    from: Path<'a>,
}

struct RangeFromToIterController<'a> {
    path_gen: PathGenerator,
    from: Path<'a>,
    to: Path<'a>,
}

fn strip_prefix<'a>(path: Path<'a>, prefix: &[Bytes]) -> Option<Path<'a>> {
    if path.len() < prefix.len() {
        return None;
    }
    for (a, b) in path.iter().zip(prefix.iter()) {
        if a != b {
            return None;
        }
    }
    Some(&path[prefix.len()..])
}

impl<'a> IterController<'a> for RangeFromIterController<'a> {
    type Item = (OwnedPath, Value);

    fn down(&mut self) {
        self.path_gen.push(EMPTY_PATH);
    }

    // go one level up in the directory tree
    fn up(&mut self) {
        self.path_gen.pop();
        self.from = &[];
    }

    // set the current last component of the path.
    //
    // this will be called after down()
    fn set_last(&mut self, component: &[u8]) {
        self.path_gen.set_last(component);
        if self.from.get(0) == Some(&component) {
            self.from = &self.from[1..];
        } else {
            self.from = &[];
        }
    }

    // get the range when iterating over the directory below the current path
    fn range(&self, inode: INode) -> RcRange<BlobKey<'a>> {
        let start = self.from.get(0).map(|x| *x).unwrap_or_default();
        ((inode, start)..(inode + 1, EMPTY_PATH)).into()
    }

    // produce an item, given a value
    fn item(&self, value: Option<&Value>) -> Option<Self::Item> {
        if self.from.is_empty() {
            Some((self.path_gen.path().clone(), *value?))
        } else {
            None
        }
    }
}

impl<'a> IterController<'a> for RangeFromToIterController<'a> {
    type Item = (OwnedPath, Value);

    fn down(&mut self) {
        self.path_gen.push(EMPTY_PATH);
    }

    // go one level up in the directory tree
    fn up(&mut self) {
        self.path_gen.pop();
        self.from = &[];
    }

    // set the current last component of the path.
    //
    // this will be called after down()
    fn set_last(&mut self, component: &[u8]) {
        self.path_gen.set_last(component);
        if self.from.get(0) == Some(&component) {
            self.from = &self.from[1..];
        } else {
            self.from = &[];
        }
    }

    // get the range when iterating over the directory below the current path
    fn range(&self, inode: INode) -> RcRange<BlobKey<'a>> {
        let start = self
            .from
            .get(0)
            .map(|x| (inode, *x))
            .unwrap_or((inode, EMPTY_PATH));
        // todo: calling strip_prefix on every item is not efficient
        let end1 = strip_prefix(self.to, self.path_gen.path().as_ref());
        match end1 {
            Some(end) if end.len() == 1 => (start..(inode, end[0])).into(),
            Some(end) if end.len() > 1 => (start..=(inode, end[0])).into(),
            _ => (start..(inode + 1, EMPTY_PATH)).into(),
        }
    }

    // produce an item, given a value
    fn item(&self, value: Option<&Value>) -> Option<Self::Item> {
        if self.from.is_empty() {
            Some((self.path_gen.path().clone(), *value?))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
enum RcRange<T> {
    Range(std::ops::Range<T>),
    RangeInclusive(std::ops::RangeInclusive<T>),
}

impl<T> From<std::ops::Range<T>> for RcRange<T> {
    fn from(r: std::ops::Range<T>) -> Self {
        RcRange::Range(r)
    }
}

impl<T> From<std::ops::RangeInclusive<T>> for RcRange<T> {
    fn from(r: std::ops::RangeInclusive<T>) -> Self {
        RcRange::RangeInclusive(r)
    }
}

impl<T> RangeBounds<T> for RcRange<T> {
    fn start_bound(&self) -> Bound<&T> {
        match self {
            RcRange::Range(r) => r.start_bound(),
            RcRange::RangeInclusive(r) => r.start_bound(),
        }
    }

    fn end_bound(&self) -> Bound<&T> {
        match self {
            RcRange::Range(r) => r.end_bound(),
            RcRange::RangeInclusive(r) => r.end_bound(),
        }
    }
}

impl<'a> IterController<'a> for AllPathsIterController {
    type Item = (OwnedPath, Value);

    fn down(&mut self) {
        self.path_gen.push(EMPTY_PATH);
    }

    // go one level up in the directory tree
    fn up(&mut self) {
        self.path_gen.pop();
    }

    // set the current last component of the path.
    //
    // this will be called after down()
    fn set_last(&mut self, component: &[u8]) {
        self.path_gen.set_last(component);
    }

    // get the range when iterating over the directory below the current path
    fn range(&self, inode: INode) -> RcRange<BlobKey<'static>> {
        complete_range(inode).into()
    }

    // produce an item, given a value
    fn item(&self, value: Option<&Value>) -> Option<Self::Item> {
        Some((self.path_gen.path().clone(), *value?))
    }
}

/// Next non-prefix path for a given path.
fn next_non_prefix<'a>(path: &[u8], buffer: &'a mut Vec<u8>) -> Option<&'a [u8]> {
    buffer.clear();
    buffer.extend_from_slice(path);
    while buffer.last() == Some(&0xFF) {
        buffer.pop();
    }
    if let Some(last) = buffer.last_mut() {
        *last -= 1;
        Some(buffer.as_slice())
    } else {
        None
    }
}

/// Increment the path in place. To turn inclusive ranges into exclusive ranges.
fn inc<'a>(path: &'a [u8], buffer: &'a mut Vec<u8>) -> &'a [u8] {
    buffer.clear();
    buffer.extend_from_slice(path);
    for i in (0..buffer.len()).rev() {
        if buffer[i] == 0xFF {
            buffer[i] = 0;
        } else {
            buffer[i] += 1;
            return buffer.as_slice();
        }
    }
    buffer.push(0);
    buffer.as_slice()
}

fn get_level_range<'a>(dir: INode, range: &'a std::ops::Range<Path<'a>>, buf: &'a mut Vec<u8>) -> std::ops::Range<BlobKey<'a>> {
    if dir == ROOT_INODE {
        complete_range(dir)
    } else {
        // if start is not set, use (dir, EMPTY_PATH), which is the lowest possible path for the given dir, inclusive.
        let start = match range.start.first() {
            Some(component) => (dir, *component),
            None => (dir, EMPTY_PATH),
        };
        // if end is not set, use (dir + 1, EMPTY_PATH), which is the lowest possible path for the next dir, exclusive.
        let end = match range.end.first() {
            Some(component) if range.end.len() == 1 => (dir, *component),
            Some(component) => {
                let component_exclusive = inc(component, buf);
                (dir, component_exclusive)
            }
            None => (dir + 1, EMPTY_PATH),
        };
        start..end
    }
}

    fn fingerprint(
        tables: &mut Tables,
        range: std::ops::Range<Path>,
    ) -> Result<Fingerprint, redb::StorageError> {
        assert!(range.start < range.end);
        let mut to_store = Vec::new();
        let res = fingerprint_rec(tables, ROOT_INODE, range, &mut to_store)?;
        for (path, fp) in to_store {
            println!("storing fingerprint");
            tables.fingerprints.insert(path, fp)?;
            // insert an empty blob for the path to mark that we have a fingerprint for it.
            if tables.blobs.get(path)?.is_none() {
                tables.blobs.insert(path, BlobValue::default())?;
            }
        }
        Ok(res.0)
    }

    fn fingerprint_rec(
        tables: &Tables,
        dir: INode,
        range: std::ops::Range<Path>,
        to_store: &mut Vec<(BlobKey, Fingerprint)>
    ) -> Result<(Fingerprint, bool), redb::StorageError> {
        let mut res = Fingerprint::default();
        let mut buf = Vec::new();
        let mut buf2 = Vec::new();
        let current_range = get_level_range(dir, &range, &mut buf);
        let unrestricted = current_range.start.1 == EMPTY_PATH && current_range.start.0 != current_range.end.0;
        let current_range_end = current_range.end;
        let mut iter = tables.blobs.range(current_range)?;
        while let Some(item) = iter.next() {
            let (k, v) = item?;
            let path@(_, component) = k.value();
            if let Some(fp) = tables.fingerprints.get(path)? {
                // we have a fingerprint for this path, check if it fits into the current range.
                let fingerprint_end = next_non_prefix(component, &mut buf2).map(|x| (dir, x)).unwrap_or((dir + 1, EMPTY_PATH));
                match fingerprint_end.cmp(&current_range_end) {
                    Ordering::Less => {
                        println!("using fingerprint");
                        res ^= fp.value();
                        iter = tables.blobs.range(fingerprint_end..current_range_end)?;
                    }
                    Ordering::Equal => {
                        println!("using fingerprint until end");
                        // the fingerprint is exactly the current range
                        res ^= fp.value();
                        // we could just create the iter, but it would be empty, so we can immediately exit the loop.
                        break;
                    }
                    Ordering::Greater => {
                        // the fingerprint is not in the current range
                    }
                }
            }
            let v = v.value();
            if let Some(value) = v.value {
                res ^= Fingerprint::from(value);
            }
            if let Some(subdir) = v.dir {
                if dir == ROOT_INODE {
                    // special case for the root directory, 
                    res ^= fingerprint_rec(tables, subdir, range.clone(), to_store)?.0;
                } else {
                    let start = range.start.strip_prefix(&[component]).unwrap_or_default();
                    let end = range.end.strip_prefix(&[component]).unwrap_or_default();
                    let (child_fp, unrestricted) = fingerprint_rec(tables, subdir, start..end, to_store)?;
                    if unrestricted && !tables.fingerprints.get((subdir, EMPTY_PATH))?.is_some() {
                        to_store.push(((subdir, EMPTY_PATH), child_fp));
                    }
                    res ^= child_fp;
                }
            }
        }
        Ok((res, unrestricted))
    }

    fn fingerprint_rec_reference(
        tables: &Tables,
        dir: INode,
        range: std::ops::Range<Path>,
    ) -> Result<Fingerprint, redb::StorageError> {
        let mut res = Fingerprint::default();
        let mut buf = Vec::new();
        let current_range = if dir == ROOT_INODE {
            complete_range(dir)
        } else {
            // if start is not set, use (dir, EMPTY_PATH), which is the lowest possible path for the given dir, inclusive.
            let start = match range.start.first() {
                Some(component) => (dir, *component),
                None => (dir, EMPTY_PATH),
            };
            // if end is not set, use (dir + 1, EMPTY_PATH), which is the lowest possible path for the next dir, exclusive.
            let end = match range.end.first() {
                Some(component) if range.end.len() == 1 => (dir, *component),
                Some(component) => {
                    let component_exclusive = inc(component, &mut buf);
                    (dir, component_exclusive)
                }
                None => (dir + 1, EMPTY_PATH),
            };
            start..end
        };
        let mut iter = tables.blobs.range(current_range)?;
        while let Some(item) = iter.next() {
            let (k, v) = item?;
            let (_, component) = k.value();
            let v = v.value();
            if let Some(value) = v.value {
                res ^= Fingerprint::from(value);
            }
            if let Some(subdir) = v.dir {
                if dir == ROOT_INODE {
                    // special case for the root directory, 
                    res ^= fingerprint_rec_reference(tables, subdir, range.clone())?;
                } else {
                    let start = range.start.strip_prefix(&[component]).unwrap_or_default();
                    let end = range.end.strip_prefix(&[component]).unwrap_or_default();
                    res ^= fingerprint_rec_reference(tables, subdir, start..end)?;
                }
            }
        }
        Ok(res)
    }

fn to_ref(path: &[impl AsRef<[u8]>]) -> Vec<&[u8]>
{
    let mut t = Vec::with_capacity(path.len());
    for p in path {
        t.push(p.as_ref());
    }
    t
}

impl TreeStore {
    fn memory() -> Self {
        let db = redb::Database::builder()
            .create_with_backend(redb::backends::InMemoryBackend::new())
            .unwrap();
        let mut res = Self {
            db,
            current_transaction: CurrentTransaction::None,
        };
        res.modify(|tables| {
            tables
                .blobs
                .insert((ROOT_INODE, EMPTY_PATH), BlobValue::default())?;
            Ok(())
        })
        .unwrap();
        res
    }

    async fn iter_impl<'a, C>(
        tables: &'a Tables<'a>,
        mut controller: C,
        co: &Co<std::result::Result<C::Item, redb::StorageError>>,
    ) -> std::result::Result<(), redb::StorageError>
    where
        C: IterController<'a>,
    {
        let range = tables.blobs.range(complete_range(ROOT_INODE))?;
        let mut stack = vec![range];
        while let Some(current) = stack.last_mut() {
            match current.next() {
                Some(Ok((k, v))) => {
                    let (_, component) = k.value();
                    let v = v.value();
                    if stack.len() > 1 {
                        controller.set_last(component);
                    }
                    if let Some(value) = controller.item(v.value.as_ref()) {
                        co.yield_(Ok(value)).await;
                    }
                    if let Some(dir) = v.dir {
                        let range = controller.range(dir);
                        let range = tables.blobs.range(range)?;
                        stack.push(range);
                        controller.down();
                    }
                }
                Some(Err(e)) => {
                    co.yield_(Err(e)).await;
                }
                None => {
                    stack.pop();
                    controller.up();
                }
            }
        }
        Ok(())
    }

    fn iter_from<'a>(
        &'a mut self,
        from: Path<'a>,
    ) -> std::result::Result<
        impl IntoIterator<Item = std::result::Result<(OwnedPath, Value), redb::StorageError>> + '_,
        redb::Error,
    > {
        let tables = self.tables()?;
        let controller = RangeFromIterController {
            path_gen: PathGenerator::new(),
            from,
        };
        Ok(Gen::new(|co| async move {
            if let Err(cause) = Self::iter_impl(tables, controller, &co).await {
                co.yield_(Err(cause)).await;
            }
        }))
    }

    fn iter_from_to<'a>(
        &'a mut self,
        from: Path<'a>,
        to: Path<'a>,
    ) -> std::result::Result<
        impl IntoIterator<Item = std::result::Result<(OwnedPath, Value), redb::StorageError>> + '_,
        redb::Error,
    > {
        let tables = self.tables()?;
        let controller = RangeFromToIterController {
            path_gen: PathGenerator::new(),
            from,
            to,
        };
        Ok(Gen::new(|co| async move {
            if let Err(cause) = Self::iter_impl(tables, controller, &co).await {
                co.yield_(Err(cause)).await;
            }
        }))
    }

    fn iter_to<'a>(
        &'a mut self,
        to: Path<'a>,
    ) -> std::result::Result<
        impl IntoIterator<Item = std::result::Result<(OwnedPath, Value), redb::StorageError>> + '_,
        redb::Error,
    > {
        let tables = self.tables()?;
        let controller = RangeFromToIterController {
            path_gen: PathGenerator::new(),
            from: &[],
            to,
        };
        Ok(Gen::new(|co| async move {
            if let Err(cause) = Self::iter_impl(tables, controller, &co).await {
                co.yield_(Err(cause)).await;
            }
        }))
    }

    fn fingerprint(&mut self, range: std::ops::Range<Path>) -> Result<Fingerprint, redb::Error> {
        self.modify(|tables| {
            Ok(fingerprint(tables, range)?)
        })
    }

    fn fingerprint_reference(
        &mut self,
        range: std::ops::Range<Path>,
    ) -> Result<Fingerprint, redb::Error> {
        let mut res = Fingerprint::default();
        for item in self.iter_from_to(range.start, range.end)?.into_iter() {
            let (_, value) = item?;
            res ^= Fingerprint::from(value);
        }
        Ok(res)
    }

    fn iter(
        &mut self,
    ) -> std::result::Result<
        impl IntoIterator<Item = std::result::Result<(OwnedPath, Value), redb::StorageError>> + '_,
        redb::Error,
    > {
        let tables = self.tables()?;
        let controller = AllPathsIterController {
            path_gen: PathGenerator::new(),
        };
        Ok(Gen::new(|co| async move {
            if let Err(cause) = Self::iter_impl(tables, controller, &co).await {
                co.yield_(Err(cause)).await;
            }
        }))
    }

    fn tables(&mut self) -> std::result::Result<&Tables, redb::Error> {
        let guard = &mut self.current_transaction;
        let tables = match std::mem::take(guard) {
            CurrentTransaction::None => {
                let tx = self.db.begin_write()?;
                TransactionAndTables::new(tx)?
            }
            CurrentTransaction::Write(w) => w,
        };
        *guard = CurrentTransaction::Write(tables);
        match guard {
            CurrentTransaction::Write(ref mut tables) => Ok(tables.tables()),
            _ => unreachable!(),
        }
    }

    fn dump(&mut self) -> std::result::Result<(), redb::Error> {
        let tables = self.tables()?;
        for entry in tables.blobs.iter()? {
            let (k, v) = entry?;
            let (inode, path) = k.value();
            let v = v.value();
            println!("{}.{} => {:?}", inode, hex::encode(path), v);
        }
        Ok(())
    }

    fn modify<T>(
        &mut self,
        f: impl FnOnce(&mut Tables) -> Result<T, redb::Error>,
    ) -> Result<T, redb::Error> {
        let guard = &mut self.current_transaction;
        let tables = match std::mem::take(guard) {
            CurrentTransaction::None => {
                let tx = self.db.begin_write()?;
                TransactionAndTables::new(tx)?
            }
            CurrentTransaction::Write(w) => w,
        };
        *guard = CurrentTransaction::Write(tables);
        match guard {
            CurrentTransaction::Write(ref mut tables) => {
                tables.with_tables_mut(|tables| f(tables))
            },
            _ => unreachable!(),
        }
    }

    fn new_inode(
        blobs: &impl ReadableTable<BlobKey<'static>, BlobValue>,
    ) -> Result<INode, redb::Error> {
        let (k, v) = blobs.last()?.unwrap();
        Ok(k.value().0 + 1)
    }

    pub fn get(&mut self, path: &[&[u8]]) -> Result<Option<Value>, redb::Error> {
        let tables = self.tables()?;
        let mut current = tables.blobs.get((ROOT_INODE, EMPTY_PATH))?.unwrap().value();
        for component in path {
            let Some(inode) = current.dir else {
                return Ok(None);
            };
            let Some(entry) = tables.blobs.get((inode, *component))? else {
                return Ok(None);
            };
            current = entry.value();
        }
        let Some(value) = current.value else {
            return Ok(None);
        };
        Ok(Some(value))
    }

    /// Insert a new entry into the blobs table, and invalidate the corresponding fingerprints.
    fn insert_entry(
        tables: &mut Tables,
        path: BlobKey,
        value: BlobValue,
    ) -> Result<(), redb::Error> {
        let prev = tables.blobs.insert(path, value)?;
        if let Some(prev) = prev {
            if prev.value() == value {
                return Ok(());
            }
        }
        let (path_inode, path_path) = path;
        let fp_range = (path_inode, EMPTY_PATH)..=path;
        // Remove all fingerprints that are prefixes of the new path.
        tables
            .fingerprints
            .retain_in(fp_range, |fullpath @ (_, fp_path), _| {
                let keep = !path_path.starts_with(fp_path);
                if !keep {
                    println!("removing fingerprint for {:?}", fullpath);
                }
                keep
            })?;
        Ok(())
    }

    pub fn insert2(&mut self, path: &[impl AsRef<[u8]>], value: Value) -> Result<(), redb::Error> {
        let mut t = Vec::with_capacity(path.len());
        for p in path {
            t.push(p.as_ref());
        }
        self.insert(&t, value)
    }

    pub fn insert(&mut self, path: &[&[u8]], value: Value) -> Result<(), redb::Error> {
        self.modify(|tables| {
            let mut parent = (ROOT_INODE, EMPTY_PATH);
            let mut current = tables.blobs.get(parent)?.unwrap().value();
            for component in path {
                let path = match current.dir {
                    Some(inode) => (inode, *component),
                    None => {
                        let inode = Self::new_inode(&tables.blobs)?;
                        current.dir = Some(inode);
                        // Update the parent entry to point to the new directory.
                        // There is not yet any entry for the new directory, but that
                        // will be created below.
                        Self::insert_entry(tables, parent, current)?;
                        (inode, *component)
                    }
                };
                let entry = tables.blobs.get(path)?.map(|entry| entry.value());
                let entry = match entry {
                    Some(entry) => entry,
                    None => {
                        // This path will be hit either if the directory was just created,
                        // or if the entry is missing from the directory.
                        let res = BlobValue::default();
                        Self::insert_entry(tables, path, res)?;
                        res
                    }
                };
                parent = path;
                current = entry;
            }
            current.value = Some(value);
            Self::insert_entry(tables, parent, current)?;
            Ok(())
        })
    }
}

fn mk_path(i: u64, j: u64) -> [[u8;8];2] {
    [i.to_be_bytes(), j.to_be_bytes()]
}

fn mk_value(i: u64, j: u64) -> Value {
    let mut t = [0u8; 16];
    t[..8].copy_from_slice(&i.to_be_bytes());
    t[8..].copy_from_slice(&j.to_be_bytes());
    blake3::hash(t.as_slice()).into()
}

fn format_path(path: Path) -> String {
    let parts = path.iter().map(|x| format!("{}", hex::encode(x))).collect::<Vec<_>>();
    format!("[{}]", parts.join(","))
}

fn format_range(range: std::ops::Range<Path>) -> String {
    format!("range: {} .. {}", format_path(range.start), format_path(range.end))
}


fn main() -> std::result::Result<(), redb::Error> {
    let mut store = TreeStore::memory();
    let v = [0u8; 32];
    for i in 0..10u64 {
        for j in 0..10u64 {
            store.insert2(&mk_path(i, j), mk_value(i, j))?;
        }
    }
    store.insert(&[b"this", b"is"], v)?;
    store.insert(&[b"this", b"is", b"a"], v)?;
    store.insert(&[b"this", b"is", b"a", b"test"], v)?;
    store.dump().unwrap();
    for item in store.iter()? {
        let (path, value) = item?;
        println!("{:?} => {}", path, hex::encode(value));
    }
    println!("this / is / a");
    for item in store.iter_from(&[b"this", b"is", b"a"])? {
        let (path, value) = item?;
        println!("{:?} => {}", path, hex::encode(value));
    }
    println!("this / is / a .. this / is / a / test");
    for item in store.iter_from_to(&[b"this", b"is", b"a"], &[b"this", b"is", b"a", b"test"])? {
        let (path, value) = item?;
        println!("{:?} => {}", path, hex::encode(value));
    }
    println!("fp");
    let start = mk_path(0, 1);
    let end = mk_path(4, 3);
    let start = to_ref(&start);
    let end = to_ref(&end);
    store.insert(&[b"this", b"is", b"a", b"test"], v)?;

    println!("\nfirst call: {}", format_range(&start..&end));
    println!("fingerprint:\n{}", store.fingerprint(&start .. &end)?);
    println!("fingerprint reference:\n{}", store.fingerprint_reference(&start .. &end)?);

    println!("\nsecond call:");
    println!("fingerprint:\n{}", store.fingerprint(&start .. &end)?);
    println!("fingerprint reference:\n{}", store.fingerprint_reference(&start .. &end)?);

    store.insert(&to_ref(&mk_path(2, 5)), mk_value(2, 1000))?;

    println!("\nfirst call after change:");
    println!("fingerprint:\n{}", store.fingerprint(&start .. &end)?);
    println!("fingerprint reference:\n{}", store.fingerprint_reference(&start .. &end)?);

    println!("\nsecond call after change:");
    println!("fingerprint:\n{}", store.fingerprint(&start .. &end)?);
    println!("fingerprint reference:\n{}", store.fingerprint_reference(&start .. &end)?);
    Ok(())
}
