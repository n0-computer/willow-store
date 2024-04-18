use std::{fmt::Debug, sync::Arc};

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
type Fingerprint = [u8; 32];
type BlobKey<'a> = (INode, &'a [u8]);
type FingerprintKey<'a> = (INode, &'a [u8]);
const BLOBS_TABLE: redb::TableDefinition<BlobKey<'static>, BlobValue> =
    redb::TableDefinition::new("blobs-v0");
const FINGERPRINTS_TABLE: redb::TableDefinition<FingerprintKey<'static>, Fingerprint> =
    redb::TableDefinition::new("fingerprints-v0");

type Path<'a> = &'a [&'a [u8]];

#[derive(Clone)]
struct OwnedPath(Arc<Vec<Bytes>>);

impl Default for OwnedPath {
    fn default() -> Self {
        Self(Arc::new(vec![Bytes::new()]))
    }
}

impl Debug for OwnedPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.as_ref().iter().map(|x| hex::encode(x))).finish()
    }

}

impl AsRef<[Bytes]> for OwnedPath {
    fn as_ref(&self) -> &[Bytes] {
        &self.0[1..]
    }
}

impl OwnedPath {
    fn get_mut(&mut self) -> &mut Vec<Bytes> {
        Arc::make_mut(&mut self.0)
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
struct BlobValue {
    value: Option<Value>,
    dir: Option<INode>,
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
        self.current_path
            .get_mut()
            .push(component);
    }

    fn path(&self) -> OwnedPath {
        self.current_path.clone()
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

pub struct TransactionAndTables {
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

trait IterController<'a> {
    type Item;
    fn down(&mut self);
    fn up(&mut self);
    fn set(&mut self, component: &[u8]);
    fn range(&self, inode: INode) -> std::result::Result<redb::Range<'a, BlobKey<'static>, BlobValue>, redb::StorageError>;
    fn item(&self, value: &Value) -> Self::Item;
}

struct AllPathsIterController<'a> {
    path_gen: PathGenerator,
    tables: &'a Tables<'a>,
}

impl<'a> IterController<'a> for AllPathsIterController<'a> {
    type Item = (OwnedPath, Value);

    fn down(&mut self) {
        self.path_gen.push(EMPTY_PATH);
    }

    fn up(&mut self) {
        self.path_gen.pop();
    }

    fn set(&mut self, component: &[u8]) {
        self.path_gen.set_last(component);
    }

    fn range(&self, inode: INode) -> std::result::Result<redb::Range<'a, BlobKey<'static>, BlobValue>, redb::StorageError> {
        Ok(self.tables.blobs.range(complete_range(inode))?)
    }

    fn item(&self, value: &Value) -> Self::Item {
        (self.path_gen.path(), *value)
    }

}

impl TreeStore {
    fn memory() -> Self {
        let db = redb::Database::builder()
            .create_with_backend(redb::backends::InMemoryBackend::new())
            .unwrap();
        let mut res = Self { db, current_transaction: CurrentTransaction::None };
        res.modify(|tables| {
            tables
                .blobs
                .insert((ROOT_INODE, EMPTY_PATH), BlobValue::default())?;
            Ok(())
        })
        .unwrap();
        res
    }

    async fn iter_inner_2<'a, C>(mut controller: C, co: &Co<std::result::Result<C::Item, redb::StorageError>>) -> std::result::Result<(), redb::StorageError>
        where C: IterController<'a>
    {
        let range = controller.range(ROOT_INODE)?;
        let mut stack = vec![range];
        while let Some(current) = stack.last_mut() {
            match current.next() {
                Some(Ok((k, v))) => {
                    let (_, component) = k.value();
                    let v = v.value();
                    controller.set(component);
                    if let Some(value) = v.value.as_ref() {
                        co.yield_(Ok(controller.item(value))).await;
                    }
                    if let Some(dir) = v.dir {
                        let range = controller.range(dir)?;
                        stack.push(range);
                        controller.down();
                    }
                }
                Some(Err(e)) => {
                    co.yield_(Err(e)).await;
                },
                None => {
                    stack.pop();
                    controller.up();
                },
            }
        }
        Ok(())
    }


    async fn iter_inner(tables: &Tables<'_>, co: &Co<std::result::Result<(OwnedPath, Value), redb::StorageError>>) -> std::result::Result<(), redb::StorageError> {
        let range = tables.blobs.range(complete_range(ROOT_INODE))?;
        let mut path_gen = PathGenerator::new();
        let mut stack = vec![range];
        while let Some(current) = stack.last_mut() {
            match current.next() {
                Some(Ok((k, v))) => {
                    let (_, component) = k.value();
                    let v = v.value();
                    path_gen.set_last(component);
                    if let Some(value) = v.value {
                        co.yield_(Ok((path_gen.path(), value))).await;
                    }
                    if let Some(dir) = v.dir {
                        let range = tables.blobs.range(complete_range(dir))?;
                        stack.push(range);
                        path_gen.push(EMPTY_PATH);
                    }
                }
                Some(Err(e)) => {
                    co.yield_(Err(e)).await;
                },
                None => {
                    stack.pop();
                    path_gen.pop();
                },
            }
        }
        Ok(())
    }

    fn iter2(&mut self) -> std::result::Result<impl IntoIterator<Item = std::result::Result<(OwnedPath, Value), redb::StorageError>> + '_, redb::Error> {
        let tables = self.tables()?;
        let controller = AllPathsIterController { path_gen: PathGenerator::new(), tables };
        Ok(Gen::new(|co| async move {
            if let Err(cause) = Self::iter_inner_2(controller, &co).await {
                co.yield_(Err(cause)).await;
            }
        }))
    }

    fn iter(&mut self) -> std::result::Result<impl IntoIterator<Item = std::result::Result<(OwnedPath, Value), redb::StorageError>> + '_, redb::Error> {
        let tables = self.tables()?;
        Ok(Gen::new(|co| async move {
            if let Err(cause) = Self::iter_inner(tables, &co).await {
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
            CurrentTransaction::Write(w) => {
                w
            }
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

    fn modify(
        &mut self,
        f: impl FnOnce(&mut Tables) -> Result<(), redb::Error>,
    ) -> Result<(), redb::Error> {
        let txn = self.db.begin_write()?;
        let mut tables = Tables::new(&txn)?;
        let res = f(&mut tables);
        drop(tables);
        if res.is_ok() {
            txn.commit()?;
        } else {
            txn.abort()?;
        }
        res
    }

    fn new_inode(
        blobs: &impl ReadableTable<BlobKey<'static>, BlobValue>,
    ) -> Result<INode, redb::Error> {
        let (k, v) = blobs.last()?.unwrap();
        Ok(k.value().0 + 1)
    }

    fn get(&mut self, path: &[&[u8]]) -> Result<Option<Value>, redb::Error> {
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
        tables.blobs.insert(path, value)?;
        tables.fingerprints.remove((path.0, EMPTY_PATH))?;
        Ok(())
    }

    fn insert(&mut self, path: &[&[u8]], value: Value) -> Result<(), redb::Error> {
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

fn main() -> std::result::Result<(), redb::Error> {
    let mut store = TreeStore::memory();
    for i in 0..3u64 {
        for j in 0..3u64 {
            store
                .insert(&[&i.to_be_bytes(), &j.to_be_bytes()], [0u8; 32])?;
        }
    }
    store
        .insert(&[b"this", b"is", b"a", b"test"], [0u8; 32])?;
    store.dump().unwrap();
    for item in store.iter()? {
        let (path, value) = item?;
        println!("{:?} => {:?}", path, value);
    }
    for item in store.iter2()? {
        let (path, value) = item?;
        println!("{:?} => {:?}", path, value);
    }
    Ok(())
}
