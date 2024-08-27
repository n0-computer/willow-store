use std::{borrow::Borrow, marker::PhantomData, num::NonZeroU64, ops::Deref, sync::Arc};

use bytes::Bytes;
use redb::{AccessGuard, ReadTransaction, ReadableTable, WriteTransaction};
use tracing::{info, trace};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

type RedbResult<T> = std::result::Result<T, redb::Error>;

trait StoreParams {
    type Value;
    type Key;
    type Summary;
}

const EMPTY_PATH: &'static [u8] = &[];
const ROOT_INODE: u64 = 0;

const ESCAPE: u8 = 1;
const SEPARATOR: u8 = 0;

type Id = [u8; 8];

// trait Summary<T>: Default {
//     fn combine_value(&mut self, other: &T);
//     fn combine(&mut self, other: &Self);
// }

// trait TreeParams {
//     // max unescaped component length
//     const MAX_COMPONENT_LEN: usize = 256;
//     // max number of components in a path
//     const MAX_COMPONENTS: usize = 32;
//     // 2 bytes per byte, plus one byte for the separator for each component
//     const MAX_ESCAPED_SIZE: usize = Self::MAX_COMPONENTS * Self::MAX_COMPONENT_LEN * 2 + Self::MAX_COMPONENTS;
//     type Value: FromBytes + AsBytes + Copy;
//     type Summary: FromBytes + AsBytes + Copy + Summary<Self::Value>;
// }

#[repr(C)]
#[derive(Clone, Copy, Default, FromZeroes, FromBytes, AsBytes, PartialEq, Eq)]
struct Entry {
    // first character of the prefix, for cheap lookup in case prefix is not inlined
    first_char: u8,
    // if prefix_len <= 8, the prefix is stored inline, otherwise it is an id
    prefix_len: u8,
    // if has_value is 1, the entry has a value
    has_value: u8,
    // inline prefix or prefix id
    prefix: Id,
    // children id
    children: Id,
}

impl std::fmt::Debug for Entry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry")
            .field(
                "prefix",
                &self.prefix_id().map_err(|p| (self.first_char, p)),
            )
            .field("children", &self.children())
            .field("has_value", &(self.has_value != 0))
            .finish()
    }
}

/// For each directory, there can be at most 256 children, each of which can
/// have a value. So we compute the value id from the directory id and the
/// first character of the prefix.
fn value_id(dir: NonZeroU64, first_char: u8) -> NonZeroU64 {
    NonZeroU64::new(dir.get() * 256 + first_char as u64).unwrap()
}

impl Entry {
    fn children(&self) -> Option<NonZeroU64> {
        NonZeroU64::new(u64::from_le_bytes(self.children))
    }

    fn set_children(&mut self, children: Option<NonZeroU64>) {
        self.children = children.map(|x| x.get()).unwrap_or_default().to_le_bytes();
    }

    fn value_id(&self, dir: NonZeroU64) -> NonZeroU64 {
        value_id(dir, self.first_char)
    }

    fn value(&self, dir: NonZeroU64) -> Option<NonZeroU64> {
        if self.has_value != 0 {
            Some(self.value_id(dir))
        } else {
            None
        }
    }

    fn set_has_value(&mut self, has_value: bool) {
        self.has_value = has_value as u8;
    }

    fn prefix_id(&self) -> std::result::Result<&[u8], NonZeroU64> {
        if self.prefix_len <= 8 {
            Ok(self.prefix[0..self.prefix_len as usize].try_into().unwrap())
        } else {
            let id = u64::from_le_bytes(self.prefix);
            Err(NonZeroU64::new(id).unwrap())
        }
    }
}

struct RedbStore {
    db: redb::Database,
    current_transaction: CurrentTransaction,
    next_dir_id: NonZeroU64,
}

impl RedbStore {
    fn memory() -> Self {
        let db = redb::Database::builder()
            .create_with_backend(redb::backends::InMemoryBackend::new())
            .unwrap();
        let mut res = Self {
            db,
            current_transaction: CurrentTransaction::None,
            next_dir_id: NonZeroU64::new(1).unwrap(),
        };
        res.modify(|tables| {
            tables.paths.insert(1, EMPTY_PATH)?;
            Ok(())
        })
        .unwrap();
        let next_dir_id = NonZeroU64::new(
            res.tables()
                .unwrap()
                .paths
                .last()
                .unwrap()
                .unwrap()
                .0
                .value()
                + 1,
        )
        .unwrap();
        res.next_dir_id = next_dir_id;
        res
    }

    fn traverse(&mut self) -> RedbResult<()> {
        self.traverse_rec(NonZeroU64::new(1).unwrap(), &[])
    }

    fn traverse_rec(&mut self, dir: NonZeroU64, path: &[u8]) -> RedbResult<()> {
        let entries = self.get_dir(dir)?.into_owned();
        let mut buf = Vec::new();
        for entry in entries {
            let prefix = entry.prefix(self)?;
            buf.clear();
            buf.extend_from_slice(path);
            buf.extend_from_slice(prefix.as_ref());
            drop(prefix);
            let value = entry.value(dir);
            if let Some(value) = value {
                let value = self.get_value(value.get())?;
                println!("{}: {:?}", hex::encode(&buf), value.value());
            }
            if let Some(dir) = entry.children() {
                self.traverse_rec(dir, &buf)?;
            }
        }
        Ok(())
    }

    fn dump(&mut self) -> RedbResult<()> {
        let tables = self.tables()?;
        println!("paths:");
        for item in tables.paths.iter()? {
            let (k, v) = item?;
            let v = Entry::slice_from(v.value()).unwrap();
            println!("{}: {:?}", k.value(), v);
        }
        println!("prefixes:");
        for item in tables.prefixes.iter()? {
            let (k, v) = item?;
            println!("{}: {:?}", k.value(), hex::encode(v.value()));
        }
        println!("values:");
        for item in tables.values.iter()? {
            let (k, v) = item?;
            println!("{}: {:?}", k.value(), v.value());
        }
        println!("summaries:");
        for item in tables.summaries.iter()? {
            let (k, v) = item?;
            println!("{}: {:?}", k.value(), v.value());
        }

        Ok(())
    }

    fn snapshot(&mut self) -> RedbResult<ReadonlyTables> {
        let guard = &mut self.current_transaction;
        match std::mem::take(guard) {
            CurrentTransaction::Read(tx, _tables) => {
                Ok(ReadonlyTables::new(&tx)?)
            }
            CurrentTransaction::Write(w) => {
                todo!()
            }
            CurrentTransaction::None => {
                let tx = self.db.begin_read()?;
                Ok(ReadonlyTables::new(&tx)?)
            }
        }
    }

    fn tables(&mut self) -> RedbResult<&Tables> {
        let guard = &mut self.current_transaction;
        let tables = match std::mem::take(guard) {
            CurrentTransaction::None | CurrentTransaction::Read(_, _) => {
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

    fn modify<T>(
        &mut self,
        f: impl FnOnce(&mut Tables) -> RedbResult<T>,
    ) -> RedbResult<T> {
        let guard = &mut self.current_transaction;
        let tables = match std::mem::take(guard) {
            CurrentTransaction::None | CurrentTransaction::Read(_, _) => {
                let tx = self.db.begin_write()?;
                TransactionAndTables::new(tx)?
            }
            CurrentTransaction::Write(w) => w,
        };
        *guard = CurrentTransaction::Write(tables);
        match guard {
            CurrentTransaction::Write(ref mut tables) => tables.with_tables_mut(|tables| f(tables)),
            _ => unreachable!(),
        }
    }
}

struct EntriesGuard<'a>(AccessGuard<'a, &'static [u8]>);

impl Deref for EntriesGuard<'_> {
    type Target = [Entry];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl EntriesGuard<'_> {
    fn to_buffer(self, target: &mut Vec<Entry>) -> &mut Vec<Entry> {
        target.clear();
        target.extend_from_slice(self.as_ref());
        target
    }

    fn into_owned(self) -> Vec<Entry> {
        self.as_ref().to_vec()
    }
}

impl<'a> AsRef<[Entry]> for EntriesGuard<'a> {
    fn as_ref(&self) -> &[Entry] {
        let data = self.0.value();
        Entry::slice_from(data).unwrap()
    }
}

struct OwnedX<X: FromBytes>(Bytes, PhantomData<X>);

impl<X: FromBytes> AsRef<X> for OwnedX<X> {
    fn as_ref(&self) -> &X {
        X::ref_from(&self.0).unwrap()
    }
}

struct PrefixGuard<'a>(std::result::Result<(u8, [u8; 8]), AccessGuard<'a, &'static [u8]>>);

impl PrefixGuard<'_> {
    fn to_buffer(self, target: &mut Vec<u8>) -> &mut Vec<u8> {
        target.clear();
        target.extend_from_slice(self.as_ref());
        target
    }

    fn into_owned(self) -> Vec<u8> {
        self.as_ref().to_vec()
    }
}

impl Deref for PrefixGuard<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a> AsRef<[u8]> for PrefixGuard<'a> {
    fn as_ref(&self) -> &[u8] {
        match &self.0 {
            Ok((size, inline)) => &inline[..*size as usize],
            Err(guard) => guard.value(),
        }
    }
}

impl Entry {
    fn prefix<'a>(
        &self,
        store: &'a mut RedbStore,
    ) -> RedbResult<PrefixGuard<'a>> {
        if self.prefix_len <= 8 {
            Ok(PrefixGuard(Ok((self.prefix_len, self.prefix))))
        } else {
            let id = u64::from_le_bytes(self.prefix);
            store.get_prefix(id)
        }
    }
}

/// Key for iteration of radixtree prefixes
///
/// This uses copy on write to allow iterating over all prefixes without allocations, provided that the keys are not stored somewhere.
#[derive(Debug, Clone, Default)]
pub struct IterKey(Arc<Vec<u8>>);

impl IterKey {
    fn new(root: &[u8]) -> Self {
        Self(Arc::new(root.to_vec()))
    }

    fn append(&mut self, data: &[u8]) {
        // for typical iterator use, a reference is not kept for a long time, so this will be very cheap
        //
        // in the case a reference is kept, this will make a copy.
        let elems = Arc::make_mut(&mut self.0);
        elems.extend_from_slice(data);
    }

    fn pop(&mut self, n: usize) {
        let elems = Arc::make_mut(&mut self.0);
        elems.truncate(elems.len().saturating_sub(n));
    }
}

impl AsRef<[u8]> for IterKey {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl Borrow<[u8]> for IterKey {
    fn borrow(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl Deref for IterKey {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

struct KeyValueIter {
    path: IterKey,
    tables: ReadonlyTables,
    stack: Vec<(NonZeroU64, usize)>,
}

impl KeyValueIter {
    fn empty(tables: ReadonlyTables) -> Self {
        Self {
            stack: Vec::new(),
            path: IterKey::default(),
            tables,
        }
    }
}

impl RedbStore {
    fn get_raw(&mut self, path: &[u8]) -> RedbResult<Option<Value>> {
        let root = NonZeroU64::new(1).unwrap();
        self.get_rec(root, path)
    }

    fn get_rec(&mut self, dir: NonZeroU64, path: &[u8]) -> RedbResult<Option<Value>> {
        assert!(!path.is_empty());
        let entries = self.get_dir(dir)?;
        if let Ok(index) = entries.binary_search_by_key(&path[0], |x| x.first_char) {
            let entry = entries[index];
            drop(entries);
            let entry_prefix_guard = entry.prefix(self)?;
            let entry_prefix = entry_prefix_guard.as_ref();
            let entry_prefix_len = entry_prefix.len();
            let n = common_prefix(entry_prefix, path);
            drop(entry_prefix_guard);
            if n == path.len() && n == entry_prefix_len {
                // found the value
                if let Some(value) = entry.value(dir) {
                    let value = self.get_value(value.get())?;
                    return Ok(Some(value.value()));
                }
            } else if n == entry_prefix_len {
                // entry prefix is a prefix of path, recurse
                if let Some(subdir) = entry.children() {
                    return self.get_rec(subdir, &path[n..]);
                }
            }
        }
        Ok(None)
    }

    fn get<P, C>(&mut self, path: P) -> RedbResult<Option<Value>>
    where
        P: AsRef<[C]>,
        C: AsRef<[u8]>,
    {
        let path = escape(path);
        info!(get = ?hex::encode(&path));
        self.get_raw(&path)
    }

    fn range_summary<P, C>(&mut self, start: P, end: Option<P>) -> RedbResult<Summary>
    where
        P: AsRef<[C]>,
        C: AsRef<[u8]>
    {
        let start = escape(start);
        let end = end.map(escape);
        let mut summary = Summary::default();
        Ok(summary)
    }

    fn range_iter<P, C>(&self, start: P, end: Option<P>) -> impl Iterator<Item = RedbResult<(Path, Value)>> + 'static
    where
        P: AsRef<[C]>,
        C: AsRef<[u8]>
    {
        std::iter::empty()
    }

    fn iter(&mut self) -> RedbResult<impl Iterator<Item = RedbResult<(Path, Value)>> + 'static> {
        let snapshot = self.snapshot().unwrap();
        let root = NonZeroU64::new(1).unwrap();
        Ok(std::iter::empty())
    }

    fn insert<P, C>(&mut self, path: P, value: Value) -> RedbResult<()>
    where
        P: AsRef<[C]>,
        C: AsRef<[u8]>,
    {
        let path = escape(path);
        tracing::info!("insert {:?}", hex::encode(&path));
        self.insert_raw(&path, value)
    }

    fn insert_raw(&mut self, path: &[u8], value: Value) -> RedbResult<()> {
        let root = NonZeroU64::new(1).unwrap();
        let mut buffers = Buffers::default();
        self.insert_rec(root, path, value, &mut buffers)?;
        Ok(())
    }

    fn get_or_create_subdir(
        &mut self,
        dir: NonZeroU64,
        entries: EntriesGuard,
        index: usize,
    ) -> RedbResult<NonZeroU64> {
        let entry = &entries.as_ref()[index];
        if let Some(dir) = entry.children() {
            return Ok(dir);
        } else {
            let mut entries1 = entries.as_ref().to_vec();
            let subdir = self.next_dir_id()?;
            self.put_children(subdir, &[])?;
            entries1[index].set_children(Some(subdir));
            self.put_children(dir, &entries1)?;
            Ok(subdir)
        }
    }

    fn insert_rec(
        &mut self,
        dir: NonZeroU64,
        path: &[u8],
        value: Value,
        buffers: &mut Buffers,
    ) -> RedbResult<bool> {
        assert!(!path.is_empty());
        let entries = self.get_dir(dir)?;
        let res = match entries.binary_search_by_key(&path[0], |x| x.first_char) {
            Ok(index) => {
                let entries = entries.to_buffer(&mut buffers.entries);
                let entry = entries[index];
                let entry_prefix = entry.prefix(self)?;
                let ep = entry_prefix.as_ref();
                let n = common_prefix(ep, path);
                if n == path.len() && n == ep.len() {
                    // replace the value
                    drop(entry_prefix);
                    self.put_value(entry.value_id(dir), Some(&value))?;
                    if entry.has_value == 0 {
                        entries[index].set_has_value(true);
                        self.put_children(dir, &entries)?;
                    }
                    true
                } else if n == ep.len() {
                    // entry prefix is a prefix of path
                    drop(entry_prefix);
                    // ensure that the entry has a subdir
                    let subdir = if let Some(subdir) = entries[index].children() {
                        subdir
                    } else {
                        // create an empty subdir
                        let subdir = self.next_dir_id()?;
                        self.put_children(subdir, &[])?;
                        // update the entry
                        entries[index].set_children(Some(subdir));
                        // save the updated entries
                        self.put_children(dir, &entries)?;
                        subdir
                    };
                    // recurse
                    self.insert_rec(subdir, &path[n..], value, buffers)?
                } else if n == path.len() {
                    // path is a prefix of entry prefix
                    let mut entry0 = entry;
                    let src = (dir, entry0.first_char);
                    let entry_prefix = entry_prefix.to_buffer(&mut buffers.entry_path);
                    self.set_prefix(&mut entry0, &entry_prefix[n..])?;
                    let subdir = self.next_dir_id()?;
                    let tgt = (subdir, entry0.first_char);
                    if entry0.has_value != 0 {
                        self.move_child(src, tgt)?;
                    }
                    self.put_children(subdir, &[entry0])?;
                    entries[index] = self.create_entry(dir, path, Some(&value), Some(subdir))?;
                    self.put_children(dir, &entries)?;
                    true
                } else {
                    // disjoint prefixes
                    let mut entry0 = entries[index];
                    let src = (dir, entry0.first_char);
                    let entry_prefix = entry_prefix.to_buffer(&mut buffers.entry_path);
                    let common = &path[..n];
                    let subdir = self.next_dir_id()?;
                    let entry1 = self.create_entry(subdir, &path[n..], Some(&value), None)?;
                    self.set_prefix(&mut entry0, &entry_prefix[n..])?;
                    let tgt = (subdir, entry0.first_char);
                    if entry0.has_value != 0 {
                        self.move_child(src, tgt)?;
                    }
                    let mut children = [entry0, entry1];
                    children.sort_by_key(|x| x.first_char);
                    self.put_children(subdir, &children)?;
                    entries[index] = self.create_entry(dir, common, None, Some(subdir))?;
                    self.put_children(dir, &entries)?;
                    true
                }
            }
            Err(ip) => {
                // no entry with the same first character
                let entries = entries.to_buffer(&mut buffers.entries);
                let entry1 = self.create_entry(dir, path, Some(&value), None)?;
                entries.insert(ip, entry1);
                self.put_children(dir, &entries)?;
                true
            }
        };
        if res {
            // invalidate the summary
            self.modify(|tables| {
                tables.summaries.remove(dir.get())?;
                Ok(())
            })?;
        }
        Ok(res)
    }

    fn move_child(
        &mut self,
        from: (NonZeroU64, u8),
        to: (NonZeroU64, u8),
    ) -> RedbResult<()> {
        let from_id = value_id(from.0, from.1).get();
        let to_id = value_id(to.0, to.1).get();
        trace!("move_child {} {}", from_id, to_id);
        self.modify(|tables| {
            let Some(guard) = tables.values.remove(from_id)? else {
                return Ok(());
            };
            let value = guard.value();
            drop(guard);
            tables.values.insert(to_id, value)?;
            Ok(())
        })
    }

    fn set_prefix(&mut self, entry: &mut Entry, value: &[u8]) -> RedbResult<()> {
        entry.first_char = value[0];
        let old_id = entry.prefix_id().err();
        if value.len() <= 8 {
            self.put_prefix(old_id, None)?;
            entry.prefix_len = value.len() as u8;
            entry.prefix = [0; 8];
            entry.prefix[..value.len()].copy_from_slice(value);
        } else {
            let new_id = self.put_prefix(old_id, Some(value))?.unwrap();
            entry.prefix_len = 9;
            entry.prefix = new_id.get().to_le_bytes();
        }
        Ok(())
    }

    fn set_children(
        &mut self,
        entry: &mut Entry,
        children: Option<NonZeroU64>,
    ) -> RedbResult<()> {
        entry.children = children.map(|x| x.get()).unwrap_or_default().to_le_bytes();
        Ok(())
    }

    fn create_entry(
        &mut self,
        dir: NonZeroU64,
        path: &[u8],
        value: Option<&Value>,
        children: Option<NonZeroU64>,
    ) -> RedbResult<Entry> {
        let mut res = Entry::default();
        self.set_prefix(&mut res, path)?;
        if let Some(value) = value {
            res.set_has_value(true);
            self.put_value(value_id(dir, path[0]), Some(value))?;
        };
        self.set_children(&mut res, children)?;
        Ok(res)
    }

    fn put_children(&mut self, id: NonZeroU64, data: &[Entry]) -> RedbResult<()> {
        self.modify(|tables| {
            let data = <[Entry]>::as_bytes(data);
            let mut guard = tables.paths.insert_reserve(id.get(), data.len() as u32)?;
            guard.as_mut().copy_from_slice(data);
            Ok(())
        })
    }

    fn next_dir_id(&mut self) -> RedbResult<NonZeroU64> {
        let res = self.next_dir_id;
        self.next_dir_id = NonZeroU64::new(res.get() + 1).unwrap();
        Ok(res)
    }

    fn get_dir(&mut self, id: NonZeroU64) -> RedbResult<EntriesGuard> {
        let Some(value) = self.tables()?.paths.get(id.get())? else {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        };
        Ok(EntriesGuard(value))
    }

    fn rm_dir(&mut self, id: u64) -> RedbResult<()> {
        self.modify(|tables| {
            tables.paths.remove(id)?;
            tables.summaries.remove(id)?;
            Ok(())
        })
    }

    fn put_prefix(
        &mut self,
        key: Option<NonZeroU64>,
        data: Option<&[u8]>,
    ) -> RedbResult<Option<NonZeroU64>> {
        self.modify(|tables: &mut Tables| match (key, data) {
            (Some(key), Some(data)) => {
                tables
                    .prefixes
                    .insert_reserve(key.get(), data.len() as u32)?
                    .as_mut()
                    .copy_from_slice(data);
                Ok(Some(key))
            }
            (Some(key), None) => {
                tables.prefixes.remove(key.get())?;
                Ok(None)
            }
            (None, Some(data)) => {
                let id = tables
                    .prefixes
                    .last()?
                    .map(|(k, _)| k.value())
                    .unwrap_or_default()
                    + 1;
                tables.prefixes.insert(id, data)?;
                Ok(Some(id.try_into().unwrap()))
            }
            (None, None) => Ok(None),
        })
    }

    fn get_prefix(&mut self, id: u64) -> RedbResult<PrefixGuard> {
        let Some(value) = self.tables()?.prefixes.get(id)? else {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        };
        Ok(PrefixGuard(Err(value)))
    }

    fn del_prefix(&mut self, id: u64) -> RedbResult<()> {
        self.modify(|tables: &mut Tables| {
            tables.prefixes.remove(id)?;
            Ok(())
        })
    }

    fn put_value(
        &mut self,
        id: NonZeroU64,
        value: Option<&Value>,
    ) -> RedbResult<PutValueResult> {
        trace!("put_value {} {:?}", id, value.is_some());
        self.modify(|tables| match (id, value) {
            (id, Some(value)) => {
                let prev = tables.values.insert(id.get(), value)?;
                if let Some(prev) = prev {
                    if prev.value() == *value {
                        return Ok(PutValueResult::Unchanged(id));
                    }
                }
                Ok(PutValueResult::Updated(id))
            }
            (id, None) => {
                let res = tables.values.remove(id.get())?;
                if res.is_some() {
                    Ok(PutValueResult::Deleted(id))
                } else {
                    Ok(PutValueResult::NotPresent)
                }
            }
        })
    }

    fn get_value(&mut self, id: u64) -> RedbResult<AccessGuard<Value>> {
        let Some(value) = self.tables()?.values.get(id)? else {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        };
        Ok(value)
    }

    fn del_value(&mut self, id: u64) -> RedbResult<()> {
        self.modify(|tables| {
            tables.values.remove(id)?;
            Ok(())
        })
    }

    fn put_summary(&mut self, data: &Summary) -> RedbResult<u64> {
        self.modify(|tables| {
            let id = tables
                .summaries
                .last()?
                .map(|(k, _)| k.value())
                .unwrap_or_default()
                + 1;
            tables.summaries.insert(id, data)?;
            Ok(id)
        })
    }

    fn get_summary(&mut self, id: u64) -> RedbResult<AccessGuard<Summary>> {
        let Some(value) = self.tables()?.summaries.get(id)? else {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        };
        Ok(value)
    }

    fn del_summary(&mut self, id: u64) -> RedbResult<()> {
        self.modify(|tables| {
            tables.summaries.remove(id)?;
            Ok(())
        })
    }
}

#[derive(Debug, Default)]
struct Buffers {
    entries: Vec<Entry>,
    entry_path: Vec<u8>,
}

enum PutValueResult {
    Inserted(NonZeroU64),
    Updated(NonZeroU64),
    Deleted(NonZeroU64),
    Unchanged(NonZeroU64),
    NotPresent,
}

impl PutValueResult {
    fn current(&self) -> Option<NonZeroU64> {
        match self {
            PutValueResult::Inserted(id) => Some(*id),
            PutValueResult::Updated(id) => Some(*id),
            PutValueResult::Unchanged(id) => Some(*id),
            PutValueResult::Deleted(id) => None,
            PutValueResult::NotPresent => None,
        }
    }
}

#[derive(Debug, Default)]
struct Summary {
    min: [u8; 8],
    max: [u8; 8],
}

#[derive(Debug, Clone, Default)]
struct WillowSummary {
    timestamp_range: std::ops::Range<u64>,
    fingerprint: [u8; 32],
}

impl redb::Value for Value {
    type SelfType<'a> = Value;
    type AsBytes<'a> = [u8; 8];

    fn fixed_width() -> Option<usize> {
        Some(8)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        let data: [u8; 8] = data.try_into().unwrap();
        Self(data)
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a> {
        value.0
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("Value")
    }
}

impl redb::Value for WillowValue {
    type SelfType<'a> = WillowValue;

    type AsBytes<'a> = [u8; 48];

    fn fixed_width() -> Option<usize> {
        Some(48)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self
    where
        Self: 'a,
    {
        let data: [u8; 48] = data.try_into().unwrap();
        let timestamp = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let fingerprint = data[8..40].try_into().unwrap();
        let size = u64::from_le_bytes(data[40..48].try_into().unwrap());
        Self {
            timestamp,
            fingerprint,
            size,
        }
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'a,
        Self: 'b,
    {
        let mut data = [0; 48];
        data[0..8].copy_from_slice(&value.timestamp.to_le_bytes());
        data[8..40].copy_from_slice(&value.fingerprint);
        data[40..48].copy_from_slice(&value.size.to_le_bytes());
        data
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("Value")
    }
}


impl redb::Value for Summary {
    type SelfType<'a> = Summary;
    type AsBytes<'a> = [u8; 16];

    fn fixed_width() -> Option<usize> {
        Some(16)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        let data: [u8; 16] = data.try_into().unwrap();
        let min = data[0..8].try_into().unwrap();
        let max = data[8..16].try_into().unwrap();
        Self { min, max }
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a> {
        let mut data = [0; 16];
        data[0..8].copy_from_slice(&value.min);
        data[8..16].copy_from_slice(&value.max);
        data
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("Summary")
    }
}

impl redb::Value for WillowSummary {
    type SelfType<'a> = WillowSummary;
    type AsBytes<'a> = [u8; 48];

    fn fixed_width() -> Option<usize> {
        Some(48)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        let data: [u8; 48] = data.try_into().unwrap();
        let start = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let end = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let fingerprint = data[16..48].try_into().unwrap();
        Self {
            timestamp_range: start..end,
            fingerprint,
        }
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a> {
        let mut data = [0; 48];
        data[0..8].copy_from_slice(&value.timestamp_range.start.to_le_bytes());
        data[8..16].copy_from_slice(&value.timestamp_range.end.to_le_bytes());
        data[16..48].copy_from_slice(&value.fingerprint);
        data
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("Summary")
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct WillowValue {
    timestamp: u64,
    fingerprint: [u8; 32],
    size: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Value([u8; 8]);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RawValue<const C: usize = 48>([u8; C]);

type INode = u64;
type PathKey<'a> = (u64, &'a [u8]);

const PATHS_TABLE: redb::TableDefinition<u64, &[u8]> = redb::TableDefinition::new("paths-v0");
const VALUES_TABLE: redb::TableDefinition<u64, Value> = redb::TableDefinition::new("values-v0");
const SUMMARIES_TABLE: redb::TableDefinition<u64, Summary> =
    redb::TableDefinition::new("summaries-v0");
const PREFIXES_TABLE: redb::TableDefinition<u64, &[u8]> = redb::TableDefinition::new("prefixes-v0");

struct Tables<'txn> {
    paths: redb::Table<'txn, u64, &'static [u8]>,
    values: redb::Table<'txn, u64, Value>,
    summaries: redb::Table<'txn, u64, Summary>,
    prefixes: redb::Table<'txn, u64, &'static [u8]>,
}

#[derive(Debug)]
struct ReadonlyTables {
    paths: redb::ReadOnlyTable<u64, &'static [u8]>,
    values: redb::ReadOnlyTable<u64, Value>,
    summaries: redb::ReadOnlyTable<u64, Summary>,
    prefixes: redb::ReadOnlyTable<u64, &'static [u8]>,
}

impl ReadonlyTables {
    fn new(tx: &ReadTransaction) -> std::result::Result<Self, redb::TableError> {
        Ok(Self  {
            paths: tx.open_table(PATHS_TABLE)?,
            values: tx.open_table(VALUES_TABLE)?,
            summaries: tx.open_table(SUMMARIES_TABLE)?,
            prefixes: tx.open_table(PREFIXES_TABLE)?,
        })
    }
}

impl<'a> Tables<'a> {
    fn new(txn: &'a WriteTransaction) -> std::result::Result<Self, redb::TableError> {
        Ok(Self {
            paths: txn.open_table(PATHS_TABLE)?,
            values: txn.open_table(VALUES_TABLE)?,
            summaries: txn.open_table(SUMMARIES_TABLE)?,
            prefixes: txn.open_table(PREFIXES_TABLE)?,
        })
    }
}

impl std::fmt::Debug for TransactionAndTables {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransactionAndTables").finish()
    }
}

self_cell::self_cell! {
    struct TransactionAndTablesInner {
        owner: WriteTransaction,
        #[covariant]
        dependent: Tables,
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
    Read(ReadTransaction, ReadonlyTables),
}

// common prefix of two slices.
fn common_prefix<'a, T: Eq>(a: &'a [T], b: &'a [T]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

fn escape<P, C>(path: P) -> Vec<u8>
where
    P: AsRef<[C]>,
    C: AsRef<[u8]>,
{
    let mut result = Vec::new();
    result.push(0);
    for segment in path.as_ref() {
        for &byte in segment.as_ref() {
            match byte {
                ESCAPE => result.extend([ESCAPE, ESCAPE]),
                SEPARATOR => result.extend([ESCAPE, SEPARATOR]),
                _ => result.push(byte),
            }
        }
        result.push(SEPARATOR);
    }
    result
}

fn unescape(path: &[u8]) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    let mut segment = Vec::new();
    let mut escape = false;
    assert!(path.len() > 0 && path[0] == 0);
    let path = &path[1..];
    for &byte in path {
        if escape {
            segment.push(byte);
            escape = false;
        } else {
            match byte {
                ESCAPE => escape = true,
                SEPARATOR => {
                    result.push(segment);
                    segment = Vec::new();
                }
                _ => segment.push(byte),
            }
        }
    }
    result
}

fn main() {
    let mut store = RedbStore::memory();
    let v = Value::default();
    store.insert_raw(b"ab", v).unwrap();
    store.insert_raw(b"a", v).unwrap();
    store.dump().unwrap();
    println!("---");

    let mut store = RedbStore::memory();
    let v = Value::default();
    store.insert_raw(b"a", v).unwrap();
    store.insert_raw(b"ab", v).unwrap();
    store.dump().unwrap();
    println!("---");

    let mut store = RedbStore::memory();
    let v = Value::default();
    store.insert_raw(b"ab", v).unwrap();
    store.insert_raw(b"ab", v).unwrap();
    store.dump().unwrap();
    println!("---");

    let mut store = RedbStore::memory();
    let v = Value::default();
    store.insert_raw(b"abc", v).unwrap();
    store.insert_raw(b"abd", v).unwrap();
    store.dump().unwrap();
    println!("---");

    let mut store = RedbStore::memory();
    let v = Value::default();
    store.insert_raw(b"abababababababac", v).unwrap();
    store.insert_raw(b"abababababababad", v).unwrap();
    store.dump().unwrap();
    println!("---");

    let mut store = RedbStore::memory();
    let n = 1000000;
    let t0 = std::time::Instant::now();
    for i in 0..n {
        let text = i.to_string();
        // println!("{}", hex::encode(text.as_bytes()));
        store.insert_raw(text.as_bytes(), v).unwrap();
    }
    println!("create {} {}", n, t0.elapsed().as_secs_f64());

    let t0 = std::time::Instant::now();
    for i in 0..n {
        let text = i.to_string();
        store.get_raw(text.as_bytes()).unwrap().unwrap();
    }
    println!("get {} {}", n, t0.elapsed().as_secs_f64());
    // store.dump().unwrap();
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
struct Path(Vec<Vec<u8>>);

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::prelude::*;
    use rand::SeedableRng;
    use redb::ReadableTableMetadata;
    use test_strategy::proptest;

    use crate::{RawValue, RedbStore, Value, WillowValue};

    fn init_logging() {
        tracing_subscriber::fmt::try_init().ok();
    }

    #[derive(Debug)]
    struct RawPath(Vec<u8>);

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
    struct Path(Vec<Vec<u8>>);

    fn path_value_range() -> std::ops::Range<u8> {
        // having more than 3 or 4 values does not add much value to the test
        // 0 and 1 are special values (escape and separator)
        // 3 and 4 are normal values
        0..4
    }

    fn arb_raw_path(max_size: usize) -> impl Strategy<Value = RawPath> {
        prop::collection::vec(path_value_range(), 0..max_size).prop_map(RawPath)
    }

    fn arb_path(max_components: usize, max_component_size: usize) -> impl Strategy<Value = Path> {
        prop::collection::vec(
            prop::collection::vec(path_value_range(), 0..max_component_size),
            0..max_components,
        )
        .prop_map(Path)
    }

    impl Arbitrary for RawPath {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb_raw_path(32).boxed()
        }
    }

    impl Arbitrary for Value {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb_value().boxed()
        }
    }

    impl Arbitrary for Path {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb_path(4, 12).boxed()
        }
    }

    fn arb_raw_value() -> impl Strategy<Value = RawValue> {
        any::<u64>().prop_map(|seed| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut data = [0u8; 48];
            rng.fill_bytes(&mut data);
            RawValue(data)
        })
    }

    fn arb_willow_value() -> impl Strategy<Value = WillowValue> {
        (any::<u64>(), any::<[u8; 32]>(), any::<u64>()).prop_map(
            |(timestamp, fingerprint, size)| WillowValue {
                timestamp,
                fingerprint,
                size,
            },
        )
    }

    fn arb_value() -> impl Strategy<Value = Value> {
        any::<u64>().prop_map(|v| Value(v.to_le_bytes()))
    }

    #[proptest]
    fn test_escape_roundtrip(path: Path) {
        let path = path.0;
        let escaped = crate::escape(&path);
        let unescaped = crate::unescape(&escaped);
        assert_eq!(path, unescaped);
    }

    #[proptest]
    fn test_escape_preserves_order(a: Path, b: Path) {
        let ae = crate::escape(&a.0);
        let be = crate::escape(&b.0);
        assert_eq!(ae.cmp(&be), a.cmp(&b));
    }

    fn test_insert_get_impl(paths: Vec<(Path, Value)>) {
        let mut store = RedbStore::memory();
        let mut reference = BTreeMap::new();
        // insert all values, overwriting any existing values
        for (path, value) in paths {
            store.insert(&path.0, value.clone()).unwrap();
            reference.insert(path.clone(), value.clone());
        }
        let n = store.tables().unwrap().prefixes.len().unwrap();
        if n > 0 {
            println!("prefixes: {n}");
        }
        // check that the store basically behaves like a BTreeMap
        for (path, value) in reference {
            tracing::trace!("checking path {:?}", path);
            let got = store.get(&path.0).unwrap();
            assert_eq!(got, Some(value));
        }
    }

    #[proptest]
    fn prop_insert_get(paths: Vec<(Path, Value)>) {
        test_insert_get_impl(paths);
    }

    #[test]
    fn test_insert_get() {
        init_logging();
        let paths = vec![
            vec![(Path(vec![]), Value::default())],
            vec![
                (Path(vec![vec![1]]), Value::default()),
                (Path(vec![]), Value::default()),
            ],
            vec![
                (Path(vec![]), Value::default()),
                (Path(vec![vec![1]]), Value::default()),
            ],
            vec![
                (Path(vec![vec![]]), Value::default()),
                (Path(vec![vec![0]]), Value::default()),
            ],
            vec![
                (Path(vec![vec![0]]), Value::default()),
                (Path(vec![vec![]]), Value::default()),
            ],
            vec![
                (Path(vec![vec![]]), Value::default()),
                (Path(vec![vec![0]]), Value::default()),
                (Path(vec![]), Value::default()),
            ],
        ];
        for paths in paths {
            test_insert_get_impl(paths);
        }
    }
}
