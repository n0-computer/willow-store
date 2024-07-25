use std::{marker::PhantomData, num::NonZeroU64, ops::Deref};

use bytes::Bytes;
use redb::{AccessGuard, ReadableTable, WriteTransaction};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

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
    first_char: u8,
    prefix_len: u8,
    has_value: u8,
    prefix: Id,
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
    entry_buf: Vec<Entry>,
    path_buf: Vec<u8>,
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
            entry_buf: Vec::new(),
            path_buf: Vec::new(),
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

    fn traverse(&mut self) -> Result<(), redb::Error> {
        self.traverse_rec(NonZeroU64::new(1).unwrap(), &[])
    }

    fn traverse_rec(&mut self, dir: NonZeroU64, path: &[u8]) -> Result<(), redb::Error> {
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

    fn dump(&mut self) -> Result<(), redb::Error> {
        let tables = self.tables()?;
        for item in tables.paths.iter()? {
            let (k, v) = item?;
            let v = Entry::slice_from(v.value()).unwrap();
            println!("{}: {:?}", k.value(), v);
        }
        for item in tables.prefixes.iter()? {
            let (k, v) = item?;
            println!("{}: {:?}", k.value(), hex::encode(v.value()));
        }
        for item in tables.values.iter()? {
            let (k, v) = item?;
            println!("{}: {:?}", k.value(), v.value());
        }
        for item in tables.summaries.iter()? {
            let (k, v) = item?;
            println!("{}: {:?}", k.value(), v.value());
        }

        Ok(())
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
    ) -> std::result::Result<PrefixGuard<'a>, redb::Error> {
        if self.prefix_len <= 8 {
            Ok(PrefixGuard(Ok((self.prefix_len, self.prefix))))
        } else {
            let id = u64::from_le_bytes(self.prefix);
            store.get_prefix(id)
        }
    }
}

impl RedbStore {
    fn get(&mut self, path: &[&[u8]]) -> Result<Option<Value>, redb::Error> {
        let path = escape(path);
        self.get_raw(&path)
    }

    fn get_raw(&mut self, path: &[u8]) -> Result<Option<Value>, redb::Error> {
        let root = NonZeroU64::new(1).unwrap();
        self.get_rec(root, path)
    }

    fn get_rec(&mut self, dir: NonZeroU64, path: &[u8]) -> Result<Option<Value>, redb::Error> {
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

    fn insert(&mut self, path: &[&[u8]], value: Value) -> Result<(), redb::Error> {
        let path = escape(path);
        self.insert_raw(&path, value)
    }

    fn insert_raw(&mut self, path: &[u8], value: Value) -> Result<(), redb::Error> {
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
    ) -> Result<NonZeroU64, redb::Error> {
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
    ) -> Result<bool, redb::Error> {
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
                    let entry_prefix = entry_prefix.to_buffer(&mut buffers.entry_path);
                    self.set_prefix(&mut entry0, &entry_prefix[n..])?;
                    let subdir = self.next_dir_id()?;
                    self.move_child(entry0.first_char, dir, subdir)?;
                    self.put_children(subdir, &[entry0])?;
                    entries[index] = self.create_entry(dir, path, Some(&value), Some(subdir))?;
                    self.put_children(dir, &entries)?;
                    true
                } else {
                    // disjoint prefixes
                    let mut entry0 = entries[index];
                    let entry_prefix = entry_prefix.to_buffer(&mut buffers.entry_path);
                    let common = &path[..n];
                    let subdir = self.next_dir_id()?;
                    self.move_child(entry0.first_char, dir, subdir)?;
                    let entry1 = self.create_entry(subdir, &path[n..], Some(&value), None)?;
                    self.set_prefix(&mut entry0, &entry_prefix[n..])?;
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
        first_char: u8,
        from: NonZeroU64,
        to: NonZeroU64,
    ) -> Result<(), redb::Error> {
        self.modify(|tables| {
            let Some(guard) = tables.values.get(value_id(from, first_char).get())? else {
                return Ok(());
            };
            let value = guard.value();
            drop(guard);
            tables
                .values
                .insert(value_id(to, first_char).get(), value)?;
            Ok(())
        })
    }

    fn set_prefix(&mut self, entry: &mut Entry, value: &[u8]) -> Result<(), redb::Error> {
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
    ) -> Result<(), redb::Error> {
        entry.children = children.map(|x| x.get()).unwrap_or_default().to_le_bytes();
        Ok(())
    }

    fn create_entry(
        &mut self,
        dir: NonZeroU64,
        path: &[u8],
        value: Option<&Value>,
        children: Option<NonZeroU64>,
    ) -> std::result::Result<Entry, redb::Error> {
        let mut res = Entry::default();
        self.set_prefix(&mut res, path)?;
        if let Some(value) = value {
            res.set_has_value(true);
            self.put_value(value_id(dir, path[0]), Some(value))?;
        };
        self.set_children(&mut res, children)?;
        Ok(res)
    }

    fn put_children(&mut self, id: NonZeroU64, data: &[Entry]) -> Result<(), redb::Error> {
        self.modify(|tables| {
            let data = <[Entry]>::as_bytes(data);
            let mut guard = tables.paths.insert_reserve(id.get(), data.len() as u32)?;
            guard.as_mut().copy_from_slice(data);
            Ok(())
        })
    }

    fn next_dir_id(&mut self) -> Result<NonZeroU64, redb::Error> {
        let res = self.next_dir_id;
        self.next_dir_id = NonZeroU64::new(res.get() + 1).unwrap();
        Ok(res)
    }

    fn get_dir(&mut self, id: NonZeroU64) -> Result<EntriesGuard, redb::Error> {
        let Some(value) = self.tables()?.paths.get(id.get())? else {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        };
        Ok(EntriesGuard(value))
    }

    fn rm_dir(&mut self, id: u64) -> Result<(), redb::Error> {
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
    ) -> Result<Option<NonZeroU64>, redb::Error> {
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

    fn get_prefix(&mut self, id: u64) -> Result<PrefixGuard, redb::Error> {
        let Some(value) = self.tables()?.prefixes.get(id)? else {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        };
        Ok(PrefixGuard(Err(value)))
    }

    fn del_prefix(&mut self, id: u64) -> Result<(), redb::Error> {
        self.modify(|tables: &mut Tables| {
            tables.prefixes.remove(id)?;
            Ok(())
        })
    }

    fn put_value(
        &mut self,
        id: NonZeroU64,
        value: Option<&Value>,
    ) -> Result<PutValueResult, redb::Error> {
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

    fn get_value(&mut self, id: u64) -> Result<AccessGuard<Value>, redb::Error> {
        let Some(value) = self.tables()?.values.get(id)? else {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        };
        Ok(value)
    }

    fn del_value(&mut self, id: u64) -> Result<(), redb::Error> {
        self.modify(|tables| {
            tables.values.remove(id)?;
            Ok(())
        })
    }

    fn put_summary(&mut self, data: &Summary) -> Result<u64, redb::Error> {
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

    fn get_summary(&mut self, id: u64) -> Result<AccessGuard<Summary>, redb::Error> {
        let Some(value) = self.tables()?.summaries.get(id)? else {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        };
        Ok(value)
    }

    fn del_summary(&mut self, id: u64) -> Result<(), redb::Error> {
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
#[derive(Debug, Clone, Default)]
struct Summary {
    timestamp_range: std::ops::Range<u64>,
    fingerprint: [u8; 32],
}

impl redb::Value for Value {
    type SelfType<'a> = Value;

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
struct Value {
    timestamp: u64,
    fingerprint: [u8; 32],
    size: u64,
}

type INode = u64;
type PathKey<'a> = (u64, &'a [u8]);

const PATHS_TABLE: redb::TableDefinition<u64, &[u8]> = redb::TableDefinition::new("paths-v0");
const VALUES_TABLE: redb::TableDefinition<u64, Value> = redb::TableDefinition::new("values-v0");
const SUMARIES_TABLE: redb::TableDefinition<u64, Summary> =
    redb::TableDefinition::new("summaries-v0");

struct Tables<'txn> {
    paths: redb::Table<'txn, u64, &'static [u8]>,
    values: redb::Table<'txn, u64, Value>,
    summaries: redb::Table<'txn, u64, Summary>,
    prefixes: redb::Table<'txn, u64, &'static [u8]>,
}

impl<'a> Tables<'a> {
    fn new(txn: &'a WriteTransaction) -> std::result::Result<Self, redb::TableError> {
        Ok(Self {
            paths: txn.open_table(PATHS_TABLE)?,
            values: txn.open_table(VALUES_TABLE)?,
            summaries: txn.open_table(SUMARIES_TABLE)?,
            prefixes: txn.open_table(redb::TableDefinition::new("prefixes-v0"))?,
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
}

// common prefix of two slices.
fn common_prefix<'a, T: Eq>(a: &'a [T], b: &'a [T]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

fn escape<T, U>(path: T) -> Vec<u8>
where
    T: AsRef<[U]>,
    U: AsRef<[u8]>,
{
    let mut result = Vec::new();
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
