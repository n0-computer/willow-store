use redb::{
    Database, ReadTransaction, ReadableTable, ReadableTableMetadata, TableDefinition,
    WriteTransaction,
};
use zerocopy::{AsBytes, FromBytes};

use super::{BlobStore, BlobStoreRead, NodeId, Result};

pub struct RedbBlobStore {
    db: Database,
}

const AUTOINC_TABLE: TableDefinition<&str, u64> = TableDefinition::new("autoinc");
const BLOB_TABLE: TableDefinition<NodeId, &[u8]> = TableDefinition::new("blobs");

impl redb::Value for NodeId {
    type SelfType<'b> = NodeId where Self: 'b;

    type AsBytes<'a> = &'a [u8];

    fn fixed_width() -> Option<usize> {
        Some(8)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        NodeId::read_from(data).unwrap()
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'a,
        Self: 'b,
    {
        AsBytes::as_bytes(value)
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("NodeId")
    }
}

impl<'a> redb::Key for NodeId {
    fn compare(data1: &[u8], data2: &[u8]) -> std::cmp::Ordering {
        data1.cmp(data2)
    }
}

impl RedbBlobStore {
    /// Create a new instance of the RedbBlobStore.
    pub fn new(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let db = Database::create(path.as_ref())?;
        Self::create_tables(&db)?;
        Ok(Self { db })
    }

    pub fn memory() -> Result<Self> {
        let db =
            Database::builder().create_with_backend(redb::backends::InMemoryBackend::default())?;
        Self::create_tables(&db)?;
        Ok(Self { db })
    }

    fn create_tables(db: &redb::Database) -> Result<()> {
        let txn = db.begin_write()?;
        Tables::open(&txn)?;
        txn.commit()?;
        Ok(())
    }

    pub fn snapshot(&self) -> Result<Snapshot> {
        let txn = self.db.begin_read()?;
        Ok(Snapshot {
            blob_table: txn.open_table(BLOB_TABLE)?,
        })
    }

    pub fn txn(&self) -> Result<WriteBatch> {
        let txn = self.db.begin_write()?;
        let inner = WriteTransactionInner::try_new(txn, |txn| {
            let autoinc_table = txn.open_table(AUTOINC_TABLE)?;
            let blob_table = txn.open_table(BLOB_TABLE)?;
            anyhow::Ok(Tables {
                autoinc: autoinc_table,
                blobs: blob_table,
            })
        })?;
        Ok(WriteBatch(inner))
    }

    pub fn blob_count(&self) -> Result<u64> {
        let txn = self.db.begin_read().unwrap();
        let table = txn.open_table(BLOB_TABLE).unwrap();
        Ok(table.len()?)
    }
}

pub struct Snapshot {
    blob_table: redb::ReadOnlyTable<NodeId, &'static [u8]>,
}

impl Snapshot {
    pub fn open(txn: &ReadTransaction) -> Result<Self> {
        Ok(Self {
            blob_table: txn.open_table(BLOB_TABLE)?,
        })
    }
}

impl BlobStoreRead for Snapshot {
    fn read(&self, id: NodeId) -> Result<Vec<u8>> {
        self.peek(id, |x| x.to_vec())
    }

    fn peek<T>(&self, id: NodeId, f: impl Fn(&[u8]) -> T) -> Result<T> {
        match self.blob_table.get(&id)? {
            Some(value) => Ok(f(value.value())),
            None => Err(anyhow::anyhow!("Node not found")),
        }
    }
}

pub struct Tables<'a> {
    autoinc: redb::Table<'a, &'static str, u64>,
    blobs: redb::Table<'a, NodeId, &'static [u8]>,
}

impl<'a> Tables<'a> {
    pub fn open(txn: &'a WriteTransaction) -> Result<Self> {
        let autoinc = txn.open_table(AUTOINC_TABLE)?;
        let blobs = txn.open_table(BLOB_TABLE)?;
        Ok(Self { autoinc, blobs })
    }
}

impl<'a> BlobStoreRead for Tables<'a> {
    fn read(&self, id: NodeId) -> Result<Vec<u8>> {
        let value = self
            .blobs
            .get(&id)?
            .ok_or_else(|| anyhow::anyhow!("Node not found"))?;
        Ok(value.value().to_vec())
    }

    fn peek<T>(&self, id: NodeId, f: impl Fn(&[u8]) -> T) -> Result<T> {
        let value = self
            .blobs
            .get(&id)?
            .ok_or_else(|| anyhow::anyhow!("Node not found"))?;
        Ok(f(value.value()))
    }
}

impl<'a> BlobStore for Tables<'a> {
    fn create(&mut self, node: &[u8]) -> Result<NodeId> {
        let current_id = match self.autoinc.get("id")? {
            Some(id) => id.value(),
            None => 0u64,
        };
        let new_id = current_id + 1;
        self.autoinc.insert("id", &new_id)?;
        let new_node_id = NodeId::from(new_id);
        self.blobs.insert(&new_node_id, node)?;
        Ok(new_node_id)
    }

    fn update(&mut self, id: NodeId, node: &[u8]) -> Result<()> {
        self.blobs.insert(id, node)?;
        Ok(())
    }

    fn delete(&mut self, id: NodeId) -> Result<()> {
        self.blobs.remove(id)?;
        Ok(())
    }
}

self_cell::self_cell!(
    struct WriteTransactionInner {
        owner: WriteTransaction,

        #[covariant]
        dependent: Tables,
    }
);

pub struct WriteBatch(WriteTransactionInner);

impl WriteBatch {
    pub fn commit(self) -> Result<()> {
        self.0.into_owner().commit()?;
        Ok(())
    }
}

impl BlobStoreRead for WriteBatch {
    fn peek<T>(&self, id: NodeId, f: impl Fn(&[u8]) -> T) -> Result<T> {
        self.0.with_dependent(|_db, tables| tables.peek(id, f))
    }

    fn read(&self, id: NodeId) -> Result<Vec<u8>> {
        self.0.with_dependent(|_db, tables| tables.read(id))
    }
}

impl BlobStore for WriteBatch {
    fn create(&mut self, node: &[u8]) -> Result<NodeId> {
        let new_node_id = self
            .0
            .with_dependent_mut(|_db, tables| tables.create(node))?;
        Ok(new_node_id)
    }

    fn update(&mut self, id: NodeId, node: &[u8]) -> Result<()> {
        self.0
            .with_dependent_mut(|_db, tables| tables.update(id, node))
    }

    fn delete(&mut self, id: NodeId) -> Result<()> {
        self.0.with_dependent_mut(|_db, tables| tables.delete(id))
    }
}

impl BlobStoreRead for RedbBlobStore {
    fn read(&self, id: NodeId) -> Result<Vec<u8>> {
        let read_txn = self.db.begin_read()?;
        Snapshot::open(&read_txn)?.read(id)
    }

    fn peek<T>(&self, id: NodeId, f: impl Fn(&[u8]) -> T) -> Result<T> {
        let read_txn = self.db.begin_read()?;
        Snapshot::open(&read_txn)?.peek(id, f)
    }
}

impl BlobStore for RedbBlobStore {
    fn create(&mut self, node: &[u8]) -> Result<NodeId> {
        let write_txn = self.db.begin_write()?;
        let new_node_id = Tables::open(&write_txn)?.create(node)?;
        write_txn.commit()?;
        Ok(new_node_id)
    }

    fn update(&mut self, id: NodeId, node: &[u8]) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        Tables::open(&write_txn)?.update(id, node)?;
        write_txn.commit()?;
        Ok(())
    }

    fn delete(&mut self, id: NodeId) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        Tables::open(&write_txn)?.delete(id)?;
        write_txn.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::mock_willow::{Subspace, TNode, TPoint, Timestamp, WillowTreeParams, WillowValue};
    use std::{borrow::Borrow, str::FromStr};

    use crate::{Node, Point, QueryRange, QueryRange3d};

    use super::*;
    use crate::blob_seq::BlobSeq;
    use testresult::TestResult;

    fn init(
        x: Vec<(
            impl Into<Subspace>,
            Vec<(impl Into<Timestamp>, impl Into<String>, impl Into<String>)>,
        )>,
    ) -> (TNode, RedbBlobStore) {
        let store = RedbBlobStore::memory().unwrap();
        let mut txn = store.txn().unwrap();
        let mut node = TNode::EMPTY;
        for (subspace, data) in x {
            let subspace = subspace.into();
            for (timestamp, path, value) in data {
                let path = BlobSeq::from_str(&path.into()).unwrap();
                let key = TPoint::new(&subspace, &timestamp.into(), path.borrow());
                let value = WillowValue::hash(value.into().as_bytes());
                node.insert(&key, &value, &mut txn).unwrap();
            }
        }
        txn.commit().unwrap();
        (node, store)
    }

    #[test]
    fn db_ops_test() -> TestResult<()> {
        let (node, store) = init(vec![(
            1u64,
            vec![
                (5, r#""a"/"b"/"c""#, "data1"),
                (4, r#""a"/"b"/"d""#, "data2"),
                (3, r#""a"/"b"/"e""#, "data3"),
                (2, r#""a"/"b"/"f""#, "data4"),
                (1, r#""a"/"b"/"g""#, "data5"),
            ],
        )]);
        node.dump(&store)?;
        println!("iter");
        for item in node.iter(&store) {
            println!("{:?}", item?);
        }
        let query = QueryRange3d {
            x: QueryRange::all(),
            y: QueryRange::all(),
            z: QueryRange::from(BlobSeq::from_str(r#""a"/"b"/"e""#)?..),
        };
        println!("query by path range");
        for item in node.query(&query, &store) {
            println!("{:?}", item?);
        }
        println!("query by path range, sorted by path");
        for item in node.query_ordered(&query, crate::SortOrder::ZXY, &store) {
            println!("{:?}", item?);
        }
        println!("query by path range, sorted by timestamp");
        for item in node.query_ordered(&query, crate::SortOrder::YZX, &store) {
            println!("{:?}", item?);
        }
        println!("split");
        for item in node.split_range(query.clone(), 2, &store) {
            println!("{:?}", item?);
        }
        println!("count by path range");
        let count = node.range_count(&query, &store)?;
        println!("{:?}", count);
        println!("fingerprint by path range");
        let fingerprint = node.range_summary(&query, &store)?;
        println!("{:?}", fingerprint);
        return Ok(());
    }

    #[test]
    fn db_smoke_test() -> TestResult<()> {
        let db = RedbBlobStore::memory()?;
        let mut txn = db.txn()?;
        let mut node = Node::<WillowTreeParams>::EMPTY;
        let v1 = WillowValue::hash(b"data1");
        let v2 = WillowValue::hash(b"data2");
        let t = Timestamp::now();
        let s = Subspace::ZERO;
        let path = BlobSeq::from_str(r#""a"/"b"/"c""#)?;
        let key = Point::new(&s, &t, path.borrow());
        node.insert(&key, &v1, &mut txn)?;
        let path = BlobSeq::from_str(r#""a"/"b"/"d""#)?;
        let key = Point::new(&s, &t, path.borrow());
        node.insert(&key, &v2, &mut txn)?;
        txn.commit()?;
        node.dump(&db)?;
        let ss = db.snapshot()?;
        node.dump(&ss)?;
        for item in node.iter(&ss) {
            println!("{:?}", item?);
        }
        Ok(())
    }
}
