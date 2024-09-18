use std::path::Path;

use redb::{Database, ReadableTable, TableDefinition};
use zerocopy::{AsBytes, FromBytes};

use super::{BlobStore, NodeId, Result};

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
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let db = Database::create(path)?;
        Ok(Self { db })
    }

    pub fn memory() -> Result<Self> {
        let db =
            Database::builder().create_with_backend(redb::backends::InMemoryBackend::default())?;
        Ok(Self { db })
    }

    pub fn snapshot(&self) -> Result<Snapshot> {
        let txn = self.db.begin_read()?;
        Ok(Snapshot {
            blob_table: txn.open_table(BLOB_TABLE)?,
        })
    }
}

pub struct Snapshot {
    blob_table: redb::ReadOnlyTable<NodeId, &'static [u8]>,
}

impl BlobStore for Snapshot {
    fn read(&self, id: NodeId) -> Result<Vec<u8>> {
        match self.blob_table.get(&id)? {
            Some(value) => Ok(value.value().to_vec()),
            None => Err(anyhow::anyhow!("Node not found")),
        }
    }

    fn create(&mut self, _node: &[u8]) -> Result<NodeId> {
        anyhow::bail!("Cannot create nodes in a snapshot")
    }

    fn update(&mut self, _id: NodeId, _node: &[u8]) -> Result<()> {
        anyhow::bail!("Cannot update nodes in a snapshot")
    }

    fn delete(&mut self, _id: NodeId) -> Result<()> {
        anyhow::bail!("Cannot delete nodes in a snapshot")
    }
}

impl BlobStore for RedbBlobStore {
    fn create(&mut self, node: &[u8]) -> Result<NodeId> {
        let write_txn = self.db.begin_write()?;
        let new_node_id: NodeId;
        {
            // Open the autoincrement table
            let mut autoinc_table = write_txn.open_table(AUTOINC_TABLE)?;
            // Get the current autoincrement value
            let current_id = match autoinc_table.get("id")? {
                Some(id) => id.value(),
                None => 0u64,
            };
            let new_id = current_id + 1;
            // Update the autoincrement value
            autoinc_table.insert("id", &new_id)?;
            // Create a NodeId from the new_id
            new_node_id = NodeId(new_id.to_le_bytes());
            // Open the blobs table and insert the new node
            let mut blob_table = write_txn.open_table(BLOB_TABLE)?;
            blob_table.insert(&new_node_id, node)?;
        }
        write_txn.commit()?;
        Ok(new_node_id)
    }

    fn read(&self, id: NodeId) -> Result<Vec<u8>> {
        let read_txn = self.db.begin_read()?;
        let blob_table = read_txn.open_table(BLOB_TABLE)?;
        match blob_table.get(&id)? {
            Some(value) => Ok(value.value().to_vec()),
            None => Err(anyhow::anyhow!("Node not found")),
        }
    }

    fn update(&mut self, id: NodeId, node: &[u8]) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut blob_table = write_txn.open_table(BLOB_TABLE)?;
            blob_table.insert(&id, node)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn delete(&mut self, id: NodeId) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut blob_table = write_txn.open_table(BLOB_TABLE)?;
            blob_table.remove(&id)?;
        }
        write_txn.commit()?;
        Ok(())
    }
}
