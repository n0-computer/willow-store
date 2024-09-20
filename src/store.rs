use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use anyhow::Result;
use std::fmt::{Debug, Display};
use zerocopy::{AsBytes, FromBytes, FromZeroes};
pub mod redb;

/// A simple store trait for storing blobs.
pub trait BlobStore {
    /// Create a new node in the store. The generated ids should not be reused.
    fn create(&mut self, node: &[u8]) -> Result<NodeId>;
    /// Read a node from the store.
    fn read(&self, id: NodeId) -> Result<Vec<u8>>;
    /// Peek at a node in the store and project it into a value.
    fn peek<T>(&self, id: NodeId, f: impl Fn(&[u8]) -> T) -> Result<T>;
    /// Update a node in the store.
    fn update(&mut self, id: NodeId, node: &[u8]) -> Result<()>;
    /// Delete a node from the store.
    fn delete(&mut self, id: NodeId) -> Result<()>;
}

pub struct MemStore {
    nodes: HashMap<NodeId, Arc<[u8]>>,
}

impl MemStore {
    pub fn new() -> Self {
        MemStore {
            nodes: Default::default(),
        }
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn total_bytes(&self) -> usize {
        self.nodes.values().map(|v| v.len()).sum()
    }
}

impl BlobStore for MemStore {
    fn create(&mut self, node: &[u8]) -> Result<NodeId> {
        let id = NodeId::from((self.nodes.len() as u64) + 1);
        assert!(!id.is_empty());
        self.nodes.insert(id, node.to_vec().into());
        Ok(id)
    }

    fn update(&mut self, id: NodeId, node: &[u8]) -> Result<()> {
        assert!(!id.is_empty());
        self.nodes.insert(id, node.to_vec().into());
        Ok(())
    }

    fn read(&self, id: NodeId) -> Result<Vec<u8>> {
        assert!(!id.is_empty());
        match self.nodes.get(&id) {
            Some(data) => Ok(data.to_vec()),
            None => Err(anyhow::anyhow!("Node not found")),
        }
    }

    fn peek<T>(&self, id: NodeId, f: impl Fn(&[u8]) -> T) -> Result<T> {
        assert!(!id.is_empty());
        match self.nodes.get(&id) {
            Some(data) => Ok(f(data)),
            None => Err(anyhow::anyhow!("Node not found")),
        }
    }

    fn delete(&mut self, id: NodeId) -> Result<()> {
        assert!(!id.is_empty());
        self.nodes.remove(&id);
        Ok(())
    }
}

/// We implement the zero copy traits with native u64. This means that the storage
/// will use the native endianess of the platform, and the DBs will not be compatible
/// between platforms with different endianess.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, AsBytes, FromZeroes, FromBytes, Hash)]
#[repr(transparent)]
pub struct NodeId(u64);

impl Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id = self.0;
        write!(f, "NodeId({})", id)
    }
}

impl Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id = self.0;
        write!(f, "{}", id)
    }
}

impl From<u64> for NodeId {
    fn from(id: u64) -> Self {
        NodeId(id)
    }
}

impl NodeId {
    pub const EMPTY: Self = NodeId(0);

    pub fn is_empty(&self) -> bool {
        self == &Self::EMPTY
    }
}
