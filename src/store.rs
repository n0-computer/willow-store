use std::{collections::BTreeMap, sync::Arc};

use anyhow::Result;
use std::fmt::{Debug, Display};
use zerocopy::{AsBytes, FromBytes, FromZeroes};
mod redb;

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
    nodes: BTreeMap<NodeId, Arc<[u8]>>,
}

impl MemStore {
    pub fn new() -> Self {
        MemStore {
            nodes: BTreeMap::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, AsBytes, FromZeroes, FromBytes)]
#[repr(transparent)]
pub struct NodeId([u8; 8]);

impl Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id = u64::from_be_bytes(self.0);
        write!(f, "NodeId({})", id)
    }
}

impl Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id = u64::from_be_bytes(self.0);
        write!(f, "{}", id)
    }
}

impl From<u64> for NodeId {
    fn from(id: u64) -> Self {
        NodeId(id.to_be_bytes())
    }
}

impl NodeId {
    pub const EMPTY: Self = NodeId([0; 8]);

    pub fn is_empty(&self) -> bool {
        self == &Self::EMPTY
    }
}
