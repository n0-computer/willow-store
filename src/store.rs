use std::collections::BTreeMap;

use crate::VariableSize;
use anyhow::Result;
use std::fmt::{Debug, Display};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

/// A simple store trait for storing blobs.
pub trait Store<T: VariableSize> {
    fn create(&mut self, node: &[u8]) -> Result<NodeId>;
    fn read(&self, id: NodeId) -> Result<T>;
    fn update(&mut self, id: NodeId, node: &[u8]) -> Result<()>;
    fn delete(&mut self, id: NodeId) -> Result<()>;
}

impl<T: VariableSize> Store<T> for Box<dyn Store<T>> {
    fn create(&mut self, node: &[u8]) -> Result<NodeId> {
        self.as_mut().create(node)
    }

    fn read(&self, id: NodeId) -> Result<T> {
        self.as_ref().read(id)
    }

    fn update(&mut self, id: NodeId, node: &[u8]) -> Result<()> {
        self.as_mut().update(id, node)
    }

    fn delete(&mut self, id: NodeId) -> Result<()> {
        self.as_mut().delete(id)
    }
}

pub struct MemStore {
    nodes: BTreeMap<NodeId, Vec<u8>>,
}

impl MemStore {
    pub fn new() -> Self {
        MemStore {
            nodes: BTreeMap::new(),
        }
    }
}

impl<T: VariableSize> Store<T> for MemStore {
    fn create(&mut self, node: &[u8]) -> Result<NodeId> {
        let id = NodeId::from((self.nodes.len() as u64) + 1);
        assert!(!id.is_empty());
        self.nodes.insert(id, node.to_vec());
        Ok(id)
    }

    fn update(&mut self, id: NodeId, node: &[u8]) -> Result<()> {
        assert!(!id.is_empty());
        self.nodes.insert(id, node.to_vec());
        Ok(())
    }

    fn read(&self, id: NodeId) -> Result<T> {
        assert!(!id.is_empty());
        match self.nodes.get(&id) {
            Some(data) => Ok(T::read(data)),
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
