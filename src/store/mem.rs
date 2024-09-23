use std::{collections::HashMap, sync::Arc};

use super::{BlobStore, NodeId, Result};

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
