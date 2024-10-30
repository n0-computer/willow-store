use std::{collections::HashMap, sync::Arc};

use super::{BlobStore, BlobStoreRead, NodeId, Result};

#[derive(Debug, Clone)]
pub struct MemStore {
    nodes: HashMap<NodeId, Arc<[u8]>>,
    max_id: u64,
}

impl MemStore {
    pub fn new() -> Self {
        MemStore {
            nodes: Default::default(),
            max_id: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn max_id(&self) -> u64 {
        self.max_id
    }

    pub fn total_bytes(&self) -> usize {
        self.nodes.values().map(|v| v.len()).sum()
    }
}

impl BlobStore for MemStore {
    fn create(&mut self, node: &[u8]) -> Result<NodeId> {
        self.max_id += 1;
        let id = NodeId(self.max_id);
        assert!(!id.is_empty());
        self.nodes.insert(id, node.to_vec().into());
        Ok(id)
    }

    fn update(&mut self, id: NodeId, node: &[u8]) -> Result<()> {
        assert!(!id.is_empty());
        self.nodes.insert(id, node.to_vec().into());
        Ok(())
    }

    fn delete(&mut self, id: NodeId) -> Result<()> {
        assert!(!id.is_empty());
        self.nodes.remove(&id);
        Ok(())
    }
}

impl BlobStoreRead for MemStore {
    fn peek<T>(&self, id: NodeId, f: impl Fn(&[u8]) -> T) -> Result<T> {
        assert!(!id.is_empty());
        match self.nodes.get(&id) {
            Some(data) => Ok(f(data)),
            None => Err(anyhow::anyhow!("Node not found")),
        }
    }
}
