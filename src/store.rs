use anyhow::Result;
use std::fmt::{Debug, Display};
use zerocopy::{AsBytes, FromBytes, FromZeroes};
pub mod mem;
pub mod redb;

/// A simple store trait for reading & storing blobs.
pub trait BlobStore: BlobStoreRead {
    /// Create a new node in the store. The generated ids should not be reused.
    fn create(&mut self, node: &[u8]) -> Result<NodeId>;
    /// Update a node in the store.
    fn update(&mut self, id: NodeId, node: &[u8]) -> Result<()>;
    /// Delete a node from the store.
    fn delete(&mut self, id: NodeId) -> Result<()>;
}

/// A simple store trait for reading blobs.
pub trait BlobStoreRead {
    /// Read a node from the store.
    fn read(&self, id: NodeId) -> Result<Vec<u8>>;
    /// Peek at a node in the store and project it into a value.
    fn peek<T>(&self, id: NodeId, f: impl Fn(&[u8]) -> T) -> Result<T>;
}

/// We implement the zero copy traits with native u64. This means that the storage
/// will use the native endianess of the platform, and the DBs will not be compatible
/// between platforms with different endianess.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, AsBytes, FromZeroes, FromBytes, Hash,
)]
#[repr(transparent)]
pub struct NodeId(u64);

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
