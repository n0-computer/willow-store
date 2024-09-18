//! Defines the layout of a node in the database.
//!
//! The layout is:
//!
//! left:       8 bytes
//! right:      8 bytes
//! rank:       8 bytes // padded to 8 bytes so the rest is aligned
//! count:      8 bytes
//! value:      V::SIZE bytes
//! summary:    M::SIZE bytes
//! key:        X::SIZE+Y::SIZE+z.size() bytes (variable size)
//!
//! Count is stored separately so we can do ops such as splits efficiently.
use crate::{FixedSize, KeyParams, TreeParams};

#[inline(always)]
pub fn min_key_size<P: KeyParams>() -> usize {
    P::X::SIZE + P::Y::SIZE
}

#[inline(always)]
pub fn min_data_size<P: TreeParams>() -> usize {
    key_offset::<P>() + min_key_size::<P>()
}

pub const LEFT_OFFSET: usize = 0;
pub const RIGHT_OFFSET: usize = 8;
pub const RANK_OFFSET: usize = 16;
pub const COUNT_OFFSET: usize = 24;
pub const VALUE_OFFSET: usize = 32;

#[inline(always)]
pub fn summary_offset<P: TreeParams>() -> usize {
    VALUE_OFFSET + P::V::SIZE
}

#[inline(always)]
pub fn key_offset<P: TreeParams>() -> usize {
    VALUE_OFFSET + P::V::SIZE + P::M::SIZE
}