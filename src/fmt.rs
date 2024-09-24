//! Utilities for formatting
use std::fmt::Debug;

pub(crate) struct NoQuotes<'a>(pub &'a str);

impl<'a> Debug for NoQuotes<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// pub(crate) struct DD<T>(pub T);

// impl<T: Display> Debug for DD<T> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", self.0)
//     }
// }
