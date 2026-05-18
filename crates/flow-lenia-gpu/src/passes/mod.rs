//! Compute-pass implementations (M2.3 onwards).
//!
//! Each pass is a small `pub struct` owning its compiled pipeline and
//! bind-group layout, with a stateless `record(...)` method that
//! appends a single `dispatch_workgroups` to a caller-supplied
//! [`wgpu::CommandEncoder`]. This factoring keeps per-step ordering
//! and synchronisation visible at the M2.10 simulator loop, instead
//! of being hidden inside each pass.

pub mod convolve;

pub use convolve::ConvolvePass;
