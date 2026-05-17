#![deny(warnings)]
//! Flow-Lenia core: platform-independent CA logic and CPU reference implementation.
//!
//! Implements Plantec et al. 2025 (arXiv:2506.08569v1) per the design contract
//! in `DESIGN.md` (Rev. 4.1). JAX implementation cross-reference is documented
//! in `references/JAX_NOTES.md` (commit `dce428c` of `erwanplantec/FlowLenia`).
//!
//! Contents are populated incrementally from M1.2 onwards.

pub mod config;
pub mod convolve;
pub mod growth;
pub mod kernel;
pub mod params;
pub mod sobel;

pub use config::{BorderMode, FlowLeniaConfig, MixRule};
pub use convolve::convolve2d;
pub use kernel::{compute_kernel, effective_radius, sigmoid};
pub use params::{KernelEntry, KernelParams, SamplingSettings};
pub use sobel::{sobel, sobel_x, sobel_y, SobelGradients};
// `growth::growth` would shadow the module name when re-exported, so
// callers use `flow_lenia_core::growth::{bell, growth}` directly.
