#![deny(warnings)]
//! Flow-Lenia core: platform-independent CA logic and CPU reference implementation.
//!
//! Implements Plantec et al. 2025 (arXiv:2506.08569v1) per the design contract
//! in `DESIGN.md` (Rev. 4.1). JAX implementation cross-reference is documented
//! in `references/JAX_NOTES.md` (commit `dce428c` of `erwanplantec/FlowLenia`).
//!
//! Contents are populated incrementally from M1.2 onwards.

pub mod affinity;
pub mod alpha;
pub mod config;
pub mod convolve;
pub mod flow;
pub mod growth;
pub mod kernel;
pub mod overlap;
pub mod params;
pub mod reintegrate;
pub mod sobel;
pub mod state;

pub use affinity::{affinity_with_constant_weights, affinity_with_localized_weights};
pub use alpha::alpha;
pub use config::{BorderMode, FlowLeniaConfig, MixRule};
pub use convolve::convolve2d;
pub use flow::flow;
pub use kernel::{compute_kernel, effective_radius, sigmoid, KernelMeta};
pub use overlap::overlap_area;
pub use params::{KernelEntry, KernelParams, SamplingSettings};
pub use reintegrate::reintegrate;
pub use sobel::{grad_a_sum, sobel, sobel_per_channel, sobel_x, sobel_y, SobelGradients};
pub use state::{
    sum_channels, ActivationField, AlphaField, FlowField, FlowFieldExt, UField, AXIS_C, AXIS_FLOW,
    AXIS_H, AXIS_W, FLOW_DX, FLOW_DY,
};
// `growth::growth` would shadow the module name when re-exported, so
// callers use `flow_lenia_core::growth::{bell, growth}` directly.
