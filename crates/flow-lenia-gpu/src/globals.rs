//! Compute-shader global uniform (M2.3).
//!
//! Used by the convolve pass (M2.3) and will be reused by every later
//! per-step compute shader (M2.4 affinity_growth, M2.5 sobel, …) — the
//! grid + kernel-bank dimensions and the border mode are the same
//! across every pass.
//!
//! Layout pinned by a `const _: ()` assert so any future field
//! reorder/add can't silently drift the WGSL `Globals` struct.

use crate::config_border::BorderCode;
use bytemuck::{Pod, Zeroable};
use flow_lenia_core::config::BorderMode;

/// `@group(0) @binding(4) var<uniform> globals: Globals` in WGSL.
///
/// Layout (32 bytes, multiple of 16 for uniform alignment):
/// ```text
///   0..4   h          : u32   grid height
///   4..8   w          : u32   grid width
///   8..12  c          : u32   number of channels
///  12..16  k          : u32   number of kernels
///  16..20  max_side   : u32   shared padded kernel side
///  20..24  half_side  : u32   = (max_side - 1) / 2
///  24..28  border     : u32   0 = Torus, 1 = Wall  (see BorderCode)
///  28..32  _pad       : u32   alignment padding
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuGlobals {
    pub h: u32,
    pub w: u32,
    pub c: u32,
    pub k: u32,
    pub max_side: u32,
    pub half_side: u32,
    pub border: u32,
    pub _pad: u32,
}

const _: () = {
    assert!(std::mem::size_of::<GpuGlobals>() == 32);
    assert!(std::mem::align_of::<GpuGlobals>() == 4);
};

impl GpuGlobals {
    /// Construct from per-pass parameters. `max_side` typically comes
    /// from `GpuKernelBuffers::max_side`; `border` is taken from
    /// `cfg.border` (CPU enum → GPU u32 via [`BorderCode`]).
    #[must_use]
    pub fn new(h: u32, w: u32, c: u32, k: u32, max_side: u32, border: BorderMode) -> Self {
        assert!(max_side >= 1, "max_side must be ≥ 1");
        assert!(max_side % 2 == 1, "max_side must be odd (got {max_side})");
        Self {
            h,
            w,
            c,
            k,
            max_side,
            half_side: (max_side - 1) / 2,
            border: BorderCode::from(border) as u32,
            _pad: 0,
        }
    }
}
