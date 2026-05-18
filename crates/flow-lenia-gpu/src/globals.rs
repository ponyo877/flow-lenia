//! Compute-shader global uniform (M2.3, extended in M2.6).
//!
//! Used by every per-step compute shader (convolve M2.3,
//! affinity_growth M2.4, gradient M2.5, flow M2.6, reintegrate
//! M2.7) — the grid + kernel-bank dimensions, border policy, and
//! the mode flags are all the same across passes.
//!
//! Layout pinned by `const _: ()` asserts so any future field
//! reorder/add can't silently drift the WGSL `Globals` struct.

use crate::config_border::BorderCode;
use bytemuck::{Pod, Zeroable};
use flow_lenia_core::config::BorderMode;

/// `@group(0) @binding(N) var<uniform> globals: Globals` in WGSL.
///
/// Layout (64 bytes, multiple of 16 for uniform alignment):
/// ```text
///   0..4   h            : u32   grid height
///   4..8   w            : u32   grid width
///   8..12  c            : u32   number of channels
///  12..16  k            : u32   number of kernels
///  16..20  max_side     : u32   shared padded kernel side
///  20..24  half_side    : u32   = (max_side - 1) / 2
///  24..28  border       : u32   0 = Torus, 1 = Wall  (see BorderCode)
///  28..32  paper_strict : u32   0 = JAX compat, 1 = paper Eq. 5
///  32..36  beta_a       : f32   critical mass β_A (M2.6 flow pass)
///  36..40  n            : f32   α exponent (paper Eq. 5; M2.6)
///  40..64  _pad[6]      : u32[] alignment padding (24 bytes)
/// ```
///
/// The pre-M2.6 layout was 32 bytes (no `paper_strict / beta_a / n` +
/// a single `_pad: u32`). M2.6 extends the struct so the M2.4/M2.5
/// passes that don't read the new fields just leave them at default (0)
/// — see [`GpuGlobals::new`] + the `with_*` builder methods.
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
    pub paper_strict: u32,
    pub beta_a: f32,
    pub n: f32,
    pub _pad: [u32; 6],
}

const _: () = {
    assert!(std::mem::size_of::<GpuGlobals>() == 64);
    assert!(std::mem::align_of::<GpuGlobals>() == 4);
};

impl GpuGlobals {
    /// Construct from the parameters every pass needs. Mode flags and
    /// physical parameters (`paper_strict`, `beta_a`, `n`) default to
    /// `0 / 0.0 / 0.0`; set them with the [`with_paper_strict`],
    /// [`with_beta_a`], [`with_n`] builder methods. Passes that don't
    /// read the M2.6 fields (convolve M2.3, affinity_growth M2.4,
    /// gradient M2.5) can use this minimal constructor unchanged.
    ///
    /// [`with_paper_strict`]: Self::with_paper_strict
    /// [`with_beta_a`]: Self::with_beta_a
    /// [`with_n`]: Self::with_n
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
            paper_strict: 0,
            beta_a: 0.0,
            n: 0.0,
            _pad: [0; 6],
        }
    }

    /// Set the paper-strict / JAX-compat mode flag. CPU enum is the
    /// authority; the WGSL shader reads `paper_strict == 1u`.
    #[must_use]
    pub fn with_paper_strict(mut self, on: bool) -> Self {
        self.paper_strict = u32::from(on);
        self
    }

    /// Set the critical mass `β_A` (paper Eq. 5; M2.6).
    #[must_use]
    pub fn with_beta_a(mut self, beta_a: f32) -> Self {
        self.beta_a = beta_a;
        self
    }

    /// Set the `α` exponent `n` (paper Eq. 5; M2.6).
    #[must_use]
    pub fn with_n(mut self, n: f32) -> Self {
        self.n = n;
        self
    }
}
