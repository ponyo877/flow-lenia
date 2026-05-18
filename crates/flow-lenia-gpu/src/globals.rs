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
///  32..36  beta_a       : f32   critical mass β_A (paper Eq. 5; M2.6)
///  36..40  n            : f32   α exponent (paper Eq. 5; M2.6)
///  40..44  dd           : u32   Chebyshev neighbourhood radius (M2.7)
///  44..48  sigma        : f32   reintegration σ (paper Eq. 6; M2.7)
///  48..52  dt           : f32   time step (paper Eq. 6; M2.7)
///  52..64  _pad[3]      : u32[] alignment padding (12 bytes)
/// ```
///
/// History:
/// - Pre-M2.6: 32 bytes (no paper_strict / beta_a / n + single `_pad: u32`).
/// - M2.6: added `paper_strict`, `beta_a`, `n` — 64 bytes.
/// - M2.7: added `dd`, `sigma`, `dt` — stayed 64 bytes by shrinking
///   `_pad` from `[u32; 6]` to `[u32; 3]`.
///
/// Builder methods (`with_*`) keep callers that don't need the M2.6+
/// fields at zero diff against the pre-extension API.
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
    pub dd: u32,
    pub sigma: f32,
    pub dt: f32,
    pub _pad: [u32; 3],
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
            dd: 0,
            sigma: 0.0,
            dt: 0.0,
            _pad: [0; 3],
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

    /// Set the Chebyshev neighbourhood radius `dd` (M2.7 reintegrate).
    #[must_use]
    pub fn with_dd(mut self, dd: u32) -> Self {
        self.dd = dd;
        self
    }

    /// Set the reintegration distribution width `σ` (paper Eq. 6; M2.7).
    #[must_use]
    pub fn with_sigma(mut self, sigma: f32) -> Self {
        self.sigma = sigma;
        self
    }

    /// Set the integration time step `dt` (paper Eq. 6; M2.7).
    #[must_use]
    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }
}
