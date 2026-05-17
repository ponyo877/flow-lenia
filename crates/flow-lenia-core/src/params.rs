//! Per-kernel parameters for Flow-Lenia.
//!
//! M1.2 establishes only the type definitions. Sampling (M1.3) will reproduce
//! the ranges from JAX `flowlenia.py:55-64` (and `flowlenia_params.py:60-66`),
//! with the per-field adjustments documented in `references/JAX_NOTES.md` §6
//! and DESIGN.md §6.

/// Per-kernel parameters defining one `(K_i, G_i)` pair.
///
/// Field meanings (paper Eq. 1, 2 and JAX `flowlenia.py:55-64`):
///
/// - `c0` / `c1`: source / target channels (paper §2, connectivity matrix `M`)
/// - `r`: kernel-`i` radius scale `r_i ∈ [0.2, 1.0]` (paper Eq. 1)
/// - `a`, `b`, `w`: Gaussian-bump-ring parameters (3 rings — `k=3` in paper Eq. 1)
/// - `h`: kernel weight (paper Eq. 3); ignored when parameter embedding
///   (paper Eq. 7) is enabled, since the per-cell map `P_i(x)` takes its place
/// - `mu`, `sigma`: growth function parameters (paper Eq. 2 `μ_i`, `σ_i`).
///   Note: `sigma` here is the *growth* width, not the *reintegration*
///   distribution width (which lives in [`crate::config::FlowLeniaConfig::sigma`]).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KernelEntry {
    /// Source channel index (paper `c_i^0`).
    pub c0: u32,
    /// Target channel index (paper `c_i^1`).
    pub c1: u32,
    /// Kernel scale factor `r_i`.
    pub r: f32,
    /// Centre offsets of the three Gaussian rings.
    pub a: [f32; 3],
    /// Amplitudes of the three Gaussian rings.
    pub b: [f32; 3],
    /// Widths of the three Gaussian rings.
    pub w: [f32; 3],
    /// Kernel weight (paper Eq. 3 `h_i`; used only when parameter embedding
    /// is disabled).
    pub h: f32,
    /// Growth function mean (paper Eq. 2 `μ_i`).
    pub mu: f32,
    /// Growth function width (paper Eq. 2 `σ_i`; **not** the reintegration σ).
    pub sigma: f32,
}

/// Full kernel set for one Flow-Lenia configuration.
///
/// `r_global` is the paper-`R` global maximum neighbourhood radius
/// (JAX `flowlenia.py:57`, range `[2.0, 25.0]`).
#[derive(Clone, Debug, PartialEq)]
pub struct KernelParams {
    /// Paper `R`, the global maximum neighbourhood radius.
    pub r_global: f32,
    /// Per-kernel entries; length is `|K|`.
    pub kernels: Vec<KernelEntry>,
}

impl KernelParams {
    /// Number of kernels `|K|`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether the kernel set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke-test that the structs are constructible and the `len` / `is_empty`
    /// helpers behave as expected. Full sampling is exercised in M1.3.
    #[test]
    fn kernel_params_helpers() {
        let empty = KernelParams {
            r_global: 10.0,
            kernels: Vec::new(),
        };
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let one = KernelParams {
            r_global: 10.0,
            kernels: vec![KernelEntry {
                c0: 0,
                c1: 0,
                r: 0.5,
                a: [0.3, 0.6, 0.9],
                b: [1.0, 0.5, 0.25],
                w: [0.1, 0.1, 0.1],
                h: 1.0,
                mu: 0.15,
                sigma: 0.02,
            }],
        };
        assert!(!one.is_empty());
        assert_eq!(one.len(), 1);
    }
}
