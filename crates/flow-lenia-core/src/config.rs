//! Configuration types for Flow-Lenia.
//!
//! See `DESIGN.md` (Rev. 4.1) §6 for the canonical definition of each field
//! and §4.1.5 / §4.1.6 for the mode-switching rationale.

/// Boundary condition for convolution, gradient, and reintegration tracking.
///
/// `Torus` is the default per DESIGN.md §6 — it preserves mass exactly under
/// reintegration, whereas `Wall` introduces clipping at the boundary (JAX
/// `reintegration_tracking.py:64-65`) that can lose mass.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BorderMode {
    /// Periodic boundary (default).
    Torus,
    /// Hard wall: distribution centres are clamped to `[σ, W-σ]`.
    Wall,
}

/// Parameter-mixing rule for Eq. 8 (parameter-embedding mode).
///
/// `Stochastic` samples from the softmax / normalised distribution; this is
/// the default per DESIGN.md §6 and matches JAX `mix="stoch"`
/// (`reintegration_tracking.py:125-133`).
/// `Deterministic` picks the argmax (JAX `mix="det"`-equivalent branch).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MixRule {
    /// Sample from the (softmax) probability distribution.
    Stochastic,
    /// Pick the highest-probability neighbour deterministically.
    Deterministic,
}

/// Top-level Flow-Lenia configuration.
///
/// Per-kernel parameters (kernel radii, growth μ/σ, etc.) live in
/// [`crate::params::KernelParams`]; this struct holds only the global
/// physical parameters and the mode flags.
///
/// **Mode switching** — a single `paper_strict` toggle controls both the α
/// formula (DESIGN.md §4.1.5) and the Eq. 8 softmax (DESIGN.md §4.1.6),
/// so the two cannot be set independently:
///
/// | `paper_strict` | α formula                                     | Eq. 8 softmax                       |
/// |----------------|------------------------------------------------|--------------------------------------|
/// | `false` (default) | `clip((A_c / β_A)^n, 0, 1)` per-channel    | `(A_Σ · I) / Σ(A_Σ · I)`            |
/// | `true`            | `clip((A_Σ / β_A)^n, 0, 1)` shared over C  | `exp(A_Σ · I) / Σ exp(A_Σ · I)`     |
///
/// JAX cross-references:
/// - α (`paper_strict=false`): `flowlenia.py:98` / `flowlenia_params.py:101`
/// - softmax (`paper_strict=false`): `reintegration_tracking.py:127`
///   (see `JAX_NOTES.md` §11 NOTE-11-A / NOTE-11-B)
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FlowLeniaConfig {
    // --- Physical parameters ---
    /// Grid width in cells.
    pub grid_width: u32,
    /// Grid height in cells.
    pub grid_height: u32,
    /// Number of channels (paper symbol: `C`).
    pub channels: u32,
    /// Time step (paper Eq. 6 `dt`; JAX `Config.dt = 0.2`).
    pub dt: f32,
    /// Reintegration distribution width / "temperature"
    /// (paper Eq. 6 `s`, JAX `Config.sigma = 0.65`).
    pub sigma: f32,
    /// α exponent in paper Eq. 5. Only meaningful when `paper_strict = true`;
    /// when `paper_strict = false` the JAX-compatible path hard-codes 2.0
    /// (JAX `flowlenia.py:98`). Stored even in JAX mode so that toggling
    /// `paper_strict` does not silently reset the slider.
    pub n: f32,
    /// Critical mass β_A in paper Eq. 5
    /// (JAX `flowlenia_params.py:101` uses 2.0).
    pub beta_a: f32,
    /// Chebyshev neighbourhood radius for reintegration tracking
    /// (JAX `Config.dd = 5`; allowed UI range 3..=7;
    /// see `JAX_NOTES.md` §1).
    pub dd: u32,
    /// Number of kernels `|K|` in the kernel bank. Drives random
    /// sampling in the simulator (M1.14); does not directly constrain
    /// any function in `step()`, which receives the kernels as an
    /// explicit slice.
    pub num_kernels: u32,

    // --- Mode switches ---
    /// If `true`, use the paper Eq. 5 / Eq. 8 formulas verbatim
    /// (shared-channel α with `A_Σ`, exp-softmax for parameter mixing).
    /// If `false` (default), use the JAX-compatible per-channel α and the
    /// exp-less normalised softmax. See the table in the type-level doc
    /// comment for details.
    pub paper_strict: bool,
    /// Boundary condition for all spatial operators.
    pub border: BorderMode,
    /// Parameter mixing rule (only relevant when parameter embedding,
    /// i.e. paper Eq. 7, is enabled).
    pub mix_rule: MixRule,
}

impl Default for FlowLeniaConfig {
    /// Defaults match DESIGN.md (Rev. 4.1) §6.
    fn default() -> Self {
        Self {
            grid_width: 128,
            grid_height: 128,
            channels: 3,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            num_kernels: 10,
            paper_strict: false,
            border: BorderMode::Torus,
            mix_rule: MixRule::Stochastic,
        }
    }
}

impl FlowLeniaConfig {
    /// Smaller config for demos / interactive runs (M1.14 binary and
    /// the M2 GPU window). 64×64 keeps frame times manageable on a
    /// pure-CPU step loop while still exposing creature-scale
    /// structure.
    ///
    /// Differs from [`FlowLeniaConfig::default`] only in `grid_*` and
    /// `num_kernels` — physical parameters, modes, and `dd` are the
    /// same so toggling between default and demo doesn't silently
    /// re-tune the dynamics.
    #[must_use]
    pub fn demo_default() -> Self {
        Self {
            grid_width: 64,
            grid_height: 64,
            num_kernels: 10,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that [`FlowLeniaConfig::default`] matches DESIGN.md (Rev. 4.1)
    /// §6 exactly. Any change to these defaults must be reflected in the
    /// design document.
    #[test]
    fn default_values_match_design() {
        let cfg = FlowLeniaConfig::default();

        // Grid dimensions (DESIGN.md §7 UI "Grid" group defaults).
        assert_eq!(cfg.grid_width, 128, "grid_width default must be 128");
        assert_eq!(cfg.grid_height, 128, "grid_height default must be 128");
        assert_eq!(cfg.channels, 3, "channels default must be 3");

        // Physical parameters (DESIGN.md §6).
        approx::assert_relative_eq!(cfg.dt, 0.2, epsilon = 1e-9);
        approx::assert_relative_eq!(cfg.sigma, 0.65, epsilon = 1e-9);
        approx::assert_relative_eq!(cfg.n, 2.0, epsilon = 1e-9);
        approx::assert_relative_eq!(cfg.beta_a, 2.0, epsilon = 1e-9);
        assert_eq!(cfg.dd, 5, "dd default must be 5 (JAX Config.dd)");
        assert_eq!(cfg.num_kernels, 10, "num_kernels default must be 10");

        // Mode flags — paper_strict=false is "JAX compatible" for both α
        // (DESIGN.md §4.1.5) and Eq. 8 softmax (DESIGN.md §4.1.6).
        assert!(
            !cfg.paper_strict,
            "paper_strict default must be false (JAX compat)"
        );
        assert_eq!(cfg.border, BorderMode::Torus);
        assert_eq!(cfg.mix_rule, MixRule::Stochastic);
    }

    /// `demo_default()` differs only in grid size; everything else
    /// matches `default()` so toggling between presets does not change
    /// the dynamics.
    #[test]
    fn demo_default_matches_default_for_physical_params() {
        let demo = FlowLeniaConfig::demo_default();
        let dflt = FlowLeniaConfig::default();
        assert_eq!(demo.grid_width, 64);
        assert_eq!(demo.grid_height, 64);
        assert_eq!(demo.channels, dflt.channels);
        approx::assert_relative_eq!(demo.dt, dflt.dt, epsilon = 1e-9);
        approx::assert_relative_eq!(demo.sigma, dflt.sigma, epsilon = 1e-9);
        approx::assert_relative_eq!(demo.n, dflt.n, epsilon = 1e-9);
        approx::assert_relative_eq!(demo.beta_a, dflt.beta_a, epsilon = 1e-9);
        assert_eq!(demo.dd, dflt.dd);
        assert_eq!(demo.num_kernels, dflt.num_kernels);
        assert_eq!(demo.paper_strict, dflt.paper_strict);
        assert_eq!(demo.border, dflt.border);
        assert_eq!(demo.mix_rule, dflt.mix_rule);
    }

    /// Sanity-check that the enums are `Copy` + `Eq`-comparable (these are
    /// relied on by downstream pattern matching and uniform-buffer encoding).
    #[test]
    fn mode_enums_are_copy_and_eq() {
        let a = BorderMode::Torus;
        let b = a;
        assert_eq!(a, b);

        let m = MixRule::Stochastic;
        let n = m;
        assert_eq!(m, n);
    }
}
