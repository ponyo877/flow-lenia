//! Stateful Flow-Lenia simulator (M1.14).
//!
//! Wraps the pure-function [`step`](crate::step::step) with the longer-lived
//! state that a real run needs:
//! - the current activation field `A`;
//! - the kernel bank and its metadata (precomputed once, reused per step);
//! - the configuration;
//! - an optional cell-local parameter map `P_i(x)` for Eq. 7
//!   (parameter embedding);
//! - a step counter.
//!
//! The simulator is intentionally a thin container — every per-step
//! computation still goes through `crate::step::step`, so any bug there
//! affects this module by extension and the simulator's own tests can be
//! short. M1.13's mass-conservation tests already exercise the full
//! pipeline; the tests in this module focus on the *state machine* the
//! simulator adds (seed reproducibility, step counting, weight-mode
//! toggling, alignment with pure `step`).
//!
//! **Initial state recipe** (DESIGN.md §6): a central square patch (30×30
//! by default, capped to the grid size) where each cell of each channel
//! is independently drawn from `U(0, 1)`. The rest of the grid is zero.
//! This is the standard "spark" used by both the JAX reference
//! (`examples/example_fl.py` initial blob) and the paper Table 1 figures.
//!
//! **RNG**: `ChaCha8Rng` so that a single `u64` seed determines the entire
//! initial state and the sampled kernel parameters bit-for-bit on any
//! platform. We do *not* persist the RNG across steps — Eq. 8
//! (stochastic parameter mixing, M1.15+) will own a separate RNG that
//! advances per step.

use crate::config::FlowLeniaConfig;
use crate::kernel::{compute_kernel, KernelMeta};
use crate::params::{KernelParams, SamplingSettings};
use crate::state::ActivationField;
use crate::step::{step, WeightsRef};
use ndarray::{Array2, Array3, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Stateful Flow-Lenia simulator owning one running instance of the
/// dynamics.
///
/// See the module-level docs for the initial-state recipe and the RNG
/// design.
///
/// All fields are `pub(crate)` so the in-crate tests can introspect
/// without going through the accessor methods; downstream consumers
/// must use the public API.
pub struct FlowLeniaSimulator {
    pub(crate) a: ActivationField,
    pub(crate) kernels: Vec<Array2<f32>>,
    pub(crate) kernel_meta: Vec<KernelMeta>,
    pub(crate) kernel_params: KernelParams,
    pub(crate) cfg: FlowLeniaConfig,
    pub(crate) p_map: Option<Array3<f32>>,
    pub(crate) step_count: u64,
    /// Cached `h_i` slice for the `Constant` weights path — re-derived
    /// from `kernel_params` on construction / `reset_with_seed` so
    /// `step()` can borrow it as `&[f32]` without per-step allocation.
    h_cache: Vec<f32>,
}

/// Side length (cells) of the central activation patch used by
/// [`FlowLeniaSimulator::new`] / [`reset_with_seed`]. Within the
/// DESIGN.md §6 "central 20..=40" range; sits in the middle so smaller
/// grids (e.g. 32×32 for tests) still get a non-trivial seed without
/// hitting the boundary.
const INITIAL_PATCH_SIDE: usize = 30;

impl FlowLeniaSimulator {
    /// Construct with random initial state (central patch) and random
    /// kernel parameters from the given seed.
    ///
    /// The same `(cfg, seed)` pair always produces the same simulator
    /// state — verified by `simulator_new_with_seed_is_reproducible`.
    ///
    /// # Panics
    ///
    /// Panics if `cfg.channels == 0` (no kernels to sample) or if
    /// either grid dimension is zero.
    #[must_use]
    pub fn new(cfg: FlowLeniaConfig, seed: u64) -> Self {
        assert!(cfg.channels > 0, "channels must be > 0");
        assert!(
            cfg.grid_height > 0 && cfg.grid_width > 0,
            "grid dims must be > 0"
        );

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let kernel_params = KernelParams::sample_random(
            &mut rng,
            SamplingSettings {
                num_kernels: cfg.num_kernels,
                num_channels: cfg.channels,
            },
        );
        let a = build_initial_state(&cfg, &mut rng);
        Self::from_components_internal(a, kernel_params, cfg)
    }

    /// Construct from a specific initial state and a pre-sampled kernel
    /// set. Useful for tests, fixture replay, and the M1.15 regression
    /// suite where exact reproducibility against a saved trajectory is
    /// the point.
    ///
    /// # Panics
    ///
    /// Panics if `a.shape() != (cfg.grid_height, cfg.grid_width, cfg.channels)`
    /// or `kernel_params.kernels.len() != cfg.num_kernels`.
    #[must_use]
    pub fn from_components(
        a: ActivationField,
        kernel_params: KernelParams,
        cfg: FlowLeniaConfig,
    ) -> Self {
        let (h, w, c) = a.dim();
        assert_eq!(
            (h, w, c),
            (
                cfg.grid_height as usize,
                cfg.grid_width as usize,
                cfg.channels as usize
            ),
            "a shape {:?} != cfg grid ({}, {}, {})",
            a.dim(),
            cfg.grid_height,
            cfg.grid_width,
            cfg.channels
        );
        assert_eq!(
            kernel_params.kernels.len(),
            cfg.num_kernels as usize,
            "kernel_params has {} kernels, cfg expects {}",
            kernel_params.kernels.len(),
            cfg.num_kernels
        );
        Self::from_components_internal(a, kernel_params, cfg)
    }

    fn from_components_internal(
        a: ActivationField,
        kernel_params: KernelParams,
        cfg: FlowLeniaConfig,
    ) -> Self {
        let kernels: Vec<Array2<f32>> = kernel_params
            .kernels
            .iter()
            .map(|entry| compute_kernel(kernel_params.r_global, entry))
            .collect();
        let kernel_meta: Vec<KernelMeta> = (0..kernel_params.kernels.len())
            .map(|i| KernelMeta::from_params(&kernel_params, i))
            .collect();
        let h_cache: Vec<f32> = kernel_params.kernels.iter().map(|e| e.h).collect();
        Self {
            a,
            kernels,
            kernel_meta,
            kernel_params,
            cfg,
            p_map: None,
            step_count: 0,
            h_cache,
        }
    }

    /// Advance one time-step.
    pub fn step(&mut self) {
        let weights = match &self.p_map {
            Some(p) => WeightsRef::Localized(p),
            None => WeightsRef::Constant(self.h_slice()),
        };
        // Borrow `a` immutably to compute the new field, then swap.
        let next = step(
            &self.a,
            &self.kernels,
            &self.kernel_meta,
            weights,
            &self.cfg,
        );
        self.a = next;
        self.step_count += 1;
    }

    /// Advance `n` time-steps. Equivalent to calling [`step`](Self::step)
    /// `n` times — bit-identical, verified by
    /// `simulator_step_many_equals_step_in_loop`.
    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Current activation field `A`.
    #[must_use]
    pub fn activation(&self) -> &ActivationField {
        &self.a
    }

    /// Number of `step`s taken since construction / last `reset_with_seed`.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Current configuration.
    #[must_use]
    pub fn config(&self) -> &FlowLeniaConfig {
        &self.cfg
    }

    /// Per-channel total mass `Σ_{y,x} A_c(y, x)`. Accumulated in `f64`
    /// internally to keep the result robust against grid-size growth,
    /// then cast back to `f32` per the value's natural type.
    ///
    /// Returns one entry per channel, in `0..C` order. Empty when
    /// `cfg.channels == 0` (currently impossible — `new` rejects it).
    #[must_use]
    pub fn total_mass(&self) -> Vec<f32> {
        let (_, _, c) = self.a.dim();
        (0..c)
            .map(|ci| {
                let sum_f64: f64 = self
                    .a
                    .index_axis(Axis(2), ci)
                    .iter()
                    .map(|&v| f64::from(v))
                    .sum();
                sum_f64 as f32
            })
            .collect()
    }

    /// Switch to the parameter-embedding (paper Eq. 7) variant by
    /// supplying a per-cell-per-kernel weight map `P_i(x)`. Subsequent
    /// `step` calls will use [`WeightsRef::Localized`] instead of the
    /// default [`WeightsRef::Constant`] (`h_i` from the kernel
    /// parameters).
    ///
    /// # Panics
    ///
    /// Panics if `p_map.shape() != (H, W, |K|)`.
    pub fn enable_localized_weights(&mut self, p_map: Array3<f32>) {
        let (h, w, _) = self.a.dim();
        assert_eq!(
            p_map.dim(),
            (h, w, self.kernels.len()),
            "p_map shape {:?} != expected ({}, {}, {})",
            p_map.dim(),
            h,
            w,
            self.kernels.len()
        );
        self.p_map = Some(p_map);
    }

    /// Revert to the constant per-kernel weight `h_i` form (paper Eq. 3).
    /// No-op if localized weights were not enabled.
    pub fn disable_localized_weights(&mut self) {
        self.p_map = None;
    }

    /// Reset to a fresh random state under the given seed. Reuses the
    /// existing `cfg` — the kernel bank is re-sampled and the
    /// activation patch redrawn.
    ///
    /// `step_count` is reset to 0; any active `p_map` is cleared.
    pub fn reset_with_seed(&mut self, seed: u64) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        self.kernel_params = KernelParams::sample_random(
            &mut rng,
            SamplingSettings {
                num_kernels: self.cfg.num_kernels,
                num_channels: self.cfg.channels,
            },
        );
        self.kernels = self
            .kernel_params
            .kernels
            .iter()
            .map(|entry| compute_kernel(self.kernel_params.r_global, entry))
            .collect();
        self.kernel_meta = (0..self.kernel_params.kernels.len())
            .map(|i| KernelMeta::from_params(&self.kernel_params, i))
            .collect();
        self.h_cache = self.kernel_params.kernels.iter().map(|e| e.h).collect();
        self.a = build_initial_state(&self.cfg, &mut rng);
        self.p_map = None;
        self.step_count = 0;
    }

    /// Cached slice of per-kernel `h_i` for the `Constant` step path.
    /// The slice borrow keeps the `step()` signature free of per-step
    /// allocations.
    fn h_slice(&self) -> &[f32] {
        self.h_cache.as_slice()
    }
}

fn build_initial_state(cfg: &FlowLeniaConfig, rng: &mut ChaCha8Rng) -> ActivationField {
    let h = cfg.grid_height as usize;
    let w = cfg.grid_width as usize;
    let c = cfg.channels as usize;
    let mut a = Array3::<f32>::zeros((h, w, c));

    let side = INITIAL_PATCH_SIDE.min(h).min(w);
    let y0 = (h - side) / 2;
    let x0 = (w - side) / 2;
    for y in y0..y0 + side {
        for x in x0..x0 + side {
            for ci in 0..c {
                a[[y, x, ci]] = rng.gen_range(0.0_f32..1.0);
            }
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BorderMode;
    use approx::assert_relative_eq;

    fn small_cfg() -> FlowLeniaConfig {
        FlowLeniaConfig {
            grid_width: 32,
            grid_height: 32,
            channels: 3,
            num_kernels: 6,
            border: BorderMode::Torus,
            ..FlowLeniaConfig::default()
        }
    }

    /// Same `(cfg, seed)` produces the same state — both kernels and
    /// initial activation. Asserted bit-equal for the kernels (deterministic
    /// f64 sum normalisation, see M1.4) and the activation (deterministic
    /// `gen_range` on `ChaCha8Rng`).
    #[test]
    fn simulator_new_with_seed_is_reproducible() {
        let s1 = FlowLeniaSimulator::new(small_cfg(), 0xDEADBEEF);
        let s2 = FlowLeniaSimulator::new(small_cfg(), 0xDEADBEEF);
        for (k1, k2) in s1.kernels.iter().zip(s2.kernels.iter()) {
            for (a, b) in k1.iter().zip(k2.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "kernel value diverged");
            }
        }
        for (a, b) in s1.a.iter().zip(s2.a.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "initial A diverged");
        }
        // And after 50 steps it should still be in lockstep.
        let mut s1 = s1;
        let mut s2 = s2;
        s1.step_many(50);
        s2.step_many(50);
        for (a, b) in s1.a.iter().zip(s2.a.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "A after 50 steps diverged");
        }
        assert_eq!(s1.step_count(), 50);
    }

    /// `simulator.step()` produces the same A_next as calling the
    /// pure-function `step()` directly on the same inputs.
    ///
    /// Asserted bit-equal: the simulator is a *thin* wrapper, so any
    /// mismatch is a bug in the wrapping (e.g. wrong weights variant,
    /// stale state).
    #[test]
    fn simulator_step_matches_pure_step() {
        let mut sim = FlowLeniaSimulator::new(small_cfg(), 0xC0FFEE);
        let a0 = sim.activation().clone();
        let h: Vec<f32> = sim.kernel_params.kernels.iter().map(|e| e.h).collect();
        let expected = step(
            &a0,
            &sim.kernels,
            &sim.kernel_meta,
            WeightsRef::Constant(&h),
            &sim.cfg,
        );
        sim.step();
        for (a, b) in sim.activation().iter().zip(expected.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(sim.step_count(), 1);
    }

    /// `step_many(n)` is bit-identical to calling `step()` `n` times.
    #[test]
    fn simulator_step_many_equals_step_in_loop() {
        let mut a = FlowLeniaSimulator::new(small_cfg(), 0xBEEF_1234);
        let mut b = FlowLeniaSimulator::new(small_cfg(), 0xBEEF_1234);
        a.step_many(10);
        for _ in 0..10 {
            b.step();
        }
        assert_eq!(a.step_count(), b.step_count());
        for (x, y) in a.activation().iter().zip(b.activation().iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    /// 100-step mass conservation, same standard as M1.13. The simulator
    /// must not introduce drift on top of `step`.
    #[test]
    fn simulator_mass_conservation_100_steps() {
        let mut sim = FlowLeniaSimulator::new(small_cfg(), 0x5EED);
        let m0: f64 = sim.total_mass().iter().map(|&v| f64::from(v)).sum();
        assert!(m0 > 0.0, "initial mass should be positive");
        let mut max_rel = 0.0_f64;
        for _ in 0..100 {
            sim.step();
            let m: f64 = sim.total_mass().iter().map(|&v| f64::from(v)).sum();
            let rel = (m - m0).abs() / m0;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        // M1.13 step-level table reports ~4.5e-6; same scale expected here.
        assert!(max_rel < 1e-3, "max_rel = {max_rel:.3e}");
    }

    /// `enable_localized_weights` with `P_i ≡ h_i` reproduces the
    /// `Constant`-mode trajectory (within 1e-6 — see the M1.13
    /// `step_localized_with_uniform_p_equals_constant_h` rationale).
    /// Toggling back to disabled restores `Constant`-mode behaviour.
    #[test]
    fn simulator_enable_disable_localized_weights() {
        let mut sim_a = FlowLeniaSimulator::new(small_cfg(), 0xAB1234);
        let mut sim_b = FlowLeniaSimulator::new(small_cfg(), 0xAB1234);

        // Build P ≡ h.
        let (h, w, _) = sim_b.a.dim();
        let k = sim_b.kernels.len();
        let mut p: Array3<f32> = Array3::zeros((h, w, k));
        for i in 0..k {
            let hi = sim_b.kernel_params.kernels[i].h;
            for y in 0..h {
                for x in 0..w {
                    p[[y, x, i]] = hi;
                }
            }
        }
        sim_b.enable_localized_weights(p);

        // 5 localized steps then disable, then 5 constant steps. The
        // result should match 10 constant steps on sim_a within 1e-6.
        sim_a.step_many(10);
        sim_b.step_many(5);
        sim_b.disable_localized_weights();
        sim_b.step_many(5);
        assert_eq!(sim_a.step_count(), sim_b.step_count());
        for (x, y) in sim_a.activation().iter().zip(sim_b.activation().iter()) {
            assert_relative_eq!(*x, *y, epsilon = 1e-6);
        }
    }

    /// `reset_with_seed` rebuilds both A and kernels deterministically.
    #[test]
    fn simulator_reset_with_seed_is_deterministic() {
        let mut sim = FlowLeniaSimulator::new(small_cfg(), 1);
        sim.step_many(20);
        sim.reset_with_seed(0xFACE);
        let s2 = FlowLeniaSimulator::new(small_cfg(), 0xFACE);
        assert_eq!(sim.step_count(), 0);
        for (a, b) in sim.activation().iter().zip(s2.activation().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (k1, k2) in sim.kernels.iter().zip(s2.kernels.iter()) {
            for (a, b) in k1.iter().zip(k2.iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    /// `total_mass` returns one entry per channel.
    #[test]
    fn simulator_total_mass_is_per_channel() {
        let sim = FlowLeniaSimulator::new(small_cfg(), 0x7);
        let m = sim.total_mass();
        assert_eq!(m.len(), small_cfg().channels as usize);
        // Initial state is a central uniform patch — every channel gets
        // its own draw, so all three should be positive and similar
        // (within a factor of 2 for the random distribution).
        for &v in &m {
            assert!(v > 0.0, "channel mass should be positive on init");
        }
    }
}
