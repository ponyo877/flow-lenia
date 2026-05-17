//! Reintegration tracking — paper Eq. 6, mass-conserving update step.
//!
//! ```text
//! A^{t+dt}_i(x) = Σ_{x' ∈ N(x)}  A^t_i(x')  ·  I_i(x', x)
//! I_i(x', x)    = ∫_{Ω(x)} D(x' + dt·F^t_i(x'), σ) dA
//! ```
//!
//! where `N(x)` is the **receiver-side** neighbourhood — JAX
//! (`reintegration_tracking.py:39`) uses `dd = 5` ⇒ an 11×11 patch
//! around each target cell `x`. DESIGN.md §4.1.3 fixes this for our
//! implementation as well (Q1A confirmed).
//!
//! Three correctness-critical pieces other than the overlap area itself
//! (which lives in [`crate::overlap`]):
//!
//! 1. **`ma = dd − σ` clip on `dt·F`** (JAX
//!    `reintegration_tracking.py:62`). Bounds the distribution centre's
//!    displacement so that, even at the maximum reach `|dt·F| + σ = dd`,
//!    the receiver-side `dd`-neighbourhood still sees every cell that
//!    can contribute mass. Without this, fast-flow regimes leak mass
//!    out of the neighbourhood window.
//!
//! 2. **Wall μ-clip** (JAX `reintegration_tracking.py:64-65`): in
//!    `BorderMode::Wall` the distribution centre `μ = src + dt·F` is
//!    clipped to `[σ, side − σ]` so the distribution stays inside the
//!    grid. Torus does not need this — wrap handles displacement
//!    beyond the boundary.
//!
//! 3. **Receiver-side `dpmu` is computed in *logical* coordinates**
//!    (`target − source_logical_unwrapped`), *not* using the
//!    grid-wrapped source index. The wrap is applied only to the array
//!    index used to *read* `A` and `F`. This keeps the neighbourhood
//!    semantics correct on small grids where `dd > H/2`, where a
//!    single physical cell appears at multiple offsets (each carrying
//!    its own `dpmu`).
//!
//! Two implementations live here:
//! - [`reintegrate`] (public) — `cfg.dd`-bounded neighbourhood loop.
//! - [`reintegrate_naive`] (`pub(crate)`) — `dd = max(H, W)` (i.e. the
//!   full grid as the neighbourhood). Used by
//!   `reintegrate_naive_matches_pruned_neighborhood` to regression-lock
//!   the equivalence (the `ma` clip plus the overlap area's far-distance
//!   zero already constrain mass to the `dd`-neighbourhood, so the two
//!   implementations are *bit-for-bit equal*).

use crate::config::{BorderMode, FlowLeniaConfig};
use crate::overlap::overlap_area;
use crate::state::{ActivationField, FlowField, FLOW_DX, FLOW_DY};
use ndarray::Array3;

/// Yields the receiver-side neighbourhood offsets `(dy, dx)` in row-major
/// order (`dy` outer, `dx` inner), each in `-dd..=dd`. Used by both Eq. 6
/// (mass) and the upcoming Eq. 8 (parameter mixing) loops, so the order
/// is pinned here once.
pub(crate) fn neighborhood_offsets(dd: i32) -> impl Iterator<Item = (i32, i32)> {
    (-dd..=dd).flat_map(move |dy| (-dd..=dd).map(move |dx| (dy, dx)))
}

/// Resolve a logical `(ty + dy_off, tx + dx_off)` source position to the
/// actual grid index, returning `None` if the cell is out of the grid
/// in `Wall` mode. In `Torus` mode the wrap is always defined.
#[inline]
fn resolve_source(
    ty: i32,
    tx: i32,
    dy_off: i32,
    dx_off: i32,
    h: usize,
    w: usize,
    border: BorderMode,
) -> Option<(usize, usize)> {
    let logical_sy = ty + dy_off;
    let logical_sx = tx + dx_off;
    match border {
        BorderMode::Torus => Some((
            logical_sy.rem_euclid(h as i32) as usize,
            logical_sx.rem_euclid(w as i32) as usize,
        )),
        BorderMode::Wall => {
            if logical_sy < 0 || logical_sy >= h as i32 || logical_sx < 0 || logical_sx >= w as i32
            {
                None
            } else {
                Some((logical_sy as usize, logical_sx as usize))
            }
        }
    }
}

/// Core receiver-side reintegration loop, parameterised by the
/// neighbourhood half-width.
///
/// `neighborhood_dd` is the *loop bound*; `cfg.dd` is the *physics bound*
/// that determines the `ma` clip. The two coincide for the public
/// [`reintegrate`] (`neighborhood_dd = cfg.dd`); the `naive` variant
/// just enlarges the loop bound to cover the full grid.
fn reintegrate_impl(
    a: &ActivationField,
    flow: &FlowField,
    cfg: &FlowLeniaConfig,
    neighborhood_dd: i32,
) -> ActivationField {
    let (h, w, c) = a.dim();
    debug_assert_eq!(flow.dim(), (h, w, 2, c));

    let sigma = cfg.sigma;
    let dt = cfg.dt;
    let ma = (cfg.dd as f32) - sigma; // physics flow-magnitude clip

    let mut out: ActivationField = Array3::zeros((h, w, c));

    let h_i = h as i32;
    let w_i = w as i32;
    let h_f = h as f32;
    let w_f = w as f32;

    for ty in 0..h_i {
        for tx in 0..w_i {
            let target_y = ty as f32 + 0.5;
            let target_x = tx as f32 + 0.5;

            // For each target cell we accumulate per-channel sums in a
            // small stack buffer of size C, then write the row out at the
            // end. ndarray indexed writes inside the inner loop would be
            // unnecessarily checked; this hoists C bounds checks.
            // (Profile this when M1.15 hits, not now.)
            for ci in 0..c {
                let mut sum = 0.0_f32;
                for (dy_off, dx_off) in neighborhood_offsets(neighborhood_dd) {
                    let Some((sy_idx, sx_idx)) =
                        resolve_source(ty, tx, dy_off, dx_off, h, w, cfg.border)
                    else {
                        // Wall: out-of-grid source contributes nothing
                        // (zero-pad on A; the distribution centre clip
                        // below handles any flow extending out of the
                        // grid for in-grid sources).
                        continue;
                    };

                    // `ma`-clipped per-cell flow.
                    let dt_fy = (dt * flow[[sy_idx, sx_idx, FLOW_DY, ci]]).clamp(-ma, ma);
                    let dt_fx = (dt * flow[[sy_idx, sx_idx, FLOW_DX, ci]]).clamp(-ma, ma);

                    // Distribution centre μ, in *logical* (unwrapped)
                    // coordinates — this is what gives the correct dpmu
                    // even when wrap collapses two `dy_off` values to the
                    // same `sy_idx` (small-grid case).
                    let logical_sy = ty + dy_off;
                    let logical_sx = tx + dx_off;
                    let mu_y = logical_sy as f32 + 0.5 + dt_fy;
                    let mu_x = logical_sx as f32 + 0.5 + dt_fx;

                    // Wall μ-clip — JAX `reintegration_tracking.py:64-65`.
                    let (mu_y, mu_x) = match cfg.border {
                        BorderMode::Wall => (
                            mu_y.clamp(sigma, h_f - sigma),
                            mu_x.clamp(sigma, w_f - sigma),
                        ),
                        BorderMode::Torus => (mu_y, mu_x),
                    };

                    let dpmu_y = target_y - mu_y;
                    let dpmu_x = target_x - mu_x;
                    let i_val = overlap_area(dpmu_y, dpmu_x, sigma);

                    sum += a[[sy_idx, sx_idx, ci]] * i_val;
                }
                out[[ty as usize, tx as usize, ci]] = sum;
            }
        }
    }
    out
}

/// One Flow-Lenia step using `cfg.dd`-bounded receiver-side reintegration
/// (paper Eq. 6).
///
/// Iterates the 11×11 (default `dd = 5`) neighbourhood per target cell.
/// The full update rule is then this function chained with
/// [`crate::alpha::alpha`], [`crate::sobel::grad_a_sum`] /
/// [`crate::sobel::sobel_per_channel`], and [`crate::flow::flow`];
/// M1.13 will assemble these into a single `step()`.
#[must_use]
pub fn reintegrate(
    a: &ActivationField,
    flow: &FlowField,
    cfg: &FlowLeniaConfig,
) -> ActivationField {
    reintegrate_impl(a, flow, cfg, cfg.dd as i32)
}

/// Full-grid (un-pruned) receiver-side reintegration. Used by tests to
/// pin the equivalence with the production [`reintegrate`] — the `ma`
/// clip plus the overlap area's far-distance zero confine all
/// non-zero contributions to the `cfg.dd`-neighbourhood, so the two
/// implementations are mathematically (and numerically, since the
/// summation order is unchanged for the in-neighbourhood cells)
/// identical.
#[cfg(test)]
#[must_use]
pub(crate) fn reintegrate_naive(
    a: &ActivationField,
    flow: &FlowField,
    cfg: &FlowLeniaConfig,
) -> ActivationField {
    let (h, w, _c) = a.dim();
    let max_dd = (h.max(w)) as i32;
    reintegrate_impl(a, flow, cfg, max_dd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BorderMode, MixRule};
    use crate::state::AXIS_C;
    use approx::assert_relative_eq;
    use ndarray::{Array3, Array4};

    fn cfg_torus(sigma: f32, dt: f32, dd: u32, channels: u32) -> FlowLeniaConfig {
        FlowLeniaConfig {
            grid_width: 16,
            grid_height: 16,
            channels,
            dt,
            sigma,
            n: 2.0,
            beta_a: 2.0,
            dd,
            paper_strict: false,
            border: BorderMode::Torus,
            mix_rule: MixRule::Stochastic,
        }
    }

    // ─── Stage 1: translation tests (the most important M1.11 coverage) ──

    /// `F = 0` + `σ < 0.5` → distribution narrower than a cell + no
    /// displacement → reintegration is the identity. M1.10's
    /// `overlap_area_sigma_below_half_uses_2sigma_clip` pins the centre
    /// cell `I = 1`; here we exercise the integration of that property
    /// into the full step.
    #[test]
    fn reintegrate_zero_flow_with_narrow_sigma_is_identity() {
        let h = 8;
        let w = 8;
        let c = 1;
        let mut a: ActivationField = Array3::zeros((h, w, c));
        a[[4, 4, 0]] = 1.0;
        a[[3, 5, 0]] = 0.5;
        let flow: FlowField = Array4::zeros((h, w, 2, c));

        let cfg = cfg_torus(0.3, 1.0, 5, c as u32); // σ < 0.5 → δ-like
        let next = reintegrate(&a, &flow, &cfg);

        for ((y, x, ci), &v) in next.indexed_iter() {
            assert_relative_eq!(v, a[[y, x, ci]], epsilon = 1e-6);
        }
    }

    /// Translation by exactly one cell to the right, `σ = 0.5`. With
    /// the matched cell/distribution width, **all** of the mass at
    /// `(4, 4)` should arrive at `(4, 5)`, leaving the rest of the grid
    /// at zero.
    #[test]
    fn reintegrate_uniform_translation_one_cell_right_torus() {
        let h = 8;
        let w = 8;
        let c = 1;
        let mut a: ActivationField = Array3::zeros((h, w, c));
        a[[4, 4, 0]] = 1.0;

        // dt·F = (0, 1) uniformly. σ = 0.5 (exactly one cell wide).
        let mut flow: FlowField = Array4::zeros((h, w, 2, c));
        flow.fill(0.0);
        flow.slice_mut(ndarray::s![.., .., FLOW_DX, ..]).fill(1.0);

        let cfg = cfg_torus(0.5, 1.0, 5, c as u32);
        let next = reintegrate(&a, &flow, &cfg);

        // All of A[4, 4] arrived at A[4, 5].
        assert_relative_eq!(next[[4, 5, 0]], 1.0, epsilon = 1e-6);
        // Everything else is zero (no spread).
        for ((y, x, ci), &v) in next.indexed_iter() {
            if (y, x, ci) != (4, 5, 0) {
                assert!(
                    v.abs() < 1e-6,
                    "spurious mass at ({y}, {x}, {ci}) = {v} (expected 0)"
                );
            }
        }
    }

    /// Translation by one cell down (axis sanity, transpose of the
    /// previous test).
    #[test]
    fn reintegrate_uniform_translation_one_cell_down_torus() {
        let h = 8;
        let w = 8;
        let c = 1;
        let mut a: ActivationField = Array3::zeros((h, w, c));
        a[[4, 4, 0]] = 1.0;

        let mut flow: FlowField = Array4::zeros((h, w, 2, c));
        flow.slice_mut(ndarray::s![.., .., FLOW_DY, ..]).fill(1.0);

        let cfg = cfg_torus(0.5, 1.0, 5, c as u32);
        let next = reintegrate(&a, &flow, &cfg);

        assert_relative_eq!(next[[5, 4, 0]], 1.0, epsilon = 1e-6);
        for ((y, x, ci), &v) in next.indexed_iter() {
            if (y, x, ci) != (5, 4, 0) {
                assert!(v.abs() < 1e-6, "spurious mass at ({y}, {x}, {ci}) = {v}");
            }
        }
    }

    /// Translation diagonally by (1, 1). The distribution lands centred
    /// on `(5, 5)`.
    #[test]
    fn reintegrate_uniform_translation_diagonal() {
        let h = 8;
        let w = 8;
        let c = 1;
        let mut a: ActivationField = Array3::zeros((h, w, c));
        a[[4, 4, 0]] = 1.0;

        let mut flow: FlowField = Array4::zeros((h, w, 2, c));
        flow.slice_mut(ndarray::s![.., .., FLOW_DY, ..]).fill(1.0);
        flow.slice_mut(ndarray::s![.., .., FLOW_DX, ..]).fill(1.0);

        let cfg = cfg_torus(0.5, 1.0, 5, c as u32);
        let next = reintegrate(&a, &flow, &cfg);

        assert_relative_eq!(next[[5, 5, 0]], 1.0, epsilon = 1e-6);
        for ((y, x, ci), &v) in next.indexed_iter() {
            if (y, x, ci) != (5, 5, 0) {
                assert!(v.abs() < 1e-6, "spurious mass at ({y}, {x}, {ci}) = {v}");
            }
        }
    }

    /// **Sub-cell translation** — the most important test for reintegration
    /// correctness. With `dt·F = (0, 0.5)`, the distribution at source
    /// `(4, 4)` has centre `(4.5, 5.0)`. With `σ = 0.5`:
    ///
    /// - target `(4, 4)`, centre `(4.5, 4.5)`: dpmu = `(0, -0.5)`,
    ///   `sz_x = clamp(0.5 - 0.5 + 0.5, 0, 1) = 0.5`, `sz_y = 1`,
    ///   `I = (0.5 · 1) / (4·0.25) = 0.5`
    /// - target `(4, 5)`, centre `(4.5, 5.5)`: dpmu = `(0, 0.5)`,
    ///   symmetric → `I = 0.5`
    /// - everything else: `I = 0`
    ///
    /// Expected output: mass at `(4, 4)` is split exactly 50/50 between
    /// `(4, 4)` and `(4, 5)`. This is the classic case that catches
    /// integer-step-only bugs.
    #[test]
    fn reintegrate_uniform_translation_subcell() {
        let h = 8;
        let w = 8;
        let c = 1;
        let mut a: ActivationField = Array3::zeros((h, w, c));
        a[[4, 4, 0]] = 1.0;

        let mut flow: FlowField = Array4::zeros((h, w, 2, c));
        flow.slice_mut(ndarray::s![.., .., FLOW_DX, ..]).fill(0.5);

        let cfg = cfg_torus(0.5, 1.0, 5, c as u32);
        let next = reintegrate(&a, &flow, &cfg);

        assert_relative_eq!(next[[4, 4, 0]], 0.5, epsilon = 1e-6);
        assert_relative_eq!(next[[4, 5, 0]], 0.5, epsilon = 1e-6);
        for ((y, x, ci), &v) in next.indexed_iter() {
            if (y, x, ci) != (4, 4, 0) && (y, x, ci) != (4, 5, 0) {
                assert!(v.abs() < 1e-6, "spurious mass at ({y}, {x}, {ci}) = {v}");
            }
        }
    }

    // ─── Stage 2: mass conservation across multiple steps ─────────────

    /// Build a per-channel divergence-free random flow scaled so
    /// `|dt·F| < ma`, then iterate `n_steps` of `reintegrate` and assert
    /// per-channel mass is preserved.
    fn run_mass_conservation(
        h: usize,
        w: usize,
        c: usize,
        n_steps: usize,
        border: BorderMode,
        seed: u64,
    ) -> Vec<(f32, f32)> {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut a: ActivationField =
            Array3::from_shape_fn((h, w, c), |_| rng.gen_range(0.0_f32..1.0));

        // Per-channel "spinning" flow at modest magnitude (well within ma).
        let flow: FlowField = Array4::from_shape_fn((h, w, 2, c), |(_, _, fi, _)| {
            // Encourage motion but keep ‖dt·F‖ small enough that the test
            // doesn't accidentally test the ma clip.
            if fi == FLOW_DY {
                0.2_f32
            } else {
                -0.2_f32
            }
        });

        let cfg = FlowLeniaConfig {
            grid_width: w as u32,
            grid_height: h as u32,
            channels: c as u32,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            paper_strict: false,
            border,
            mix_rule: MixRule::Stochastic,
        };

        let initial_mass_per_channel: Vec<f32> =
            (0..c).map(|ci| a.index_axis(AXIS_C, ci).sum()).collect();

        for _ in 0..n_steps {
            a = reintegrate(&a, &flow, &cfg);
        }

        let final_mass_per_channel: Vec<f32> =
            (0..c).map(|ci| a.index_axis(AXIS_C, ci).sum()).collect();

        initial_mass_per_channel
            .into_iter()
            .zip(final_mass_per_channel)
            .collect()
    }

    /// Mass is preserved across 100 steps on a 32×32 / C=1 torus grid
    /// (relative error < 1e-3 per channel — DESIGN.md §5.3 target).
    #[test]
    fn reintegrate_mass_conservation_torus_c1() {
        for &seed in &[7_u64, 42, 0xDEAD_BEEF] {
            let pairs = run_mass_conservation(32, 32, 1, 100, BorderMode::Torus, seed);
            for (i, (m0, m1)) in pairs.iter().enumerate() {
                let rel = (m1 - m0).abs() / m0.abs().max(1e-6);
                assert!(
                    rel < 1e-3,
                    "torus C=1 seed={seed} ch={i}: m0 = {m0}, m1 = {m1}, rel = {rel:e}"
                );
            }
        }
    }

    /// Same on C=3 — independent per-channel mass should each be
    /// preserved (we are *not* yet using parameter mixing / Eq. 8).
    #[test]
    fn reintegrate_mass_conservation_torus_c3() {
        for &seed in &[11_u64, 99] {
            let pairs = run_mass_conservation(32, 32, 3, 100, BorderMode::Torus, seed);
            for (i, (m0, m1)) in pairs.iter().enumerate() {
                let rel = (m1 - m0).abs() / m0.abs().max(1e-6);
                assert!(
                    rel < 1e-3,
                    "torus C=3 seed={seed} ch={i}: m0 = {m0}, m1 = {m1}, rel = {rel:e}"
                );
            }
        }
    }

    /// Wall border: the JAX μ-clip (`reintegration_tracking.py:64-65`) is
    /// *non-conservative*. It re-centres the distribution inside the
    /// grid, which can both gain (boundary concentration) and lose
    /// (out-of-grid source zero-pad) small amounts of mass per step.
    /// JAX exhibits the same drift.
    ///
    /// We test that the drift stays within an order-of-magnitude bound
    /// (1 % over 100 steps for this seed / flow pattern) so a
    /// regression that broke the μ-clip would still be caught — the
    /// uncorrected behaviour would diverge much further. The lower
    /// bound (≥ 50 % retention) catches the catastrophic case where
    /// most mass leaks out.
    #[test]
    fn reintegrate_mass_conservation_wall_c1() {
        let pairs = run_mass_conservation(32, 32, 1, 100, BorderMode::Wall, 13);
        for (i, (m0, m1)) in pairs.iter().enumerate() {
            let rel = (m1 - m0).abs() / m0.abs().max(1e-6);
            assert!(
                rel < 1e-2,
                "wall ch={i}: m0 = {m0} → m1 = {m1} (rel = {rel:e})"
            );
            assert!(
                *m1 > 0.5 * m0,
                "wall ch={i}: lost too much mass m0={m0} → m1={m1}"
            );
        }
    }

    // ─── Stage 3: naive ↔ pruned equivalence ────────────────────────

    /// The `ma = dd − σ` clip plus the overlap area's far-distance zero
    /// guarantee that mass cannot land outside the `dd`-neighbourhood,
    /// so the production `reintegrate` (loop bound `cfg.dd`) and
    /// `reintegrate_naive` (loop bound `max(H, W)`) must produce the
    /// same result — exercised here for both torus and wall.
    #[test]
    fn reintegrate_naive_matches_pruned_neighborhood() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(2024);
        let h = 16;
        let w = 16;
        let c = 2;
        let a: ActivationField = Array3::from_shape_fn((h, w, c), |_| rng.gen_range(0.0_f32..1.0));
        let flow: FlowField = Array4::from_shape_fn((h, w, 2, c), |_| rng.gen_range(-1.0_f32..1.0));

        for &border in &[BorderMode::Torus, BorderMode::Wall] {
            let cfg = FlowLeniaConfig {
                border,
                ..cfg_torus(0.65, 0.2, 5, c as u32)
            };
            let pruned = reintegrate(&a, &flow, &cfg);
            let naive = reintegrate_naive(&a, &flow, &cfg);
            for ((y, x, ci), &p) in pruned.indexed_iter() {
                let n = naive[[y, x, ci]];
                assert_relative_eq!(p, n, epsilon = 1e-6);
            }
        }
    }

    /// **Diagnostic** — print the measured relative mass-conservation
    /// drift across the configurations exercised by the main mass tests.
    /// `#[ignore]` so it does not slow normal `cargo test`. Run with:
    ///
    ///   `cargo test reintegrate diagnose_mass_drift -- --nocapture --include-ignored`
    #[test]
    #[ignore = "diagnostic only"]
    fn diagnose_mass_drift() {
        let cases: [(BorderMode, usize, &str); 4] = [
            (BorderMode::Torus, 1, "torus C=1"),
            (BorderMode::Torus, 3, "torus C=3"),
            (BorderMode::Wall, 1, "wall  C=1"),
            (BorderMode::Wall, 3, "wall  C=3"),
        ];
        for (border, c, label) in cases {
            for seed in [7_u64, 42, 0xDEAD_BEEF] {
                let pairs = run_mass_conservation(32, 32, c, 100, border, seed);
                let mut max_rel = 0.0_f32;
                for (m0, m1) in &pairs {
                    let rel = (m1 - m0).abs() / m0.abs().max(1e-6);
                    if rel > max_rel {
                        max_rel = rel;
                    }
                }
                println!("{label}  seed={seed:#018x}  max_rel = {max_rel:.3e}");
            }
        }
    }

    // ─── Stage 4: ma clip behaviour ──────────────────────────────────

    /// Huge flow magnitudes get clipped by `ma = dd − σ`; the
    /// distribution stays within the neighbourhood window and mass is
    /// preserved on torus. Without the clip, a `dt·F = 100` flow would
    /// place the distribution centre 100 cells away and the
    /// neighbourhood window would miss most of it — mass would *appear*
    /// to vanish (since the receiver-side loop only looks 11 cells out).
    #[test]
    fn reintegrate_flow_clipped_at_ma_boundary() {
        let h = 16;
        let w = 16;
        let c = 1;
        let mut a: ActivationField = Array3::zeros((h, w, c));
        for y in 6..10 {
            for x in 6..10 {
                a[[y, x, 0]] = 1.0;
            }
        }
        let initial_mass: f32 = a.iter().sum();

        // |dt·F| = 100 ≫ ma = 5 − 0.65 = 4.35, so the clip must kick in.
        let flow: FlowField = Array4::from_elem((h, w, 2, c), 100.0);

        let cfg = cfg_torus(0.65, 1.0, 5, c as u32);
        let next = reintegrate(&a, &flow, &cfg);

        let new_mass: f32 = next.iter().sum();
        let rel = (new_mass - initial_mass).abs() / initial_mass.abs();
        assert!(
            rel < 1e-4,
            "ma clip failure: initial_mass = {initial_mass}, new_mass = {new_mass}, rel = {rel:e}"
        );
    }
}
