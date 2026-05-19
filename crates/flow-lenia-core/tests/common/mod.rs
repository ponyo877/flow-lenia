//! Shared helpers for the flow-lenia-core integration tests.
//!
//! This file lives at `tests/common/mod.rs` rather than as a top-level
//! `tests/common.rs` so cargo's test harness treats it as auxiliary
//! code rather than a standalone test binary. Individual tests pull
//! it in via `mod common;`.
//!
//! `#![allow(dead_code)]` is intentional: not every test binary uses
//! every helper. Lints still apply to the helper bodies themselves.

#![allow(dead_code)]

use flow_lenia_core::FlowLeniaConfig;

/// M6.A.11 — sanity-check the activation field. Used by the M6 mass
/// conservation suite (and any future long-horizon C=3 test) to catch
/// the failure modes that *don't* show up in a relative-error mass
/// comparison:
///
/// 1. **NaN / Inf** — `is_finite` reports false. f32 chaos at C=3 can
///    push values to ±Inf if a kernel re-init goes wrong, and a NaN
///    mass passes a `rel < 1e-3` check vacuously (NaN comparisons
///    return false).
/// 2. **Creature died** — every cell collapsed to zero. Mass would
///    still satisfy `rel < tol` (it's just `m₀ - 0 / m₀ = 1`, which
///    is technically above 1e-3, but if the death is slow it may sit
///    just under the threshold for a while).
/// 3. **Runaway scale** — total mass blew up by orders of magnitude.
///    Indicates the simulation diverged numerically. The legitimate
///    activation range under bounded paper-spec initial patches is
///    O(grid × channels); we accept up to 100× growth before flagging.
///
/// Panics on any of the above with a message that names the failure
/// mode plus the offending statistic, so a future M6 step that
/// introduces (say) an FFT bug at the kernel-padding boundary points
/// at the right place immediately.
pub fn assert_creature_alive(activation: &[f32], cfg: &FlowLeniaConfig) {
    // (1) — no non-finite values.
    let non_finite = activation.iter().filter(|v| !v.is_finite()).count();
    assert!(
        non_finite == 0,
        "creature exploded: {non_finite} non-finite values out of {} \
         (grid={}×{}, channels={})",
        activation.len(),
        cfg.grid_width,
        cfg.grid_height,
        cfg.channels,
    );

    // (2) and (3) combined — total mass within sane range. Scale is
    // `grid² × channels`; bound on either side leaves room for natural
    // density swings without trapping a real bug.
    let total: f32 = activation.iter().sum();
    let scale = (cfg.grid_width * cfg.grid_height * cfg.channels) as f32;
    assert!(
        total > scale * 1e-3 && total < scale * 100.0,
        "creature mass anomaly: total={total:.3e}, scale~{scale:.3e} \
         (expected total / scale ∈ [1e-3, 100])",
    );

    // (2 strict) — at least one cell above the death threshold. Catches
    // the case where mass is finite but every cell sits at noise (1e-6
    // ish) so the creature is effectively gone even if `total` is
    // still measurable.
    let max_val = activation.iter().copied().fold(0.0f32, f32::max);
    assert!(
        max_val > 1e-3,
        "creature died: max activation = {max_val:.3e} (need > 1e-3)",
    );
}
