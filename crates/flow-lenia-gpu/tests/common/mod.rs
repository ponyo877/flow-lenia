//! Shared test helpers for `flow-lenia-gpu` integration tests.
//!
//! Mirrors `crates/flow-lenia-core/tests/common/mod.rs`. The two
//! copies are intentionally identical — Rust integration tests cannot
//! depend on auxiliary modules across crate boundaries without
//! exposing a `#[cfg(test)]`-only public API on the library side,
//! which would pollute the production crate signature for what is a
//! pure test-support concern. Long-term, if a third consumer
//! appears, this lifts to a `flow-lenia-testkit` dev-only crate; for
//! M6.A the duplication cost is minimal.
//!
//! `#![allow(dead_code)]` is intentional: not every test binary uses
//! every helper. Lints still apply to the helper bodies themselves.

#![allow(dead_code)]

use flow_lenia_core::FlowLeniaConfig;

/// M6.A.11 — sanity-check the activation field. See the
/// `flow-lenia-core` copy for the full rationale.
pub fn assert_creature_alive(activation: &[f32], cfg: &FlowLeniaConfig) {
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

    let total: f32 = activation.iter().sum();
    let scale = (cfg.grid_width * cfg.grid_height * cfg.channels) as f32;
    assert!(
        total > scale * 1e-3 && total < scale * 100.0,
        "creature mass anomaly: total={total:.3e}, scale~{scale:.3e} \
         (expected total / scale ∈ [1e-3, 100])",
    );

    let max_val = activation.iter().copied().fold(0.0f32, f32::max);
    assert!(
        max_val > 1e-3,
        "creature died: max activation = {max_val:.3e} (need > 1e-3)",
    );
}
