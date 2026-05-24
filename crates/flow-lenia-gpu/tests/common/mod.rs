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
use flow_lenia_gpu::{validation::ValidationGuard, GpuContext};

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

/// M6.A.7 — test-only GPU context constructor with optional WebGPU
/// validation. Returns `Some(ValidationGuard)` when the
/// `FLOW_LENIA_VALIDATE` env var is set (any value); otherwise `None`.
///
/// All flow-lenia-gpu integration tests that exercise the full
/// pipeline call this instead of constructing `GpuContext` directly,
/// so a single env var flip enables the validation sweep across the
/// whole suite without per-test edits.
///
/// Caller pattern:
///
/// ```ignore
/// let (ctx, guard) = test_ctx();
/// // ... run pipeline ...
/// if let Some(g) = &guard {
///     g.assert_no_errors();
/// }
/// ```
///
/// Production callers (`flow-lenia-app` native binary,
/// `flow-lenia-web`) construct `GpuContext` directly via
/// `new_blocking` and never opt into validation — see CLAUDE.md
/// "production code への validation 不適用" rationale.
#[must_use]
pub fn test_ctx() -> (GpuContext, Option<ValidationGuard>) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);
    let guard = if std::env::var("FLOW_LENIA_VALIDATE").is_ok() {
        Some(ValidationGuard::new(&ctx.device))
    } else {
        None
    };
    (ctx, guard)
}
