//! M6.A.7 — test-only WebGPU validation error capture.
//!
//! `wgpu` validation surfaces shader/binding/usage errors through
//! `Device::on_uncaptured_error`. By default the callback panics; for
//! the M6.A.7 test sweep we want to (a) collect them into a vector
//! so the test can decide whether to fail, and (b) leave production
//! `GpuContext` callers (`flow-lenia-app` native, `flow-lenia-web`)
//! untouched so end-users don't pay the validation cost.
//!
//! The full design:
//!
//! - This module is opt-in. `GpuContext::new_blocking` does not call
//!   it. Tests construct a `ValidationGuard` after they have a
//!   `wgpu::Device`, register the callback, run their workload,
//!   then call `assert_no_errors` at the end.
//! - The guard owns an `Arc<Mutex<Vec<String>>>` because the wgpu
//!   callback signature (`Box<dyn Fn(_) + Send + Sync>`) outlives any
//!   borrow we could give it.
//! - The error formatter uses `Debug` rather than `Display` so the
//!   full wgpu error chain (validation message + source location)
//!   lands in the test failure output.
//!
//! `tests/common/mod.rs`'s `test_ctx()` returns this guard wrapped in
//! `Option`, gated on the `FLOW_LENIA_VALIDATE=1` env var.

use std::sync::{Arc, Mutex};

/// Collects `Device::on_uncaptured_error` callbacks (the wgpu
/// `Validation` / `OutOfMemory` / `Internal` enum variants) into a
/// vector and surfaces them on demand. Hold one per test that wants
/// validation coverage; drop without calling `assert_no_errors` if
/// you only want the side-effect (rare).
///
/// The M6.A.7 smoke test (`tests/validation_smoke.rs`) currently
/// exercises only the `Validation`-via-buffer-copy-overrun path. The
/// `Arc<Mutex<…>>` collector handles all error variants uniformly, but
/// the empirical coverage of "ValidationGuard catches *every* class of
/// wgpu error" is bounded by what the smoke tests trigger. Add a
/// new should-panic smoke when a different class becomes relevant
/// (e.g. bind-group layout mismatch, shader OOB).
///
/// Intended for use after pipeline construction, *not* during
/// steady-state perf measurement: the `Mutex` serialises every error
/// callback, and a high-throughput perf run that surfaces hundreds of
/// errors per second would contribute lock contention to the very
/// overhead the perf test is measuring. The M6.A.7 sweep (zero errors
/// across the M6.A.7 integration-test surface — see BENCH.md §10
/// "Coverage scope" for the per-binary count and the lib unit tests
/// still uncovered) trivially satisfies this constraint.
pub struct ValidationGuard {
    errors: Arc<Mutex<Vec<String>>>,
}

impl ValidationGuard {
    /// Install the uncaptured-error callback on `device`. There can
    /// be only one callback per device; if a previous `ValidationGuard`
    /// targeted the same device, it is replaced (wgpu does not stack
    /// callbacks).
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let errors_for_callback = Arc::clone(&errors);
        // wgpu 29 expects `Arc<dyn UncapturedErrorHandler>` (not Box);
        // the trait has a blanket impl for `Fn(Error) + Send + Sync`,
        // so an `Arc<closure>` satisfies it directly.
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            // `Debug` instead of `Display` so the full wgpu error
            // chain (validation message + source location) lands in
            // the test failure output.
            errors_for_callback
                .lock()
                .expect("validation errors mutex poisoned")
                .push(format!("{error:?}"));
        }));
        Self { errors }
    }

    /// Panic if any validation error was captured since this guard was
    /// constructed. The message lists every error with a 0-based index
    /// so a multi-error run still surfaces all of them.
    pub fn assert_no_errors(&self) {
        let errors = self
            .errors
            .lock()
            .expect("validation errors mutex poisoned");
        if !errors.is_empty() {
            let listed = errors
                .iter()
                .enumerate()
                .map(|(i, e)| format!("  [{i}] {e}"))
                .collect::<Vec<_>>()
                .join("\n");
            panic!("WebGPU validation errors detected:\n{listed}");
        }
    }

    /// Read-only access to the captured error list. Intended for the
    /// intentional-error smoke test (`tests/validation_smoke.rs`)
    /// that wants to inspect the contents rather than panic.
    #[must_use]
    pub fn errors(&self) -> Vec<String> {
        self.errors
            .lock()
            .expect("validation errors mutex poisoned")
            .clone()
    }
}

/// M6.C-0 — lib-unit-test counterpart of
/// `flow-lenia-gpu/tests/common/mod.rs`'s `test_ctx()`. Both helpers
/// share the same shape (`(GpuContext, Option<ValidationGuard>)`
/// gated on `FLOW_LENIA_VALIDATE`) but live in separate places: this
/// one is `#[cfg(test)] pub(crate)` so the 7 src-side `#[cfg(test)]
/// mod tests` blocks can construct a validating context without
/// re-importing the integration-test `common` module (Rust forbids
/// cross-binary test-helper sharing without exposing helpers in the
/// production crate signature).
///
/// **Intentional duplication, not oversight.** The 8-line body of
/// `tests/common::test_ctx()` is verbatim the body below. Lifting
/// both into a single source location would require either (a) a
/// public-API symbol on the production crate (rejected — pollutes
/// the production surface for a test-only concern) or (b) a separate
/// `flow-lenia-testkit` dev-only crate (BENCH.md §10 "long-term
/// option"; deferred until a third consumer appears). With only two
/// consumers, the duplication cost is two `env::var` lookups; a
/// future env-var rename touches both sites mechanically.
///
/// `#[cfg(test)]` guarantees zero footprint in production binaries
/// (`flow-lenia-app`, `flow-lenia-web`); these are compiled without
/// the test harness and therefore without this symbol. Production
/// callers continue to construct `GpuContext::new_blocking` directly
/// and never opt into validation — see CLAUDE.md "production code
/// への validation 不適用".
///
/// Caller pattern, mirroring the integration-test helper:
///
/// ```ignore
/// let (ctx, guard) = test_ctx_for_lib();
/// // ... pipeline construction + dispatch + readback ...
/// if let Some(g) = &guard {
///     g.assert_no_errors();
/// }
/// ```
#[cfg(test)]
#[must_use]
pub(crate) fn test_ctx_for_lib() -> (crate::GpuContext, Option<ValidationGuard>) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = crate::GpuContext::new_blocking(instance, None);
    let guard = if std::env::var("FLOW_LENIA_VALIDATE").is_ok() {
        Some(ValidationGuard::new(&ctx.device))
    } else {
        None
    };
    (ctx, guard)
}
