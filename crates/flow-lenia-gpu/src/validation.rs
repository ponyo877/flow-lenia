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
