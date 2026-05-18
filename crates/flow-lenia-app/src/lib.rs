#![deny(warnings)]
//! Shared support code for the Flow-Lenia application binaries.
//!
//! M1.14 ships only [`render_terminal`] (ANSI 256-colour visualisation
//! used by the `native_cpu` binary). The M2.10 `native_gpu` binary will
//! add a `winit` event loop alongside this module.

pub mod render_terminal;
