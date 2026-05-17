//! Type aliases and axis conventions for Flow-Lenia 3D fields.
//!
//! All 3D fields (activations, α, U, F components, etc.) use the
//! **(H, W, C)** axis order — matching the JAX implementation's
//! `Float[Array, "X Y C"]` convention in `flowlenia.py:80`. Sticking to
//! one convention everywhere keeps `KernelEntry::{c0, c1}` indexing,
//! `sum_axis(AXIS_C)`, and JAX cross-references all unambiguous.
//!
//! Subsequent milestones will add more aliases here (e.g. `UField`,
//! `GradientField`); each new alias should keep the same (H, W, C) order.

use ndarray::{Array3, Axis};

/// 3D activation field. Shape `(H, W, C)`. Paper symbol: `A`.
pub type ActivationField = Array3<f32>;

/// 3D α (diffusion weight) field. Shape `(H, W, C)`.
///
/// In `paper_strict` mode every cell `(y, x)` carries the same value
/// across all C channels (broadcast from a single shared scalar).
/// In JAX-compat (default) mode each channel has its own α value.
pub type AlphaField = Array3<f32>;

/// Height (row) axis of a `(H, W, C)` field.
pub const AXIS_H: Axis = Axis(0);
/// Width (column) axis of a `(H, W, C)` field.
pub const AXIS_W: Axis = Axis(1);
/// Channel axis of a `(H, W, C)` field.
pub const AXIS_C: Axis = Axis(2);
