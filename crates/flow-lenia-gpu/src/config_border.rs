//! CPU↔GPU bridge for [`flow_lenia_core::config::BorderMode`].
//!
//! Centralises the `BorderMode::Torus = 0`, `BorderMode::Wall = 1`
//! mapping so every shader and host-side dispatch stays in agreement
//! with the WGSL `BORDER_TORUS` / `BORDER_WALL` constants.

use flow_lenia_core::config::BorderMode;

/// `u32` form of [`BorderMode`] passed to compute shaders.
///
/// Match the constants in `crates/flow-lenia-gpu/src/shaders/*.wgsl`:
/// ```wgsl
/// const BORDER_TORUS: u32 = 0u;
/// const BORDER_WALL:  u32 = 1u;
/// ```
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BorderCode {
    Torus = 0,
    Wall = 1,
}

impl From<BorderMode> for BorderCode {
    fn from(m: BorderMode) -> Self {
        match m {
            BorderMode::Torus => Self::Torus,
            BorderMode::Wall => Self::Wall,
        }
    }
}
