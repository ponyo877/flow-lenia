//! Compute-pass implementations (M2.3 onwards).
//!
//! Each pass is a small `pub struct` owning its compiled pipeline and
//! bind-group layout, with a stateless `record(...)` method that
//! appends a single `dispatch_workgroups` to a caller-supplied
//! [`wgpu::CommandEncoder`]. This factoring keeps per-step ordering
//! and synchronisation visible at the M2.10 simulator loop, instead
//! of being hidden inside each pass.

pub mod affinity_growth;
pub mod convolve;
pub mod fft;
pub mod flow;
pub mod gradient;
pub mod kernel_fft;
pub mod reintegrate;
pub mod spectral_multiply;
pub mod visualize;

pub use affinity_growth::{
    upload_constant_weights, upload_localized_weights, AffinityGrowthPass, GpuConstantWeights,
    MAX_KERNELS,
};
pub use convolve::ConvolvePass;
pub use fft::{
    is_supported_n, precompute_twiddles_1d, Fft2dPass, FftAxis, FftDirection, FftParams, FftPass,
    SUPPORTED_N,
};
pub use kernel_fft::{build_padded_kernel, precompute_kernel_ffts, KernelFftBuffers};
pub use spectral_multiply::{SpectralMultiplyParams, SpectralMultiplyPass};
pub use flow::FlowPass;
pub use gradient::GradientPass;
pub use reintegrate::ReintegratePass;
pub use visualize::{VisualizeGlobals, VisualizePass};
