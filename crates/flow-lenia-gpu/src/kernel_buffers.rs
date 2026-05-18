//! Kernel-bank GPU buffer upload (M2.2).
//!
//! Layout: **Plan A — fixed stride, zero-padded** (DESIGN.md §3.2,
//! M2.2 design judgment).
//!
//! Per-kernel arrays have side length `2·er_i + 1` which varies with
//! the sampled `r_i`. We pad every kernel to a single `max_side =
//! 2·max(er_i) + 1` and pack them contiguously:
//!
//! ```text
//! kernels storage buffer ((H, W, …, C) JAX axis convention is for
//! activations; this buffer is per-kernel 2D and row-major in (y, x)):
//!
//!   element_index(k, y, x) = k * max_side² + y * max_side + x
//!   total elements         = num_kernels * max_side²
//!   total bytes            = total_elements * 4
//! ```
//!
//! WGSL can index this with a single multiply + add, no per-kernel
//! branching, no offset table lookup. The cost is a worst-case 4.5 MB
//! for K=45 kernels at R=25 (max_side=81) — well under WebGPU's
//! 128 MB `max_storage_buffer_binding_size`. Variable-length packing
//! is left as a M6 performance lever.
//!
//! Per-kernel metadata (source/target channel + growth μ, σ) lives in
//! a separate uniform buffer of `GpuKernelMeta` records. The natural
//! alignment of `[u32; 2] + [f32; 2]` is 16 bytes — no padding fields
//! needed.

use crate::{readback::readback_buffer, GpuContext};
use bytemuck::{Pod, Zeroable};
use flow_lenia_core::{compute_kernel, effective_radius, KernelParams};
use wgpu::util::DeviceExt;

/// GPU-side copy of [`flow_lenia_core::KernelMeta`].
///
/// Field order is chosen so the natural alignment is exactly 16 bytes
/// (matching WGSL `uniform` requirements without an explicit
/// `[padding: u32; …]` array) and so the WGSL struct can mirror the
/// Rust struct field-for-field without manual `@align(...)` decorators.
///
/// We deliberately omit `effective_radius` (Plan A uses one shared
/// `max_side` instead) and the kernel weight `h_i` (lives in the
/// separate per-step weights array; future M2.4).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuKernelMeta {
    /// Source channel index `c_i^0`.
    pub source_channel: u32,
    /// Target channel index `c_i^1`.
    pub target_channel: u32,
    /// Growth function mean `μ_i` (paper Eq. 2).
    pub mu: f32,
    /// Growth function width `σ_i` (paper Eq. 2).
    pub sigma: f32,
}

const _: () = {
    // Compile-time guards for the layout claim made in this module's
    // doc comment. If either assertion stops holding, the WGSL `meta`
    // struct in M2.3+ will silently misalign and produce garbage —
    // catch it here at compile time.
    assert!(std::mem::size_of::<GpuKernelMeta>() == 16);
    assert!(std::mem::align_of::<GpuKernelMeta>() == 4);
    // Note: alignment of the struct itself is 4 (max of field
    // alignments), but the **array of structs** in the uniform buffer
    // remains 16-byte-strided because 4 × 4-byte fields == 16 bytes.
    // WGSL `array<Meta>` requires 16-byte stride for uniform buffers,
    // which is the property the size_of() check above pins.
};

/// Owns the two GPU buffers used by every compute pass that reads the
/// kernel bank (M2.3 convolve, M2.4 affinity, …).
///
/// `count` and `max_side` are kept on the host side rather than in a
/// uniform — the shader receives them through pipeline-constants
/// (M2.3 onwards) so they participate in shader specialisation.
pub struct GpuKernelBuffers {
    /// Storage buffer with the zero-padded packed kernel arrays.
    /// Usage: `STORAGE | COPY_DST | COPY_SRC` (COPY_SRC for readback
    /// tests / debugging).
    pub kernels: wgpu::Buffer,
    /// Uniform buffer with the per-kernel `GpuKernelMeta` array.
    /// Usage: `UNIFORM | COPY_DST | COPY_SRC`.
    pub meta: wgpu::Buffer,
    /// `|K|` — number of kernels.
    pub count: u32,
    /// Side length each kernel is padded to (`2·max(er_i) + 1`).
    pub max_side: u32,
}

impl GpuKernelBuffers {
    /// Number of `f32` elements per padded kernel slot.
    #[must_use]
    pub fn padded_kernel_len(&self) -> u32 {
        self.max_side * self.max_side
    }

    /// Total size of the kernels buffer in bytes.
    #[must_use]
    pub fn kernels_bytes(&self) -> wgpu::BufferAddress {
        wgpu::BufferAddress::from(self.count)
            * wgpu::BufferAddress::from(self.padded_kernel_len())
            * 4
    }

    /// Total size of the meta buffer in bytes.
    #[must_use]
    pub fn meta_bytes(&self) -> wgpu::BufferAddress {
        wgpu::BufferAddress::from(self.count)
            * std::mem::size_of::<GpuKernelMeta>() as wgpu::BufferAddress
    }
}

/// Precompute every kernel (M1.4) on the CPU, zero-pad each to a
/// shared `max_side × max_side`, and upload the packed bank + meta
/// to GPU buffers.
///
/// Caller still owns `params`; the function reads it and produces
/// fresh GPU buffers.
///
/// Empty `params.kernels` is rejected — there is nothing meaningful
/// to upload, and an empty buffer trips wgpu's "size must be > 0"
/// validation.
#[must_use]
pub fn upload_kernels(ctx: &GpuContext, params: &KernelParams) -> GpuKernelBuffers {
    assert!(!params.is_empty(), "upload_kernels: KernelParams is empty");

    let count = params.kernels.len() as u32;

    // Compute every kernel on the CPU first; this also lets us
    // measure the true max_side without redoing the formula.
    let cpu_kernels: Vec<ndarray::Array2<f32>> = params
        .kernels
        .iter()
        .map(|entry| compute_kernel(params.r_global, entry))
        .collect();

    let max_side = cpu_kernels
        .iter()
        .map(|k| k.shape()[0] as u32)
        .max()
        .expect("non-empty KernelParams must produce at least one kernel");

    // Cross-check against the JAX formula (defence in depth — also
    // serves as a smoke test for `effective_radius` itself).
    let expected_max_side = 2 * params
        .kernels
        .iter()
        .map(|e| effective_radius(params.r_global, e.r))
        .max()
        .unwrap()
        + 1;
    assert_eq!(
        max_side, expected_max_side,
        "kernel side mismatch: got {max_side}, expected {expected_max_side}"
    );

    // Pad + pack into a single Vec<f32> of size `count * max_side²`.
    let stride = (max_side * max_side) as usize;
    let mut packed = vec![0.0_f32; count as usize * stride];
    for (k, kernel) in cpu_kernels.iter().enumerate() {
        let k_side = kernel.shape()[0] as u32;
        let offset = (max_side - k_side) / 2;
        let base = k * stride;
        for y in 0..k_side as usize {
            for x in 0..k_side as usize {
                let dst = base + (y + offset as usize) * max_side as usize + (x + offset as usize);
                packed[dst] = kernel[[y, x]];
            }
        }
    }

    // Build the per-kernel meta vector.
    let meta_vec: Vec<GpuKernelMeta> = params
        .kernels
        .iter()
        .map(|e| GpuKernelMeta {
            source_channel: e.c0,
            target_channel: e.c1,
            mu: e.mu,
            sigma: e.sigma,
        })
        .collect();

    // Upload both via `create_buffer_init` (one-shot creation + copy
    // through the staging belt). `write_buffer` would also work, but
    // `create_buffer_init` avoids a separate zeroed allocation step.
    let kernels = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flow-lenia kernels (padded)"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
    // Note (M2.3): `STORAGE` is OR'd in alongside `UNIFORM` so the
    // WGSL convolve shader can bind this as `var<storage, read>
    // array<Meta>` (runtime-sized array). A buffer with both flags
    // can be bound either way; the runtime-array form is required
    // because `K` (= `cfg.num_kernels`) varies without pipeline
    // rebuild, and WGSL `var<uniform>` arrays must have a
    // compile-time fixed size.
    let meta = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flow-lenia kernel meta"),
            contents: bytemuck::cast_slice(&meta_vec),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

    GpuKernelBuffers {
        kernels,
        meta,
        count,
        max_side,
    }
}

/// Convenience: read the kernels buffer back as a flat `Vec<f32>`.
/// Used by the M2.2 round-trip test and by future debugging.
#[must_use]
pub fn readback_kernels(ctx: &GpuContext, buffers: &GpuKernelBuffers) -> Vec<f32> {
    let elements = (buffers.count * buffers.padded_kernel_len()) as usize;
    readback_buffer::<f32>(ctx, &buffers.kernels, elements)
}

/// Convenience: read the meta buffer back as a `Vec<GpuKernelMeta>`.
#[must_use]
pub fn readback_meta(ctx: &GpuContext, buffers: &GpuKernelBuffers) -> Vec<GpuKernelMeta> {
    readback_buffer::<GpuKernelMeta>(ctx, &buffers.meta, buffers.count as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flow_lenia_core::{KernelEntry, SamplingSettings};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    fn headless_ctx() -> GpuContext {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        GpuContext::new_blocking(instance, None)
    }

    /// Round-trip: upload a deterministically-sampled kernel bank,
    /// read both buffers back, assert **bit-equal** match against the
    /// CPU originals. Plan A's zero-padding means cells outside the
    /// per-kernel array must read back as exactly `0.0_f32` (bit-equal
    /// `0x00000000`), so a single `to_bits()` compare covers both
    /// "values match" and "padding is correct".
    #[test]
    fn upload_then_readback_is_bit_equal() {
        let ctx = headless_ctx();

        // Small bank so the test runs fast without skimping on the
        // mixed-radius coverage. `num_kernels=6` exercises padding for
        // every kernel that is smaller than the max.
        let mut rng = ChaCha8Rng::seed_from_u64(0x1234_5678);
        let params = KernelParams::sample_random(
            &mut rng,
            SamplingSettings {
                num_kernels: 6,
                num_channels: 3,
            },
        );

        let upload_started = Instant::now();
        let buffers = upload_kernels(&ctx, &params);
        let upload_ms = upload_started.elapsed().as_secs_f64() * 1000.0;

        let readback_started = Instant::now();
        let gpu_kernels = readback_kernels(&ctx, &buffers);
        let gpu_meta = readback_meta(&ctx, &buffers);
        let readback_ms = readback_started.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "[M2.2] upload={upload_ms:.2}ms readback={readback_ms:.2}ms \
             count={count} max_side={max_side} \
             kernels_bytes={kb} meta_bytes={mb}",
            count = buffers.count,
            max_side = buffers.max_side,
            kb = buffers.kernels_bytes(),
            mb = buffers.meta_bytes(),
        );

        // Rebuild the CPU-side expectation matching the packing.
        let stride = (buffers.max_side * buffers.max_side) as usize;
        let mut cpu_packed = vec![0.0_f32; buffers.count as usize * stride];
        for (k, entry) in params.kernels.iter().enumerate() {
            let kernel = compute_kernel(params.r_global, entry);
            let k_side = kernel.shape()[0] as u32;
            let offset = (buffers.max_side - k_side) / 2;
            let base = k * stride;
            for y in 0..k_side as usize {
                for x in 0..k_side as usize {
                    let dst = base
                        + (y + offset as usize) * buffers.max_side as usize
                        + (x + offset as usize);
                    cpu_packed[dst] = kernel[[y, x]];
                }
            }
        }

        assert_eq!(
            cpu_packed.len(),
            gpu_kernels.len(),
            "element count mismatch"
        );
        for (i, (&cpu, &gpu)) in cpu_packed.iter().zip(gpu_kernels.iter()).enumerate() {
            assert_eq!(
                cpu.to_bits(),
                gpu.to_bits(),
                "kernel element {i} diverged: cpu={cpu} gpu={gpu}"
            );
        }

        // Meta: hand-rebuild and compare.
        let cpu_meta: Vec<GpuKernelMeta> = params
            .kernels
            .iter()
            .map(|e| GpuKernelMeta {
                source_channel: e.c0,
                target_channel: e.c1,
                mu: e.mu,
                sigma: e.sigma,
            })
            .collect();
        assert_eq!(cpu_meta, gpu_meta);
    }

    /// `GpuKernelMeta` layout pin: keeps the M2.3 WGSL struct layout
    /// from drifting under us. If this assertion fails, update both
    /// the WGSL `Meta` struct and any pipeline that uses it.
    #[test]
    fn gpu_kernel_meta_layout() {
        assert_eq!(std::mem::size_of::<GpuKernelMeta>(), 16);
        assert_eq!(std::mem::align_of::<GpuKernelMeta>(), 4);
    }

    /// Constant-bank upload: confirm `max_side` matches when every
    /// kernel shares the same radius (no padding needed).
    #[test]
    fn upload_constant_radius_has_no_padding() {
        let ctx = headless_ctx();
        let entry = KernelEntry {
            c0: 0,
            c1: 0,
            r: 0.5,
            a: [0.25, 0.5, 0.75],
            b: [1.0, 0.7, 0.4],
            w: [0.05, 0.05, 0.05],
            h: 1.0,
            mu: 0.15,
            sigma: 0.02,
        };
        let params = KernelParams {
            r_global: 10.0,
            kernels: vec![entry; 4],
        };
        let buffers = upload_kernels(&ctx, &params);

        // All kernels have side = 2·er+1 = 2·13+1 = 27.
        assert_eq!(buffers.max_side, 27);
        assert_eq!(buffers.count, 4);

        let gpu = readback_kernels(&ctx, &buffers);
        let cpu_one = compute_kernel(10.0, &entry);
        let stride = 27 * 27;
        for k in 0..4 {
            for (i, &v) in cpu_one.iter().enumerate() {
                let dst = k * stride + i;
                assert_eq!(v.to_bits(), gpu[dst].to_bits());
            }
        }
    }
}
