//! Host-side helpers for the activation field `A` on the GPU (M2.3).
//!
//! Layout on the GPU: **channel-major flat `f32`** —
//!
//! ```text
//! flat[c * H * W + y * W + x] == cpu_a[[y, x, c]]
//! ```
//!
//! Channel-major is chosen so the convolve inner loop (M2.3
//! `convolve.wgsl`) reads adjacent `(dy, dx)` neighbours from a
//! contiguous `[c · H · W .. (c+1) · H · W]` slab. With `C = 3` the
//! `(H, W, C)` ndarray layout would put adjacent x's 12 bytes apart
//! (`C · sizeof(f32)`), defeating the L1/texture caches.
//!
//! The PSize of the buffer is `H · W · C · sizeof(f32)`. At the
//! demo-default 64×64×3 = 49 KB; at the 128×128×3 default 196 KB.
//! Both are well within WebGPU's `max_storage_buffer_binding_size`.

use crate::{readback::readback_buffer, GpuContext};
use flow_lenia_core::state::ActivationField;
use ndarray::Array3;
use wgpu::util::DeviceExt;

/// Flatten a `(H, W, C)` [`ActivationField`] into the channel-major
/// `Vec<f32>` consumed by the GPU.
#[must_use]
pub fn flatten_activation_channel_major(a: &ActivationField) -> Vec<f32> {
    let (h, w, c) = a.dim();
    let mut flat = vec![0.0_f32; h * w * c];
    for ci in 0..c {
        for y in 0..h {
            for x in 0..w {
                flat[ci * h * w + y * w + x] = a[[y, x, ci]];
            }
        }
    }
    flat
}

/// Inverse of [`flatten_activation_channel_major`]. Used by readback
/// paths (M2.11 regression tests) to compare directly against an
/// `ActivationField`.
#[must_use]
pub fn unflatten_activation_channel_major(
    flat: &[f32],
    h: usize,
    w: usize,
    c: usize,
) -> ActivationField {
    assert_eq!(flat.len(), h * w * c, "flat length / shape mismatch");
    let mut a: ActivationField = Array3::zeros((h, w, c));
    for ci in 0..c {
        for y in 0..h {
            for x in 0..w {
                a[[y, x, ci]] = flat[ci * h * w + y * w + x];
            }
        }
    }
    a
}

/// Upload an activation field as a fresh GPU storage buffer.
///
/// `usage` defaults to `STORAGE | COPY_DST | COPY_SRC` — read by
/// compute, writable from CPU side via `queue.write_buffer`, and
/// readback-able for tests/debug. Callers that need a different
/// usage mask (e.g. no `COPY_SRC` for memory savings) should build
/// the buffer themselves and feed it `flatten_activation_channel_major`'s
/// output.
#[must_use]
pub fn upload_activation(ctx: &GpuContext, a: &ActivationField) -> wgpu::Buffer {
    let flat = flatten_activation_channel_major(a);
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flow-lenia activation"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
}

/// Convenience: read the activation buffer back into an
/// [`ActivationField`]. Asserts the buffer holds exactly `H · W · C`
/// `f32`s.
#[must_use]
pub fn readback_activation(
    ctx: &GpuContext,
    buffer: &wgpu::Buffer,
    h: usize,
    w: usize,
    c: usize,
) -> ActivationField {
    let flat = readback_buffer::<f32>(ctx, buffer, h * w * c);
    unflatten_activation_channel_major(&flat, h, w, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `unflatten ∘ flatten` is the identity (round-trip pin).
    #[test]
    fn flatten_unflatten_round_trip() {
        let (h, w, c) = (5, 7, 3);
        let mut a: ActivationField = Array3::zeros((h, w, c));
        for y in 0..h {
            for x in 0..w {
                for ci in 0..c {
                    a[[y, x, ci]] = ((y * 13 + x * 7 + ci * 31) % 17) as f32 / 17.0;
                }
            }
        }
        let flat = flatten_activation_channel_major(&a);
        let back = unflatten_activation_channel_major(&flat, h, w, c);
        for (orig, recovered) in a.iter().zip(back.iter()) {
            assert_eq!(orig.to_bits(), recovered.to_bits());
        }
    }
}
