//! Generic GPU buffer readback (M2.2).
//!
//! The pattern is the same for every "copy a GPU buffer back to the
//! CPU" operation: allocate a staging buffer with `MAP_READ | COPY_DST`,
//! issue a `copy_buffer_to_buffer`, submit, `map_async` + poll until
//! the mapping is ready, then `bytemuck::cast_slice` into the caller's
//! type. We need this in M2.2 (verify upload bit-equal), M2.11 (CPU vs
//! GPU regression), and most of the M2.3..M2.8 unit tests, so it is
//! worth extracting once instead of duplicating in each call site.

use crate::GpuContext;
use bytemuck::Pod;
use std::sync::mpsc;

/// Copy `element_count` elements of type `T` out of `source` and back
/// to the CPU.
///
/// `source` must be created with the [`wgpu::BufferUsages::COPY_SRC`]
/// bit set. Element count is in *elements of `T`*, not bytes — internally
/// this function multiplies by `std::mem::size_of::<T>()`.
///
/// Returns a fresh `Vec<T>`. The staging buffer is dropped before
/// return so no GPU memory leaks.
///
/// # Panics
///
/// Panics if:
/// - the staging mapping fails (driver-level error — there is no
///   sensible recovery in the unit-test contexts that use this);
/// - the source buffer is smaller than `element_count * size_of::<T>()`.
///   The latter is checked implicitly by the `copy_buffer_to_buffer`
///   bounds check inside wgpu and surfaced via the device's validation
///   layer.
#[must_use]
pub fn readback_buffer<T: Pod>(
    ctx: &GpuContext,
    source: &wgpu::Buffer,
    element_count: usize,
) -> Vec<T> {
    let bytes = (element_count * std::mem::size_of::<T>()) as wgpu::BufferAddress;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder"),
        });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
    ctx.queue.submit([encoder.finish()]);

    // Synchronous map: post a sender, then block on the device to drain
    // submitted work + invoke pending map callbacks.
    let (tx, rx) = mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            // `send` only fails if the receiver was dropped — which would
            // mean this thread already gave up waiting, so the error is
            // discardable.
            let _ = tx.send(result);
        });

    // `Wait` blocks until the most recent submission completes and the
    // map callback above fires.
    ctx.device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .expect("device.poll(Wait) failed");
    rx.recv()
        .expect("readback channel disconnected")
        .expect("buffer map_async reported failure");

    let view = staging.slice(..).get_mapped_range();
    let out: Vec<T> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    out
}
