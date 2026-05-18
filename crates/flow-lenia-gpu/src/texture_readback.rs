//! Texture → CPU readback helper (M2.9).
//!
//! Companion to `readback::readback_buffer` for the texture path:
//! copy a 2D `wgpu::Texture` of `Rgba8Unorm` / `Rgba8UnormSrgb` format
//! back to the CPU as a `Vec<u8>` of length `4 · w · h`.
//!
//! Used by the M2.9 visualisation PNG export tests; the production
//! `native_gpu` binary (M2.10) presents the texture directly and
//! never readback-s.

use crate::GpuContext;
use std::sync::mpsc;

/// Copy an `Rgba8`-format 2D texture back to the CPU as a tightly-packed
/// `Vec<u8>` of length `4 · width · height` (row-major, top-left origin).
///
/// `bytes_per_row` in `copy_texture_to_buffer` must be a multiple of
/// `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT (256)`; this function handles
/// that padding internally and trims the result back to `4 · width`
/// per row.
///
/// # Panics
///
/// Panics if `texture.format()` is neither `Rgba8Unorm` nor
/// `Rgba8UnormSrgb`, if the `width` or `height` exceeds the device's
/// `max_buffer_size` limit divided by `4 · padded_row`, or if the
/// staging mapping fails.
#[must_use]
pub fn readback_rgba8_texture(
    ctx: &GpuContext,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let fmt = texture.format();
    assert!(
        matches!(
            fmt,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
        ),
        "readback_rgba8_texture requires an Rgba8 format (got {fmt:?})"
    );

    // `bytes_per_row` for `copy_texture_to_buffer` must be a multiple
    // of `COPY_BYTES_PER_ROW_ALIGNMENT` (256 bytes).
    let unpadded_row = (width * 4) as usize;
    let row_align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded_row = unpadded_row.div_ceil(row_align) * row_align;
    let buffer_size = (padded_row * height as usize) as wgpu::BufferAddress;

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rgba8 readback staging"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rgba8 readback encoder"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row as u32),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([encoder.finish()]);

    let (tx, rx) = mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::Wait)
        .expect("device.poll(Wait) failed");
    rx.recv()
        .expect("readback channel disconnected")
        .expect("buffer map_async reported failure");

    let view = staging.slice(..).get_mapped_range();
    let mut out = vec![0_u8; unpadded_row * height as usize];
    for y in 0..height as usize {
        let src_start = y * padded_row;
        let dst_start = y * unpadded_row;
        out[dst_start..dst_start + unpadded_row]
            .copy_from_slice(&view[src_start..src_start + unpadded_row]);
    }
    drop(view);
    staging.unmap();
    out
}
