//! M6.A.7 — intentional WebGPU validation-error smoke test.
//!
//! Confirms that `ValidationGuard` actually captures
//! `Device::on_uncaptured_error` callbacks. Trigger: a
//! `copy_buffer_to_buffer` where the source range overruns the
//! source buffer's size. wgpu's validation layer rejects this with
//! a `Validation` error that the guard collects, and
//! `assert_no_errors` then panics with the documented message.
//!
//! The test is `#[should_panic(expected = "WebGPU validation errors")]`
//! so a missing capture (panic with a different message, no panic at
//! all) fails the test — that's the inverse safety check.
//!
//! Unlike the rest of the M6.A.7 sweep this test does **not** read
//! `FLOW_LENIA_VALIDATE`; it always installs the guard, because the
//! guard *is* what's being tested.

mod common;

use flow_lenia_gpu::{validation::ValidationGuard, GpuContext};

#[test]
#[should_panic(expected = "WebGPU validation errors")]
fn validation_guard_catches_bad_buffer_copy() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);
    let guard = ValidationGuard::new(&ctx.device);

    // Two small staging buffers — enough room for src to exist and
    // dst to receive, but the copy below intentionally overruns src.
    let src = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("validation_smoke src (16 bytes)"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dst = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("validation_smoke dst (256 bytes)"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("validation_smoke encoder"),
        });
    // Copy 64 bytes from a 16-byte source — wgpu rejects this at
    // `copy_buffer_to_buffer` validation.
    encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, 64);
    ctx.queue.submit([encoder.finish()]);

    // Drain submitted work so the async validation callback definitely
    // fires before we check.
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("device.poll(Wait) failed");

    // Sanity: at least one error landed. The `assert_no_errors`
    // panic is what `#[should_panic]` matches against.
    let captured = guard.errors();
    assert!(
        !captured.is_empty(),
        "validation guard captured zero errors despite the deliberate \
         overrunning copy — `on_uncaptured_error` may not have fired"
    );
    eprintln!(
        "[M6.A.7 smoke] captured {} validation error(s); first: {}",
        captured.len(),
        captured[0]
    );
    guard.assert_no_errors();
}
