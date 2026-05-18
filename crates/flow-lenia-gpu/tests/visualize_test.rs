//! M2.9 visualisation tests — exercise [`VisualizePass`] end to end
//! by rendering an off-screen `Rgba8Unorm` texture and reading it
//! back to the CPU as a flat `Vec<u8>` (`readback_rgba8_texture`).
//!
//! Two `cargo test`-default tests pin the channel→RGB mapping:
//! constant-field and 3-channel mapping. A third test renders a
//! 200-step trajectory and writes a PNG to
//! `target/m2_9_visualize_test.png` for human visual inspection
//! (no assertion on pixel content).

use flow_lenia_core::{
    config::{BorderMode, MixRule},
    state::ActivationField,
    FlowLeniaConfig, FlowLeniaSimulator,
};
use flow_lenia_gpu::{
    activation_buffer::upload_activation, readback_rgba8_texture, GpuContext, GpuStepPipeline,
    VisualizePass,
};
use ndarray::Array3;

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn headless_ctx() -> GpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    GpuContext::new_blocking(instance, None)
}

/// Build an off-screen `Rgba8Unorm` colour target sized
/// `grid × upscale` on each axis.
fn make_target(ctx: &GpuContext, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("visualize target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Render `activation` directly (no compute step) at `upscale ×`, read
/// the texture back, and return the raw RGBA bytes (`4·W·H` length,
/// row-major, top-left origin).
fn render_to_rgba(
    ctx: &GpuContext,
    activation: &ActivationField,
    upscale: u32,
) -> (u32, u32, Vec<u8>) {
    let (h, w, c) = activation.dim();
    let render_w = w as u32 * upscale;
    let render_h = h as u32 * upscale;

    let a_buf = upload_activation(ctx, activation);
    let pass = VisualizePass::new(ctx, TEXTURE_FORMAT, upscale);
    let globals_buf = pass.upload_globals(ctx, h as u32, w as u32, c as u32);
    let bg = pass.make_bind_group(ctx, &a_buf, &globals_buf);

    let (texture, view) = make_target(ctx, render_w, render_h);
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("visualize test encoder"),
        });
    pass.record(&mut enc, &bg, &view);
    ctx.queue.submit([enc.finish()]);
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

    let bytes = readback_rgba8_texture(ctx, &texture, render_w, render_h);
    (render_w, render_h, bytes)
}

/// Unpack a single pixel's RGBA bytes into `(r, g, b, a)` in `[0, 1]`.
fn pixel_at(bytes: &[u8], width: u32, x: u32, y: u32) -> (f32, f32, f32, f32) {
    let i = ((y * width + x) * 4) as usize;
    (
        bytes[i] as f32 / 255.0,
        bytes[i + 1] as f32 / 255.0,
        bytes[i + 2] as f32 / 255.0,
        bytes[i + 3] as f32 / 255.0,
    )
}

/// Constant per-channel field → every fragment should produce the same
/// RGBA triple. `Rgba8Unorm` quantises the f32 fragment output to 8-bit
/// (`round(v · 255)`), so the per-channel tolerance is `1/255 ≈ 4e-3`.
#[test]
fn visualize_constant_field_yields_uniform_color() {
    let ctx = headless_ctx();
    let (h, w, c) = (16, 16, 3);
    let a: ActivationField = Array3::from_elem((h, w, c), 0.5);
    let upscale = 4;
    let (_width, _height, bytes) = render_to_rgba(&ctx, &a, upscale);

    let expected = 0.5_f32;
    let tol = 1.5 / 255.0; // ≤ 1 ulp of an 8-bit channel.
                           // Spot-check 4 corners + centre — the shader's per-pixel logic is
                           // pure, so anything other than uniform output is a bug.
    let render_w = w as u32 * upscale;
    let render_h = h as u32 * upscale;
    let probe_points = [
        (0, 0),
        (render_w - 1, 0),
        (0, render_h - 1),
        (render_w - 1, render_h - 1),
        (render_w / 2, render_h / 2),
    ];
    for (px, py) in probe_points {
        let (r, g, b, a_out) = pixel_at(&bytes, render_w, px, py);
        assert!((r - expected).abs() < tol, "R@{px},{py} = {r}");
        assert!((g - expected).abs() < tol, "G@{px},{py} = {g}");
        assert!((b - expected).abs() < tol, "B@{px},{py} = {b}");
        assert!((a_out - 1.0).abs() < tol, "A@{px},{py} = {a_out}");
    }
}

/// Channel→RGB mapping with C∈{1, 2, 3}. Each test case fills the
/// channels with distinct values; render → read back → check the
/// centre pixel matches the expected `(r, g, b)`.
#[test]
fn visualize_channels_map_to_rgb_correctly() {
    let ctx = headless_ctx();
    let upscale = 4;
    let tol = 1.5 / 255.0;

    // C=1
    {
        let (h, w) = (8, 8);
        let a: ActivationField = Array3::from_elem((h, w, 1), 0.7);
        let (rw, _, bytes) = render_to_rgba(&ctx, &a, upscale);
        let (r, g, b, _) = pixel_at(&bytes, rw, rw / 2, rw / 2);
        assert!((r - 0.7).abs() < tol, "C=1: R = {r}");
        assert!(g < tol, "C=1: G = {g}");
        assert!(b < tol, "C=1: B = {b}");
    }
    // C=2
    {
        let (h, w) = (8, 8);
        let mut a: ActivationField = Array3::zeros((h, w, 2));
        a.slice_mut(ndarray::s![.., .., 0]).fill(0.5);
        a.slice_mut(ndarray::s![.., .., 1]).fill(0.3);
        let (rw, _, bytes) = render_to_rgba(&ctx, &a, upscale);
        let (r, g, b, _) = pixel_at(&bytes, rw, rw / 2, rw / 2);
        assert!((r - 0.5).abs() < tol, "C=2: R = {r}");
        assert!((g - 0.3).abs() < tol, "C=2: G = {g}");
        assert!(b < tol, "C=2: B = {b}");
    }
    // C=3
    {
        let (h, w) = (8, 8);
        let mut a: ActivationField = Array3::zeros((h, w, 3));
        a.slice_mut(ndarray::s![.., .., 0]).fill(0.2);
        a.slice_mut(ndarray::s![.., .., 1]).fill(0.4);
        a.slice_mut(ndarray::s![.., .., 2]).fill(0.6);
        let (rw, _, bytes) = render_to_rgba(&ctx, &a, upscale);
        let (r, g, b, _) = pixel_at(&bytes, rw, rw / 2, rw / 2);
        assert!((r - 0.2).abs() < tol, "C=3: R = {r}");
        assert!((g - 0.4).abs() < tol, "C=3: G = {g}");
        assert!((b - 0.6).abs() < tol, "C=3: B = {b}");
    }
}

/// Render a real simulator state and save it as a PNG for human visual
/// inspection. M1.14's terminal viewer at `seed = 1729` showed a clear
/// donut-shaped emergent structure at ~step 200 — this test reproduces
/// the same trajectory via `GpuStepPipeline` and saves the image to
/// `target/m2_9_visualize_test.png`. The test only asserts that the
/// PNG was written; the visual check is for the human reviewer.
#[test]
fn visualize_writes_png_for_visual_inspection() {
    let ctx = headless_ctx();
    let cfg = FlowLeniaConfig {
        grid_width: 64,
        grid_height: 64,
        channels: 3,
        dt: 0.2,
        sigma: 0.65,
        n: 2.0,
        beta_a: 2.0,
        dd: 5,
        num_kernels: 10,
        paper_strict: false,
        border: BorderMode::Torus,
        mix_rule: MixRule::Stochastic,
    };
    let cpu_sim = FlowLeniaSimulator::new(cfg, 1729);
    let initial_a = cpu_sim.activation().clone();
    let kernel_params = cpu_sim.kernel_params().clone();
    let mut pipeline = GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);
    pipeline.run_steps(&ctx, 200);
    let a_after = pipeline.readback_activation(&ctx);

    let upscale = 8; // 64 × 8 = 512
    let (rw, rh, bytes) = render_to_rgba(&ctx, &a_after, upscale);

    // Write into the workspace's top-level `target/` directory so the
    // path is predictable regardless of cargo's per-test tempdir layout.
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("target")
        .join("m2_9_visualize_test.png");
    let img = image::RgbaImage::from_raw(rw, rh, bytes).expect("PNG buffer size mismatch");
    img.save_with_format(&path, image::ImageFormat::Png)
        .expect("PNG save failed");

    eprintln!("[M2.9] wrote {rw}×{rh} PNG to {}", path.display());
    assert!(path.exists(), "PNG was not actually written");
}
