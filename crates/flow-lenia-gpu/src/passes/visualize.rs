//! M2.9 visualisation render pass.
//!
//! Reads a channel-major activation buffer (the same shape as
//! `GpuStepPipeline::current_activation_buffer`) and renders it into
//! a colour target with a nearest-neighbour `upscale` factor.
//!
//! No intermediate compute texture: the fragment shader indexes
//! directly into the storage buffer. See `shaders/visualize.wgsl`
//! for the channel → RGB mapping rules.

use crate::GpuContext;
use bytemuck::{cast_slice, Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Host-side uniform consumed by `visualize.wgsl`.
///
/// 16 bytes total — four `u32`s with natural alignment. Compile-time
/// pinned so the WGSL `VisualizeGlobals` struct layout stays in sync.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct VisualizeGlobals {
    pub h: u32,
    pub w: u32,
    pub c: u32,
    pub upscale: u32,
}

const _: () = {
    assert!(std::mem::size_of::<VisualizeGlobals>() == 16);
    assert!(std::mem::align_of::<VisualizeGlobals>() == 4);
};

/// Compiled visualisation pass.
pub struct VisualizePass {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    upscale: u32,
}

impl VisualizePass {
    /// `format` is the colour-target format the host will provide;
    /// `upscale` is the nearest-neighbour zoom factor (e.g. 8 for
    /// a 64×64 grid → 512×512 image).
    #[must_use]
    pub fn new(ctx: &GpuContext, format: wgpu::TextureFormat, upscale: u32) -> Self {
        assert!(upscale > 0, "upscale must be > 0");
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("visualize.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/visualize.wgsl").into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("visualize bind group layout"),
                    entries: &[
                        // a_in (storage, read) — visible to fragment stage.
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // globals (uniform) — visible to fragment stage.
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visualize pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("visualize render pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                // wgpu 28 renamed the pipeline-side multiview field;
                // type stayed `Option<NonZero<u32>>` (the RenderPass-
                // side field below is a plain u32, easy to mix up).
                multiview_mask: None,
                cache: None,
            });

        Self {
            pipeline,
            bind_group_layout,
            upscale,
        }
    }

    /// Configured upscale factor.
    #[must_use]
    pub fn upscale(&self) -> u32 {
        self.upscale
    }

    /// Upload a [`VisualizeGlobals`] into a fresh GPU uniform buffer.
    /// `c` is the number of channels; `h, w` come from the activation
    /// buffer's shape.
    #[must_use]
    pub fn upload_globals(
        &self,
        ctx: &GpuContext,
        height: u32,
        width: u32,
        channels: u32,
    ) -> wgpu::Buffer {
        let globals = VisualizeGlobals {
            h: height,
            w: width,
            c: channels,
            upscale: self.upscale,
        };
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("visualize globals"),
                contents: cast_slice(std::slice::from_ref(&globals)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    #[must_use]
    pub fn make_bind_group(
        &self,
        ctx: &GpuContext,
        a_in: &wgpu::Buffer,
        globals: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visualize bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: globals.as_entire_binding(),
                },
            ],
        })
    }

    /// Record a single draw of the fullscreen triangle into `encoder`,
    /// targeting `target_view`. The caller controls clear / load
    /// behaviour and submission timing.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        target_view: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("visualize render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                // wgpu 26 added `depth_slice` for 3D / volumetric
                // attachments — None for 2D targets like ours.
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            // wgpu 28 multiview: None = single-view (matches the
            // pipeline's multiview_mask above).
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        // 3 vertices, 1 instance — oversize triangle.
        pass.draw(0..3, 0..1);
    }
}
