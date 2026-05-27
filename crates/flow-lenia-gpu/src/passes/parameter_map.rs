//! M6.C-2-4-a — parameter map P storage + initialization helpers
//! for Plantec 2025 Eq. 7 per-cell kernel weighting `h`.
//!
//! Flow-Lenia paper §3.1 (Plantec 2025): the constant `h_i` vector
//! from Eq. 3 is replaced by a per-cell function `P_i(x)` from
//! `parameter map P : L → Θ` with `Θ ≡ ℝ^|K|`. Each cell in the
//! grid carries a length-`K` vector of kernel weighting coefficients.
//! Multi-creature behavior is realised by initialising distinct
//! `P` vectors in distinct grid patches (Plantec §4.3.2 Vanilla:
//! 64 creatures × 20×20 patches × per-patch normal-distributed P);
//! Flow-Lenia M6.C-2-4 specialises this to 4 creatures with
//! Ponyo877-san strategic decision 2026-05-27.
//!
//! ## Layout
//!
//! ```text
//! parameter_map_p[(y * W + x) * K + ki] = P_ki(y, x)
//! ```
//!
//! Row-major (y, x), then per-cell kernel-index `ki`. Matches the
//! WGSL binding `array<f32>` length `H * W * K` that the
//! Eq. 7-mode `AffinityGrowthPass` (lands in C-2-4-b) reads.
//!
//! Memory cost: `H * W * K * 4` bytes.
//! - N=64,  K=10 → 160 KB
//! - N=256, K=10 → 2.50 MB (acceptable per scope-guardian C-2-4
//!   memory budget)
//!
//! ## Eq. 8 (parameter inheritance during reintegration) — deferred to M5
//!
//! Per Ponyo877-san strategic decision (judgment 2): the M6.C-2-4
//! scope is **infrastructure only**. The `ReintegratePass` will
//! flow `P` along with matter (C-2-4-c hook point) but the
//! softmax-over-incoming-mass parameter selection is M5 evolutionary
//! search work, NOT this milestone.
//!
//! ## Caller pattern (planned, C-2-4-b / C-2-4-c)
//!
//! ```text
//! // Startup, per-creature P vectors hardcoded or RNG-seeded
//! let map = parameter_map::build_for_patches(n, k, &patches);
//! let p_buf = parameter_map::upload(&ctx, &map);
//! // ... bind p_buf into AffinityGrowthPass localised-mode bind group
//! ```

use crate::GpuContext;
use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

/// One creature's seed parameters: bounding box on the grid + the
/// length-K vector of `h` weights that the creature carries.
///
/// `bbox`: `(y0, x0, y1, x1)` inclusive-exclusive (`y0..y1`,
/// `x0..x1`). Out-of-bounds cells are clamped silently (no panic);
/// overlapping creatures use **last-write-wins** by call order in
/// `build_for_patches` — Plantec §4.3.2 uses random uniform sampling
/// of patch position so overlaps are rare but possible. The
/// last-write semantics matches a "writer wins" interpretation that
/// later sub-steps (C-2-4-c reintegrate hook, M5 Eq. 8) will replace
/// with stochastic sampling.
#[derive(Clone, Debug)]
pub struct CreaturePatch {
    pub bbox: (u32, u32, u32, u32),
    pub p_vector: Vec<f32>,
}

/// Build a flat `H * W * K` parameter map on the CPU from a list of
/// creature patches. Background cells receive `default_p` (typically
/// a copy of the project's existing constant `h` vector so that
/// outside-patch cells fall back to "Eq. 3 default behaviour"; or
/// zeros for a true "no kernel response outside patches" mode).
///
/// `default_p.len()` and each `patch.p_vector.len()` must equal
/// `num_kernels` (asserted).
#[must_use]
pub fn build_for_patches(
    n: u32,
    num_kernels: u32,
    default_p: &[f32],
    patches: &[CreaturePatch],
) -> Vec<f32> {
    let n_usize = n as usize;
    let k_usize = num_kernels as usize;
    assert_eq!(
        default_p.len(),
        k_usize,
        "default_p length must equal num_kernels ({k_usize}, got {})",
        default_p.len()
    );
    for (i, patch) in patches.iter().enumerate() {
        assert_eq!(
            patch.p_vector.len(),
            k_usize,
            "patch[{i}].p_vector length must equal num_kernels ({k_usize}, got {})",
            patch.p_vector.len()
        );
    }
    let cells = n_usize * n_usize;
    let mut map = Vec::with_capacity(cells * k_usize);
    for _ in 0..cells {
        map.extend_from_slice(default_p);
    }
    for patch in patches {
        let (y0, x0, y1, x1) = patch.bbox;
        let y0c = y0.min(n) as usize;
        let x0c = x0.min(n) as usize;
        let y1c = y1.min(n) as usize;
        let x1c = x1.min(n) as usize;
        for y in y0c..y1c {
            for x in x0c..x1c {
                let cell_base = (y * n_usize + x) * k_usize;
                map[cell_base..cell_base + k_usize].copy_from_slice(&patch.p_vector);
            }
        }
    }
    map
}

/// Upload a CPU-built parameter map as a STORAGE buffer suitable
/// for AffinityGrowthPass binding (read-only). Lifecycle: built
/// once at simulator init; rebuilt + re-uploaded whenever
/// parameter painting or creature respawn changes the per-patch
/// `P` vectors (UI-driven invalidate is C-4 / M5 work — see
/// `GpuStepPipeline::update_globals` TODO from C-1-5-b).
#[must_use]
pub fn upload(ctx: &GpuContext, map: &[f32]) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("parameter_map_p buffer"),
            contents: cast_slice(map),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_for_patches_default_fills_background() {
        let n: u32 = 8;
        let k: u32 = 3;
        let default_p = vec![0.1_f32, 0.2, 0.3];
        let map = build_for_patches(n, k, &default_p, &[]);
        assert_eq!(map.len(), (n * n * k) as usize);
        // Every cell == default_p
        for y in 0..n as usize {
            for x in 0..n as usize {
                let base = (y * n as usize + x) * k as usize;
                assert_eq!(&map[base..base + 3], &default_p[..]);
            }
        }
    }

    #[test]
    fn build_for_patches_writes_patch_p_vector() {
        let n: u32 = 8;
        let k: u32 = 3;
        let default_p = vec![0.0_f32; k as usize];
        let patch_p = vec![1.0_f32, 2.0, 3.0];
        let patches = vec![CreaturePatch {
            bbox: (2, 2, 5, 5),
            p_vector: patch_p.clone(),
        }];
        let map = build_for_patches(n, k, &default_p, &patches);
        for y in 0..n as usize {
            for x in 0..n as usize {
                let base = (y * n as usize + x) * k as usize;
                if (2..5).contains(&y) && (2..5).contains(&x) {
                    assert_eq!(
                        &map[base..base + 3],
                        &patch_p[..],
                        "cell ({y}, {x}) should hold patch P"
                    );
                } else {
                    assert_eq!(
                        &map[base..base + 3],
                        &default_p[..],
                        "cell ({y}, {x}) should hold default P"
                    );
                }
            }
        }
    }

    #[test]
    fn build_for_patches_four_creatures_distinct_p() {
        // M6.C-2-4 4 creature setup at N=64, K=10. Each creature
        // gets a distinct P vector to exercise the per-creature
        // routing semantics that AffinityGrowthPass localised mode
        // will read in C-2-4-b.
        let n: u32 = 64;
        let k: u32 = 10;
        let default_p = vec![0.0_f32; k as usize];
        let patches: Vec<CreaturePatch> = (0..4u32)
            .map(|c| {
                let (y0, x0) = match c {
                    0 => (4, 4),    // upper-left
                    1 => (4, 44),   // upper-right
                    2 => (44, 4),   // lower-left
                    3 => (44, 44),  // lower-right
                    _ => unreachable!(),
                };
                let p_vector: Vec<f32> = (0..k)
                    .map(|ki| (c as f32 + 1.0) * (ki as f32 + 1.0) * 0.01)
                    .collect();
                CreaturePatch {
                    bbox: (y0, x0, y0 + 20, x0 + 20),
                    p_vector,
                }
            })
            .collect();
        let map = build_for_patches(n, k, &default_p, &patches);
        // Spot-check: cell (4, 4) is creature 0's corner
        let creature0_first_p: Vec<f32> =
            (0..k).map(|ki| 1.0 * (ki as f32 + 1.0) * 0.01).collect();
        let base = (4 * n as usize + 4) * k as usize;
        assert_eq!(&map[base..base + k as usize], &creature0_first_p[..]);
        // Spot-check: cell (44, 44) is creature 3's corner
        let creature3_first_p: Vec<f32> =
            (0..k).map(|ki| 4.0 * (ki as f32 + 1.0) * 0.01).collect();
        let base = (44 * n as usize + 44) * k as usize;
        assert_eq!(&map[base..base + k as usize], &creature3_first_p[..]);
        // Outside any patch (e.g. center): default
        let base = (32 * n as usize + 32) * k as usize;
        assert_eq!(&map[base..base + k as usize], &default_p[..]);
    }

    #[test]
    fn build_for_patches_overlap_last_wins() {
        let n: u32 = 8;
        let k: u32 = 2;
        let default_p = vec![0.0_f32; k as usize];
        let patches = vec![
            CreaturePatch {
                bbox: (0, 0, 4, 4),
                p_vector: vec![1.0, 1.0],
            },
            CreaturePatch {
                bbox: (2, 2, 6, 6),
                p_vector: vec![2.0, 2.0],
            },
        ];
        let map = build_for_patches(n, k, &default_p, &patches);
        // Overlap region (2..4, 2..4) should hold patch[1] (last write).
        let base = (3 * n as usize + 3) * k as usize;
        assert_eq!(&map[base..base + 2], &[2.0_f32, 2.0]);
        // patch[0]-only region (0..2, 0..2)
        let base = (1 * n as usize + 1) * k as usize;
        assert_eq!(&map[base..base + 2], &[1.0_f32, 1.0]);
    }
}
