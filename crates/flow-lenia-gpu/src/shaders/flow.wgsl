// Flow-Lenia α + F integrated pass (M2.6).
//
// Computes the flow field per paper Eq. 5:
//
//     F_i(x) = (1 - α(x)) · ∇U_i(x)  -  α(x) · ∇A_Σ(x)
//
// α is mode-switched via `globals.paper_strict`:
//
//   paper_strict = 0  (JAX-compat, DESIGN.md §4.1.5):
//       α_c(x) = clamp((A_c(x) / β_A)², 0, 1)
//       — per-channel, n is hardcoded to 2 (JAX flowlenia.py:98).
//   paper_strict = 1  (paper Eq. 5 verbatim):
//       α(x)   = clamp((A_Σ(x) / β_A)^n, 0, 1)
//       — shared across channels, n is `globals.n`.
//
// CPU reference:
//   flow_lenia_core::alpha::alpha  (M1.8, dispatches on cfg.paper_strict)
//   flow_lenia_core::flow::flow    (M1.9)
//
// ──────────────────────────────────────────────────────────────────
// Design rationale (vs M2.4 affinity_growth split into 2 shaders):
//
// M2.4 split Eq. 3 / Eq. 7 into separate shaders because:
//   - Different bindings (h_weights vs p_map).
//   - Structurally different formulas (constant vs per-cell map).
//
// M2.6 uses a SINGLE shader for both paper_strict modes because:
//   - Same bindings (a, grad_u, grad_a_sum, F, globals).
//   - α(A_c, β_A, 2) vs α(A_Σ, β_A, n) is a uniform branch — every
//     invocation takes the same path, so wavefront divergence cost
//     is zero.
//   - Eliminates the need for an intermediate α buffer.
// ──────────────────────────────────────────────────────────────────
//
// Layout / binding contract (must agree with passes/flow.rs):
//   @binding(0) a_in:           storage<read>,  channel-major (C, H, W)
//   @binding(1) grad_u:         storage<read>,  (C, H, W, 2) axis-flow inner
//   @binding(2) grad_a_sum:     storage<read>,  (H, W, 2) axis-flow inner
//   @binding(3) f_out:          storage<read_write>, (C, H, W, 2) axis-flow inner
//   @binding(4) globals:        uniform<Globals>

@group(0) @binding(0) var<storage, read>       a_in:       array<f32>;
@group(0) @binding(1) var<storage, read>       grad_u:     array<f32>;
@group(0) @binding(2) var<storage, read>       grad_a_sum: array<f32>;
@group(0) @binding(3) var<storage, read_write> f_out:      array<f32>;
@group(0) @binding(4) var<uniform>             globals:    Globals;

// clamp((mass / β_A)^n, 0, 1). Used by both branches; the JAX-compat
// branch passes n = 2.0 explicitly.
fn alpha_from_mass(mass: f32, beta_a: f32, n: f32) -> f32 {
    let ratio = mass / beta_a;
    // `pow(0.0, n) == 0` for `n > 0` per WGSL spec, so the floor at 0
    // is automatic when A is non-negative. We still clamp for the
    // upper bound (1.0) and as belt-and-braces against tiny f32
    // negatives that could in principle arise post-reintegration.
    let powered = pow(ratio, n);
    return clamp(powered, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= globals.w || gid.y >= globals.h) {
        return;
    }
    let x = gid.x;
    let y = gid.y;
    let plane: u32 = globals.h * globals.w;

    let gas_idx = y * globals.w * 2u + x * 2u;
    let gas_dy = grad_a_sum[gas_idx + 0u];
    let gas_dx = grad_a_sum[gas_idx + 1u];

    // Shared-α branch (paper Eq. 5): compute α(A_Σ) once, reuse for
    // every target channel.
    if (globals.paper_strict == 1u) {
        var a_sum: f32 = 0.0;
        for (var cc: u32 = 0u; cc < globals.c; cc = cc + 1u) {
            a_sum = a_sum + a_in[cc * plane + y * globals.w + x];
        }
        let alpha = alpha_from_mass(a_sum, globals.beta_a, globals.n);
        let one_minus_alpha = 1.0 - alpha;

        for (var cc: u32 = 0u; cc < globals.c; cc = cc + 1u) {
            let gu_base = cc * plane * 2u + y * globals.w * 2u + x * 2u;
            let f_y = one_minus_alpha * grad_u[gu_base + 0u] - alpha * gas_dy;
            let f_x = one_minus_alpha * grad_u[gu_base + 1u] - alpha * gas_dx;
            f_out[gu_base + 0u] = f_y;
            f_out[gu_base + 1u] = f_x;
        }
        return;
    }

    // Per-channel α branch (JAX-compat): n hardcoded to 2.0 to match
    // JAX flowlenia.py:98 `(A[:, :, None, :] / cfg.C) ** 2`.
    //   (Note: JAX divides by `cfg.C` rather than `β_A`. Our CPU
    //    `alpha_jax_compat` keeps `β_A` configurable — see DESIGN.md
    //    §4.1.5; the WGSL shader follows the CPU reference, not the
    //    raw JAX line.)
    for (var cc: u32 = 0u; cc < globals.c; cc = cc + 1u) {
        let a_c = a_in[cc * plane + y * globals.w + x];
        let alpha_c = alpha_from_mass(a_c, globals.beta_a, 2.0);
        let one_minus_alpha = 1.0 - alpha_c;
        let gu_base = cc * plane * 2u + y * globals.w * 2u + x * 2u;
        let f_y = one_minus_alpha * grad_u[gu_base + 0u] - alpha_c * gas_dy;
        let f_x = one_minus_alpha * grad_u[gu_base + 1u] - alpha_c * gas_dx;
        f_out[gu_base + 0u] = f_y;
        f_out[gu_base + 1u] = f_x;
    }
}
