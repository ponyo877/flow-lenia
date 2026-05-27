// Flow-Lenia ParameterFlowPass — M6.C-2-4-c (case δ infrastructure).
//
// **Status: identity copy + M5 hook block.** This pass exists to
// establish the binding contract and ping-pong scaffolding for
// Plantec 2025 Eq. 8 (parameter inheritance during reintegration);
// the actual stochastic-sampling algorithm is deferred to M5 per
// Ponyo877-san strategic decision 2026-05-27.
//
// ## Binding contract (frozen for M5)
//
// 0: `p_in`           read       — `array<f32>` length H*W*K, row-major
//                                  cell-major `(y, x, ki)` (matches
//                                  `parameter_map::build_for_patches`
//                                  output and `affinity_growth_localized`
//                                  `p_map` binding).
// 1: `p_out`          read_write — same shape as p_in. Caller ping-
//                                  pongs between two buffers across
//                                  step boundaries (same pattern as
//                                  ReintegratePass `a_in` / `a_out`).
// 2: `matter_flow`    read       — `(C, H, W, 2)` channel-major flat,
//                                  matches `FlowPass::allocate_f_out`
//                                  layout (.x = dy, .y = dx).
//                                  Unused in case (a) identity copy;
//                                  M5 reads incoming flow weights to
//                                  build the softmax distribution.
// 3: `kernel_routing` read       — `array<u32>` length K,
//                                  `kernel_routing[k] = source_channel`
//                                  (creature ID proxy when C-2-4-c→M5
//                                  generalises P per kernel rather than
//                                  per cell). Unused in case (a); M5
//                                  uses it for creature-competition
//                                  semantics.
// 4: `globals`        uniform    — standard Globals (W, H, C, K, …).
//
// ## M5 hook (Eq. 8 stochastic sampling)
//
// Replace the identity-copy body with:
//
// ```
// for each cell x:
//   let in_mass = Σ_y  incoming_mass(y -> x)   // from matter_flow
//   let weights[y] = softmax( ... )            // Plantec Eq. 8
//   sample y* ~ Categorical(weights)
//   P_out[x] = P_in[y*]
// ```
//
// The bindings above are sufficient for that rewrite — no Rust-side
// pipeline shape changes are required at M5.
//
// See `docs/M6_C2_4_creature_design.md` §"M5 hook specification".

@group(0) @binding(0) var<storage, read>       p_in:           array<f32>;
@group(0) @binding(1) var<storage, read_write> p_out:          array<f32>;
@group(0) @binding(2) var<storage, read>       matter_flow:    array<f32>;
@group(0) @binding(3) var<storage, read>       kernel_routing: array<u32>;
@group(0) @binding(4) var<uniform>             globals:        Globals;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= globals.w || y >= globals.h) {
        return;
    }
    let k = globals.k;
    let cell_base = (y * globals.w + x) * k;

    // M6.C-2-4-c: identity copy.
    // This is a hook block for M5's stochastic sampling (Eq. 8).
    // Currently P_out[x] = P_in[x], deferring parameter inheritance
    // algorithm to M5.
    //
    // M5 plan: Replace this with softmax sampling using local matter
    // flow weights (binding 2) and creature parameter competition
    // via kernel_routing (binding 3). See
    // docs/M6_C2_4_creature_design.md for the M5 hook specification.
    for (var ki: u32 = 0u; ki < k; ki = ki + 1u) {
        p_out[cell_base + ki] = p_in[cell_base + ki];
    }
}
