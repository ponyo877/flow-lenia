//! Terminal visualisation of a Flow-Lenia activation field.
//!
//! Uses ANSI 256-colour background escapes (`\x1b[48;5;Nm`) — supported by
//! every modern terminal emulator without 24-bit-colour negotiation, which
//! some older TTYs (vt100 over serial, plain `xterm`) still lack. Each
//! cell is one space character with its background painted from the
//! 6×6×6 RGB colour cube at palette indices `16 .. 232`:
//!
//! ```text
//! index = 16 + 36·r + 6·g + b      // r, g, b ∈ [0, 5]
//! ```
//!
//! Channel mapping: up to 3 channels go to R / G / B in that order;
//! `C = 1` shows up as a red-only display, `C = 2` as red/green,
//! `C ≥ 3` as full RGB (extra channels are silently dropped — the
//! palette is exhausted at 3). Each channel value is clamped to
//! `[0, 1]` then quantised to the 6-level palette axis.
//!
//! `viuer` (true 24-bit RGB image rendering via Sixel / Kitty / iTerm
//! protocols) was considered but rejected for M1.14 — it adds heavy
//! deps and per-terminal protocol negotiation. A `feature = "viuer"`
//! flag can be added later if higher-resolution preview becomes
//! useful.

use flow_lenia_core::FlowLeniaSimulator;
use std::io::Write;

/// Clear the terminal and move the cursor home. Uses
/// `\x1b[2J` (clear entire screen) followed by `\x1b[H` (cursor to
/// `(1, 1)`). Both are standard ECMA-48 escapes.
pub fn clear_terminal() {
    // Write directly to stdout, bypassing any line-buffering layered
    // on top — this is called once per frame and we want it visible
    // *before* the new frame's cells start landing.
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(b"\x1b[2J\x1b[H");
    let _ = out.flush();
}

/// Render the simulator's current activation field to stdout.
///
/// Output layout: one terminal *row* per grid row, each grid cell
/// rendered as one space character with a coloured background. After
/// the last cell of each row we emit a reset (`\x1b[0m`) and a
/// newline; after the last row we leave the cursor on the line
/// *after* the field so subsequent `println!` calls (the metrics
/// line) appear cleanly below.
///
/// On the default 64×64 demo grid this produces ~4 KB per frame —
/// well under the per-flush cost on a modern terminal.
pub fn render_to_terminal(sim: &FlowLeniaSimulator) {
    let a = sim.activation();
    let (h, w, c) = a.dim();
    let channels_used = c.min(3);

    let mut buf = String::with_capacity(h * (w * 24 + 8));
    for y in 0..h {
        for x in 0..w {
            // Gather up to 3 channels for the RGB triplet; pad with 0.
            let mut rgb = [0_u8; 3];
            for ci in 0..channels_used {
                rgb[ci] = quantise_to_palette_axis(a[[y, x, ci]]);
            }
            let palette = 16 + 36 * rgb[0] + 6 * rgb[1] + rgb[2];
            // Two-space cell so terminal cells appear roughly square
            // (most fonts are ~2:1 tall:wide).
            buf.push_str(&format!("\x1b[48;5;{palette}m  "));
        }
        // Reset background and end the line.
        buf.push_str("\x1b[0m\n");
    }
    let _ = std::io::stdout().lock().write_all(buf.as_bytes());
}

/// Quantise an `f32` activation value in `[0, 1]` (out-of-range values
/// are clamped) onto the 6-level colour-cube axis (`0..=5`). Anything
/// `< 0` maps to 0, `≥ 1` to 5; the in-range mapping is uniform.
///
/// Exposed `pub(crate)` so a future `render_to_image` companion can
/// share the quantisation rule; not part of the public API.
fn quantise_to_palette_axis(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    // Map [0, 1] → {0, 1, 2, 3, 4, 5}. `(v * 5.999).floor()` distributes
    // the input range across all 6 buckets without putting `v == 1.0`
    // into bucket 6.
    (v * 5.999_f32) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantise_endpoints_and_midpoints() {
        assert_eq!(quantise_to_palette_axis(0.0), 0);
        assert_eq!(quantise_to_palette_axis(1.0), 5);
        // Out-of-range clamp.
        assert_eq!(quantise_to_palette_axis(-1.0), 0);
        assert_eq!(quantise_to_palette_axis(2.0), 5);
        // Midpoint: 0.5 · 5.999 = 2.9995 → 2.
        assert_eq!(quantise_to_palette_axis(0.5), 2);
        // Each bucket boundary lands where expected.
        assert_eq!(quantise_to_palette_axis(0.167), 1); // 1.001 → 1
        assert_eq!(quantise_to_palette_axis(0.833), 4); // 4.997 → 4
    }
}
