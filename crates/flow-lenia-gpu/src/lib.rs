#![deny(warnings)]
//! Flow-Lenia GPU: `wgpu` compute pipeline implementation.
//!
//! Implements the WebGPU/wgpu side of the design specified in `DESIGN.md` (Rev. 4),
//! sections §3 (data layout) and §4 (shader pipeline). Populated incrementally
//! during M2 (see `DESIGN.md` §8 for milestones).
