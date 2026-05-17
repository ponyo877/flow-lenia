#![deny(warnings)]
//! Flow-Lenia core: platform-independent CA logic and CPU reference implementation.
//!
//! Implements Plantec et al. 2025 (arXiv:2506.08569v1) per the design contract
//! in `DESIGN.md` (Rev. 4). JAX implementation cross-reference is documented in
//! `references/JAX_NOTES.md` (commit `dce428c` of `erwanplantec/FlowLenia`).
//!
//! Contents are populated incrementally from M1.2 onwards. M1.1 establishes
//! only the crate skeleton.
