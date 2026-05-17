//! Per-kernel parameters and sampling for Flow-Lenia.
//!
//! Sampling reproduces the parameter ranges of JAX `flowlenia.py:55-64`
//! (`erwanplantec/FlowLenia`, commit `dce428c`), with the per-field deltas
//! vs. paper Table 1 documented in `references/JAX_NOTES.md` §6.

use rand::Rng;

/// Per-kernel parameters defining one `(K_i, G_i)` pair.
///
/// Field meanings (paper Eq. 1, 2 and JAX `flowlenia.py:55-64`):
///
/// - `c0` / `c1`: source / target channels (paper §2, connectivity matrix `M`)
/// - `r`: kernel-`i` radius scale `r_i ∈ [0.2, 1.0]` (paper Eq. 1)
/// - `a`, `b`, `w`: Gaussian-bump-ring parameters (3 rings — `k=3` in paper Eq. 1)
/// - `h`: kernel weight (paper Eq. 3); ignored when parameter embedding
///   (paper Eq. 7) is enabled, since the per-cell map `P_i(x)` takes its place
/// - `mu`, `sigma`: growth function parameters (paper Eq. 2 `μ_i`, `σ_i`).
///   Note: `sigma` here is the *growth* width, not the *reintegration*
///   distribution width (which lives in [`crate::config::FlowLeniaConfig::sigma`]).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KernelEntry {
    /// Source channel index (paper `c_i^0`).
    pub c0: u32,
    /// Target channel index (paper `c_i^1`).
    pub c1: u32,
    /// Kernel scale factor `r_i`.
    pub r: f32,
    /// Centre offsets of the three Gaussian rings.
    pub a: [f32; 3],
    /// Amplitudes of the three Gaussian rings.
    pub b: [f32; 3],
    /// Widths of the three Gaussian rings.
    pub w: [f32; 3],
    /// Kernel weight (paper Eq. 3 `h_i`; used only when parameter embedding
    /// is disabled).
    pub h: f32,
    /// Growth function mean (paper Eq. 2 `μ_i`).
    pub mu: f32,
    /// Growth function width (paper Eq. 2 `σ_i`; **not** the reintegration σ).
    pub sigma: f32,
}

/// Full kernel set for one Flow-Lenia configuration.
///
/// `r_global` is the paper-`R` global maximum neighbourhood radius
/// (JAX `flowlenia.py:57`, range `[2.0, 25.0]`).
#[derive(Clone, Debug, PartialEq)]
pub struct KernelParams {
    /// Paper `R`, the global maximum neighbourhood radius.
    pub r_global: f32,
    /// Per-kernel entries; length is `|K|`.
    pub kernels: Vec<KernelEntry>,
}

/// Inputs required to draw a random [`KernelParams`].
///
/// The channel connectivity (`c_i^0`, `c_i^1`) is sampled uniformly from
/// `0..num_channels` per kernel — this is a deliberate simplification of the
/// JAX reference, where connectivity is supplied as an explicit channel-pair
/// matrix `M` and converted via `conn_from_matrix` (`utils.py:73-85`). The
/// uniform-random form is the default specified in DESIGN.md §6; a
/// matrix-based constructor can be added in a later milestone.
#[derive(Copy, Clone, Debug)]
pub struct SamplingSettings {
    /// Number of kernels `|K|` to draw.
    pub num_kernels: u32,
    /// Number of channels `C`. Must be `> 0` whenever `num_kernels > 0`.
    pub num_channels: u32,
}

impl KernelParams {
    /// Number of kernels `|K|`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether the kernel set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Sample a random kernel set.
    ///
    /// Reproduces JAX `flowlenia.py:55-64` (`erwanplantec/FlowLenia`, commit
    /// `dce428c`):
    ///
    /// ```text
    /// R   ∈ [2.000, 25.0)
    /// per kernel:
    ///   r ∈ [0.200, 1.00)
    ///   μ ∈ [0.050, 0.50)
    ///   σ ∈ [0.001, 0.18)   # narrower than paper Table 1 (0.2 upper)
    ///   h ∈ [0.010, 1.00)   # paper [0, 1] — JAX excludes 0
    ///   a ∈ [0.000, 1.00)³
    ///   b ∈ [0.001, 1.00)³  # paper [0, 1]³ — JAX excludes 0
    ///   w ∈ [0.010, 0.50)³
    /// ```
    ///
    /// Upper bounds are *exclusive* to match `jax.random.uniform` semantics.
    ///
    /// # Panics
    ///
    /// Panics if `settings.num_channels == 0` while `settings.num_kernels > 0`
    /// (cannot sample a channel index from an empty range).
    pub fn sample_random(rng: &mut impl Rng, settings: SamplingSettings) -> Self {
        if settings.num_kernels > 0 {
            assert!(
                settings.num_channels > 0,
                "num_channels must be > 0 when sampling kernels (got num_kernels={}, num_channels=0)",
                settings.num_kernels
            );
        }

        // R is drawn once globally — see JAX flowlenia.py:57.
        let r_global = rng.gen_range(2.0_f32..25.0);

        let kernels = (0..settings.num_kernels)
            .map(|_| KernelEntry::sample_random(rng, settings.num_channels))
            .collect();

        Self { r_global, kernels }
    }
}

impl KernelEntry {
    /// Sample a single kernel entry. See [`KernelParams::sample_random`] for
    /// the range table.
    ///
    /// # Panics
    ///
    /// Panics if `num_channels == 0`.
    pub fn sample_random(rng: &mut impl Rng, num_channels: u32) -> Self {
        assert!(num_channels > 0, "num_channels must be > 0");

        // Connectivity (DESIGN.md §6 — uniform-random simplification of
        // JAX `conn_from_matrix`).
        let c0 = rng.gen_range(0..num_channels);
        let c1 = rng.gen_range(0..num_channels);

        // Scalar params (JAX flowlenia.py:58-61).
        let r = rng.gen_range(0.20_f32..1.00);
        let mu = rng.gen_range(0.05_f32..0.50);
        let sigma = rng.gen_range(0.001_f32..0.18);
        let h = rng.gen_range(0.010_f32..1.00);

        // Three-ring params (JAX flowlenia.py:62-64).
        let a = [
            rng.gen_range(0.000_f32..1.00),
            rng.gen_range(0.000_f32..1.00),
            rng.gen_range(0.000_f32..1.00),
        ];
        let b = [
            rng.gen_range(0.001_f32..1.00),
            rng.gen_range(0.001_f32..1.00),
            rng.gen_range(0.001_f32..1.00),
        ];
        let w = [
            rng.gen_range(0.010_f32..0.50),
            rng.gen_range(0.010_f32..0.50),
            rng.gen_range(0.010_f32..0.50),
        ];

        Self {
            c0,
            c1,
            r,
            a,
            b,
            w,
            h,
            mu,
            sigma,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn settings(num_kernels: u32, num_channels: u32) -> SamplingSettings {
        SamplingSettings {
            num_kernels,
            num_channels,
        }
    }

    /// Smoke-test that the structs are constructible and the `len` /
    /// `is_empty` helpers behave as expected.
    #[test]
    fn kernel_params_helpers() {
        let empty = KernelParams {
            r_global: 10.0,
            kernels: Vec::new(),
        };
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let one = KernelParams {
            r_global: 10.0,
            kernels: vec![KernelEntry {
                c0: 0,
                c1: 0,
                r: 0.5,
                a: [0.3, 0.6, 0.9],
                b: [1.0, 0.5, 0.25],
                w: [0.1, 0.1, 0.1],
                h: 1.0,
                mu: 0.15,
                sigma: 0.02,
            }],
        };
        assert!(!one.is_empty());
        assert_eq!(one.len(), 1);
    }

    /// All sampled fields fall within the documented JAX ranges
    /// (`flowlenia.py:55-64`). A sample of 200 kernels per fixed seed gives
    /// enough draws to push each U(low, high) close to its bounds without
    /// being flaky.
    #[test]
    fn sampling_within_jax_ranges() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p = KernelParams::sample_random(&mut rng, settings(200, 3));

        assert!(
            (2.0..25.0).contains(&p.r_global),
            "R out of range: {}",
            p.r_global
        );

        for (i, k) in p.kernels.iter().enumerate() {
            assert!(k.c0 < 3, "kernel {i}: c0={} >= 3", k.c0);
            assert!(k.c1 < 3, "kernel {i}: c1={} >= 3", k.c1);
            assert!((0.20..1.00).contains(&k.r), "kernel {i}: r={}", k.r);
            assert!((0.05..0.50).contains(&k.mu), "kernel {i}: mu={}", k.mu);
            assert!(
                (0.001..0.18).contains(&k.sigma),
                "kernel {i}: sigma={}",
                k.sigma
            );
            assert!((0.010..1.00).contains(&k.h), "kernel {i}: h={}", k.h);
            for (j, v) in k.a.iter().enumerate() {
                assert!((0.000..1.00).contains(v), "kernel {i} a[{j}]={v}");
            }
            for (j, v) in k.b.iter().enumerate() {
                assert!((0.001..1.00).contains(v), "kernel {i} b[{j}]={v}");
            }
            for (j, v) in k.w.iter().enumerate() {
                assert!((0.010..0.50).contains(v), "kernel {i} w[{j}]={v}");
            }
        }
    }

    /// Same seed → bit-identical output (required for reproducible UI seeds
    /// per DESIGN.md §6).
    #[test]
    fn sampling_is_deterministic_with_seed() {
        let mut rng1 = ChaCha8Rng::seed_from_u64(0xDEAD_BEEF);
        let p1 = KernelParams::sample_random(&mut rng1, settings(10, 3));

        let mut rng2 = ChaCha8Rng::seed_from_u64(0xDEAD_BEEF);
        let p2 = KernelParams::sample_random(&mut rng2, settings(10, 3));

        assert_eq!(p1, p2);
    }

    /// Different seeds produce different parameter sets (basic sanity).
    #[test]
    fn sampling_distinct_seeds_differ() {
        let mut rng1 = ChaCha8Rng::seed_from_u64(1);
        let p1 = KernelParams::sample_random(&mut rng1, settings(5, 3));

        let mut rng2 = ChaCha8Rng::seed_from_u64(2);
        let p2 = KernelParams::sample_random(&mut rng2, settings(5, 3));

        assert_ne!(p1, p2);
    }

    /// Output length matches `num_kernels`.
    #[test]
    fn sampling_respects_num_kernels() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let p = KernelParams::sample_random(&mut rng, settings(13, 3));
        assert_eq!(p.len(), 13);
    }

    /// `num_kernels = 0` produces an empty kernel list without panicking.
    /// (`num_channels = 0` would not be exercised because the
    /// channel-index sampling is skipped entirely.)
    #[test]
    fn sampling_zero_kernels_is_empty() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let p = KernelParams::sample_random(&mut rng, settings(0, 3));
        assert!(p.is_empty());
        // R is still drawn — that's expected behaviour (deterministic RNG
        // state advance regardless of |K|).
        assert!((2.0..25.0).contains(&p.r_global));
    }

    /// Sampling with `num_channels = 0` and `num_kernels > 0` must panic
    /// (cannot pick a channel index from an empty range).
    #[test]
    #[should_panic(expected = "num_channels must be > 0")]
    fn sampling_panics_on_zero_channels() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let _ = KernelParams::sample_random(&mut rng, settings(5, 0));
    }

    /// Channel indices cover the full `0..num_channels` range across enough
    /// samples (defensive — guards against an off-by-one in `gen_range`).
    #[test]
    fn sampling_covers_all_channels() {
        let mut rng = ChaCha8Rng::seed_from_u64(2024);
        let p = KernelParams::sample_random(&mut rng, settings(500, 4));

        let mut seen_c0 = [false; 4];
        let mut seen_c1 = [false; 4];
        for k in &p.kernels {
            seen_c0[k.c0 as usize] = true;
            seen_c1[k.c1 as usize] = true;
        }
        assert!(
            seen_c0.iter().all(|&b| b),
            "not all source channels were sampled: {seen_c0:?}"
        );
        assert!(
            seen_c1.iter().all(|&b| b),
            "not all target channels were sampled: {seen_c1:?}"
        );
    }
}
