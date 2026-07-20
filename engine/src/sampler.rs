/// Minimal PCG32 (O'Neill) — deterministic, per-worker, no dependencies.
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64) -> Self {
        let mut s = Self { state: 0, inc: (seed << 1) | 1 };
        s.next_u32();
        s.state = s.state.wrapping_add(seed);
        s.next_u32();
        s
    }

    /// `pub(crate)` for `episode::reset_episode`'s replay-pool index: deriving
    /// it from `next_f32` instead can round up to exactly `len` in f32 at
    /// million-element pool sizes and panic out of bounds.
    pub(crate) fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    pub fn next_f32(&mut self) -> f32 {
        // 24 high bits -> [0,1)
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }
}

/// Softmax-CDF sample; numerically stable (max-subtracted).
/// Returns (index, ln softmax(logits)[index]).
pub fn sample_categorical(logits: &[f32], rng: &mut Pcg32) -> (usize, f32) {
    assert!(!logits.is_empty(), "sample_categorical: empty logits");
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0f32;
    // two passes: normalizer, then CDF walk
    for &l in logits {
        z += (l - max).exp();
    }
    let target = rng.next_f32() * z;
    let mut acc = 0f32;
    let mut idx = logits.len() - 1; // fallback: last index (fp round-off)
    for (i, &l) in logits.iter().enumerate() {
        acc += (l - max).exp();
        if acc > target {
            idx = i;
            break;
        }
    }
    let logprob = (logits[idx] - max) - z.ln();
    (idx, logprob)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_given_seed() {
        let logits = vec![0.1f32, 2.0, -1.0, 0.5];
        let (mut a, mut b) = (Pcg32::new(42), Pcg32::new(42));
        for _ in 0..100 {
            assert_eq!(sample_categorical(&logits, &mut a).0,
                       sample_categorical(&logits, &mut b).0);
        }
    }

    #[test]
    fn samples_follow_distribution() {
        // logits [0, ln(3)]: p = [0.25, 0.75]
        let logits = vec![0.0f32, (3.0f32).ln()];
        let mut rng = Pcg32::new(7);
        let n = 100_000;
        let ones = (0..n).filter(|_| sample_categorical(&logits, &mut rng).0 == 1).count();
        let frac = ones as f32 / n as f32;
        assert!((frac - 0.75).abs() < 0.01, "got {frac}");
    }

    #[test]
    fn logprob_matches_log_softmax() {
        let logits = vec![1.0f32, 2.0, 3.0];
        // log_softmax reference computed by hand: subtract max, exp, normalize
        let max = 3.0f32;
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let z: f32 = exps.iter().sum();
        let mut rng = Pcg32::new(1);
        for _ in 0..50 {
            let (a, lp) = sample_categorical(&logits, &mut rng);
            let expected = (exps[a] / z).ln();
            assert!((lp - expected).abs() < 1e-6, "a={a} lp={lp} expected={expected}");
        }
    }

    #[test]
    fn extreme_logits_do_not_nan() {
        let logits = vec![1000.0f32, -1000.0, 0.0];
        let mut rng = Pcg32::new(3);
        let (a, lp) = sample_categorical(&logits, &mut rng);
        assert_eq!(a, 0);
        assert!(lp.is_finite() && lp <= 0.0);
    }

    #[test]
    #[should_panic(expected = "empty logits")]
    fn empty_logits_panics_with_clear_message() {
        let mut rng = Pcg32::new(1);
        sample_categorical(&[], &mut rng);
    }
}
