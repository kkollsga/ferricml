//! Mean functions connecting a raw score to the quantity a loss compares.

use crate::numeric::sigmoid_f64;

/// A link function, kept separate from the loss that uses it.
///
/// A loss composes with a link instead of hard-coding one, so the same
/// objective family can be re-pointed at a different mean function without
/// rewriting its derivatives. The trait is stateless: every method is an
/// associated function, which keeps a consumer's dispatch entirely at compile
/// time.
///
/// FerricML names the directions the way generalized-linear-model literature
/// does. The *link* maps a mean onto the unconstrained raw scale an optimizer
/// works in, and [`Link::inverse`] maps back.
pub(crate) trait Link {
    /// Maps a raw score onto the mean the loss compares with its target.
    fn inverse(raw: f64) -> f64;
}

/// The logit link, whose inverse is the logistic sigmoid.
///
/// Saturation is the sigmoid's documented `0`/`1` boundary, so a raw score of
/// large magnitude in either sign yields an exact probability rather than a
/// non-finite intermediate.
pub(crate) enum Logit {}

impl Link for Logit {
    fn inverse(raw: f64) -> f64 {
        sigmoid_f64(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logit_inverse_is_the_shared_sigmoid_at_every_magnitude() {
        for step in -2_000..=2_000 {
            let raw = f64::from(step) / 20.0;
            assert_eq!(Logit::inverse(raw).to_bits(), sigmoid_f64(raw).to_bits());
        }
        for &raw in &[f64::MAX, -f64::MAX, 1.0e300, -1.0e300, 0.0] {
            assert_eq!(Logit::inverse(raw).to_bits(), sigmoid_f64(raw).to_bits());
        }
    }
}
