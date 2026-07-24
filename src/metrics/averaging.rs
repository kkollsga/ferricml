//! The typed vocabulary for combining per-class scores into one number.
//!
//! Averaging is a separate vocabulary from the metrics that use it so that one
//! set of names covers binary and multiclass evaluation. A metric asks for an
//! [`Averaging`], never for a string or a boolean pair.

/// How per-class scores combine into a single value.
///
/// The variants are the standard averaging conventions. For single-label
/// classification — every row has exactly one true and one predicted label —
/// [`Average::Micro`] pools all classes into one count pair, which makes
/// micro-averaged precision, recall, and F-score all equal to accuracy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Average {
    /// Report the positive class, label `1`, alone.
    ///
    /// Valid only when every observed label is `0` or `1`; a wider label set
    /// is [`MetricError::NotBinary`](super::MetricError::NotBinary) rather
    /// than a silently reinterpreted one-vs-rest score.
    Binary,
    /// Pool every class's counts, then score the pooled counts once.
    Micro,
    /// Unweighted mean of the per-class scores.
    Macro,
    /// Mean of the per-class scores weighted by each class's true support.
    Weighted,
}

/// What an averaged score does with a class whose denominator is empty.
///
/// A class that is never predicted has no precision, and a class with no true
/// rows has no recall. FerricML reports that as an error by default rather
/// than substituting a value, because the substituted value is indisputably
/// not a measurement. The other variants exist so a caller that wants one of
/// the conventional substitutions states it explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ZeroDivision {
    /// Report [`MetricError::Undefined`](super::MetricError::Undefined).
    Error,
    /// Score the affected class as zero and keep it in the average.
    Zero,
    /// Leave the affected class out of the average entirely.
    ///
    /// The average is undefined if this removes every class.
    Skip,
}

/// A complete averaging request: how classes combine, and what an empty
/// denominator means.
///
/// [`Average`] converts into this, so the common case stays short and the
/// zero-denominator policy is a visible opt-in:
///
/// ```
/// use ferricml::metrics::{Average, Averaging, ConfusionMatrix, ZeroDivision};
///
/// let matrix = ConfusionMatrix::new(&[0, 1, 2], &[0, 1, 1])?;
/// // Class 2 is never predicted, so its precision does not exist.
/// assert!(matrix.precision(Average::Macro).is_err());
/// assert_eq!(
///     matrix.precision(Averaging::new(Average::Macro).with_zero_division(ZeroDivision::Skip)),
///     Ok(0.75),
/// );
/// # Ok::<(), ferricml::metrics::MetricError>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Averaging {
    average: Average,
    zero_division: ZeroDivision,
}

impl Averaging {
    /// Averages the requested way and reports an empty denominator as an error.
    pub const fn new(average: Average) -> Self {
        Self {
            average,
            zero_division: ZeroDivision::Error,
        }
    }

    /// Sets what happens to a class whose denominator is empty.
    #[must_use]
    pub const fn with_zero_division(mut self, zero_division: ZeroDivision) -> Self {
        self.zero_division = zero_division;
        self
    }

    /// Returns how per-class scores combine.
    pub const fn average(self) -> Average {
        self.average
    }

    /// Returns the empty-denominator policy.
    pub const fn zero_division(self) -> ZeroDivision {
        self.zero_division
    }
}

impl From<Average> for Averaging {
    fn from(average: Average) -> Self {
        Self::new(average)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_average_converts_into_the_strict_request() {
        let converted: Averaging = Average::Macro.into();
        assert_eq!(converted, Averaging::new(Average::Macro));
        assert_eq!(converted.average(), Average::Macro);
        assert_eq!(converted.zero_division(), ZeroDivision::Error);
    }

    #[test]
    fn the_zero_division_policy_is_the_only_thing_the_builder_changes() {
        const REQUEST: Averaging =
            Averaging::new(Average::Weighted).with_zero_division(ZeroDivision::Skip);
        assert_eq!(REQUEST.average(), Average::Weighted);
        assert_eq!(REQUEST.zero_division(), ZeroDivision::Skip);
        assert_eq!(
            REQUEST.with_zero_division(ZeroDivision::Error),
            Averaging::new(Average::Weighted)
        );
    }
}
