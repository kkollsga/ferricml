//! Declared, compile-time capability descriptors for fitted estimator types.

use super::Estimator;

/// What a fitted estimator type declares it can do.
///
/// The descriptor is a compile-time constant carried by [`HasCapabilities`], so
/// meta-layers ask what an estimator supports instead of matching on its
/// concrete type. It is deliberately small, and stays small by one rule: a
/// capability belongs here only when it varies between estimator types *and* is
/// not already guaranteed by the type system. Non-finite inputs, for example,
/// are already impossible inside a
/// [`MatrixView`](crate::data::MatrixView), so tolerating them is not a field
/// here. Producing probabilities *was* in that category until margin-based
/// classifiers arrived: it is now [`probability`](Self::probability), because
/// [`Classifier`](super::Classifier) no longer requires it.
///
/// Fields are private and read through `const` accessors so that declaring a
/// further capability later stays a compatible change. Construction starts from
/// [`Capabilities::NONE`] and opts in explicitly:
///
/// ```
/// use ferricml::api::{Capabilities, Estimator, HasCapabilities};
///
/// struct MeanBaseline;
///
/// impl Estimator for MeanBaseline {
///     fn n_features_in(&self) -> usize {
///         1
///     }
/// }
///
/// impl HasCapabilities for MeanBaseline {
///     const CAPABILITIES: Capabilities = Capabilities::NONE.with_sample_weights(true);
/// }
///
/// assert!(MeanBaseline::CAPABILITIES.sample_weights());
/// assert!(!MeanBaseline::CAPABILITIES.artifact());
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capabilities {
    sample_weights: bool,
    artifact: bool,
    multiclass: bool,
    decision_function: bool,
    probability: bool,
}

impl Capabilities {
    /// A descriptor that declares nothing.
    ///
    /// This is the conservative starting point and the default for any type
    /// that has not opted in, so an omitted declaration understates a type
    /// rather than promising behavior it does not have.
    pub const NONE: Self = Self {
        sample_weights: false,
        artifact: false,
        multiclass: false,
        decision_function: false,
        probability: false,
    };

    /// Declares whether fitting accepts per-sample weights.
    #[must_use]
    pub const fn with_sample_weights(mut self, supported: bool) -> Self {
        self.sample_weights = supported;
        self
    }

    /// Declares whether fitted values round trip through a stable artifact.
    #[must_use]
    pub const fn with_artifact(mut self, supported: bool) -> Self {
        self.artifact = supported;
        self
    }

    /// Declares whether fitting accepts an arbitrary observed class set.
    #[must_use]
    pub const fn with_multiclass(mut self, supported: bool) -> Self {
        self.multiclass = supported;
        self
    }

    /// Declares whether the fitted classifier exposes a raw decision score.
    #[must_use]
    pub const fn with_decision_function(mut self, supported: bool) -> Self {
        self.decision_function = supported;
        self
    }

    /// Declares whether the fitted classifier produces probabilities.
    #[must_use]
    pub const fn with_probability(mut self, supported: bool) -> Self {
        self.probability = supported;
        self
    }

    /// Whether fitting accepts per-sample weights.
    ///
    /// A type declaring this offers a `fit_weighted` entry point whose fitted
    /// result for unit weights matches its unweighted fit.
    #[must_use]
    pub const fn sample_weights(self) -> bool {
        self.sample_weights
    }

    /// Whether fitted values encode to, and decode from, a stable artifact.
    #[must_use]
    pub const fn artifact(self) -> bool {
        self.artifact
    }

    /// Whether fitting accepts an arbitrary observed class set.
    ///
    /// A type declaring this offers a multiclass fitting entry point over
    /// [`ClassTargets`](crate::data::ClassTargets) whose fitted result has one
    /// probability column per observed label, in sorted label order, for any
    /// number of classes. Whether the classifier produces probabilities *at
    /// all* is [`probability`](Self::probability) and is not what this records.
    #[must_use]
    pub const fn multiclass(self) -> bool {
        self.multiclass
    }

    /// Whether the fitted classifier exposes a raw, unsquashed decision score.
    ///
    /// A type declaring this offers a `decision_function` entry point producing
    /// one real-valued score per row, monotone in the model's confidence, whose
    /// squashing is the probability. Whether the classifier produces
    /// *probabilities* is [`probability`](Self::probability) and is not what
    /// this records; a margin-based classifier may declare this and not that.
    ///
    /// This exists because Rust has no runtime attribute lookup: a
    /// meta-estimator generic over a classifier cannot discover whether the
    /// type it holds has a decision function, and the classical formulations of
    /// probability calibration and of threshold-sweeping scores are both
    /// written against one. The declaration is what lets such a consumer select
    /// its behavior at compile time instead of assuming a method exists.
    ///
    /// Note the deliberate limit of what a tag can do. Declaring this makes the
    /// capability *discoverable*; it does not make the method *callable*,
    /// because a decision function is an inherent method rather than part of
    /// the object-safe classifier contract. A consumer that needs to call one
    /// still needs a bound naming a trait that carries it.
    #[must_use]
    pub const fn decision_function(self) -> bool {
        self.decision_function
    }

    /// Whether the fitted classifier produces a probability per class.
    ///
    /// A type declaring this implements
    /// [`ProbabilisticClassifier`](super::ProbabilisticClassifier), so the
    /// declaration is normally a restatement of a bound a generic caller can
    /// simply require. It exists as a tag for the one place that bound is
    /// unavailable: a runtime dispatch value, where the concrete type is erased
    /// by construction and the question can only be asked, not proven.
    ///
    /// Producing probabilities used to be required of every
    /// [`Classifier`](super::Classifier), which is why this was not a field
    /// before. It is one now because margin-based classifiers — ridge
    /// classification, discriminant analysis, discrete boosting — have a
    /// natural output that is a score rather than a distribution, and forcing
    /// them to squash it would have been fabricating a number they never
    /// earned.
    #[must_use]
    pub const fn probability(self) -> bool {
        self.probability
    }

    /// Capabilities declared by both descriptors.
    ///
    /// This is what a runtime dispatch enum or a fitted composition can promise
    /// without knowing which variant or part it holds, so batch dispatch
    /// validates once instead of per call site.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self {
            sample_weights: self.sample_weights && other.sample_weights,
            artifact: self.artifact && other.artifact,
            multiclass: self.multiclass && other.multiclass,
            decision_function: self.decision_function && other.decision_function,
            probability: self.probability && other.probability,
        }
    }
}

/// Compile-time capability declaration for a fitted estimator type.
///
/// This is a separate generic trait rather than an associated constant on
/// [`Estimator`] because a trait carrying an associated constant is not
/// dyn-compatible, and the estimator categories must stay object-safe for
/// batch-level dispatch. It complements them exactly as
/// [`HasParams`](super::HasParams) does.
///
/// The default declares nothing, so every capability is an explicit opt-in and
/// a type that never implements this trait is simply undeclared. The declared
/// value is part of FerricML's public, semver-relevant contract: it is asserted
/// against real behavior by the estimator conformance battery rather than
/// maintained by inspection.
pub trait HasCapabilities: Estimator {
    /// Capabilities this estimator type declares.
    const CAPABILITIES: Capabilities = Capabilities::NONE;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_conservative_descriptor_declares_nothing() {
        assert!(!Capabilities::NONE.sample_weights());
        assert!(!Capabilities::NONE.artifact());
        assert!(!Capabilities::NONE.multiclass());
        assert!(!Capabilities::NONE.decision_function());
        assert!(!Capabilities::NONE.probability());
    }

    #[test]
    fn declarations_are_independent_and_const_evaluable() {
        const WEIGHTS: Capabilities = Capabilities::NONE.with_sample_weights(true);
        const ARTIFACT: Capabilities = Capabilities::NONE.with_artifact(true);
        const MULTICLASS: Capabilities = Capabilities::NONE.with_multiclass(true);
        const DECISION: Capabilities = Capabilities::NONE.with_decision_function(true);
        const PROBABILITY: Capabilities = Capabilities::NONE.with_probability(true);
        const BOTH: Capabilities = WEIGHTS.with_artifact(true);

        assert!(WEIGHTS.sample_weights() && !WEIGHTS.artifact() && !WEIGHTS.multiclass());
        assert!(!ARTIFACT.sample_weights() && ARTIFACT.artifact() && !ARTIFACT.multiclass());
        assert!(!MULTICLASS.sample_weights() && !MULTICLASS.artifact() && MULTICLASS.multiclass());
        assert!(BOTH.sample_weights() && BOTH.artifact() && !BOTH.multiclass());
        assert!(DECISION.decision_function() && !DECISION.multiclass() && !DECISION.artifact());
        assert!(!WEIGHTS.decision_function() && !BOTH.decision_function());
        assert_eq!(BOTH.with_sample_weights(false), ARTIFACT);
        assert_eq!(MULTICLASS.with_multiclass(false), Capabilities::NONE);
        assert_eq!(DECISION.with_decision_function(false), Capabilities::NONE);
        assert!(PROBABILITY.probability() && !PROBABILITY.decision_function());
        assert!(!DECISION.probability() && !BOTH.probability());
        assert_eq!(PROBABILITY.with_probability(false), Capabilities::NONE);
    }

    #[test]
    fn intersection_keeps_only_what_both_sides_declare() {
        const WEIGHTS: Capabilities = Capabilities::NONE.with_sample_weights(true);
        const BOTH: Capabilities = WEIGHTS.with_artifact(true);

        assert_eq!(BOTH.intersection(BOTH), BOTH);
        assert_eq!(BOTH.intersection(WEIGHTS), WEIGHTS);
        assert_eq!(
            BOTH.with_multiclass(true).intersection(BOTH),
            BOTH,
            "a capability only one side declares is not promised"
        );
        assert_eq!(
            BOTH.with_decision_function(true).intersection(BOTH),
            BOTH,
            "a decision function only one side has is not promised"
        );
        assert_eq!(
            BOTH.with_probability(true).intersection(BOTH),
            BOTH,
            "probabilities only one side produces are not promised"
        );
        assert_eq!(BOTH.intersection(Capabilities::NONE), Capabilities::NONE);
        assert_eq!(WEIGHTS.intersection(BOTH), BOTH.intersection(WEIGHTS));
    }

    #[test]
    fn an_undeclared_estimator_defaults_to_declaring_nothing() {
        struct Undeclared;

        impl Estimator for Undeclared {
            fn n_features_in(&self) -> usize {
                1
            }
        }

        impl HasCapabilities for Undeclared {}

        assert_eq!(Undeclared::CAPABILITIES, Capabilities::NONE);
    }
}
