use std::error::Error;
use std::fmt;

/// Errors produced while fitting or using a FerricML model.
///
/// Data-container construction has its own [`crate::data::DataError`] because
/// it occurs before an estimator is involved. All estimators use this error
/// surface once fitting or prediction begins.
///
/// # What is deliberately absent
///
/// There is no variant for an empty target vector, a binary label outside
/// `{0, 1}`, or a non-finite regression target. Those are
/// [`DataError`](crate::data::DataError) cases, refused by
/// [`BinaryTargets::new`](crate::data::BinaryTargets::new) and
/// [`RegressionTargets::new`](crate::data::RegressionTargets::new), and the
/// refusal is total: the target containers have no unchecked constructor, and
/// `select` preserves what `new` established. An estimator taking one of them
/// cannot be handed a value that would need such a variant, so a variant for it
/// would document a failure no caller can observe.
///
/// [`ModelError::EmptyData`] and [`ModelError::NonFiniteFeature`] look like the
/// same case and are **not**: several entry points take a bare `&[f32]` rather
/// than a container — every `predict_one`, and calibration, which fits on
/// decision scores — and nothing has validated those.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelError {
    /// Training data has no samples or features.
    EmptyData,
    /// The target count does not match the sample count.
    TargetLength {
        /// Number of rows the feature matrix holds.
        rows: usize,
        /// Number of targets supplied alongside them.
        targets: usize,
    },
    /// The sample-weight count does not match the sample count.
    SampleWeightLength {
        /// Number of rows the feature matrix holds.
        rows: usize,
        /// Number of sample weights supplied alongside them.
        weights: usize,
    },
    /// A feature is NaN or infinite.
    NonFiniteFeature {
        /// Zero-based row of the first offending value, in row-major order.
        row: usize,
        /// Zero-based column of that value.
        column: usize,
    },
    /// `n_estimators` is zero.
    InvalidEstimatorCount,
    /// `max_depth` is zero.
    InvalidMaxDepth,
    /// `min_samples_split` is less than two.
    InvalidMinSamplesSplit,
    /// `min_samples_leaf` is zero.
    InvalidMinSamplesLeaf,
    /// A fixed `max_features` count exceeds the fitted feature count.
    InvalidMaxFeatures {
        /// Feature count the parameter asked each split to consider.
        requested: usize,
        /// Feature count the training data actually has.
        available: usize,
    },
    /// A fixed `n_jobs` count is zero.
    InvalidJobCount,
    /// The sample count exceeds the internal bootstrap-counter format.
    TooManyRows,
    /// The feature count exceeds the internal tree format.
    TooManyFeatures,
    /// A fitted tree exceeds the internal node-index format.
    TreeTooLarge,
    /// A fitted tree violates the internal node format's structural rules.
    ///
    /// Separate from [`ModelError::TreeTooLarge`], which is a size bound: this
    /// reports a topology or a leaf value the format cannot represent at any
    /// size — an unreachable node, a child index that does not follow its
    /// parent, a split on a column the tree was not fitted over, or a
    /// non-finite threshold or leaf. It is a backstop against an internal
    /// inconsistency rather than a statement about the caller's data.
    InvalidTreeStructure,
    /// Input feature width differs from the fitted width.
    FeatureDimension {
        /// Feature width the model was fitted on.
        expected: usize,
        /// Feature width of the batch supplied.
        actual: usize,
    },
    /// A caller-provided output buffer has the wrong length.
    OutputLength {
        /// Length the output buffer needs for this batch.
        expected: usize,
        /// Length the caller supplied.
        actual: usize,
    },
    /// Scaling a finite feature produced a non-finite `f32` output.
    NonFiniteTransform {
        /// Zero-based row of the first offending output, in row-major order.
        row: usize,
        /// Zero-based column of that output.
        column: usize,
    },
    /// The requested output matrix dimensions cannot be represented.
    OutputShapeOverflow {
        /// Row count of the requested output matrix.
        rows: usize,
        /// Column count of the requested output matrix.
        columns: usize,
    },
    /// A requested class was not observed during fitting.
    UnknownClass {
        /// The label asked for, which is absent from the fitted classes.
        class: u8,
    },
    /// A scoped model-training worker panicked.
    WorkerPanicked,
    /// A classifier requiring both binary classes observed only one.
    RequiresTwoClasses,
    /// The inverse regularization strength is not finite and positive.
    InvalidRegularization,
    /// The maximum optimization iteration count is zero.
    InvalidIterationCount,
    /// The convergence tolerance is not finite and positive.
    InvalidTolerance,
    /// A least-squares tolerance is not finite and non-negative.
    InvalidLeastSquaresTolerance,
    /// A ridge penalty is not finite and non-negative.
    InvalidRidgeAlpha,
    /// An elastic-net-family penalty is not finite and non-negative.
    ///
    /// Separate from [`ModelError::InvalidRidgeAlpha`] because the two `alpha`
    /// parameters are different quantities: a ridge penalty accompanies an
    /// undivided squared-error term, and this one accompanies a squared-error
    /// term divided by twice the total sample weight.
    InvalidPenaltyAlpha,
    /// An elastic-net mixing parameter is outside `0.0..=1.0`.
    InvalidL1Ratio,
    /// A boosting learning rate is not finite and positive.
    InvalidLearningRate,
    /// A boosting iteration count is outside its supported bound.
    InvalidBoostingIterationCount,
    /// A boosting leaf-count limit is outside its supported bound.
    InvalidMaxLeafNodes,
    /// A boosting depth limit is outside its supported bound.
    InvalidBoostingMaxDepth,
    /// A histogram bin count is outside its supported bound.
    InvalidMaxBins,
    /// A boosting L2 penalty is not finite and non-negative.
    InvalidL2Regularization,
    /// Requested boosting bounds exceed the total logical-node limit.
    BoostingModelTooLarge,
    /// Finite training values overflowed an internal `f32` result.
    NumericalOverflow,
    /// A fitted prediction accumulated to NaN or infinity.
    NonFinitePrediction {
        /// Zero-based row whose prediction accumulated to NaN or infinity.
        row: usize,
    },
    /// The linear solver encountered a non-positive-definite system, or an
    /// iterative solver could not make progress from its current iterate.
    LinearSolveFailed,
    /// An iterative solver stopped short of its `tol`.
    ///
    /// Reported instead of returning the last iterate, because an unconverged
    /// model is indistinguishable from a converged one once it is fitted.
    ///
    /// # Why neither this sentence nor the message names a cause
    ///
    /// Two different things stop a solver short of `tol`, and both report
    /// here. The `max_iter` budget can run out. Or the solver can reach a point
    /// where no step it can represent improves the objective — an L-BFGS
    /// line-search bracket that collapsed, a Newton backtracking search that
    /// found no descending step — which is the observable form of a `tol` below
    /// the objective's own numerical resolution, near `1e-9` for a log-loss of
    /// order one. To a caller both mean the same thing, that the fit is not
    /// converged, which is why one variant carries both. Naming either cause
    /// would make the message false on the other, and naming the budget sent
    /// callers to raise `max_iter` — the one parameter that cannot help a fit
    /// whose budget was never the constraint.
    ///
    /// # What separates them
    ///
    /// `iterations` does, in one direction. It never exceeds the `max_iter`
    /// that was set, so `iterations < max_iter` proves the budget was not the
    /// constraint and the remedy is a looser `tol`. Equality does not prove the
    /// converse, because a step can fail to descend on the last iteration the
    /// budget allowed. Inequality is therefore decisive and equality leaves
    /// both possible.
    SolverDidNotConverge {
        /// Iterations completed before the solver stopped.
        iterations: usize,
    },
    /// A scalar-valued operation was asked of a model that produces a vector.
    ///
    /// A multiclass fit has one raw score per class, so a method returning one
    /// value per row reports this rather than silently returning one component
    /// of the vector.
    MulticlassOutput {
        /// Number of values the fitted model produces for one row.
        columns: usize,
    },
    /// A quantile range is not a pair of percentiles in `0.0..=100.0` with the
    /// lower value first.
    InvalidQuantileRange,
    /// A min-max output range is not a finite interval with the lower bound
    /// strictly below the upper one.
    InvalidFeatureRange,
    /// A decision threshold is not finite.
    InvalidThreshold,
    /// An inverse transformation was requested of a transformer that was not
    /// given one.
    ///
    /// This is a typed error rather than a
    /// [`Capabilities`](super::Capabilities) declaration on purpose, and not
    /// because the transformer family was skipped when probability production
    /// became a declared capability. A capability is an associated constant and
    /// therefore describes a *type*; whether a
    /// [`FunctionTransformer`](crate::preprocessing::FunctionTransformer)
    /// inverts is decided per *instance*, by whether `with_inverse_func` was
    /// called, so no per-type value is true of it. The instance-level question
    /// is answerable exactly through its params' `inverse_func`, which is an
    /// `Option`.
    NoInverseFunction,
    /// A polynomial expansion would be wider than FerricML will build.
    ///
    /// Raised at *fit* time, before the expansion's term table is reserved,
    /// because width is the failure mode of this transformer rather than an
    /// unlikely edge of it: the expanded width grows as `C(n + d, d)`, so a
    /// merely unremarkable request — fifty features at degree ten — asks for
    /// seventy-five billion output columns. Both causes report here, the
    /// arithmetic overflowing and the width exceeding the supported bound,
    /// because from a caller's side they are the same mistake and the two
    /// numbers that caused it are what identifies it.
    FeatureExpansionOverflow {
        /// Number of input features the expansion was requested over.
        n_features: usize,
        /// Requested polynomial degree.
        degree: u32,
    },
    /// A feature expansion was configured to produce no columns at all.
    ///
    /// Reachable one way: degree zero with the bias column disabled, which
    /// asks for the empty expansion of every row. A zero-width output is not a
    /// transformer's answer to anything, so it is refused where it is
    /// described rather than returned as a matrix nothing can consume.
    EmptyFeatureExpansion,
    /// A multiclass linear fit would need a second-order system larger than the
    /// supported bound, which is `classes * (features + intercept)` parameters.
    MulticlassSystemTooLarge {
        /// Number of observed classes.
        classes: usize,
        /// Number of input features.
        features: usize,
        /// Largest supported parameter count.
        limit: usize,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyData => f.write_str("training data must have at least one row and column"),
            Self::TargetLength { rows, targets } => {
                write!(f, "target length {targets} does not match row count {rows}")
            }
            Self::SampleWeightLength { rows, weights } => {
                write!(
                    f,
                    "sample-weight length {weights} does not match row count {rows}"
                )
            }
            Self::NonFiniteFeature { row, column } => {
                write!(f, "feature at row {row}, column {column} is not finite")
            }
            Self::InvalidEstimatorCount => f.write_str("n_estimators must be at least one"),
            Self::InvalidMaxDepth => f.write_str("max_depth must be at least one when set"),
            Self::InvalidMinSamplesSplit => f.write_str("min_samples_split must be at least two"),
            Self::InvalidMinSamplesLeaf => f.write_str("min_samples_leaf must be at least one"),
            Self::InvalidMaxFeatures {
                requested,
                available,
            } => write!(
                f,
                "max_features count {requested} must be in 1..={available}"
            ),
            Self::InvalidJobCount => f.write_str("n_jobs count must be at least one"),
            Self::TooManyRows => f.write_str("row count exceeds the bootstrap counter format"),
            Self::TooManyFeatures => f.write_str("feature count exceeds the packed node format"),
            Self::TreeTooLarge => f.write_str("tree exceeds the packed node format"),
            Self::InvalidTreeStructure => {
                f.write_str("tree topology or value is invalid for the packed node format")
            }
            Self::FeatureDimension { expected, actual } => {
                write!(f, "expected {expected} features, got {actual}")
            }
            Self::OutputLength { expected, actual } => {
                write!(f, "expected output length {expected}, got {actual}")
            }
            Self::NonFiniteTransform { row, column } => {
                write!(
                    f,
                    "transformed value at row {row}, column {column} is not finite"
                )
            }
            Self::OutputShapeOverflow { rows, columns } => {
                write!(f, "output shape overflows usize: {rows} x {columns}")
            }
            Self::FeatureExpansionOverflow { n_features, degree } => {
                write!(
                    f,
                    "a degree-{degree} expansion of {n_features} features is \
                     wider than the supported bound"
                )
            }
            Self::EmptyFeatureExpansion => {
                f.write_str("a feature expansion must produce at least one column")
            }
            Self::UnknownClass { class } => {
                write!(f, "class {class} was not observed during fitting")
            }
            Self::WorkerPanicked => f.write_str("a model training worker panicked"),
            Self::RequiresTwoClasses => f.write_str("classifier requires both classes 0 and 1"),
            Self::InvalidRegularization => {
                f.write_str("inverse regularization strength must be finite and positive")
            }
            Self::InvalidIterationCount => f.write_str("max_iter must be at least one"),
            Self::InvalidTolerance => f.write_str("tolerance must be finite and positive"),
            Self::InvalidLeastSquaresTolerance => {
                f.write_str("least-squares tolerance must be finite and non-negative")
            }
            Self::InvalidRidgeAlpha => f.write_str("ridge alpha must be finite and non-negative"),
            Self::InvalidPenaltyAlpha => {
                f.write_str("penalty alpha must be finite and non-negative")
            }
            Self::InvalidL1Ratio => f.write_str("l1_ratio must be finite and in 0..=1"),
            Self::InvalidLearningRate => {
                f.write_str("boosting learning rate must be finite and positive")
            }
            Self::InvalidBoostingIterationCount => {
                f.write_str("boosting max_iter is outside the supported bound")
            }
            Self::InvalidMaxLeafNodes => {
                f.write_str("boosting max_leaf_nodes is outside the supported bound")
            }
            Self::InvalidBoostingMaxDepth => {
                f.write_str("boosting max_depth is outside the supported bound")
            }
            Self::InvalidMaxBins => f.write_str("boosting max_bins must be in 2..=255"),
            Self::InvalidL2Regularization => {
                f.write_str("boosting L2 regularization must be finite and non-negative")
            }
            Self::BoostingModelTooLarge => {
                f.write_str("requested boosting model exceeds the logical-node limit")
            }
            Self::NumericalOverflow => {
                f.write_str("finite training values overflowed numerical model state")
            }
            Self::NonFinitePrediction { row } => {
                write!(f, "prediction for row {row} is not finite")
            }
            Self::LinearSolveFailed => f.write_str("linear solve failed during optimization"),
            // Neither cause is named: the same variant reports an exhausted
            // budget and a step that could no longer descend, and a sentence
            // naming one of them is false wherever the other applies.
            Self::SolverDidNotConverge { iterations } => write!(
                f,
                "solver stopped after {iterations} iterations without meeting tol"
            ),
            Self::MulticlassOutput { columns } => write!(
                f,
                "operation returns one value per row, but this model produces {columns}"
            ),
            Self::InvalidQuantileRange => {
                f.write_str("quantile range must be two percentiles in 0..=100, lowest first")
            }
            Self::InvalidFeatureRange => {
                f.write_str("feature range must be finite with its minimum below its maximum")
            }
            Self::InvalidThreshold => f.write_str("threshold must be finite"),
            Self::NoInverseFunction => {
                f.write_str("this transformer was not given an inverse function")
            }
            Self::MulticlassSystemTooLarge {
                classes,
                features,
                limit,
            } => write!(
                f,
                "{classes} classes over {features} features exceed the {limit}-parameter \
                 multiclass solver bound"
            ),
        }
    }
}

impl Error for ModelError {}
