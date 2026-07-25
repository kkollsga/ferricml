use std::error::Error;
use std::fmt;

/// Errors produced while fitting or using a FerricML model.
///
/// Data-container construction has its own [`crate::data::DataError`] because
/// it occurs before an estimator is involved. All estimators use this error
/// surface once fitting or prediction begins.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelError {
    /// Training data has no samples or features.
    EmptyData,
    /// The target vector is empty.
    EmptyTargets,
    /// The target count does not match the sample count.
    TargetLength { rows: usize, targets: usize },
    /// The sample-weight count does not match the sample count.
    SampleWeightLength { rows: usize, weights: usize },
    /// A feature is NaN or infinite.
    NonFiniteFeature { row: usize, column: usize },
    /// A binary target is not zero or one.
    InvalidBinaryTarget { index: usize, value: u8 },
    /// A regression target is NaN or infinite.
    NonFiniteTarget { index: usize },
    /// `n_estimators` is zero.
    InvalidEstimatorCount,
    /// `max_depth` is zero.
    InvalidMaxDepth,
    /// `min_samples_split` is less than two.
    InvalidMinSamplesSplit,
    /// `min_samples_leaf` is zero.
    InvalidMinSamplesLeaf,
    /// A fixed `max_features` count exceeds the fitted feature count.
    InvalidMaxFeatures { requested: usize, available: usize },
    /// A fixed `n_jobs` count is zero.
    InvalidJobCount,
    /// The sample count exceeds the internal bootstrap-counter format.
    TooManyRows,
    /// The feature count exceeds the internal tree format.
    TooManyFeatures,
    /// A fitted tree exceeds the internal node-index format.
    TreeTooLarge,
    /// Input feature width differs from the fitted width.
    FeatureDimension { expected: usize, actual: usize },
    /// A caller-provided output buffer has the wrong length.
    OutputLength { expected: usize, actual: usize },
    /// Scaling a finite feature produced a non-finite `f32` output.
    NonFiniteTransform { row: usize, column: usize },
    /// The requested output matrix dimensions cannot be represented.
    OutputShapeOverflow { rows: usize, columns: usize },
    /// A requested class was not observed during fitting.
    UnknownClass { class: u8 },
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
    NonFinitePrediction { row: usize },
    /// The linear solver encountered a non-positive-definite system, or an
    /// iterative solver could not make progress from its current iterate.
    LinearSolveFailed,
    /// An iterative solver exhausted `max_iter` without meeting its tolerance.
    ///
    /// Reported instead of returning the last iterate, because an unconverged
    /// model is indistinguishable from a converged one once it is fitted.
    SolverDidNotConverge {
        /// Iterations performed before the budget ran out.
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
    /// A decision threshold is not finite.
    InvalidThreshold,
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
            Self::EmptyTargets => f.write_str("targets must not be empty"),
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
            Self::InvalidBinaryTarget { index, value } => write!(
                f,
                "binary target at index {index} is {value}, expected 0 or 1"
            ),
            Self::NonFiniteTarget { index } => {
                write!(f, "regression target at index {index} is not finite")
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
            Self::SolverDidNotConverge { iterations } => write!(
                f,
                "solver reached max_iter after {iterations} iterations without converging"
            ),
            Self::MulticlassOutput { columns } => write!(
                f,
                "operation returns one value per row, but this model produces {columns}"
            ),
            Self::InvalidQuantileRange => {
                f.write_str("quantile range must be two percentiles in 0..=100, lowest first")
            }
            Self::InvalidThreshold => f.write_str("threshold must be finite"),
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
