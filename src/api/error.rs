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
    /// The linear solver encountered a non-positive-definite system.
    LinearSolveFailed,
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
            Self::LinearSolveFailed => f.write_str("linear solve failed during optimization"),
        }
    }
}

impl Error for ModelError {}
