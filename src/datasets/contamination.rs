//! The knobs that make a clean problem a realistic one, and the deterministic
//! weight patterns that exercise the weighted-fit surface.
//!
//! # Why contamination is orthogonal to the task
//!
//! Every task family here draws a target a model could in principle recover
//! exactly. Real data does not arrive that way: labels are mis-recorded, a few
//! rows are instrument failures, the noise is fatter than a uniform, its scale
//! tracks a feature, rows are duplicated by a join, columns arrive constant, and
//! two columns are the same measurement in two units.
//!
//! Those are properties of a *dataset*, not of a *task*, so they live in one
//! type that composes with every family rather than as parameters repeated on
//! each of them. A consumer sweeping robustness holds the task fixed and moves
//! the contamination, which is the comparison that isolates what it is
//! measuring.
//!
//! # Where a knob is refused rather than ignored
//!
//! Three of these reshape a target's additive noise and one reshapes labels, so
//! they do not all apply to every family. A knob that silently did nothing would
//! be the worst outcome — a robustness sweep would report a model as robust to a
//! contamination it never received. So
//! [`Recipe::with_contamination`](super::Recipe::with_contamination) refuses the
//! combination instead, with a typed error naming the knob.
//!
//! The four design-shaping knobs — duplicate rows, constant columns, collinear
//! pairs and the per-column scale spread — apply to every recipe, including one
//! carrying no task at all, because they are properties of the design matrix.
//!
//! # Determinism envelope
//!
//! Every knob here is transcendental-free except the per-column scale spread,
//! which is a real power of ten. [`Contamination::portability`] reports that,
//! and [`Recipe::portability`](super::Recipe::portability) combines it with the
//! task's, so a caller reads one answer rather than reasoning about two.

use super::error::{DatasetError, Parameter};
use super::task::scale_columns;
use crate::data::SampleWeights;

/// The multiple that relates a collinear pair's two columns.
///
/// Two rather than one, so a duplicated column and a collinear pair are
/// different findings: an exact copy is what
/// [`Task::IllConditioned`](super::Task::IllConditioned) produces, and a scaled
/// copy is what a unit conversion produces. A model that only detects the first
/// has not detected collinearity.
const COLLINEAR_FACTOR: f32 = 2.0;

/// The value a constant column takes.
const CONSTANT_COLUMN_VALUE: f32 = 1.0;

/// How a generated dataset departs from the clean problem its task describes.
///
/// A `Contamination` is a *request*: its setters are total and do not validate,
/// and [`Recipe::with_contamination`](super::Recipe::with_contamination) is the
/// single boundary where a request becomes a promise. That is deliberate — half
/// of these knobs are only checkable against the design's shape or the task's
/// response, which a free-standing contamination does not know.
///
/// ```
/// use ferricml::datasets::{Contamination, Recipe, Task};
///
/// let task = Task::LinearRegression {
///     informative: 3,
///     coefficient_scale: 1.0,
///     intercept: 0.0,
///     noise_scale: 0.05,
/// };
/// let contamination = Contamination::none()
///     .with_outlier_fraction(0.02)
///     .with_duplicate_rows(0.1)
///     .with_constant_columns(1);
///
/// let recipe = Recipe::seeded(256, 8, 5)?
///     .with_task(task)?
///     .with_contamination(contamination)?;
/// let dataset = recipe.generate();
///
/// // The constant column is the design's last, and it is constant.
/// let design = dataset.features();
/// assert!(design.iter_rows().all(|row| row[7] == 1.0));
/// // A tenth of the rows are copies of the first tenth.
/// assert_eq!(design.row(256 - 25), design.row(0));
/// # Ok::<(), ferricml::datasets::DatasetError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Contamination {
    label_noise: f32,
    outlier_fraction: f32,
    heavy_tail: f32,
    heteroscedastic: f32,
    duplicate_rows: f32,
    constant_columns: usize,
    collinear_pairs: usize,
    feature_scale_spread: f32,
}

/// Every float a contamination carries is refused unless it is finite by
/// [`Recipe::with_contamination`](super::Recipe::with_contamination), and the
/// default is all zeros, so `PartialEq` is reflexive on every value that reaches
/// a recipe. `Eq` is implemented rather than derived for that reason.
impl Eq for Contamination {}

impl Contamination {
    /// The clean dataset: every knob at zero.
    ///
    /// Identical to [`Contamination::default`], and named because
    /// `Contamination::none()` reads as a statement at a call site where
    /// `Default::default()` reads as an omission.
    #[inline]
    pub const fn none() -> Self {
        Self {
            label_noise: 0.0,
            outlier_fraction: 0.0,
            heavy_tail: 0.0,
            heteroscedastic: 0.0,
            duplicate_rows: 0.0,
            constant_columns: 0,
            collinear_pairs: 0,
            feature_scale_spread: 0.0,
        }
    }

    /// Sets the fraction of labels flipped after they are drawn.
    ///
    /// The flip is applied per row against its own draw, so the *realized* rate
    /// is binomial around the requested one rather than exactly it. That is the
    /// honest construction: a label-noise process that flipped exactly `e * n`
    /// labels would be a different process, and a classifier tuned against it
    /// would be tuned against an artifact.
    #[inline]
    #[must_use]
    pub const fn with_label_noise(mut self, fraction: f32) -> Self {
        self.label_noise = fraction;
        self
    }

    /// Sets the fraction of rows whose target is displaced by an outlier.
    #[inline]
    #[must_use]
    pub const fn with_outlier_fraction(mut self, fraction: f32) -> Self {
        self.outlier_fraction = fraction;
        self
    }

    /// Sets the fraction of rows whose noise is drawn from the heavy-tailed
    /// component.
    ///
    /// The heavy-tailed draw is the uniform one divided by `1 - u`, a truncated
    /// Pareto whose largest magnitude is `2^24` times the base noise — finite by
    /// construction, because the underlying map has 24 bits and cannot reach
    /// one.
    #[inline]
    #[must_use]
    pub const fn with_heavy_tail(mut self, fraction: f32) -> Self {
        self.heavy_tail = fraction;
        self
    }

    /// Sets how strongly the noise scale tracks the first feature's magnitude.
    ///
    /// The noise is multiplied by `1 + heteroscedastic * |x_0|`, so zero leaves
    /// the noise homoscedastic and one doubles it at the design's extremes.
    #[inline]
    #[must_use]
    pub const fn with_heteroscedastic(mut self, strength: f32) -> Self {
        self.heteroscedastic = strength;
        self
    }

    /// Sets the fraction of rows replaced by exact copies of earlier rows.
    ///
    /// Deterministic rather than drawn: the last `floor(fraction * rows)` rows
    /// become copies of the first that many, taken from a snapshot so the
    /// mapping is well defined even where the two ranges overlap. The realized
    /// fraction is therefore exactly `floor(fraction * rows) / rows`, which is
    /// what lets a test assert it rather than bound it.
    #[inline]
    #[must_use]
    pub const fn with_duplicate_rows(mut self, fraction: f32) -> Self {
        self.duplicate_rows = fraction;
        self
    }

    /// Sets how many trailing columns are replaced by a constant.
    #[inline]
    #[must_use]
    pub const fn with_constant_columns(mut self, columns: usize) -> Self {
        self.constant_columns = columns;
        self
    }

    /// Sets how many column pairs are made exactly collinear.
    ///
    /// Pair `i` overwrites the column just below the constant tail with twice
    /// the design's column `i`, so the two are proportional and neither is a
    /// copy of the other.
    #[inline]
    #[must_use]
    pub const fn with_collinear_pairs(mut self, pairs: usize) -> Self {
        self.collinear_pairs = pairs;
        self
    }

    /// Sets the decades between the largest and smallest per-column magnitude.
    ///
    /// This is the knob a scaler is measured against: at three decades the
    /// design's last column is a thousandth of its first, and any estimator
    /// whose penalty or step size is not scale-free notices.
    #[inline]
    #[must_use]
    pub const fn with_feature_scale_spread(mut self, decades: f32) -> Self {
        self.feature_scale_spread = decades;
        self
    }

    /// Returns the requested label-noise fraction.
    #[inline]
    pub const fn label_noise(&self) -> f32 {
        self.label_noise
    }

    /// Returns the requested outlier fraction.
    #[inline]
    pub const fn outlier_fraction(&self) -> f32 {
        self.outlier_fraction
    }

    /// Returns the requested heavy-tail fraction.
    #[inline]
    pub const fn heavy_tail(&self) -> f32 {
        self.heavy_tail
    }

    /// Returns the requested heteroscedasticity strength.
    #[inline]
    pub const fn heteroscedastic(&self) -> f32 {
        self.heteroscedastic
    }

    /// Returns the requested duplicate-row fraction.
    #[inline]
    pub const fn duplicate_rows(&self) -> f32 {
        self.duplicate_rows
    }

    /// Returns the requested number of constant columns.
    #[inline]
    pub const fn constant_columns(&self) -> usize {
        self.constant_columns
    }

    /// Returns the requested number of collinear pairs.
    #[inline]
    pub const fn collinear_pairs(&self) -> usize {
        self.collinear_pairs
    }

    /// Returns the requested per-column scale spread, in decades.
    #[inline]
    pub const fn feature_scale_spread(&self) -> f32 {
        self.feature_scale_spread
    }

    /// Whether this contamination reshapes a target's additive noise.
    pub(super) fn shapes_noise(&self) -> Option<Parameter> {
        if self.heavy_tail != 0.0 {
            Some(Parameter::HeavyTail)
        } else if self.heteroscedastic != 0.0 {
            Some(Parameter::Heteroscedastic)
        } else if self.outlier_fraction != 0.0 {
            Some(Parameter::OutlierFraction)
        } else {
            None
        }
    }

    /// This contamination's determinism envelope.
    ///
    /// Only the per-column scale spread costs anything: it is a real power of
    /// ten, and every other knob is an integer comparison, a copy or a
    /// multiplication.
    pub(super) fn portability(&self) -> super::Portability {
        if self.feature_scale_spread == 0.0 {
            super::Portability::BitExact
        } else {
            super::Portability::PerRunner
        }
    }

    /// Checks every knob against its range and against the design's shape.
    pub(super) fn validate(&self, columns: usize) -> Result<(), DatasetError> {
        unit_interval(self.label_noise, Parameter::LabelNoise, 0.5)?;
        half_open_unit(self.outlier_fraction, Parameter::OutlierFraction)?;
        unit_interval(self.heavy_tail, Parameter::HeavyTail, 1.0)?;
        at_least_zero(self.heteroscedastic, Parameter::Heteroscedastic)?;
        half_open_unit(self.duplicate_rows, Parameter::DuplicateRows)?;
        at_least_zero(self.feature_scale_spread, Parameter::FeatureScaleSpread)?;
        if self.constant_columns >= columns {
            return Err(DatasetError::ConstantColumnsLeaveNoSignal {
                constant_columns: self.constant_columns,
                columns,
            });
        }
        let available = columns - self.constant_columns;
        if self.collinear_pairs * 2 > available {
            return Err(DatasetError::CollinearPairsExceedDesign {
                pairs: self.collinear_pairs,
                available,
            });
        }
        Ok(())
    }

    /// Reshapes a generated design, in the fixed order the knobs compose in.
    ///
    /// Scale spread first, so a collinear pair is proportional to the *scaled*
    /// column and is therefore exactly collinear; then the pairs; then the
    /// constant tail, which is disjoint from both; then the duplicated rows,
    /// last, so a duplicate is an exact copy of a finished row rather than of an
    /// intermediate one.
    pub(super) fn shape_design(&self, rows: usize, columns: usize, values: &mut [f32]) {
        scale_columns(rows, columns, self.feature_scale_spread, values);

        let tail = columns - self.constant_columns;
        for pair in 0..self.collinear_pairs {
            let target = tail - 1 - pair;
            for row in 0..rows {
                values[row * columns + target] = COLLINEAR_FACTOR * values[row * columns + pair];
            }
        }
        for column in tail..columns {
            for row in 0..rows {
                values[row * columns + column] = CONSTANT_COLUMN_VALUE;
            }
        }

        let duplicated = self.duplicated_rows(rows);
        if duplicated > 0 {
            // A snapshot of the source rows, so the mapping is `new[rows - d +
            // i] = old[i]` even where the two ranges overlap. Writing in place
            // would make a late target read a row an earlier write had already
            // replaced, and the realized duplication would stop being the
            // requested one.
            let sources = values[..duplicated * columns].to_vec();
            values[(rows - duplicated) * columns..].copy_from_slice(&sources);
        }
    }

    /// How many rows this contamination duplicates at a given height.
    pub(super) fn duplicated_rows(&self, rows: usize) -> usize {
        if self.duplicate_rows == 0.0 {
            return 0;
        }
        ((self.duplicate_rows * rows as f32) as usize).min(rows)
    }
}

/// A deterministic per-row weight pattern.
///
/// These exist so the crate's `fit_weighted` surface can be exercised against
/// data whose weights are a known function rather than a second dataset a
/// caller has to invent. Each pattern is a pure function of the row index, or of
/// the drawn labels, so it carries no randomness of its own and no portability
/// cost.
///
/// It is `#[non_exhaustive]` because a new pattern must not be a breaking
/// change.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum WeightPattern {
    /// Every row weighs exactly one.
    ///
    /// Not the same as no weights at all: an estimator that normalizes by total
    /// weight sees the same fit either way, and one that branches on the
    /// presence of weights does not. Having the pattern makes that difference
    /// testable.
    Uniform,
    /// Weights ramp linearly from `low` at the first row to `high` at the last.
    Ramp {
        /// The first row's weight.
        low: f32,
        /// The last row's weight.
        high: f32,
    },
    /// Even rows weigh `first` and odd rows weigh `second`.
    Alternating {
        /// Weight of the even rows.
        first: f32,
        /// Weight of the odd rows.
        second: f32,
    },
    /// Each observed class carries the same total weight.
    ///
    /// A row of a class with `k` members, out of `c` observed classes, weighs
    /// `rows / (c * k)`, so every class totals `rows / c` whatever its
    /// prevalence. This is what turns
    /// [`Task::LinearBinary`](super::Task::LinearBinary)'s controlled prevalence
    /// and [`Task::Multiclass`](super::Task::Multiclass)'s controlled balance
    /// into controlled *imbalance* experiments: the same data, fitted with and
    /// without the correction.
    ///
    /// The class count is the one *observed* rather than the one requested. A
    /// class the draw never produced has no rows to weigh, and inventing a share
    /// for it would silently down-weight every class that does exist.
    ClassBalanced,
}

/// The float weights a pattern carries are refused unless finite by
/// [`Recipe::with_weights`](super::Recipe::with_weights), so `PartialEq` is
/// reflexive on every value that reaches a recipe.
impl Eq for WeightPattern {}

impl WeightPattern {
    /// Whether this pattern needs a task that draws labels.
    pub(super) const fn needs_labels(self) -> bool {
        matches!(self, Self::ClassBalanced)
    }

    /// Checks the pattern's own parameters.
    pub(super) fn validate(self) -> Result<(), DatasetError> {
        match self {
            Self::Uniform | Self::ClassBalanced => Ok(()),
            Self::Ramp { low, high } => {
                weight(low, Parameter::WeightLow)?;
                weight(high, Parameter::WeightHigh)?;
                positive_total(low + high, Parameter::WeightHigh)
            }
            Self::Alternating { first, second } => {
                weight(first, Parameter::WeightFirst)?;
                weight(second, Parameter::WeightSecond)?;
                positive_total(first + second, Parameter::WeightSecond)
            }
        }
    }

    /// Builds the weights for a generated dataset.
    ///
    /// `labels` is `Some` exactly when the task drew them, which is what
    /// [`WeightPattern::needs_labels`] made the recipe check; the `unwrap_or`
    /// below is therefore unreachable for any recipe that exists, and gives a
    /// uniform weight rather than a panic if it ever were reached.
    pub(super) fn weights(self, rows: usize, labels: Option<&[u8]>) -> SampleWeights {
        let values: Vec<f32> = match self {
            Self::Uniform => vec![1.0; rows],
            Self::Ramp { low, high } => (0..rows)
                .map(|row| {
                    if rows == 1 {
                        low
                    } else {
                        low + (high - low) * (row as f32 / (rows - 1) as f32)
                    }
                })
                .collect(),
            Self::Alternating { first, second } => (0..rows)
                .map(|row| if row % 2 == 0 { first } else { second })
                .collect(),
            Self::ClassBalanced => {
                let labels = labels.unwrap_or(&[]);
                // One pass over a 256-entry table rather than a map: labels are
                // `u8`, so the whole class space fits in a fixed array and the
                // count is linear in the rows regardless of how many classes
                // there are.
                let mut members = [0_usize; 256];
                for &label in labels {
                    members[label as usize] += 1;
                }
                let observed = members.iter().filter(|&&count| count > 0).count();
                labels
                    .iter()
                    .map(|&label| {
                        let count = members[label as usize];
                        if count == 0 || observed == 0 {
                            1.0
                        } else {
                            rows as f32 / (observed as f32 * count as f32)
                        }
                    })
                    .collect()
            }
        };
        let values = if values.len() == rows {
            values
        } else {
            vec![1.0; rows]
        };
        SampleWeights::new(values).expect("a validated weight pattern is finite and positive")
    }
}

fn at_least_zero(value: f32, parameter: Parameter) -> Result<(), DatasetError> {
    if !value.is_finite() {
        return Err(DatasetError::NonFiniteParameter { parameter });
    }
    if value < 0.0 {
        return Err(DatasetError::ParameterOutOfRange { parameter });
    }
    Ok(())
}

fn unit_interval(value: f32, parameter: Parameter, upper: f32) -> Result<(), DatasetError> {
    at_least_zero(value, parameter)?;
    if value > upper {
        return Err(DatasetError::ParameterOutOfRange { parameter });
    }
    Ok(())
}

fn half_open_unit(value: f32, parameter: Parameter) -> Result<(), DatasetError> {
    at_least_zero(value, parameter)?;
    if value >= 1.0 {
        return Err(DatasetError::ParameterOutOfRange { parameter });
    }
    Ok(())
}

fn weight(value: f32, parameter: Parameter) -> Result<(), DatasetError> {
    at_least_zero(value, parameter)
}

fn positive_total(total: f32, parameter: Parameter) -> Result<(), DatasetError> {
    if total > 0.0 {
        Ok(())
    } else {
        Err(DatasetError::ParameterOutOfRange { parameter })
    }
}
