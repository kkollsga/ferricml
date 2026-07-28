//! The families whose structure is the point: many classes, clusters, groups,
//! time order, and ranked pairs.
//!
//! # Why these live apart from `task.rs`
//!
//! `task.rs` owns families whose output is one number per row. Everything here
//! produces something *else* as well — a row of class probabilities, a cluster
//! assignment, a group label, a timestamp, a list of preference pairs — and each
//! of those is a second array that has to agree with the design row for row.
//! Keeping them together would bury four different truths in one match arm; the
//! [`Task`](super::Task) enum still names them, and this module owns what they
//! draw.
//!
//! # What each family is right about
//!
//! * [`Task::Multiclass`](super::Task::Multiclass) knows the full Bayes
//!   probability *row* `P(y = k | x)`, not just a scalar, which is what a
//!   multiclass calibration or log-loss measurement has to be compared against.
//! * [`Task::Clustered`](super::Task::Clustered) knows the assignment and the
//!   centres, and has no target at all — which is the reason
//!   [`Dataset::target`](super::Dataset::target) is an `Option`.
//! * [`Task::TimeOrdered`](super::Task::TimeOrdered) knows both ends of its
//!   drifting coefficient vector and the time of every row, so "the drift you
//!   asked for is the drift you got" is a measurement rather than a hope.
//! * [`Task::Ranking`](super::Task::Ranking) knows the utility behind every
//!   relevance grade, and emits its pairs already in the crate's own
//!   [`PairwiseObservation`] vocabulary.
//!
//! # Determinism envelopes
//!
//! Three of the four families are transcendental-free and are pinned by literal
//! values in `structural_tests.rs`. Multiclass is not: a softmax is a sum of
//! exponentials, so it carries
//! [`Portability::PerRunner`](super::Portability::PerRunner) and is held to
//! properties instead. [`GroupPattern`] is deliberately transcendental-free —
//! its unbalanced sizes interpolate linearly rather than geometrically, so
//! attaching groups to a bit-exact recipe cannot weaken its envelope.

use super::dataset::{Target, Truth};
use super::error::{DatasetError, Parameter};
use super::task::{
    STREAM_COEFFICIENTS, STREAM_LABELS, check_at_least_zero, check_informative, check_positive,
    dot, draw_coefficients, signed_draw, stream, unit_draw,
};
use crate::data::{ClassTargets, DenseMatrix};
use crate::numeric::sum_in_order;
use crate::ranking::{PairIndex, PairOutcome, PairwiseObservation};

/// The largest class count a `u8` label can express.
pub(super) const MAX_CLASSES: usize = 256;

/// Multiplicative updates applied to the per-class softmax offsets.
///
/// The update is `m_k *= π_k / mean_k`, the standard prior-correction fixed
/// point for a softmax, and it converges geometrically. Sixty-four of them leave
/// the realized mean probability of every class within `4.5e-10` of the request
/// at every case `structural_tests.rs` sweeps — six orders of magnitude below
/// the binomial spread of the labels those probabilities then produce, so the
/// solver's residual is not what any balance assertion is measuring.
const OFFSET_ITERATIONS: usize = 64;

/// Normalizes a squared design distance to a mean of one per column.
///
/// A design value and a blob centre are both uniform on `[-1, 1)` with variance
/// `1/3`, so `E‖x − c‖²` is `2/3` per column and `1.5 * 2/3 = 1`. Dividing by
/// the column count then makes `separation` mean the same thing on a four-column
/// design and on a forty-column one, instead of silently becoming a
/// hard-boundary request as the design widens.
const BLOB_DISTANCE_NORMALIZER: f64 = 1.5;

/// How a multiclass family distributes its classes.
///
/// It is `#[non_exhaustive]` because a new balance shape must not be a breaking
/// change.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ClassBalance {
    /// Every class carries the same marginal probability.
    Balanced,
    /// Class marginals decay geometrically, from `ratio` times the rarest class
    /// down to the rarest.
    ///
    /// Geometric rather than linear because that is what class imbalance looks
    /// like in practice — a long tail rather than an even slope — and because it
    /// keeps every class non-empty at any ratio, which a linear decay through
    /// zero would not.
    Imbalanced {
        /// Ratio between the most and least common class marginal. At or above
        /// `1`; exactly `1` is [`ClassBalance::Balanced`].
        ratio: f32,
    },
}

/// Every float here is refused unless finite by
/// [`Recipe::with_task`](super::Recipe::with_task), so `PartialEq` is reflexive
/// on every value that reaches a recipe.
impl Eq for ClassBalance {}

impl ClassBalance {
    /// The requested marginal probability of each class.
    ///
    /// Accumulated and normalized in `f64` so the returned vector sums to one to
    /// within a rounding, which is what the offset solver below assumes.
    fn prevalences(self, classes: usize) -> Vec<f64> {
        let weights: Vec<f64> = match self {
            Self::Balanced => vec![1.0; classes],
            Self::Imbalanced { ratio } => {
                let ratio = f64::from(ratio);
                let last = (classes - 1) as f64;
                (0..classes)
                    .map(|class| ratio.powf(-(class as f64) / last))
                    .collect()
            }
        };
        let total = sum_in_order(weights.iter().copied());
        weights.into_iter().map(|weight| weight / total).collect()
    }

    /// Checks this balance's own parameters.
    fn validate(self) -> Result<(), DatasetError> {
        match self {
            Self::Balanced => Ok(()),
            Self::Imbalanced { ratio } => at_least_one(ratio, Parameter::BalanceRatio),
        }
    }
}

/// How a multiclass family arranges its classes in feature space.
///
/// It is `#[non_exhaustive]` because a new geometry must not be a breaking
/// change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClassGeometry {
    /// Each class is a blob around its own centre, and a row's score for a class
    /// is minus its squared distance from that centre.
    ///
    /// The confusable pairs are whichever centres happen to land near each
    /// other, so the confusion structure is unstructured — which is the right
    /// null case to compare a hierarchy against.
    Blob,
    /// Classes are the leaves of a balanced binary tree of linear splits.
    ///
    /// Level `l` contributes `2^-l` times its own projection, so the top split
    /// separates two super-classes twice as strongly as the split below it
    /// separates their children. The confusion structure is therefore *nested*:
    /// a model confuses siblings far more often than cousins, which is the
    /// property a hierarchy-aware metric exists to detect and a blob geometry
    /// cannot produce.
    Hierarchical,
}

/// A deterministic assignment of rows to groups.
///
/// Groups mark rows that are not independent — the same patient, session or
/// document — and exist here because
/// [`GroupKFold`](crate::model_selection::GroupKFold) and
/// [`GroupShuffleSplit`](crate::model_selection::GroupShuffleSplit) take
/// `&[u64]`, and a generated dataset should feed them without an adapter.
///
/// Every pattern **partitions** the rows: each row carries exactly one label,
/// the labels are exactly `0..groups`, and no group is empty. That is asserted
/// rather than described, in `structural_tests.rs`.
///
/// It is `#[non_exhaustive]` because a new pattern must not be a breaking
/// change.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum GroupPattern {
    /// Rows are dealt to groups in turn, so group sizes differ by at most one
    /// and consecutive rows are never in the same group.
    ///
    /// This is the adversarial case for a splitter that leaks: a naive
    /// contiguous fold would put most of every group on both sides.
    RoundRobin {
        /// Number of groups.
        groups: usize,
    },
    /// Groups are contiguous blocks of rows, as equal as the row count allows.
    ///
    /// This is what a session or a document looks like in a table that arrived
    /// sorted, and it is the case where a splitter that ignores groups
    /// accidentally gets the right answer — so the two patterns together
    /// separate a splitter that respects groups from one that got lucky.
    Contiguous {
        /// Number of groups.
        groups: usize,
    },
    /// Contiguous groups whose sizes fall linearly, the largest `ratio` times
    /// the smallest.
    ///
    /// Linear rather than geometric, deliberately: a geometric decay is a real
    /// power, and this type is the one structural knob that must not weaken a
    /// recipe's determinism envelope. Interpolating the *weights* linearly
    /// keeps every size an exact integer computation while still realizing the
    /// requested extreme ratio, which is what the knob names.
    ///
    /// Unequal group sizes are what makes
    /// [`GroupKFold`](crate::model_selection::GroupKFold)'s balancing
    /// observable: with equal groups every assignment gives equal folds.
    Unbalanced {
        /// Number of groups.
        groups: usize,
        /// Ratio between the largest and smallest group. At or above `1`.
        ratio: f32,
    },
}

/// Every float here is refused unless finite by
/// [`Recipe::with_groups`](super::Recipe::with_groups), so `PartialEq` is
/// reflexive on every value that reaches a recipe.
impl Eq for GroupPattern {}

impl GroupPattern {
    /// How many groups this pattern asks for.
    pub(super) const fn groups(self) -> usize {
        match self {
            Self::RoundRobin { groups } | Self::Contiguous { groups } => groups,
            Self::Unbalanced { groups, .. } => groups,
        }
    }

    /// Checks the pattern against the design's row count.
    pub(super) fn validate(self, rows: usize) -> Result<(), DatasetError> {
        let groups = self.groups();
        if groups == 0 {
            return Err(DatasetError::ZeroGroups);
        }
        if groups > rows {
            return Err(DatasetError::GroupsExceedRows { groups, rows });
        }
        if let Self::Unbalanced { ratio, .. } = self {
            at_least_one(ratio, Parameter::GroupSizeRatio)?;
        }
        Ok(())
    }

    /// The group label of every row.
    pub(super) fn labels(self, rows: usize) -> Vec<u64> {
        match self {
            Self::RoundRobin { groups } => (0..rows).map(|row| (row % groups) as u64).collect(),
            // `row * groups / rows` is below `groups` for every `row` below
            // `rows`, and is non-decreasing, so the blocks are contiguous and
            // exactly `groups` of them are non-empty once `groups <= rows`.
            Self::Contiguous { groups } => {
                (0..rows).map(|row| (row * groups / rows) as u64).collect()
            }
            Self::Unbalanced { groups, ratio } => {
                let sizes = unbalanced_sizes(rows, groups, ratio);
                let mut labels = Vec::with_capacity(rows);
                for (group, size) in sizes.into_iter().enumerate() {
                    labels.extend(std::iter::repeat_n(group as u64, size));
                }
                labels
            }
        }
    }
}

/// Group sizes falling linearly from `ratio` times the smallest to the smallest,
/// summing to exactly `rows` with no empty group.
///
/// Largest-remainder apportionment: floor each ideal size, then hand the leftover
/// rows to the largest fractional parts, breaking ties by the lower index. The
/// floor is raised to one first, which can overshoot `rows`; the repair loop then
/// takes rows back from the largest group. Both loops terminate because `groups`
/// is at most `rows`, so a feasible assignment exists.
fn unbalanced_sizes(rows: usize, groups: usize, ratio: f32) -> Vec<usize> {
    if groups == 1 {
        return vec![rows];
    }
    let ratio = f64::from(ratio);
    let last = (groups - 1) as f64;
    let weights: Vec<f64> = (0..groups)
        .map(|group| 1.0 + (ratio - 1.0) * (last - group as f64) / last)
        .collect();
    let total = sum_in_order(weights.iter().copied());
    let ideal: Vec<f64> = weights
        .iter()
        .map(|weight| weight / total * rows as f64)
        .collect();
    let mut sizes: Vec<usize> = ideal.iter().map(|&value| (value as usize).max(1)).collect();

    let mut assigned: usize = sizes.iter().sum();
    while assigned < rows {
        // The group whose ideal size the floor shortchanged most.
        let target = (0..groups)
            .max_by(|&left, &right| {
                let deficit = |index: usize| ideal[index] - sizes[index] as f64;
                deficit(left)
                    .partial_cmp(&deficit(right))
                    .expect("a finite ideal size")
                    .then(right.cmp(&left))
            })
            .expect("at least one group");
        sizes[target] += 1;
        assigned += 1;
    }
    while assigned > rows {
        let target = (0..groups)
            .filter(|&index| sizes[index] > 1)
            .max_by_key(|&index| (sizes[index], groups - index))
            .expect("groups <= rows leaves a group above one");
        sizes[target] -= 1;
        assigned -= 1;
    }
    sizes
}

/// Checks a multiclass request against the design's shape.
pub(super) fn validate_multiclass(
    classes: usize,
    balance: ClassBalance,
    separation: f32,
    columns: usize,
) -> Result<(), DatasetError> {
    if classes < 2 {
        return Err(DatasetError::TooFewClasses { classes });
    }
    if classes > MAX_CLASSES {
        return Err(DatasetError::TooManyClasses {
            classes,
            limit: MAX_CLASSES,
        });
    }
    // Every geometry reads the whole design: a blob centre has one coordinate
    // per column and a split projects across all of them.
    check_informative(columns, columns)?;
    check_positive(separation, Parameter::Separation)?;
    balance.validate()
}

/// Checks a clustered request against the design's shape.
pub(super) fn validate_clustered(
    blobs: usize,
    spread: f32,
    rows: usize,
) -> Result<(), DatasetError> {
    if blobs == 0 {
        return Err(DatasetError::ZeroBlobs);
    }
    if blobs > rows {
        return Err(DatasetError::BlobsExceedRows { blobs, rows });
    }
    check_at_least_zero(spread, Parameter::Spread)
}

/// Checks a ranking request against the design's shape.
pub(super) fn validate_ranking(
    queries: usize,
    docs_per_query: usize,
    grades: usize,
    informative: usize,
    coefficient_scale: f32,
    rows: usize,
    columns: usize,
) -> Result<(), DatasetError> {
    if docs_per_query < 2 {
        return Err(DatasetError::TooFewDocumentsPerQuery { docs_per_query });
    }
    if grades < 2 {
        return Err(DatasetError::TooFewGrades { grades });
    }
    if queries.checked_mul(docs_per_query) != Some(rows) {
        return Err(DatasetError::RankingShapeMismatch {
            rows,
            queries,
            docs_per_query,
        });
    }
    check_informative(informative, columns)?;
    check_positive(coefficient_scale, Parameter::CoefficientScale)
}

/// The blob centres a clustered design is built around.
///
/// Drawn from the coefficient stream so widening the design does not move a
/// centre's leading coordinates, and shared by the reshaping pass and the truth
/// pass rather than recomputed differently in each: two derivations of one
/// number is exactly how a design and its recorded truth drift apart.
pub(super) fn cluster_centres(blobs: usize, columns: usize, digest: &[u8; 32]) -> Vec<f32> {
    let mut rng = stream(digest, STREAM_COEFFICIENTS);
    (0..blobs * columns)
        .map(|_| signed_draw(&mut rng))
        .collect()
}

/// The cluster every row belongs to.
///
/// `row % blobs` rather than a drawn assignment: the clusters are then exactly
/// as equal as the row count allows, without a second stream and without a
/// balance parameter that would duplicate [`GroupPattern`]'s. Interleaving also
/// makes a clusterer that keyed on row order visibly wrong.
pub(super) fn cluster_assignments(rows: usize, blobs: usize) -> Vec<usize> {
    (0..rows).map(|row| row % blobs).collect()
}

/// Moves every row onto its cluster's centre, keeping `spread` of its own
/// scatter.
pub(super) fn shape_clustered(
    rows: usize,
    columns: usize,
    blobs: usize,
    spread: f32,
    values: &mut [f32],
    digest: &[u8; 32],
) {
    let centres = cluster_centres(blobs, columns, digest);
    for row in 0..rows {
        let centre = (row % blobs) * columns;
        for column in 0..columns {
            let cell = row * columns + column;
            values[cell] = centres[centre + column] + spread * values[cell];
        }
    }
}

/// Records a clustered design's truth. The family draws no target at all.
pub(super) fn draw_clustered(
    design: &DenseMatrix,
    blobs: usize,
    digest: &[u8; 32],
) -> (Option<Target>, Truth) {
    (
        None,
        Truth::ClusterAssignment {
            assignments: cluster_assignments(design.rows(), blobs),
            centres: cluster_centres(blobs, design.columns(), digest),
            blobs,
        },
    )
}

/// The time of every row, on `[0, 1]`.
///
/// A division of one exactly-representable integer by another, so the sequence
/// is non-decreasing on every target and strictly increasing for any row count
/// below `2^24`. Row order *is* time order, which is what lets
/// [`TimeSeriesSplit`](crate::model_selection::TimeSeriesSplit) — which takes a
/// sample count and nothing else — be correct on this data unadapted.
pub(super) fn row_times(rows: usize) -> Vec<f32> {
    if rows == 1 {
        return vec![0.0];
    }
    let last = (rows - 1) as f32;
    (0..rows).map(|row| row as f32 / last).collect()
}

/// Draws a time-ordered family's conditional mean and records both ends of its
/// drift.
///
/// Returns the conditional mean and the truth's three vectors. The caller adds
/// the family's noise, so the noise-shaping contamination knobs reach this
/// family exactly as they reach the stationary linear one.
pub(super) fn drifting_predictor(
    design: &DenseMatrix,
    informative: usize,
    coefficient_scale: f32,
    drift: f32,
    intercept: f32,
    digest: &[u8; 32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let columns = design.columns();
    let mut rng = stream(digest, STREAM_COEFFICIENTS);
    // Both vectors consume one draw per column, informative or not, so widening
    // the informative prefix leaves the columns that were already informative
    // where they were — the discipline `draw_coefficients` follows for the same
    // reason. Start and delta are drawn in two separate passes rather than
    // interleaved, so a design with more columns does not reshuffle the drift.
    let start: Vec<f32> = (0..columns)
        .map(|column| {
            let draw = signed_draw(&mut rng);
            if column < informative {
                coefficient_scale * draw
            } else {
                0.0
            }
        })
        .collect();
    let end: Vec<f32> = (0..columns)
        .map(|column| {
            let draw = signed_draw(&mut rng);
            if column < informative {
                start[column] + drift * draw
            } else {
                0.0
            }
        })
        .collect();

    let times = row_times(design.rows());
    let mean: Vec<f32> = design
        .iter_rows()
        .zip(&times)
        .map(|(row, &time)| {
            let time = f64::from(time);
            // `β(t) = start + t (end − start)`, formed per column in `f64` and
            // narrowed once with the dot product, so the recorded conditional
            // mean is not itself a victim of the rounding it is meant to measure.
            let predictor = sum_in_order(row.iter().zip(&start).zip(&end).map(
                |((&value, &start), &end)| {
                    f64::from(value) * (f64::from(start) + time * f64::from(end - start))
                },
            ));
            (predictor + f64::from(intercept)) as f32
        })
        .collect();
    (start, end, mean, times)
}

/// Draws a ranking family's grades, utilities and within-query pairs.
pub(super) fn draw_ranking(
    design: &DenseMatrix,
    queries: usize,
    docs_per_query: usize,
    grades: usize,
    informative: usize,
    coefficient_scale: f32,
    digest: &[u8; 32],
) -> (Option<Target>, Truth, Vec<u64>, Vec<PairwiseObservation>) {
    let coefficients = draw_coefficients(design.columns(), informative, coefficient_scale, digest);
    let utilities: Vec<f32> = design
        .iter_rows()
        .map(|row| dot(row, &coefficients) as f32)
        .collect();

    let mut labels = vec![0_u8; design.rows()];
    let mut pairs = Vec::with_capacity(queries * docs_per_query * (docs_per_query - 1) / 2);
    let mut order: Vec<usize> = Vec::with_capacity(docs_per_query);
    for query in 0..queries {
        let base = query * docs_per_query;
        order.clear();
        order.extend(base..base + docs_per_query);
        // Best first. `total_cmp` rather than `partial_cmp` because it is a
        // total order on every `f32` including the ones this cannot produce, and
        // the index tie-break keeps two equal utilities in a fixed order rather
        // than in whichever order the sort happened to leave them.
        order.sort_by(|&left, &right| {
            utilities[right]
                .total_cmp(&utilities[left])
                .then(left.cmp(&right))
        });
        for (rank, &document) in order.iter().enumerate() {
            // Rank `d - 1` gives `((d - 1) * grades) / d <= grades - 1`, so the
            // grade never goes negative and the best document always gets the
            // top grade. Ties appear whenever there are fewer grades than
            // documents, which is what a real relevance judgement looks like.
            labels[document] = (grades - 1 - rank * grades / docs_per_query) as u8;
        }
        for left in 0..docs_per_query {
            for right in left + 1..docs_per_query {
                let (left, right) = (base + left, base + right);
                let outcome = match labels[left].cmp(&labels[right]) {
                    std::cmp::Ordering::Greater => PairOutcome::LeftPreferred,
                    std::cmp::Ordering::Less => PairOutcome::RightPreferred,
                    std::cmp::Ordering::Equal => PairOutcome::Tie,
                };
                let pair = PairIndex::new(left, right).expect("two distinct documents");
                pairs.push(
                    PairwiseObservation::new(pair, outcome, 1.0)
                        .expect("a unit weight is finite and non-negative"),
                );
            }
        }
    }

    let groups = (0..design.rows())
        .map(|row| (row / docs_per_query) as u64)
        .collect();
    (
        Some(Target::Class(
            ClassTargets::new(labels).expect("a query block assigns every document a grade"),
        )),
        Truth::RankingUtility {
            coefficients,
            utilities,
            grades,
        },
        groups,
        pairs,
    )
}

/// Draws a multiclass family's labels and records the whole Bayes probability
/// row of every observation.
///
/// The recorded probabilities are `P(observed label = k | x)` — after label
/// noise, exactly as the binary family records them — so a perfectly calibrated
/// model matches them and does not look mis-calibrated by the noise the caller
/// asked for.
pub(super) fn draw_multiclass(
    design: &DenseMatrix,
    classes: usize,
    balance: ClassBalance,
    geometry: ClassGeometry,
    separation: f32,
    label_noise: f32,
    digest: &[u8; 32],
) -> (Option<Target>, Truth) {
    let rows = design.rows();
    let mut weights = class_weights(design, classes, geometry, separation, digest);
    fit_class_offsets(&mut weights, classes, &balance.prevalences(classes));

    let mut rng = stream(digest, STREAM_LABELS);
    let flip_rate = f64::from(label_noise);
    // With `K` classes a flip lands on one of the other `K - 1` uniformly, so an
    // observed label is `k` either because it was drawn as `k` and survived, or
    // because it was drawn as one of the others and flipped onto `k`.
    let spill = if classes > 1 {
        flip_rate / (classes - 1) as f64
    } else {
        0.0
    };
    let mut labels = Vec::with_capacity(rows);
    let mut probabilities = Vec::with_capacity(rows * classes);
    for row in 0..rows {
        let clean = &weights[row * classes..(row + 1) * classes];
        // Every draw is unconditional, so a recipe with label noise walks the
        // same stream as the same recipe without it and the rows the noise did
        // not touch are byte-identical. Contamination overlays; it never
        // reseeds.
        let pick = f64::from(unit_draw(&mut rng));
        let flip = f64::from(unit_draw(&mut rng)) < flip_rate;
        let replacement = f64::from(unit_draw(&mut rng));

        let mut cumulative = 0.0_f64;
        let mut drawn = classes - 1;
        for (class, &probability) in clean.iter().enumerate() {
            cumulative += probability;
            if pick < cumulative {
                drawn = class;
                break;
            }
        }
        let label = if flip {
            let offset = (replacement * (classes - 1) as f64) as usize;
            (drawn + 1 + offset.min(classes - 2)) % classes
        } else {
            drawn
        };
        labels.push(label as u8);
        probabilities.extend(clean.iter().map(|&probability| {
            (probability * (1.0 - flip_rate) + (1.0 - probability) * spill) as f32
        }));
    }

    (
        Some(Target::Class(
            ClassTargets::new(labels).expect("a drawn class index is below the class count"),
        )),
        Truth::MulticlassBayes {
            probabilities,
            classes,
        },
    )
}

/// Row-major `rows * classes` unnormalized softmax weights, stabilized per row.
///
/// The maximum score of a row is subtracted before the exponential, which is
/// exact in the sense that matters — it multiplies every weight in the row by
/// one constant and the row is normalized afterwards — and keeps every weight in
/// `(0, 1]` rather than letting a large separation overflow to infinity.
fn class_weights(
    design: &DenseMatrix,
    classes: usize,
    geometry: ClassGeometry,
    separation: f32,
    digest: &[u8; 32],
) -> Vec<f64> {
    let columns = design.columns();
    let separation = f64::from(separation);
    let mut rng = stream(digest, STREAM_COEFFICIENTS);
    let mut weights = Vec::with_capacity(design.rows() * classes);

    match geometry {
        ClassGeometry::Blob => {
            let centres: Vec<f32> = (0..classes * columns)
                .map(|_| signed_draw(&mut rng))
                .collect();
            let normalizer = BLOB_DISTANCE_NORMALIZER / columns as f64;
            for row in design.iter_rows() {
                let scores: Vec<f64> = (0..classes)
                    .map(|class| {
                        let centre = &centres[class * columns..(class + 1) * columns];
                        let distance = sum_in_order(row.iter().zip(centre).map(|(&x, &c)| {
                            let gap = f64::from(x) - f64::from(c);
                            gap * gap
                        }));
                        -separation * normalizer * distance
                    })
                    .collect();
                push_stabilized(&mut weights, &scores);
            }
        }
        ClassGeometry::Hierarchical => {
            let depth = tree_depth(classes);
            let splits: Vec<f32> = (0..depth * columns)
                .map(|_| signed_draw(&mut rng))
                .collect();
            // `E[(x·w)²] = columns / 9` for two independent uniforms on
            // `[-1, 1)`, so this scales a projection to unit standard deviation
            // and `separation` keeps its meaning as the design widens.
            let normalizer = 3.0 / (columns as f64).sqrt();
            for row in design.iter_rows() {
                let projections: Vec<f64> = (0..depth)
                    .map(|level| {
                        dot(row, &splits[level * columns..(level + 1) * columns]) * normalizer
                    })
                    .collect();
                let scores: Vec<f64> = (0..classes)
                    .map(|class| {
                        separation
                            * sum_in_order(projections.iter().enumerate().map(
                                |(level, &projection)| {
                                    // Level zero is the most significant bit, and
                                    // its weight is one; every level below halves
                                    // it. Two classes agreeing on the top bits
                                    // therefore differ only in the weakly
                                    // separated coordinates, which is what makes
                                    // the confusion nested.
                                    let bit = depth - 1 - level;
                                    let sign = if (class >> bit) & 1 == 1 { 1.0 } else { -1.0 };
                                    sign * projection / (1_u64 << level) as f64
                                },
                            ))
                    })
                    .collect();
                push_stabilized(&mut weights, &scores);
            }
        }
    }
    weights
}

/// How many binary splits a balanced tree over `classes` leaves needs.
fn tree_depth(classes: usize) -> usize {
    (usize::BITS - (classes - 1).leading_zeros()) as usize
}

/// Appends `exp(score - max)` for one row.
fn push_stabilized(weights: &mut Vec<f64>, scores: &[f64]) {
    let largest = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    weights.extend(scores.iter().map(|&score| (score - largest).exp()));
}

/// Rescales the per-class softmax weights so each class's mean probability is
/// the requested one, then normalizes each row into a probability distribution.
///
/// The multiplier update `m_k *= π_k / mean_k` is the softmax prior correction:
/// it is the fixed point whose solution is exactly the offset vector a caller
/// asking for a class balance means, and there is no closed form for it because
/// the mean of a softmax over an arbitrary score matrix is not invertible. What
/// makes it worth solving rather than approximating is the same thing that makes
/// the binary family solve for its intercept: a *requested* balance has to be a
/// property of the correct answer, not an outcome of whatever the scores
/// happened to be.
fn fit_class_offsets(weights: &mut [f64], classes: usize, prevalences: &[f64]) {
    let rows = weights.len() / classes;
    let mut multipliers = vec![1.0_f64; classes];
    let mut means = vec![0.0_f64; classes];
    for _ in 0..OFFSET_ITERATIONS {
        means.fill(0.0);
        for row in 0..rows {
            let scores = &weights[row * classes..(row + 1) * classes];
            let total: f64 = scores
                .iter()
                .zip(&multipliers)
                .map(|(&weight, &multiplier)| weight * multiplier)
                .sum();
            if total <= 0.0 {
                continue;
            }
            for class in 0..classes {
                means[class] += scores[class] * multipliers[class] / total;
            }
        }
        for class in 0..classes {
            let mean = means[class] / rows as f64;
            // A class no row reaches cannot be corrected by any multiplier, and
            // dividing by its zero mean would put an infinity in the weights.
            // Leaving the multiplier alone keeps the other classes solvable and
            // lets the realized balance report the shortfall.
            if mean > 0.0 {
                multipliers[class] *= prevalences[class] / mean;
            }
        }
    }
    for row in 0..rows {
        let scores = &mut weights[row * classes..(row + 1) * classes];
        for (score, &multiplier) in scores.iter_mut().zip(&multipliers) {
            *score *= multiplier;
        }
        let total: f64 = scores.iter().sum();
        if total > 0.0 {
            for score in scores.iter_mut() {
                *score /= total;
            }
        } else {
            scores.fill(1.0 / classes as f64);
        }
    }
}

fn at_least_one(value: f32, parameter: Parameter) -> Result<(), DatasetError> {
    if !value.is_finite() {
        return Err(DatasetError::NonFiniteParameter { parameter });
    }
    if value < 1.0 {
        return Err(DatasetError::ParameterOutOfRange { parameter });
    }
    Ok(())
}
