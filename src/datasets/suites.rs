//! Two catalogues over the task families: one for measuring accuracy, one for
//! measuring throughput.
//!
//! # Why a catalogue rather than a page of examples
//!
//! Everything in this module could be written by a caller, once, by hand. The
//! reason it is not is that a hand-written sweep goes stale in one direction
//! only: the crate gains a family, nobody remembers the sweep, and the sweep
//! keeps passing while covering less of the generator than it claims to. A
//! catalogue that lives beside the families and is *checked against them* cannot
//! do that. [`Family::ALL`] is the roster, `suite_tests.rs` is where the
//! checking happens, and [`Family`]'s own documentation records exactly which
//! link in the chain is the compiler's and which is a test's.
//!
//! The two suites answer different questions and are deliberately not one:
//!
//! * [`AccuracySuite`] asks *is the answer right*. Every case is small, cheap,
//!   noise-light, and carries a [`Truth`](super::Truth) a fitted model can be
//!   scored against — the whole reason this generator exists rather than a
//!   pile of `rand` calls.
//! * [`PerformanceGrid`] asks *what does it cost*. Every family appears at every
//!   point of a rows × columns sweep, because the two dimensions do not cost the
//!   same thing: a source draws once per element, a linear family's target is a
//!   dot product per row, and a ranking family sorts within each query block.
//!
//! # The suites are tables, on purpose
//!
//! Both `cases` functions build their entries from a written-out list rather
//! than from a `match` over [`Family::ALL`]. A `match` would be compiler-checked
//! and would make the closure test vacuous — the suite would span every family
//! *by construction*, and the property nobody could then observe is the one that
//! actually matters: whether somebody adding a family thought about what a
//! meaningful case for it looks like. A table plus a test that reads the table
//! keeps that a decision. See `suite_tests.rs`.
//!
//! # Contamination is not in either suite
//!
//! [`Contamination`](super::Contamination) composes with every family, so a
//! robustness sweep is the cross product of a suite with a contamination ladder
//! rather than a third table. Folding a few contaminated cases in here would
//! make the suites neither clean baselines nor a complete robustness sweep.
//! `AccuracySuite`'s cases are therefore uncontaminated, and a caller wanting the
//! robustness question takes [`SuiteCase::recipe`] and adds the knob.

use super::dataset::Dataset;
use super::recipe::Recipe;
use super::structural::{ClassBalance, ClassGeometry};
use super::task::{BinaryKind, Family, GlmLink, NonlinearKind, Portability, Task};

/// The seed both suites derive their stream from.
///
/// Arbitrary and fixed — the value carries no meaning beyond being written down
/// once. It runs through [`Recipe::seeded`], so it is a *derived* state disjoint
/// from every stream an estimator seeded with the same number would draw from,
/// which is what stops a suite case's design correlating with the randomness of
/// the model being measured on it.
const SEED: u64 = 20_260_728;

/// Documents in every query block of a [`Task::Ranking`] case.
///
/// Eight, and every row count in either suite is a multiple of it, because
/// `queries * docs_per_query` must equal the design's rows exactly.
const DOCS_PER_QUERY: usize = 8;

/// Leading informative columns in the suites' linear families.
///
/// Four rather than "all of them", so every case carries noise features a model
/// has to decline to use. Both suites' narrowest design is eight columns wide,
/// so this is always a strict prefix.
const INFORMATIVE: usize = 4;

/// Classes in the suites' multiclass case.
const CLASSES: usize = 4;

/// Clusters in the suites' clustered case.
const BLOBS: usize = 8;

/// One member of a suite: a family and the recipe that stands for it.
///
/// The family is not a separate parameter a caller could disagree with — it is
/// read off the recipe's own task at construction, so a case cannot be filed
/// under a family it does not belong to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuiteCase {
    family: Family,
    recipe: Recipe,
}

impl SuiteCase {
    /// Builds a case, validating its recipe.
    ///
    /// # Panics
    ///
    /// Never, for the tables below: their shapes and parameters are constants,
    /// checked by [`Recipe::new`] and
    /// [`Recipe::with_task`] against each other, and
    /// `suite_tests.rs` constructs every case in both suites. A failure here
    /// would be a defect in this file rather than a caller's input.
    fn new(rows: usize, columns: usize, task: Task) -> Self {
        let recipe = Recipe::seeded(rows, columns, SEED)
            .and_then(|recipe| recipe.with_task(task))
            .expect("a suite case's shape and parameters are fixed and valid");
        Self {
            family: task.family(),
            recipe,
        }
    }

    /// Returns the family this case stands for.
    #[inline]
    pub const fn family(&self) -> Family {
        self.family
    }

    /// Returns this case's stable name, which is its family's label.
    ///
    /// A performance grid holds several cases per family, so a record naming a
    /// grid point needs the shape as well; [`SuiteCase::recipe`] carries it.
    #[inline]
    pub const fn name(&self) -> &'static str {
        self.family.label()
    }

    /// Returns the recipe this case generates from.
    ///
    /// This is the extension point: a robustness sweep adds a
    /// [`Contamination`](super::Contamination) here, and a larger run changes the
    /// shape, without either one needing a second table.
    #[inline]
    pub const fn recipe(&self) -> Recipe {
        self.recipe
    }

    /// Returns this case's determinism envelope.
    ///
    /// Half of both suites is [`Portability::PerRunner`], which is a property of
    /// the families rather than a shortcoming of the suites: a Bayes
    /// probability, a log link, a softmax and a requested condition number are
    /// all transcendental. A harness comparing two machines has to read this
    /// rather than assume, and `suite_tests.rs` pins which members carry which
    /// envelope so a family changing its answer cannot do it quietly.
    #[inline]
    pub fn portability(&self) -> Portability {
        self.recipe.portability()
    }

    /// Generates this case's dataset.
    #[inline]
    pub fn generate(&self) -> Dataset {
        self.recipe.generate()
    }
}

/// Every family as one small, clean, truth-carrying problem.
///
/// # What "accuracy" means here
///
/// Each case is a problem whose correct answer is recorded, so a consumer can
/// ask how close a fit came to *being right* rather than how close it came to
/// another implementation. What "right" is differs by family and is carried by
/// the case's [`Truth`](super::Truth): drawn coefficients for the linear
/// families, a conditional mean where no coefficient vector exists, the Bayes
/// probability behind every drawn label, a cluster assignment, both ends of a
/// drifting predictor, the utility behind every relevance grade.
///
/// Noise is deliberately low and contamination is absent. A suite that mixed
/// "can the solver recover a signal" with "does it survive an outlier" answers
/// neither question, and the second one is a sweep over
/// [`Contamination`](super::Contamination) rather than a table.
///
/// ```
/// use ferricml::datasets::{AccuracySuite, Family};
///
/// let cases = AccuracySuite::cases();
/// assert_eq!(cases.len(), Family::COUNT);
///
/// let linear = cases
///     .iter()
///     .find(|case| case.family() == Family::LinearRegression)
///     .expect("the suite spans every family");
/// let dataset = linear.generate();
///
/// assert_eq!(dataset.features().rows(), AccuracySuite::ROWS);
/// // The coefficients the target was drawn from are recorded, which is the
/// // whole reason this is an accuracy suite rather than a comparison suite.
/// let beta = dataset.truth().coefficients().expect("a linear family");
/// assert_eq!(beta.len(), AccuracySuite::COLUMNS);
/// ```
#[non_exhaustive]
pub struct AccuracySuite;

impl AccuracySuite {
    /// Rows in every accuracy case.
    ///
    /// Small on purpose: the suite is meant to run inside an ordinary test
    /// gate, so every case here fits and fits fast. `256` is a multiple of
    /// `DOCS_PER_QUERY`, which the ranking case needs, and enough rows that a
    /// realized prevalence or class balance is a measurement rather than a
    /// reading of the draw — three binomial deviations at a prevalence of `0.3`
    /// is under nine points.
    pub const ROWS: usize = 256;

    /// Columns in every accuracy case.
    ///
    /// Eight: wide enough that four informative columns leave four a model has
    /// to decline to use, and wide enough for every family's own reads —
    /// [`NonlinearKind::Friedman`] is the hungriest at five.
    pub const COLUMNS: usize = 8;

    /// Every accuracy case, one per family, in [`Family::ALL`] order.
    ///
    /// Written out rather than mapped over the roster, so that a family without
    /// a case is a failing test rather than an impossible state. The order is
    /// the roster's, and `suite_tests.rs` asserts that it stays so.
    pub fn cases() -> Vec<SuiteCase> {
        let (rows, columns) = (Self::ROWS, Self::COLUMNS);
        vec![
            // Low noise against a drawn beta: the case a least-squares path has
            // no excuse on.
            SuiteCase::new(
                rows,
                columns,
                Task::LinearRegression {
                    informative: INFORMATIVE,
                    coefficient_scale: 1.0,
                    intercept: 0.5,
                    noise_scale: 0.05,
                },
            ),
            // An interaction rather than a sinusoid, so the one nonlinear
            // regression case in the suite is bit-exact. The transcendental
            // shapes are still reachable through the recipe.
            SuiteCase::new(
                rows,
                columns,
                Task::NonlinearRegression {
                    kind: NonlinearKind::Interaction,
                    noise_scale: 0.05,
                },
            ),
            // A count response at dispersion one — an exactly Poisson target,
            // which is the specification a log-link fit is supposed to be
            // correct for. The coefficient scale is small because the link
            // exponentiates.
            SuiteCase::new(
                rows,
                columns,
                Task::GlmRegression {
                    link: GlmLink::LogCount,
                    informative: INFORMATIVE,
                    coefficient_scale: 0.4,
                    intercept: 0.5,
                    dispersion: 1.0,
                },
            ),
            // Full rank at a condition number of one hundred: hard enough to
            // separate a stable solver from a normal-equations one, and still a
            // problem whose drawn beta is the answer. A rank-deficient design
            // has an affine set of minimizers and recovering beta is not the
            // right question there, so the *accuracy* case stays full rank.
            SuiteCase::new(
                rows,
                columns,
                Task::IllConditioned {
                    condition_number: 100.0,
                    rank: Self::COLUMNS,
                    coefficient_scale: 1.0,
                    noise_scale: 0.01,
                },
            ),
            // A moderately separated boundary at a prevalence away from a half,
            // so a classifier scoring well by predicting the majority class is
            // visible.
            SuiteCase::new(
                rows,
                columns,
                Task::LinearBinary {
                    informative: INFORMATIVE,
                    separation: 3.0,
                    prevalence: 0.3,
                },
            ),
            // Exclusive-or: the boundary a linear classifier cannot represent at
            // all, so a suite result that looks fine here and nowhere else is
            // telling you something.
            SuiteCase::new(
                rows,
                columns,
                Task::NonlinearBinary {
                    kind: BinaryKind::Xor,
                    separation: 4.0,
                    prevalence: 0.5,
                },
            ),
            // Balanced blobs: the multiclass case whose confusions come from the
            // geometry rather than from a prior.
            SuiteCase::new(
                rows,
                columns,
                Task::Multiclass {
                    classes: CLASSES,
                    balance: ClassBalance::Balanced,
                    geometry: ClassGeometry::Blob,
                    separation: 3.0,
                },
            ),
            // Tight clusters, so the assignment is recoverable and a clusterer
            // that misses it has missed something easy.
            SuiteCase::new(
                rows,
                columns,
                Task::Clustered {
                    blobs: BLOBS,
                    spread: 0.15,
                },
            ),
            // A drift of one coefficient unit across the series, which is large
            // against the noise and therefore measurable by fitting two windows.
            SuiteCase::new(
                rows,
                columns,
                Task::TimeOrdered {
                    informative: INFORMATIVE,
                    coefficient_scale: 1.0,
                    drift: 1.0,
                    intercept: 0.0,
                    noise_scale: 0.05,
                },
            ),
            // Four grades over eight documents per query, so pairs tie as they
            // do in a real judgement rather than forming a total order.
            SuiteCase::new(
                rows,
                columns,
                Task::Ranking {
                    queries: Self::ROWS / DOCS_PER_QUERY,
                    docs_per_query: DOCS_PER_QUERY,
                    grades: 4,
                    informative: INFORMATIVE,
                    coefficient_scale: 1.0,
                },
            ),
        ]
    }
}

/// Every family at every point of a rows × columns sweep.
///
/// # Why both dimensions
///
/// Generation cost is not one number. A source draws once per element, so it is
/// linear in `rows * columns`; a linear family's target is a dot product per
/// row; a multiclass family's is a softmax over classes per row; a ranking
/// family sorts within each query block, so its cost grows with the block size
/// rather than with the row count; and [`Task::IllConditioned`] rescales and
/// duplicates whole columns, which is work in the column dimension that nothing
/// else does. A sweep in one dimension would attribute all of that to the wrong
/// axis.
///
/// The grid exists to be measured, not asserted: FerricML's performance protocol
/// records numbers on a registered runner rather than in a test, and the claim
/// this grid is built to support is that generation is negligible against the
/// smallest fit it feeds.
///
/// ```
/// use ferricml::datasets::{Family, PerformanceGrid};
///
/// let cases = PerformanceGrid::cases();
/// assert_eq!(
///     cases.len(),
///     PerformanceGrid::ROWS.len() * PerformanceGrid::COLUMNS.len() * Family::COUNT,
/// );
///
/// // Every case names the point it sits on, which is what a recorded row is
/// // filed under.
/// let first = cases[0];
/// let id = format!(
///     "{}/{}x{}",
///     first.name(),
///     first.recipe().rows(),
///     first.recipe().columns(),
/// );
/// assert_eq!(id, "linear-regression/256x8");
/// ```
#[non_exhaustive]
pub struct PerformanceGrid;

impl PerformanceGrid {
    /// Row counts the grid sweeps.
    ///
    /// Each is four times the one before, so a cost linear in the row count and
    /// one quadratic in it are distinguishable across three points rather than
    /// argued about. Every value is a multiple of `DOCS_PER_QUERY`, which the
    /// ranking family's shape requires.
    pub const ROWS: [usize; 3] = [256, 1_024, 4_096];

    /// Column counts the grid sweeps.
    ///
    /// The same fourfold steps, from a design narrower than most estimators care
    /// about to one wide enough that per-column work dominates. The narrowest is
    /// eight because that is the width every family's own reads fit in.
    pub const COLUMNS: [usize; 3] = [8, 32, 128];

    /// Every family at every grid point, rows-major then columns then family.
    ///
    /// The family list is written out rather than mapped over the roster, for
    /// the reason the module documentation gives: a family with no grid row must
    /// be a failing test rather than an impossible state.
    pub fn cases() -> Vec<SuiteCase> {
        let mut cases = Vec::with_capacity(Self::ROWS.len() * Self::COLUMNS.len() * Family::COUNT);
        for rows in Self::ROWS {
            for columns in Self::COLUMNS {
                cases.extend(Self::cases_at(rows, columns));
            }
        }
        cases
    }

    /// Every family at one grid point.
    ///
    /// The parameters that scale with the shape do so explicitly: the requested
    /// rank is the full column count, and the query count is whatever divides
    /// the row count into blocks of `DOCS_PER_QUERY`. Everything else is held
    /// fixed, so a difference between two grid points is the shape and not the
    /// problem.
    fn cases_at(rows: usize, columns: usize) -> Vec<SuiteCase> {
        vec![
            SuiteCase::new(
                rows,
                columns,
                Task::LinearRegression {
                    informative: INFORMATIVE,
                    coefficient_scale: 1.0,
                    intercept: 0.5,
                    noise_scale: 0.1,
                },
            ),
            SuiteCase::new(
                rows,
                columns,
                Task::NonlinearRegression {
                    kind: NonlinearKind::Interaction,
                    noise_scale: 0.1,
                },
            ),
            SuiteCase::new(
                rows,
                columns,
                Task::GlmRegression {
                    link: GlmLink::LogCount,
                    informative: INFORMATIVE,
                    coefficient_scale: 0.4,
                    intercept: 0.5,
                    dispersion: 1.0,
                },
            ),
            SuiteCase::new(
                rows,
                columns,
                Task::IllConditioned {
                    condition_number: 100.0,
                    rank: columns,
                    coefficient_scale: 1.0,
                    noise_scale: 0.1,
                },
            ),
            SuiteCase::new(
                rows,
                columns,
                Task::LinearBinary {
                    informative: INFORMATIVE,
                    separation: 3.0,
                    prevalence: 0.3,
                },
            ),
            SuiteCase::new(
                rows,
                columns,
                Task::NonlinearBinary {
                    kind: BinaryKind::Xor,
                    separation: 4.0,
                    prevalence: 0.5,
                },
            ),
            SuiteCase::new(
                rows,
                columns,
                Task::Multiclass {
                    classes: CLASSES,
                    balance: ClassBalance::Balanced,
                    geometry: ClassGeometry::Blob,
                    separation: 3.0,
                },
            ),
            SuiteCase::new(
                rows,
                columns,
                Task::Clustered {
                    blobs: BLOBS,
                    spread: 0.15,
                },
            ),
            SuiteCase::new(
                rows,
                columns,
                Task::TimeOrdered {
                    informative: INFORMATIVE,
                    coefficient_scale: 1.0,
                    drift: 1.0,
                    intercept: 0.0,
                    noise_scale: 0.1,
                },
            ),
            SuiteCase::new(
                rows,
                columns,
                Task::Ranking {
                    queries: rows / DOCS_PER_QUERY,
                    docs_per_query: DOCS_PER_QUERY,
                    grades: 4,
                    informative: INFORMATIVE,
                    coefficient_scale: 1.0,
                },
            ),
        ]
    }
}
