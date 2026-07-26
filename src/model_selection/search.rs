//! Typed hyperparameter search over FerricML's own parameter builders.

use std::error::Error;
use std::fmt;

use crate::api::{ModelError, Regressor};
use crate::data::{BinaryTargets, MatrixView, RegressionTargets};

use super::cross_validation::{validate_split_sample_count, validate_target_length};
use super::{
    ClassificationScore, CrossValidationError, CrossValidationResult, RegressionScore,
    ScorableClassifier, Split, cross_validate_classifier, cross_validate_regressor,
};

/// An ordered, explicit set of typed parameter candidates.
///
/// FerricML's whole parameter story is typed builders, so a search grid is too:
/// there are no string keys, no runtime name lookup, and no way to name a
/// parameter that does not exist. An axis is a *builder method* plus the values
/// to pass it, which is why a misspelled parameter is a compile error here and a
/// silent misconfiguration in a stringly-typed grid.
///
/// ```
/// use ferricml::linear_model::RidgeParams;
/// use ferricml::model_selection::ParameterGrid;
///
/// let grid = ParameterGrid::new(RidgeParams::default())
///     .axis([0.1_f32, 1.0], RidgeParams::with_alpha)
///     .axis([true, false], RidgeParams::with_fit_intercept);
///
/// // Two alphas by two intercept choices, and the axis added last varies
/// // fastest.
/// assert_eq!(grid.len(), 4);
/// assert_eq!(grid.candidates()[0].alpha(), 0.1);
/// assert!(grid.candidates()[0].fit_intercept());
/// assert_eq!(grid.candidates()[1].alpha(), 0.1);
/// assert!(!grid.candidates()[1].fit_intercept());
/// ```
///
/// # Candidate order
///
/// Candidates are materialized when each axis is added, so the order is fixed
/// at construction and never depends on iteration of an unordered collection.
/// Adding an axis replaces every existing candidate with one candidate per
/// value, in value order, so the **axis added last varies fastest** and the
/// first candidate is the base value with the first value of every axis.
///
/// That order is part of the contract: search reports per-candidate results in
/// it and breaks ties toward the earliest candidate, so a caller who puts the
/// cheapest or most conservative value first gets it on a tie.
///
/// # Dependent parameters
///
/// A cross product cannot express "this depth only with that leaf count".
/// [`ParameterGrid::from_candidates`] takes the list directly for that case, and
/// search treats both the same way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterGrid<P> {
    candidates: Vec<P>,
}

impl<P> ParameterGrid<P> {
    /// Starts a grid from one base parameter value.
    ///
    /// With no axes this is a one-candidate grid, which is a legitimate way to
    /// cross-validate a single configuration through the same entry point.
    pub fn new(base: P) -> Self {
        Self {
            candidates: vec![base],
        }
    }

    /// Takes an explicit candidate list in the order it will be evaluated.
    ///
    /// An empty list is accepted here and rejected by the search entry point as
    /// [`SearchError::EmptyGrid`], so a grid built by a caller's own loop is
    /// refused exactly as an axis with no values is.
    pub fn from_candidates(candidates: Vec<P>) -> Self {
        Self { candidates }
    }

    /// Candidates in evaluation order.
    pub fn candidates(&self) -> &[P] {
        &self.candidates
    }

    /// Number of candidates the grid names.
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Returns whether the grid names no candidate at all.
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

impl<P: Clone> ParameterGrid<P> {
    /// Crosses every current candidate with one typed builder setting.
    ///
    /// `set` is the parameter type's own `with_*` builder method and `values`
    /// are the values to pass it, so each axis keeps the parameter's real type.
    /// Different axes may therefore carry different value types, which is what
    /// a single erased key/value map could not do without giving up the types.
    ///
    /// An axis with no values empties the grid rather than being ignored: a grid
    /// that silently dropped an axis would search something the caller did not
    /// ask for.
    #[must_use]
    pub fn axis<T, F>(self, values: impl IntoIterator<Item = T>, set: F) -> Self
    where
        T: Clone,
        F: Fn(P, T) -> P,
    {
        let values = values.into_iter().collect::<Vec<_>>();
        let mut candidates = Vec::with_capacity(self.candidates.len().saturating_mul(values.len()));
        for candidate in self.candidates {
            for value in &values {
                candidates.push(set(candidate.clone(), value.clone()));
            }
        }
        Self { candidates }
    }
}

/// Errors produced while searching a parameter grid.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SearchError {
    /// The grid named no candidate to evaluate.
    EmptyGrid,
    /// The search was rejected before any candidate was fitted.
    ///
    /// These are the failures that do not belong to a candidate — a target
    /// length that does not match the data, no splits at all, or a split built
    /// for a different dataset. They are checked once, up front, so an
    /// unusable call costs no fitting and no candidate is blamed for it.
    Setup(CrossValidationError),
    /// One candidate's cross-validation failed.
    ///
    /// The wrapped error keeps its own zero-based fold index, so a failure is
    /// attributed to an exact candidate *and* an exact fold.
    Candidate {
        /// Zero-based index of the candidate in grid order.
        candidate: usize,
        /// Original cross-validation error.
        source: CrossValidationError,
    },
    /// A score produced a value that cannot be ordered.
    ///
    /// Cross-validation reports whatever a score returns, but search has to
    /// *rank* the results, and a non-finite score has no defensible position in
    /// that ranking. Reporting it is therefore explicit rather than letting a
    /// `NaN` comparison decide a winner silently.
    NonFiniteScore {
        /// Zero-based index of the candidate in grid order.
        candidate: usize,
        /// Zero-based fold index within that candidate.
        fold: usize,
        /// The offending score.
        score: f64,
    },
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGrid => f.write_str("parameter search requires at least one candidate"),
            Self::Setup(source) => write!(f, "search setup failed: {source}"),
            Self::Candidate { candidate, source } => {
                write!(f, "candidate {candidate} failed: {source}")
            }
            Self::NonFiniteScore {
                candidate,
                fold,
                score,
            } => write!(
                f,
                "candidate {candidate} fold {fold} scored {score}, which cannot be ranked"
            ),
        }
    }
}

impl Error for SearchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Setup(source) | Self::Candidate { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// One candidate's parameters together with its per-fold scores.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateScores<P> {
    params: P,
    folds: CrossValidationResult,
}

impl<P> CandidateScores<P> {
    /// The parameter value this candidate was evaluated with.
    pub fn params(&self) -> &P {
        &self.params
    }

    /// Every fold's score, in split order.
    ///
    /// Search reports the whole distribution rather than only its summary,
    /// because a candidate that wins on the mean while collapsing on one fold
    /// is a different thing from one that is uniformly good, and the mean alone
    /// cannot tell them apart.
    pub fn folds(&self) -> &CrossValidationResult {
        &self.folds
    }

    /// Fixed-order mean of this candidate's fold scores, which is what the
    /// winner is chosen on.
    pub fn mean_score(&self) -> f64 {
        self.folds.mean()
    }
}

/// Ordered per-candidate results from one deterministic grid search.
///
/// # Winner selection
///
/// The winner is the candidate with the best **mean fold score**: the largest
/// mean when the score declares
/// [`greater_is_better`](super::RegressionScore::greater_is_better), the
/// smallest otherwise. The comparison is strict, so a candidate replaces the
/// incumbent only by being strictly better; **ties therefore go to the earliest
/// candidate in grid order**, which is the order
/// [`ParameterGrid`] fixed at construction. A caller who lists the cheapest or
/// most conservative value first gets it whenever the data cannot tell the
/// candidates apart.
///
/// # Refitting
///
/// Search reports scores and parameters; it does not refit. The winning
/// parameters go back through the same fitting closure on whatever data the
/// caller chooses, which keeps the refit policy visible in the caller's code
/// instead of hidden in a flag.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult<P> {
    candidates: Vec<CandidateScores<P>>,
    best: usize,
}

impl<P> SearchResult<P> {
    /// Every candidate's parameters and fold scores, in grid order.
    pub fn candidates(&self) -> &[CandidateScores<P>] {
        &self.candidates
    }

    /// Number of evaluated candidates.
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Returns whether the result holds no candidate.
    ///
    /// A successful search never does; this keeps the collection-style
    /// interface explicit.
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Grid-order index of the winning candidate.
    pub fn best_index(&self) -> usize {
        self.best
    }

    /// The winning candidate's parameters and fold scores.
    pub fn best(&self) -> &CandidateScores<P> {
        &self.candidates[self.best]
    }

    /// The winning candidate's parameters.
    pub fn best_params(&self) -> &P {
        self.best().params()
    }
}

/// Searches a typed parameter grid for a classifier, serially and in order.
///
/// Every candidate is cross-validated over **the same folds**: the split
/// iterator is drained once, up front, so fold membership is identical for
/// every candidate and the comparison between them is not confounded by a
/// re-drawn partition. With a fixed seed, fixed folds, and FerricML's serial
/// fitting, the whole result is reproducible.
///
/// Scoring is not reimplemented here. Each candidate runs through
/// [`cross_validate_classifier`], which runs through the same caller-owned
/// scoring entry point batch scoring and permutation importance use, so a
/// caller-defined score behaves in search exactly as it does anywhere else.
///
/// `view` is that entry point's view argument, forwarded unchanged:
/// [`ScorableClassifier::probabilistic`] for a model that produces
/// probabilities, [`ScorableClassifier::labels_only`] for one that does not.
/// There is no separate label-only search, because there is no separate
/// label-only cross-validation.
pub fn grid_search_classifier<M, P, I, F, S, V>(
    data: &MatrixView<'_>,
    targets: &BinaryTargets,
    splits: I,
    grid: &ParameterGrid<P>,
    scorer: S,
    mut fit: F,
    view: V,
) -> Result<SearchResult<P>, SearchError>
where
    P: Clone,
    I: IntoIterator<Item = Split>,
    F: FnMut(&MatrixView<'_>, &BinaryTargets, &P) -> Result<M, ModelError>,
    S: ClassificationScore,
    V: for<'m> Fn(&'m M) -> ScorableClassifier<'m>,
{
    let splits = validated_setup(data.rows(), targets.len(), grid, splits)?;
    let mut candidates = Vec::with_capacity(grid.len());
    for (candidate, params) in grid.candidates().iter().enumerate() {
        let folds = cross_validate_classifier(
            data,
            targets,
            splits.iter().cloned(),
            &scorer,
            |train, train_targets| fit(train, train_targets, params),
            &view,
        )
        .map_err(|source| SearchError::Candidate { candidate, source })?;
        candidates.push(finish_candidate(candidate, params.clone(), folds)?);
    }
    Ok(select_best(candidates, scorer.greater_is_better()))
}

/// Searches a typed parameter grid for a regressor, serially and in order.
///
/// The same contract as [`grid_search_classifier`]: one shared fold list, one
/// scoring implementation, per-candidate fold scores, and a winner chosen by
/// the mean with ties going to the earliest candidate.
///
/// ```
/// use ferricml::data::{DenseMatrix, RegressionTargets};
/// use ferricml::linear_model::{Ridge, RidgeParams};
/// use ferricml::model_selection::{
///     KFold, ParameterGrid, RegressionScorer, grid_search_regressor,
/// };
///
/// let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1)?;
/// let targets = RegressionTargets::new(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0])?;
/// let grid = ParameterGrid::new(RidgeParams::default())
///     .axis([0.01_f32, 1.0, 100.0], RidgeParams::with_alpha);
///
/// let result = grid_search_regressor(
///     &data.as_view(),
///     &targets,
///     KFold::new(3).split(data.rows())?,
///     &grid,
///     RegressionScorer::MeanSquaredError,
///     |train, train_targets, params| Ridge::fit(train, train_targets, params.clone()),
/// )?;
///
/// // Every candidate reports every fold, not only the winner's summary.
/// assert_eq!(result.len(), 3);
/// assert!(result.candidates().iter().all(|candidate| candidate.folds().len() == 3));
/// // The weakest penalty fits this exactly linear target best.
/// assert_eq!(result.best_index(), 0);
/// assert_eq!(result.best_params().alpha(), 0.01);
///
/// // Search does not refit; the winning parameters go back through the same
/// // closure on whatever data the caller chooses.
/// let refitted = Ridge::fit(&data.as_view(), &targets, result.best_params().clone())?;
/// assert_eq!(refitted.get_params().alpha(), 0.01);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn grid_search_regressor<M, P, I, F, S>(
    data: &MatrixView<'_>,
    targets: &RegressionTargets,
    splits: I,
    grid: &ParameterGrid<P>,
    scorer: S,
    mut fit: F,
) -> Result<SearchResult<P>, SearchError>
where
    M: Regressor,
    P: Clone,
    I: IntoIterator<Item = Split>,
    F: FnMut(&MatrixView<'_>, &RegressionTargets, &P) -> Result<M, ModelError>,
    S: RegressionScore,
{
    let splits = validated_setup(data.rows(), targets.len(), grid, splits)?;
    let mut candidates = Vec::with_capacity(grid.len());
    for (candidate, params) in grid.candidates().iter().enumerate() {
        let folds = cross_validate_regressor(
            data,
            targets,
            splits.iter().cloned(),
            &scorer,
            |train, train_targets| fit(train, train_targets, params),
        )
        .map_err(|source| SearchError::Candidate { candidate, source })?;
        candidates.push(finish_candidate(candidate, params.clone(), folds)?);
    }
    Ok(select_best(candidates, scorer.greater_is_better()))
}

/// Drains the split iterator once and rejects everything that is wrong with
/// the call before any candidate is fitted.
fn validated_setup<P, I: IntoIterator<Item = Split>>(
    rows: usize,
    targets: usize,
    grid: &ParameterGrid<P>,
    splits: I,
) -> Result<Vec<Split>, SearchError> {
    if grid.is_empty() {
        return Err(SearchError::EmptyGrid);
    }
    validate_target_length(rows, targets).map_err(SearchError::Setup)?;
    let splits = splits.into_iter().collect::<Vec<_>>();
    if splits.is_empty() {
        return Err(SearchError::Setup(CrossValidationError::NoSplits));
    }
    for (fold, split) in splits.iter().enumerate() {
        validate_split_sample_count(fold, rows, split).map_err(SearchError::Setup)?;
    }
    Ok(splits)
}

/// Rejects a candidate whose scores cannot be ranked, then keeps them.
fn finish_candidate<P>(
    candidate: usize,
    params: P,
    folds: CrossValidationResult,
) -> Result<CandidateScores<P>, SearchError> {
    if let Some((fold, &score)) = folds
        .scores()
        .iter()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        return Err(SearchError::NonFiniteScore {
            candidate,
            fold,
            score,
        });
    }
    Ok(CandidateScores { params, folds })
}

/// Picks the best mean score, keeping the earliest candidate on an exact tie.
fn select_best<P>(candidates: Vec<CandidateScores<P>>, greater_is_better: bool) -> SearchResult<P> {
    let mut best = 0;
    let mut best_score = candidates[0].mean_score();
    for (candidate, scores) in candidates.iter().enumerate().skip(1) {
        let score = scores.mean_score();
        let improved = if greater_is_better {
            score > best_score
        } else {
            score < best_score
        };
        if improved {
            best = candidate;
            best_score = score;
        }
    }
    SearchResult { candidates, best }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;
    use crate::linear_model::{LogisticRegression, LogisticRegressionParams, Ridge, RidgeParams};
    use crate::model_selection::{
        ClassificationScorer, ClassifierOutput, ClassifierOutputKind, KFold, RegressionScorer,
        ScoringError, StratifiedKFold, score_regressor,
    };
    use std::cell::Cell;

    #[test]
    fn a_grid_with_no_axis_is_the_base_value_alone() {
        let grid = ParameterGrid::new(RidgeParams::default());
        assert_eq!(grid.len(), 1);
        assert!(!grid.is_empty());
        assert_eq!(grid.candidates(), &[RidgeParams::default()]);
    }

    #[test]
    fn the_axis_added_last_varies_fastest() {
        let grid = ParameterGrid::new(RidgeParams::default())
            .axis([0.5_f32, 5.0], RidgeParams::with_alpha)
            .axis([true, false], RidgeParams::with_fit_intercept);
        let observed = grid
            .candidates()
            .iter()
            .map(|params| (params.alpha(), params.fit_intercept()))
            .collect::<Vec<_>>();
        assert_eq!(
            observed,
            vec![(0.5, true), (0.5, false), (5.0, true), (5.0, false)]
        );
    }

    #[test]
    fn axes_may_carry_different_value_types() {
        let grid = ParameterGrid::new(LogisticRegressionParams::default())
            .axis([10_usize, 20], LogisticRegressionParams::with_max_iter)
            .axis([1e-3_f32, 1e-4], LogisticRegressionParams::with_tol);
        assert_eq!(grid.len(), 4);
        let observed = grid
            .candidates()
            .iter()
            .map(|params| (params.max_iter(), params.tol()))
            .collect::<Vec<_>>();
        assert_eq!(
            observed,
            vec![(10, 1e-3), (10, 1e-4), (20, 1e-3), (20, 1e-4)]
        );
    }

    #[test]
    fn a_later_axis_overrides_an_earlier_setting_of_the_same_parameter() {
        // Setting one parameter twice is the caller's business; the grid simply
        // applies the axes in order, so the last one wins per candidate.
        let grid = ParameterGrid::new(RidgeParams::default())
            .axis([1.0_f32], RidgeParams::with_alpha)
            .axis([7.0_f32], RidgeParams::with_alpha);
        assert_eq!(grid.len(), 1);
        assert_eq!(grid.candidates()[0].alpha(), 7.0);
    }

    #[test]
    fn an_axis_with_no_values_empties_the_grid_instead_of_being_ignored() {
        let grid = ParameterGrid::new(RidgeParams::default())
            .axis([0.1_f32, 1.0], RidgeParams::with_alpha)
            .axis(Vec::<bool>::new(), RidgeParams::with_fit_intercept);
        assert!(grid.is_empty());
        assert_eq!(grid.len(), 0);
        // An emptied grid stays empty however many axes follow.
        let grid = grid.axis([1.0_f32], RidgeParams::with_alpha);
        assert!(grid.is_empty());
    }

    #[test]
    fn an_explicit_candidate_list_keeps_its_own_order() {
        let candidates = vec![
            RidgeParams::default().with_alpha(9.0),
            RidgeParams::default().with_alpha(0.25),
        ];
        let grid = ParameterGrid::from_candidates(candidates.clone());
        assert_eq!(grid.candidates(), candidates.as_slice());
        assert!(ParameterGrid::<RidgeParams>::from_candidates(Vec::new()).is_empty());
    }

    #[test]
    fn construction_is_reproducible() {
        let build = || {
            ParameterGrid::new(RidgeParams::default())
                .axis([0.1_f32, 1.0, 10.0], RidgeParams::with_alpha)
                .axis([true, false], RidgeParams::with_fit_intercept)
        };
        assert_eq!(build(), build());
        assert_eq!(build().len(), 6);
    }

    fn regression_fixture() -> (DenseMatrix, RegressionTargets) {
        let data = DenseMatrix::new(
            (0..12)
                .flat_map(|row| [row as f32, ((row * 3) % 7) as f32])
                .collect(),
            12,
            2,
        )
        .unwrap();
        let targets =
            RegressionTargets::new((0..12).map(|row| (row * 2) as f32 + 1.0).collect()).unwrap();
        (data, targets)
    }

    fn folds() -> Vec<Split> {
        KFold::new(3)
            .with_shuffle(true)
            .with_random_state(29)
            .split(12)
            .unwrap()
            .collect()
    }

    fn ridge_grid() -> ParameterGrid<RidgeParams> {
        ParameterGrid::new(RidgeParams::default())
            .axis([0.01_f32, 1.0, 1_000.0], RidgeParams::with_alpha)
    }

    fn ridge_search<S: RegressionScore>(scorer: S) -> SearchResult<RidgeParams> {
        let (data, targets) = regression_fixture();
        grid_search_regressor(
            &data.as_view(),
            &targets,
            folds(),
            &ridge_grid(),
            scorer,
            |train, train_targets, params| Ridge::fit(train, train_targets, params.clone()),
        )
        .unwrap()
    }

    /// A score whose orientation is a constructor argument, so the same
    /// numbers can be ranked either way.
    struct OrientedMeanSquaredError {
        greater_is_better: bool,
    }

    impl RegressionScore for OrientedMeanSquaredError {
        fn greater_is_better(&self) -> bool {
            self.greater_is_better
        }

        fn score(&self, expected: &[f32], predicted: &[f32]) -> Result<f64, ScoringError> {
            RegressionScorer::MeanSquaredError.score(expected, predicted)
        }
    }

    struct AlwaysNaN;

    impl RegressionScore for AlwaysNaN {
        fn greater_is_better(&self) -> bool {
            true
        }

        fn score(&self, _expected: &[f32], _predicted: &[f32]) -> Result<f64, ScoringError> {
            Ok(f64::NAN)
        }
    }

    /// A classification score FerricML does not enumerate, to prove search
    /// reaches the scorer through the trait and honours its declared output.
    struct PositiveRate;

    impl ClassificationScore for PositiveRate {
        fn output_kind(&self) -> ClassifierOutputKind {
            ClassifierOutputKind::PositiveProbabilities
        }

        fn greater_is_better(&self) -> bool {
            true
        }

        fn score(
            &self,
            expected: &[u8],
            output: ClassifierOutput<'_>,
        ) -> Result<f64, ScoringError> {
            let ClassifierOutput::PositiveProbabilities(probabilities) = output else {
                return Err(ScoringError::UnsupportedOutput {
                    required: self.output_kind(),
                    supplied: output.kind(),
                });
            };
            let matched = expected
                .iter()
                .zip(probabilities)
                .filter(|&(&label, &probability)| (probability >= 0.5) == (label == 1))
                .count();
            Ok(matched as f64 / expected.len() as f64)
        }
    }

    #[test]
    fn every_candidate_reports_every_fold_in_grid_order() {
        let result = ridge_search(RegressionScorer::MeanSquaredError);
        assert_eq!(result.len(), 3);
        assert!(!result.is_empty());
        assert_eq!(result.candidates().len(), 3);
        for (candidate, expected) in result.candidates().iter().zip([0.01_f32, 1.0, 1_000.0]) {
            assert_eq!(candidate.params().alpha(), expected);
            assert_eq!(candidate.folds().len(), 3);
            assert_eq!(candidate.mean_score(), candidate.folds().mean());
            assert!(
                candidate
                    .folds()
                    .scores()
                    .iter()
                    .all(|score| score.is_finite())
            );
        }
        assert_eq!(result.best().params(), result.best_params());
        assert_eq!(result.best_index(), 0, "the weakest penalty fits best here");
    }

    /// The exit criterion: search is a loop over the existing cross-validation
    /// entry point, not a second evaluation path.
    #[test]
    fn each_candidate_equals_a_manual_cross_validation_over_the_same_folds() {
        let (data, targets) = regression_fixture();
        let result = ridge_search(RegressionScorer::MeanSquaredError);
        for (candidate, scores) in result.candidates().iter().enumerate() {
            let manual = cross_validate_regressor(
                &data.as_view(),
                &targets,
                folds(),
                RegressionScorer::MeanSquaredError,
                |train, train_targets| Ridge::fit(train, train_targets, scores.params().clone()),
            )
            .unwrap();
            assert_eq!(scores.folds(), &manual, "candidate {candidate}");
        }
    }

    /// And the same scores come out of scoring one fold directly, so search,
    /// cross-validation, and batch scoring share one implementation.
    #[test]
    fn every_fold_score_equals_scoring_that_fold_directly() {
        let (data, targets) = regression_fixture();
        let result = ridge_search(RegressionScorer::MeanSquaredError);
        for (candidate, scores) in result.candidates().iter().enumerate() {
            for (fold, split) in folds().iter().enumerate() {
                let train = data.select_rows(split.train_indices()).unwrap();
                let train_targets = targets.select(split.train_indices()).unwrap();
                let model =
                    Ridge::fit(&train.as_view(), &train_targets, scores.params().clone()).unwrap();
                let test = data.select_rows(split.test_indices()).unwrap();
                let test_targets = targets.select(split.test_indices()).unwrap();
                assert_eq!(
                    Ok(scores.folds().scores()[fold]),
                    score_regressor(
                        &model,
                        &test.as_view(),
                        &test_targets,
                        RegressionScorer::MeanSquaredError
                    ),
                    "candidate {candidate} fold {fold}"
                );
            }
        }
    }

    #[test]
    fn every_candidate_sees_exactly_the_same_folds() {
        let (data, targets) = regression_fixture();
        let expected = folds();
        let observed = std::cell::RefCell::new(Vec::new());
        let result = grid_search_regressor(
            &data.as_view(),
            &targets,
            expected.clone(),
            &ridge_grid(),
            RegressionScorer::MeanSquaredError,
            |train, train_targets, params| {
                observed
                    .borrow_mut()
                    .push(train.iter_rows().map(|row| row[0]).collect::<Vec<_>>());
                Ridge::fit(train, train_targets, params.clone())
            },
        )
        .unwrap();
        assert_eq!(result.len(), 3);

        let observed = observed.into_inner();
        assert_eq!(observed.len(), 9);
        for candidate in 1..3 {
            assert_eq!(
                observed[..3],
                observed[candidate * 3..candidate * 3 + 3],
                "candidate {candidate} was fitted on different folds"
            );
        }
    }

    #[test]
    fn the_declared_orientation_decides_the_winner() {
        let minimized = ridge_search(OrientedMeanSquaredError {
            greater_is_better: false,
        });
        let maximized = ridge_search(OrientedMeanSquaredError {
            greater_is_better: true,
        });
        // Identical fold scores, opposite winners.
        for (left, right) in minimized.candidates().iter().zip(maximized.candidates()) {
            assert_eq!(left.folds(), right.folds());
        }
        assert_eq!(minimized.best_index(), 0);
        assert_eq!(maximized.best_index(), 2);
        // The built-in enum agrees with the explicitly minimizing score.
        assert_eq!(
            ridge_search(RegressionScorer::MeanSquaredError).best_index(),
            minimized.best_index()
        );
    }

    #[test]
    fn an_exact_tie_goes_to_the_earliest_candidate() {
        let (data, targets) = regression_fixture();
        // Candidates 0 and 2 are the same configuration, so their means tie
        // exactly. The odd one out is placed so that the tie is at the winning
        // end for each orientation: worse than the pair when minimizing, better
        // than it when maximizing the same squared error.
        for (greater_is_better, odd_alpha) in [(false, 5_000.0_f32), (true, 0.001)] {
            let grid = ParameterGrid::from_candidates(vec![
                RidgeParams::default().with_alpha(0.5),
                RidgeParams::default().with_alpha(odd_alpha),
                RidgeParams::default().with_alpha(0.5),
            ]);
            let result = grid_search_regressor(
                &data.as_view(),
                &targets,
                folds(),
                &grid,
                OrientedMeanSquaredError { greater_is_better },
                |train, train_targets, params| Ridge::fit(train, train_targets, params.clone()),
            )
            .unwrap();
            let tied = result.candidates()[0].mean_score();
            assert_eq!(tied, result.candidates()[2].mean_score());
            let odd = result.candidates()[1].mean_score();
            assert!(
                if greater_is_better {
                    odd < tied
                } else {
                    odd > tied
                },
                "the tied pair must be the winning value, got tied={tied}, odd={odd}"
            );
            assert_eq!(
                result.best_index(),
                0,
                "a tie must keep the earliest candidate (greater_is_better={greater_is_better})"
            );
        }
    }

    #[test]
    fn identical_inputs_reproduce_identical_results() {
        let first = ridge_search(RegressionScorer::R2);
        let second = ridge_search(RegressionScorer::R2);
        assert_eq!(first, second);
        assert_eq!(first.best_index(), second.best_index());
    }

    #[test]
    fn a_classifier_grid_is_searched_through_the_same_path() {
        let data = DenseMatrix::new(
            (0..16)
                .flat_map(|row| [row as f32, (row % 4) as f32])
                .collect(),
            16,
            2,
        )
        .unwrap();
        let targets = BinaryTargets::new((0..16).map(|row| u8::from(row >= 8)).collect()).unwrap();
        let splits = StratifiedKFold::new(2)
            .with_shuffle(true)
            .with_random_state(5)
            .split(targets.as_slice())
            .unwrap()
            .collect::<Vec<_>>();
        let grid = ParameterGrid::new(LogisticRegressionParams::default())
            .axis([0.1_f32, 1.0], LogisticRegressionParams::with_c)
            .axis([50_usize, 200], LogisticRegressionParams::with_max_iter);

        let by_accuracy = grid_search_classifier(
            &data.as_view(),
            &targets,
            splits.clone(),
            &grid,
            ClassificationScorer::Accuracy,
            |train, train_targets, params| {
                LogisticRegression::fit(train, train_targets, params.clone())
            },
            |model| ScorableClassifier::probabilistic(model),
        )
        .unwrap();
        assert_eq!(by_accuracy.len(), 4);
        assert!(by_accuracy.candidates().iter().all(|candidate| {
            candidate.folds().len() == 2
                && candidate
                    .folds()
                    .scores()
                    .iter()
                    .all(|score| (0.0..=1.0).contains(score))
        }));

        // A caller-defined probability score routes through the same output
        // handling, so search never re-derives the class layout.
        let by_custom = grid_search_classifier(
            &data.as_view(),
            &targets,
            splits,
            &grid,
            PositiveRate,
            |train, train_targets, params| {
                LogisticRegression::fit(train, train_targets, params.clone())
            },
            |model| ScorableClassifier::probabilistic(model),
        )
        .unwrap();
        assert_eq!(by_custom.len(), 4);
        for candidate in by_custom.candidates() {
            assert!(
                candidate
                    .folds()
                    .scores()
                    .iter()
                    .all(|score| (0.0..=1.0).contains(score))
            );
        }
    }

    #[test]
    fn an_empty_grid_is_refused_before_any_fitting() {
        let (data, targets) = regression_fixture();
        let calls = Cell::new(0_usize);
        assert_eq!(
            grid_search_regressor::<Ridge, _, _, _, _>(
                &data.as_view(),
                &targets,
                folds(),
                &ParameterGrid::from_candidates(Vec::<RidgeParams>::new()),
                RegressionScorer::MeanSquaredError,
                |train, train_targets, params: &RidgeParams| {
                    calls.set(calls.get() + 1);
                    Ridge::fit(train, train_targets, params.clone())
                },
            ),
            Err(SearchError::EmptyGrid)
        );
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn setup_failures_are_reported_before_any_candidate_is_fitted() {
        let (data, targets) = regression_fixture();
        let calls = Cell::new(0_usize);
        let run = |targets: &RegressionTargets, splits: Vec<Split>| {
            grid_search_regressor::<Ridge, _, _, _, _>(
                &data.as_view(),
                targets,
                splits,
                &ridge_grid(),
                RegressionScorer::MeanSquaredError,
                |train, train_targets, params: &RidgeParams| {
                    calls.set(calls.get() + 1);
                    Ridge::fit(train, train_targets, params.clone())
                },
            )
        };

        assert_eq!(
            run(&RegressionTargets::new(vec![0.0]).unwrap(), folds()),
            Err(SearchError::Setup(CrossValidationError::TargetLength {
                rows: 12,
                targets: 1,
            }))
        );
        assert_eq!(
            run(&targets, Vec::new()),
            Err(SearchError::Setup(CrossValidationError::NoSplits))
        );
        assert_eq!(
            run(
                &targets,
                vec![Split::new(4, vec![0, 1], vec![2, 3]).unwrap()]
            ),
            Err(SearchError::Setup(CrossValidationError::SplitSampleCount {
                fold: 0,
                expected: 12,
                actual: 4,
            }))
        );
        assert_eq!(calls.get(), 0, "setup must cost no fitting");
    }

    #[test]
    fn a_failure_names_both_the_candidate_and_the_fold() {
        let (data, targets) = regression_fixture();
        let splits = KFold::new(2).split(12).unwrap().collect::<Vec<_>>();
        let calls = Cell::new(0_usize);
        // Two folds per candidate, so the fourth call is candidate 1, fold 1.
        assert_eq!(
            grid_search_regressor::<Ridge, _, _, _, _>(
                &data.as_view(),
                &targets,
                splits,
                &ridge_grid(),
                RegressionScorer::MeanSquaredError,
                |train, train_targets, params: &RidgeParams| {
                    let call = calls.get();
                    calls.set(call + 1);
                    if call == 3 {
                        Err(ModelError::LinearSolveFailed)
                    } else {
                        Ridge::fit(train, train_targets, params.clone())
                    }
                },
            ),
            Err(SearchError::Candidate {
                candidate: 1,
                source: CrossValidationError::Fit {
                    fold: 1,
                    source: ModelError::LinearSolveFailed,
                },
            })
        );
        assert_eq!(calls.get(), 4, "search must stop at the first failure");
    }

    #[test]
    fn a_score_that_cannot_be_ranked_is_a_typed_error() {
        let (data, targets) = regression_fixture();
        let error = grid_search_regressor(
            &data.as_view(),
            &targets,
            folds(),
            &ridge_grid(),
            AlwaysNaN,
            |train, train_targets, params: &RidgeParams| {
                Ridge::fit(train, train_targets, params.clone())
            },
        )
        .unwrap_err();
        // `NaN` is not equal to itself, so the variant is matched rather than
        // compared — which is exactly why it cannot be ranked either.
        let SearchError::NonFiniteScore {
            candidate,
            fold,
            score,
        } = error
        else {
            panic!("expected an unrankable score, got {error}");
        };
        assert_eq!((candidate, fold), (0, 0));
        assert!(score.is_nan());
    }

    #[test]
    fn errors_display_and_chain_to_their_source() {
        let candidate = SearchError::Candidate {
            candidate: 2,
            source: CrossValidationError::NoSplits,
        };
        assert_eq!(
            candidate.to_string(),
            "candidate 2 failed: cross-validation requires at least one split"
        );
        assert!(candidate.source().is_some());

        let setup = SearchError::Setup(CrossValidationError::NoSplits);
        assert!(setup.to_string().starts_with("search setup failed"));
        assert!(setup.source().is_some());

        assert_eq!(
            SearchError::EmptyGrid.to_string(),
            "parameter search requires at least one candidate"
        );
        assert!(SearchError::EmptyGrid.source().is_none());

        let unrankable = SearchError::NonFiniteScore {
            candidate: 1,
            fold: 0,
            score: f64::INFINITY,
        };
        assert_eq!(
            unrankable.to_string(),
            "candidate 1 fold 0 scored inf, which cannot be ranked"
        );
        assert!(unrankable.source().is_none());
    }
}
