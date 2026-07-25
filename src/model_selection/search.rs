//! Typed hyperparameter search over FerricML's own parameter builders.

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
    /// An empty list is accepted here and rejected by the search entry point,
    /// so a grid built by a caller's own loop is refused exactly as an axis with
    /// no values is.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_model::{LogisticRegressionParams, RidgeParams};

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
}
