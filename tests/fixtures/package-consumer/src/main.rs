use ferricml::api::{Classifier, Regressor};
use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets};
use ferricml::ensemble::{
    RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
    RandomForestRegressorParams,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let features = DenseMatrix::new(vec![0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 3.0, 9.0], 4, 2)?;
    let classifier = RandomForestClassifier::fit(
        &features.as_view(),
        &BinaryTargets::new(vec![0, 0, 1, 1])?,
        RandomForestClassifierParams::default()
            .with_n_estimators(3)
            .with_bootstrap(false)
            .with_random_state(7),
    )?;
    let labels = Classifier::predict(&classifier, &features.as_view())?;
    let probabilities = Classifier::predict_proba(&classifier, &features.as_view())?;
    assert_eq!(labels.len(), features.rows());
    assert_eq!(probabilities.len(), features.rows() * 2);

    let regressor = RandomForestRegressor::fit(
        &features.as_view(),
        &RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0])?,
        RandomForestRegressorParams::default()
            .with_n_estimators(3)
            .with_bootstrap(false)
            .with_random_state(7),
    )?;
    let predictions = Regressor::predict(&regressor, &features.as_view())?;
    assert_eq!(predictions.len(), features.rows());
    assert!(predictions.iter().all(|value| value.is_finite()));
    Ok(())
}
