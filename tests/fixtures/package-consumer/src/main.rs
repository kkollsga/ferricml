use ferricml::api::{Classifier, Regressor};
use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets, SampleWeights};
use ferricml::ensemble::{
    RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
use ferricml::linear_model::{LogisticRegression, LogisticRegressionParams};

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

    let logistic = LogisticRegression::fit_weighted(
        &features.as_view(),
        &BinaryTargets::new(vec![0, 0, 1, 1])?,
        &SampleWeights::new(vec![1.0, 2.0, 1.0, 2.0])?,
        LogisticRegressionParams::default(),
    )?;
    let mut decisions = vec![0.0; features.rows()];
    logistic.decision_function_into(&features.as_view(), &mut decisions)?;
    assert!(decisions.iter().all(|value| value.is_finite()));
    let schema = [7; 32];
    let encoded = logistic.to_artifact(schema)?;
    let decoded = LogisticRegression::from_artifact(&encoded, schema)?;
    assert_eq!(
        decoded.predict_proba(&features.as_view())?,
        logistic.predict_proba(&features.as_view())?
    );

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
