//! Linear estimators with scikit-style fit and prediction semantics.

use crate::api::{Classifier, Estimator, HasParams, ModelError};
use crate::artifact::{
    ArtifactError, LOGISTIC_ARTIFACT_KIND, LegacyArtifactWriter, decode_legacy_envelope,
};
use crate::data::{BinaryTargets, MatrixView};

const BINARY_CLASSES: [u8; 2] = [0, 1];
const MAX_ARTIFACT_FEATURES: usize = 1_000_000;
const LOGISTIC_FIXED_PAYLOAD_BYTES: usize = 8 * 4;

/// Parameters for [`LogisticRegression`].
#[derive(Clone, Debug, PartialEq)]
pub struct LogisticRegressionParams {
    c: f32,
    fit_intercept: bool,
    max_iter: usize,
    tol: f32,
}

impl Default for LogisticRegressionParams {
    fn default() -> Self {
        Self {
            c: 1.0,
            fit_intercept: true,
            max_iter: 100,
            tol: 1.0e-4,
        }
    }
}

impl LogisticRegressionParams {
    /// Sets inverse L2 regularization strength. Smaller values regularize more.
    #[must_use]
    pub fn with_c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    /// Enables or disables the fitted intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Sets the maximum Newton iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the maximum absolute coefficient update used for convergence.
    #[must_use]
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Returns inverse L2 regularization strength.
    pub const fn c(&self) -> f32 {
        self.c
    }

    /// Returns whether an intercept is fitted.
    pub const fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }

    /// Returns the maximum optimization iteration count.
    pub const fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the convergence tolerance.
    pub const fn tol(&self) -> f32 {
        self.tol
    }
}

/// Binary L2-regularized logistic regression.
///
/// Fitting scales features internally for numerical conditioning and centers
/// them only when an intercept is requested. The transformation is folded into
/// the stored coefficients, so prediction is one allocation-free dot product
/// and sigmoid per row.
#[derive(Clone, Debug, PartialEq)]
pub struct LogisticRegression {
    n_features_in: usize,
    params: LogisticRegressionParams,
    coefficients: Vec<f32>,
    intercept: f32,
    iterations: usize,
}

impl LogisticRegression {
    /// Fits a binary logistic classifier with deterministic Newton updates.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        params: LogisticRegressionParams,
    ) -> Result<Self, ModelError> {
        validate_fit(data, targets, &params)?;
        let rows = data.rows();
        let columns = data.columns();
        let mut means = vec![0.0_f64; columns];
        if params.fit_intercept {
            for row in data.iter_rows() {
                for (column, &value) in row.iter().enumerate() {
                    means[column] += f64::from(value);
                }
            }
            for mean in &mut means {
                *mean /= rows as f64;
            }
        }
        let mut scales = vec![0.0_f64; columns];
        for row in data.iter_rows() {
            for (column, &value) in row.iter().enumerate() {
                let centered = f64::from(value) - means[column];
                scales[column] += centered * centered;
            }
        }
        for scale in &mut scales {
            *scale = (*scale / rows as f64).sqrt();
            if *scale <= f64::EPSILON {
                *scale = 1.0;
            }
        }

        let parameter_count = columns + usize::from(params.fit_intercept);
        let intercept_index = params.fit_intercept.then_some(columns);
        let mut theta = vec![0.0_f64; parameter_count];
        let lambda = 1.0 / f64::from(params.c);
        let mut iterations = 0;
        for iteration in 0..params.max_iter {
            let mut gradient = vec![0.0_f64; parameter_count];
            let mut hessian = vec![0.0_f64; parameter_count * parameter_count];
            for (row, &target) in data.iter_rows().zip(targets.as_slice()) {
                let mut score = intercept_index.map_or(0.0, |index| theta[index]);
                for column in 0..columns {
                    score +=
                        theta[column] * (f64::from(row[column]) - means[column]) / scales[column];
                }
                let probability = sigmoid_f64(score);
                let residual = probability - f64::from(target);
                let weight = (probability * (1.0 - probability)).max(1.0e-12);
                for left in 0..parameter_count {
                    let left_value =
                        design_value(row, left, columns, &means, &scales, params.fit_intercept);
                    gradient[left] += residual * left_value;
                    for right in 0..=left {
                        let right_value = design_value(
                            row,
                            right,
                            columns,
                            &means,
                            &scales,
                            params.fit_intercept,
                        );
                        hessian[left * parameter_count + right] +=
                            weight * left_value * right_value;
                    }
                }
            }
            for column in 0..columns {
                let scaled_penalty = lambda / (scales[column] * scales[column]);
                gradient[column] += scaled_penalty * theta[column];
                hessian[column * parameter_count + column] += scaled_penalty;
            }
            for left in 0..parameter_count {
                for right in 0..left {
                    hessian[right * parameter_count + left] =
                        hessian[left * parameter_count + right];
                }
            }
            let update = solve_positive_definite(&mut hessian, &gradient, parameter_count)?;
            let max_update = update
                .iter()
                .fold(0.0_f64, |max, value| max.max(value.abs()));
            for (value, update) in theta.iter_mut().zip(update) {
                *value -= update;
            }
            iterations = iteration + 1;
            if max_update <= f64::from(params.tol) {
                break;
            }
        }

        let coefficients = (0..columns)
            .map(|column| (theta[column] / scales[column]) as f32)
            .collect::<Vec<_>>();
        let intercept = intercept_index.map_or(0.0, |index| theta[index])
            - (0..columns)
                .map(|column| theta[column] * means[column] / scales[column])
                .sum::<f64>();
        if coefficients.iter().any(|value| !value.is_finite()) || !intercept.is_finite() {
            return Err(ModelError::LinearSolveFailed);
        }
        Ok(Self {
            n_features_in: columns,
            params,
            coefficients,
            intercept: intercept as f32,
            iterations,
        })
    }

    /// Returns fitted coefficients in input-feature order.
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }

    /// Returns the fitted intercept.
    pub const fn intercept(&self) -> f32 {
        self.intercept
    }

    /// Returns the number of optimization iterations performed.
    pub const fn n_iter(&self) -> usize {
        self.iterations
    }

    /// Returns the feature width required by this model.
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact fit parameters.
    pub const fn get_params(&self) -> &LogisticRegressionParams {
        &self.params
    }

    /// Predicts one positive-class probability.
    pub fn predict_positive_proba(&self, row: &[f32]) -> Result<f32, ModelError> {
        if row.len() != self.n_features_in {
            return Err(ModelError::FeatureDimension {
                expected: self.n_features_in,
                actual: row.len(),
            });
        }
        let score = row
            .iter()
            .zip(&self.coefficients)
            .fold(self.intercept, |sum, (&value, &coefficient)| {
                sum + value * coefficient
            });
        Ok(sigmoid_f32(score))
    }

    /// Predicts one label.
    pub fn predict_one(&self, row: &[f32]) -> Result<u8, ModelError> {
        Ok(u8::from(self.predict_positive_proba(row)? > 0.5))
    }

    /// Allocating label prediction convenience method.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        <Self as Classifier>::predict(self, data)
    }

    /// Allocation-free label prediction.
    pub fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        <Self as Classifier>::predict_into(self, data, output)
    }

    /// Allocating probability prediction convenience method.
    pub fn predict_proba(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Classifier>::predict_proba(self, data)
    }

    /// Allocation-free probability prediction.
    pub fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as Classifier>::predict_proba_into(self, data, output)
    }

    /// Predicts one requested class probability column.
    pub fn predict_class_proba(
        &self,
        data: &MatrixView<'_>,
        class: u8,
    ) -> Result<Vec<f32>, ModelError> {
        <Self as Classifier>::predict_class_proba(self, data, class)
    }

    /// Encodes this model in FerricML's stable checksummed artifact format.
    pub fn to_artifact(&self, feature_schema_sha256: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        let n_features =
            u32::try_from(self.n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
        let max_iter =
            u32::try_from(self.params.max_iter).map_err(|_| ArtifactError::InvalidPayload)?;
        let iterations =
            u32::try_from(self.iterations).map_err(|_| ArtifactError::InvalidPayload)?;
        let mut writer = LegacyArtifactWriter::new(
            LOGISTIC_ARTIFACT_KIND,
            feature_schema_sha256,
            96 + self.coefficients.len() * 4,
        );
        writer.u32(n_features);
        writer.u32(u32::from(self.params.fit_intercept));
        writer.f32(self.params.c);
        writer.u32(max_iter);
        writer.f32(self.params.tol);
        writer.u32(iterations);
        writer.f32(self.intercept);
        writer.u32(n_features);
        for &coefficient in &self.coefficients {
            writer.f32(coefficient);
        }
        Ok(writer.finish())
    }

    /// Decodes a logistic model after checking integrity and feature identity.
    pub fn from_artifact(
        bytes: &[u8],
        expected_feature_schema_sha256: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let mut cursor = decode_legacy_envelope(
            bytes,
            LOGISTIC_ARTIFACT_KIND,
            expected_feature_schema_sha256,
            LOGISTIC_FIXED_PAYLOAD_BYTES,
        )?;
        let n_features_in = cursor.u32()? as usize;
        let fit_intercept = match cursor.u32()? {
            0 => false,
            1 => true,
            _ => return Err(ArtifactError::InvalidPayload),
        };
        let c = cursor.f32()?;
        let max_iter = cursor.u32()? as usize;
        let tol = cursor.f32()?;
        let iterations = cursor.u32()? as usize;
        let intercept = cursor.f32()?;
        let coefficient_count = cursor.u32()? as usize;
        if n_features_in == 0
            || n_features_in > MAX_ARTIFACT_FEATURES
            || coefficient_count != n_features_in
            || !c.is_finite()
            || c <= 0.0
            || max_iter == 0
            || !tol.is_finite()
            || tol <= 0.0
            || iterations == 0
            || iterations > max_iter
            || !intercept.is_finite()
        {
            return Err(ArtifactError::InvalidPayload);
        }
        let mut coefficients = Vec::with_capacity(coefficient_count);
        for _ in 0..coefficient_count {
            let value = cursor.f32()?;
            if !value.is_finite() {
                return Err(ArtifactError::InvalidPayload);
            }
            coefficients.push(value);
        }
        if !cursor.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        Ok(Self {
            n_features_in,
            params: LogisticRegressionParams {
                c,
                fit_intercept,
                max_iter,
                tol,
            },
            coefficients,
            intercept,
            iterations,
        })
    }
}

impl Estimator for LogisticRegression {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasParams for LogisticRegression {
    type Params = LogisticRegressionParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Classifier for LogisticRegression {
    fn classes(&self) -> &[u8] {
        &BINARY_CLASSES
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        validate_predict(data, output.len(), self.n_features_in)?;
        for (row, slot) in data.iter_rows().zip(output) {
            *slot = self.predict_one(row)?;
        }
        Ok(())
    }

    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let expected = data
            .rows()
            .checked_mul(2)
            .ok_or(ModelError::OutputShapeOverflow {
                rows: data.rows(),
                columns: 2,
            })?;
        validate_feature_width(data, self.n_features_in)?;
        if output.len() != expected {
            return Err(ModelError::OutputLength {
                expected,
                actual: output.len(),
            });
        }
        for (row, probabilities) in data.iter_rows().zip(output.chunks_exact_mut(2)) {
            let positive = self.predict_positive_proba(row)?;
            probabilities[0] = 1.0 - positive;
            probabilities[1] = positive;
        }
        Ok(())
    }

    fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        if class > 1 {
            return Err(ModelError::UnknownClass { class });
        }
        validate_predict(data, output.len(), self.n_features_in)?;
        for (row, slot) in data.iter_rows().zip(output) {
            let positive = self.predict_positive_proba(row)?;
            *slot = if class == 1 { positive } else { 1.0 - positive };
        }
        Ok(())
    }
}

fn validate_fit(
    data: &MatrixView<'_>,
    targets: &BinaryTargets,
    params: &LogisticRegressionParams,
) -> Result<(), ModelError> {
    if data.rows() != targets.len() {
        return Err(ModelError::TargetLength {
            rows: data.rows(),
            targets: targets.len(),
        });
    }
    if !params.c.is_finite() || params.c <= 0.0 {
        return Err(ModelError::InvalidRegularization);
    }
    if params.max_iter == 0 {
        return Err(ModelError::InvalidIterationCount);
    }
    if !params.tol.is_finite() || params.tol <= 0.0 {
        return Err(ModelError::InvalidTolerance);
    }
    if !targets.as_slice().contains(&0) || !targets.as_slice().contains(&1) {
        return Err(ModelError::RequiresTwoClasses);
    }
    Ok(())
}

fn validate_predict(
    data: &MatrixView<'_>,
    output_len: usize,
    features: usize,
) -> Result<(), ModelError> {
    validate_feature_width(data, features)?;
    if output_len != data.rows() {
        return Err(ModelError::OutputLength {
            expected: data.rows(),
            actual: output_len,
        });
    }
    Ok(())
}

fn validate_feature_width(data: &MatrixView<'_>, features: usize) -> Result<(), ModelError> {
    if data.columns() != features {
        return Err(ModelError::FeatureDimension {
            expected: features,
            actual: data.columns(),
        });
    }
    Ok(())
}

fn design_value(
    row: &[f32],
    index: usize,
    columns: usize,
    means: &[f64],
    scales: &[f64],
    fit_intercept: bool,
) -> f64 {
    if fit_intercept && index == columns {
        1.0
    } else {
        (f64::from(row[index]) - means[index]) / scales[index]
    }
}

fn solve_positive_definite(
    matrix: &mut [f64],
    right: &[f64],
    size: usize,
) -> Result<Vec<f64>, ModelError> {
    for row in 0..size {
        for column in 0..=row {
            let mut value = matrix[row * size + column];
            for index in 0..column {
                value -= matrix[row * size + index] * matrix[column * size + index];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(ModelError::LinearSolveFailed);
                }
                matrix[row * size + column] = value.sqrt();
            } else {
                matrix[row * size + column] = value / matrix[column * size + column];
            }
        }
    }
    let mut solution = right.to_vec();
    for row in 0..size {
        for column in 0..row {
            solution[row] -= matrix[row * size + column] * solution[column];
        }
        solution[row] /= matrix[row * size + row];
    }
    for row in (0..size).rev() {
        for column in row + 1..size {
            solution[row] -= matrix[column * size + row] * solution[column];
        }
        solution[row] /= matrix[row * size + row];
    }
    Ok(solution)
}

fn sigmoid_f64(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn sigmoid_f32(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;
    use sha2::{Digest, Sha256};

    fn simple_data() -> (DenseMatrix, BinaryTargets) {
        (
            DenseMatrix::new(vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], 6, 1).unwrap(),
            BinaryTargets::new(vec![0, 0, 0, 1, 1, 1]).unwrap(),
        )
    }

    fn phase_zero_artifact() -> (LogisticRegression, [u8; 32], Vec<u8>) {
        let model = LogisticRegression {
            n_features_in: 2,
            params: LogisticRegressionParams {
                c: 2.0,
                fit_intercept: true,
                max_iter: 100,
                tol: f32::from_bits(0x3586_37bd),
            },
            coefficients: vec![f32::from_bits(0x3fb0_56aa), f32::from_bits(0x3eab_d102)],
            intercept: f32::from_bits(0xc00e_fe0f),
            iterations: 5,
        };
        let bytes = decode_hex(
            "4645525249434d4c01000100070707070707070707070707070707070707070707070707070707070707070702000000010000000000004064000000bd378635050000000ffe0ec002000000aa56b03f02d1ab3e6d72aa073218a54d30e6d6e5fc5d19b0a8a4e0726ac51369976bfa79a3ae9ec3",
        );
        (model, [7; 32], bytes)
    }

    fn decode_hex(value: &str) -> Vec<u8> {
        assert_eq!(value.len() % 2, 0);
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let text = std::str::from_utf8(pair).expect("ASCII hex");
                u8::from_str_radix(text, 16).expect("valid hex")
            })
            .collect()
    }

    fn resign_legacy_artifact(bytes: &mut [u8]) {
        let payload_len = bytes.len() - 32;
        let checksum = Sha256::digest(&bytes[..payload_len]);
        bytes[payload_len..].copy_from_slice(&checksum);
    }

    #[test]
    fn separates_a_simple_binary_problem() {
        let (data, targets) = simple_data();
        let model = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default().with_c(10.0),
        )
        .unwrap();
        assert_eq!(model.predict(&data.as_view()).unwrap(), targets.as_slice());
        let probabilities = model.predict_proba(&data.as_view()).unwrap();
        assert!(
            probabilities
                .chunks_exact(2)
                .all(|row| (row[0] + row[1] - 1.0).abs() < 1.0e-6)
        );
        assert!(model.n_iter() > 0);
    }

    #[test]
    fn validates_parameters_classes_and_output() {
        let (data, targets) = simple_data();
        assert_eq!(
            LogisticRegression::fit(
                &data.as_view(),
                &targets,
                LogisticRegressionParams::default().with_c(0.0),
            )
            .unwrap_err(),
            ModelError::InvalidRegularization
        );
        let one_class = BinaryTargets::new(vec![0; 6]).unwrap();
        assert_eq!(
            LogisticRegression::fit(
                &data.as_view(),
                &one_class,
                LogisticRegressionParams::default(),
            )
            .unwrap_err(),
            ModelError::RequiresTwoClasses
        );
        let model = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        assert_eq!(
            model.predict_into(&data.as_view(), &mut [0; 2]),
            Err(ModelError::OutputLength {
                expected: 6,
                actual: 2
            })
        );
    }

    #[test]
    fn fit_is_deterministic_and_handles_constant_columns() {
        let data = DenseMatrix::new(vec![1.0, -2.0, 1.0, -1.0, 1.0, 1.0, 1.0, 2.0], 4, 2).unwrap();
        let targets = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
        let left = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        let right = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        assert_eq!(left, right);
        assert_eq!(left.coefficients()[0], 0.0);
    }

    #[test]
    fn matches_frozen_sklearn_lbfgs_fixture() {
        let train = DenseMatrix::new(
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 0.0, 2.0, 1.0, 3.0, 0.0, 3.0, 1.0,
            ],
            8,
            2,
        )
        .unwrap();
        let targets = BinaryTargets::new(vec![0, 0, 0, 1, 1, 1, 1, 0]).unwrap();
        let test = DenseMatrix::new(
            vec![-1.0, 0.5, 0.5, 0.5, 1.5, 0.5, 2.5, 0.5, 4.0, 0.5],
            5,
            2,
        )
        .unwrap();
        let model = LogisticRegression::fit(
            &train.as_view(),
            &targets,
            LogisticRegressionParams::default().with_tol(1.0e-8),
        )
        .unwrap();
        let expected = [
            0.815_812_1,
            0.184_187_87,
            0.644_578_2,
            0.355_421_75,
            0.5,
            0.5,
            0.355_421_78,
            0.644_578_2,
            0.184_187_83,
            0.815_812_2,
        ];
        let actual = model.predict_proba(&test.as_view()).unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "{actual} != {expected}"
            );
        }
        assert!((model.coefficients()[0] - 0.595_291_14).abs() <= 2.0e-5);
        assert!(model.coefficients()[1].abs() <= 2.0e-5);
        assert!((model.intercept() + 0.892_936_77).abs() <= 2.0e-5);
    }

    #[test]
    fn mirrored_rows_have_opposite_no_intercept_scores() {
        let data =
            DenseMatrix::new(vec![2.0, 1.0, -2.0, -1.0, 1.0, -1.0, -1.0, 1.0], 4, 2).unwrap();
        let targets = BinaryTargets::new(vec![1, 0, 1, 0]).unwrap();
        let model = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default()
                .with_fit_intercept(false)
                .with_tol(1.0e-8),
        )
        .unwrap();

        assert_eq!(model.intercept().to_bits(), 0.0_f32.to_bits());
        for rows in data.as_slice().chunks_exact(4) {
            let positive = model.predict_positive_proba(&rows[..2]).unwrap();
            let mirrored = model.predict_positive_proba(&rows[2..]).unwrap();
            assert!((positive + mirrored - 1.0).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn artifact_round_trip_checks_schema_and_corruption() {
        let (data, targets) = simple_data();
        let model = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        let schema = [7_u8; 32];
        let bytes = model.to_artifact(schema).unwrap();
        let decoded = LogisticRegression::from_artifact(&bytes, schema).unwrap();
        assert_eq!(decoded, model);
        assert_eq!(
            decoded.predict_proba(&data.as_view()).unwrap(),
            model.predict_proba(&data.as_view()).unwrap()
        );
        assert_eq!(
            LogisticRegression::from_artifact(&bytes, [8; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );

        let mut corrupted = bytes.clone();
        corrupted[20] ^= 1;
        assert_eq!(
            LogisticRegression::from_artifact(&corrupted, schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
        assert_eq!(
            LogisticRegression::from_artifact(&bytes[..40], schema).unwrap_err(),
            ArtifactError::Truncated
        );
    }

    #[test]
    fn legacy_artifact_bytes_are_frozen_exactly() {
        let (model, schema, expected) = phase_zero_artifact();
        assert_eq!(expected.len(), 116);
        assert_eq!(model.to_artifact(schema).unwrap(), expected);
        assert_eq!(
            LogisticRegression::from_artifact(&expected, schema).unwrap(),
            model
        );
    }

    #[test]
    fn legacy_artifact_error_precedence_and_payload_validation_are_frozen() {
        let (_, schema, bytes) = phase_zero_artifact();

        let mut invalid_magic_without_checksum = bytes.clone();
        invalid_magic_without_checksum[0] ^= 1;
        assert_eq!(
            LogisticRegression::from_artifact(&invalid_magic_without_checksum, schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );

        let mut invalid_magic = bytes.clone();
        invalid_magic[0] ^= 1;
        resign_legacy_artifact(&mut invalid_magic);
        assert_eq!(
            LogisticRegression::from_artifact(&invalid_magic, schema).unwrap_err(),
            ArtifactError::InvalidMagic
        );

        let mut unsupported_version = bytes.clone();
        unsupported_version[8..10].copy_from_slice(&2_u16.to_le_bytes());
        resign_legacy_artifact(&mut unsupported_version);
        assert_eq!(
            LogisticRegression::from_artifact(&unsupported_version, schema).unwrap_err(),
            ArtifactError::UnsupportedVersion { found: 2 }
        );

        let mut unsupported_kind = bytes.clone();
        unsupported_kind[10..12].copy_from_slice(&2_u16.to_le_bytes());
        resign_legacy_artifact(&mut unsupported_kind);
        assert_eq!(
            LogisticRegression::from_artifact(&unsupported_kind, schema).unwrap_err(),
            ArtifactError::UnsupportedModelKind { found: 2 }
        );

        let mut invalid_flag = bytes.clone();
        invalid_flag[48..52].copy_from_slice(&2_u32.to_le_bytes());
        resign_legacy_artifact(&mut invalid_flag);
        assert_eq!(
            LogisticRegression::from_artifact(&invalid_flag, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut count_mismatch = bytes.clone();
        count_mismatch[72..76].copy_from_slice(&1_u32.to_le_bytes());
        resign_legacy_artifact(&mut count_mismatch);
        assert_eq!(
            LogisticRegression::from_artifact(&count_mismatch, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut non_finite = bytes.clone();
        non_finite[68..72].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        resign_legacy_artifact(&mut non_finite);
        assert_eq!(
            LogisticRegression::from_artifact(&non_finite, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut trailing = bytes[..bytes.len() - 32].to_vec();
        trailing.push(0);
        trailing.extend_from_slice(&[0; 32]);
        resign_legacy_artifact(&mut trailing);
        assert_eq!(
            LogisticRegression::from_artifact(&trailing, schema).unwrap_err(),
            ArtifactError::TrailingBytes
        );

        assert_eq!(
            LogisticRegression::from_artifact(&bytes[..107], schema).unwrap_err(),
            ArtifactError::Truncated
        );
    }
}
