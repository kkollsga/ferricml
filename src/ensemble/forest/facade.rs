//! The public shape of a bagged tree ensemble, generated once.
//!
//! Two ensembles that differ only in how their members are grown must not
//! differ in their public surface, in their validation order, or in which fits
//! they persist. Generating both facades from one macro makes that structural:
//! an entry point cannot exist on one and be forgotten on the other, and a doc
//! comment cannot describe one estimator's behaviour while the other quietly
//! does something else.
//!
//! What each expansion supplies is only what genuinely differs: the type name,
//! its parameter type, its artifact kind, and its own type-level documentation.

/// Generates one public ensemble classifier over the shared core.
macro_rules! forest_classifier {
    ($name:ident, $params:ident, $kind:expr, $($doc:expr),+ $(,)?) => {
        $(#[doc = $doc])+
        ///
        /// Class labels are sorted, and probability columns follow that order.
        /// Models fitted on a single class expose one probability column
        /// containing `1.0`.
        ///
        /// [`fit`](Self::fit) takes binary targets and keeps the asymmetric
        /// scalar-leaf representation FerricML froze first.
        /// [`fit_multiclass`](Self::fit_multiclass) takes any observed class
        /// set and fits natively multiclass trees whose ensemble probability is
        /// the **mean of the per-tree probability vectors** — soft averaging,
        /// not a majority vote of per-tree labels. The two are different models
        /// even on the same two-class data. Both persist, under one artifact
        /// kind that records which leaf arithmetic it holds.
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name {
            pub(crate) params: $params,
            pub(crate) core: $crate::ensemble::forest::model::ClassifierCore,
        }

        impl $name {
            /// Returns the feature width required by this model.
            #[inline]
            pub fn n_features_in(&self) -> usize {
                self.core.n_features_in
            }

            /// Returns the exact parameters used to fit this model.
            #[inline]
            pub fn get_params(&self) -> &$params {
                &self.params
            }

            /// Returns sorted class labels observed during fitting.
            #[inline]
            pub fn classes(&self) -> &[u8] {
                &self.core.classes
            }

            /// Fits a binary classifier over `0`/`1` targets.
            ///
            /// This is the asymmetric scalar-leaf fit: each tree stores the
            /// probability of class `1` and the ensemble averages that scalar.
            pub fn fit(
                data: &$crate::data::MatrixView<'_>,
                targets: &$crate::data::BinaryTargets,
                params: $params,
            ) -> Result<Self, $crate::api::ModelError> {
                Self::fit_binary(data, targets, None, params)
            }

            /// Fits a binary classifier with per-row sample weights.
            ///
            /// A weight scales the row's contribution to every impurity and
            /// leaf statistic, and composes with the bootstrap replication
            /// count. Weights of exactly one reproduce [`Self::fit`] bit for
            /// bit, and an integer weight is the same fit as repeating that row
            /// that many times.
            pub fn fit_weighted(
                data: &$crate::data::MatrixView<'_>,
                targets: &$crate::data::BinaryTargets,
                sample_weights: &$crate::data::SampleWeights,
                params: $params,
            ) -> Result<Self, $crate::api::ModelError> {
                Self::fit_binary(data, targets, Some(sample_weights), params)
            }

            fn fit_binary(
                data: &$crate::data::MatrixView<'_>,
                targets: &$crate::data::BinaryTargets,
                sample_weights: Option<&$crate::data::SampleWeights>,
                params: $params,
            ) -> Result<Self, $crate::api::ModelError> {
                use $crate::ensemble::forest::{model, training};
                let config = training::ForestConfig::from(&params);
                model::validate_common(data, targets.as_slice().len(), sample_weights, &config)?;
                for (index, &value) in targets.as_slice().iter().enumerate() {
                    if value > 1 {
                        return Err($crate::api::ModelError::InvalidBinaryTarget { index, value });
                    }
                }
                let saw_zero = targets.as_slice().contains(&0);
                let saw_one = targets.as_slice().contains(&1);
                let classes = match (saw_zero, saw_one) {
                    (true, true) => vec![0, 1],
                    (true, false) => vec![0],
                    (false, true) => vec![1],
                    (false, false) => unreachable!("non-empty validated binary targets"),
                };
                let trees = training::train_forest(
                    data,
                    targets.as_slice(),
                    sample_weights.map($crate::data::SampleWeights::as_slice),
                    &config,
                    $crate::tree::Classification,
                )?;
                Ok(Self {
                    params,
                    core: model::ClassifierCore {
                        n_features_in: data.columns(),
                        classes,
                        forest: model::Forest::Binary(trees),
                    },
                })
            }

            /// Fits a natively multiclass classifier over any observed class
            /// set.
            ///
            /// Each tree splits on multiclass Gini impurity and stores one
            /// probability per class at every leaf. The ensemble probability is
            /// the **mean of the per-tree probability vectors**, which is a
            /// strictly different rule from a majority vote over per-tree
            /// labels — soft averaging produces values a vote cannot, and the
            /// two disagree on real data.
            ///
            /// A single observed class is accepted: the fit succeeds with one
            /// probability column containing `1.0`, matching the single-class
            /// contract the binary entry point already has. Two observed
            /// classes are also accepted and produce a vector-leaf model, which
            /// is a different — and deliberately not interchangeable — model
            /// from [`Self::fit`] on the same data.
            pub fn fit_multiclass(
                data: &$crate::data::MatrixView<'_>,
                targets: &$crate::data::ClassTargets,
                params: $params,
            ) -> Result<Self, $crate::api::ModelError> {
                Self::fit_multiclass_internal(data, targets, None, params)
            }

            /// Fits a natively multiclass classifier with per-row sample
            /// weights.
            ///
            /// The weight scales the row's contribution to the multiclass Gini
            /// statistics and to every leaf distribution, exactly as it does for
            /// the binary fit.
            pub fn fit_multiclass_weighted(
                data: &$crate::data::MatrixView<'_>,
                targets: &$crate::data::ClassTargets,
                sample_weights: &$crate::data::SampleWeights,
                params: $params,
            ) -> Result<Self, $crate::api::ModelError> {
                Self::fit_multiclass_internal(data, targets, Some(sample_weights), params)
            }

            fn fit_multiclass_internal(
                data: &$crate::data::MatrixView<'_>,
                targets: &$crate::data::ClassTargets,
                sample_weights: Option<&$crate::data::SampleWeights>,
                params: $params,
            ) -> Result<Self, $crate::api::ModelError> {
                use $crate::ensemble::forest::{model, training};
                let config = training::ForestConfig::from(&params);
                model::validate_common(data, targets.len(), sample_weights, &config)?;
                let classes = targets.classes().to_vec();
                let class_of_row = targets
                    .as_slice()
                    .iter()
                    .map(|&label| {
                        targets
                            .class_index(label)
                            .expect("every target label is an observed class")
                    })
                    .collect::<Vec<_>>();
                let trees = training::train_class_forest(
                    data,
                    &class_of_row,
                    classes.len(),
                    sample_weights.map($crate::data::SampleWeights::as_slice),
                    &config,
                )?;
                Ok(Self {
                    params,
                    core: model::ClassifierCore {
                        n_features_in: data.columns(),
                        classes,
                        forest: model::Forest::Multiclass(trees),
                    },
                })
            }

            /// Predicts the class label for one sample.
            pub fn predict_one(&self, row: &[f32]) -> Result<u8, $crate::api::ModelError> {
                self.core.predict_one(row)
            }

            /// Predicts one label per row, allocating the output vector.
            pub fn predict(
                &self,
                data: &$crate::data::MatrixView<'_>,
            ) -> Result<Vec<u8>, $crate::api::ModelError> {
                self.core.predict(data)
            }

            /// Predicts one label per row without allocating.
            pub fn predict_into(
                &self,
                data: &$crate::data::MatrixView<'_>,
                output: &mut [u8],
            ) -> Result<(), $crate::api::ModelError> {
                self.core.predict_into(data, output)
            }

            /// Predicts probabilities for one sample in [`Self::classes`] order.
            pub fn predict_proba_one(
                &self,
                row: &[f32],
            ) -> Result<Vec<f32>, $crate::api::ModelError> {
                self.core.predict_proba_one(row)
            }

            /// Predicts probabilities for one sample into caller-owned storage.
            pub fn predict_proba_one_into(
                &self,
                row: &[f32],
                output: &mut [f32],
            ) -> Result<(), $crate::api::ModelError> {
                self.core.predict_proba_one_into(row, output)
            }

            /// Predicts row-major probabilities, allocating
            /// `rows * classes().len()` values.
            pub fn predict_proba(
                &self,
                data: &$crate::data::MatrixView<'_>,
            ) -> Result<Vec<f32>, $crate::api::ModelError> {
                <Self as $crate::api::ProbabilisticClassifier>::predict_proba(self, data)
            }

            /// Predicts row-major probabilities without allocating.
            pub fn predict_proba_into(
                &self,
                data: &$crate::data::MatrixView<'_>,
                output: &mut [f32],
            ) -> Result<(), $crate::api::ModelError> {
                self.core.predict_proba_into(data, output)
            }

            /// Returns the requested fitted-class probability for one sample.
            pub fn predict_class_proba_one(
                &self,
                row: &[f32],
                class: u8,
            ) -> Result<f32, $crate::api::ModelError> {
                self.core.predict_class_proba_one(row, class)
            }

            /// Predicts one fitted-class probability column, allocating the
            /// output.
            pub fn predict_class_proba(
                &self,
                data: &$crate::data::MatrixView<'_>,
                class: u8,
            ) -> Result<Vec<f32>, $crate::api::ModelError> {
                <Self as $crate::api::ProbabilisticClassifier>::predict_class_proba(
                    self, data, class,
                )
            }

            /// Predicts one fitted-class probability column without allocating.
            pub fn predict_class_proba_into(
                &self,
                data: &$crate::data::MatrixView<'_>,
                class: u8,
                output: &mut [f32],
            ) -> Result<(), $crate::api::ModelError> {
                self.core.predict_class_proba_into(data, class, output)
            }

            /// Returns the positive-class probability for one sample.
            ///
            /// Defined only for a binary fit. A multiclass fit has no positive
            /// class and reports [`ModelError::MulticlassOutput`] instead of
            /// returning one column of a vector that has no distinguished
            /// member.
            ///
            /// [`ModelError::MulticlassOutput`]: crate::api::ModelError::MulticlassOutput
            pub fn predict_positive_proba(
                &self,
                row: &[f32],
            ) -> Result<f32, $crate::api::ModelError> {
                self.core.predict_positive_proba(row)
            }

            /// Predicts the positive-class probability for every row without
            /// allocating. `output.len()` must equal the number of input rows.
            pub fn predict_positive_proba_into(
                &self,
                data: &$crate::data::MatrixView<'_>,
                output: &mut [f32],
            ) -> Result<(), $crate::api::ModelError> {
                self.core.predict_positive_proba_into(data, output)
            }

            /// Encodes the fitted parameters, class list, and canonical logical
            /// trees.
            ///
            /// The two fits are different models with different leaf
            /// arithmetic, so the payload records which one it holds and the
            /// reader refuses to build the other. A binary fit reuses the scalar
            /// logical-tree records unchanged — the same codec the regressor and
            /// the boosted trees use. A multiclass fit writes the same topology
            /// records with a reserved zero where a scalar leaf carries its
            /// value, followed by that tree's leaf distributions in pre-order
            /// leaf rank. Storing rank rather than the runtime leaf ordinal is
            /// what keeps the encoding unique: the ordinals could be permuted
            /// together with the block to name one model twice.
            pub fn to_artifact(
                &self,
                schema: [u8; 32],
            ) -> Result<Vec<u8>, $crate::artifact::ArtifactError> {
                $crate::ensemble::forest::codec::encode_classifier(
                    $kind,
                    &self.params.artifact_fields(self.core.n_features_in),
                    &self.core.classes,
                    &self.core.forest,
                    schema,
                )
            }

            /// Decodes and revalidates a classifier before building runtime
            /// state.
            ///
            /// Counts, parameters, and the class list are checked before any
            /// tree is read, every decoded tree re-enters the same topology
            /// validator fitting uses, and every decoded probability re-enters
            /// the same class-topology invariant a fitted tree satisfies.
            pub fn from_artifact(
                bytes: &[u8],
                schema: [u8; 32],
            ) -> Result<Self, $crate::artifact::ArtifactError> {
                let (fields, classes, forest) =
                    $crate::ensemble::forest::codec::decode_classifier($kind, bytes, schema)?;
                Ok(Self {
                    params: $params::from_artifact_fields(&fields),
                    core: $crate::ensemble::forest::model::ClassifierCore {
                        n_features_in: fields.n_features_in,
                        classes,
                        forest,
                    },
                })
            }

            /// The scalar trees of a binary fit, for in-crate structural tests.
            #[cfg(test)]
            pub(crate) fn binary_trees(&self) -> &[$crate::tree::PackedTree] {
                self.core.forest.binary_trees()
            }
        }

        impl $crate::api::Estimator for $name {
            fn n_features_in(&self) -> usize {
                self.core.n_features_in
            }
        }

        impl $crate::api::Classifier for $name {
            fn classes(&self) -> &[u8] {
                &self.core.classes
            }

            fn predict_into(
                &self,
                data: &$crate::data::MatrixView<'_>,
                output: &mut [u8],
            ) -> Result<(), $crate::api::ModelError> {
                self.core.predict_into(data, output)
            }
        }

        impl $crate::api::ProbabilisticClassifier for $name {
            fn predict_proba_into(
                &self,
                data: &$crate::data::MatrixView<'_>,
                output: &mut [f32],
            ) -> Result<(), $crate::api::ModelError> {
                self.core.predict_proba_into(data, output)
            }

            fn predict_class_proba_into(
                &self,
                data: &$crate::data::MatrixView<'_>,
                class: u8,
                output: &mut [f32],
            ) -> Result<(), $crate::api::ModelError> {
                self.core.predict_class_proba_into(data, class, output)
            }
        }

        impl $crate::api::HasParams for $name {
            type Params = $params;

            fn get_params(&self) -> &Self::Params {
                &self.params
            }
        }

        /// Declares weighted fitting, multiclass fitting, and persistence. The
        /// artifact covers *both* leaf representations, so the declaration holds
        /// for every fit this type offers rather than for one of its two entry
        /// points.
        impl $crate::api::HasCapabilities for $name {
            const CAPABILITIES: $crate::api::Capabilities = $crate::api::Capabilities::NONE
                .with_sample_weights(true)
                .with_artifact(true)
                .with_multiclass(true)
                .with_probability(true);
        }
    };
}

/// Generates one public ensemble regressor over the shared core.
macro_rules! forest_regressor {
    ($name:ident, $params:ident, $kind:expr, $objective:expr, $($doc:expr),+ $(,)?) => {
        $(#[doc = $doc])+
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name {
            pub(crate) params: $params,
            pub(crate) core: $crate::ensemble::forest::model::RegressorCore,
        }

        impl $name {
            /// Returns the feature width required by this model.
            #[inline]
            pub fn n_features_in(&self) -> usize {
                self.core.n_features_in
            }

            /// Returns the exact parameters used to fit this model.
            #[inline]
            pub fn get_params(&self) -> &$params {
                &self.params
            }

            /// Fits one ensemble of regression trees.
            pub fn fit(
                data: &$crate::data::MatrixView<'_>,
                targets: &$crate::data::RegressionTargets,
                params: $params,
            ) -> Result<Self, $crate::api::ModelError> {
                Self::fit_internal(data, targets, None, params)
            }

            /// Fits with per-row sample weights.
            ///
            /// A weight scales the row's contribution to the variance and leaf
            /// mean of every node it reaches, and composes with the bootstrap
            /// replication count. Weights of exactly one reproduce [`Self::fit`]
            /// bit for bit, and an integer weight is the same fit as repeating
            /// that row that many times.
            pub fn fit_weighted(
                data: &$crate::data::MatrixView<'_>,
                targets: &$crate::data::RegressionTargets,
                sample_weights: &$crate::data::SampleWeights,
                params: $params,
            ) -> Result<Self, $crate::api::ModelError> {
                Self::fit_internal(data, targets, Some(sample_weights), params)
            }

            fn fit_internal(
                data: &$crate::data::MatrixView<'_>,
                targets: &$crate::data::RegressionTargets,
                sample_weights: Option<&$crate::data::SampleWeights>,
                params: $params,
            ) -> Result<Self, $crate::api::ModelError> {
                use $crate::ensemble::forest::{model, training};
                let config = training::ForestConfig::from(&params);
                model::validate_common(data, targets.as_slice().len(), sample_weights, &config)?;
                for (index, value) in targets.as_slice().iter().enumerate() {
                    if !value.is_finite() {
                        return Err($crate::api::ModelError::NonFiniteTarget { index });
                    }
                }
                let trees = training::train_forest(
                    data,
                    targets.as_slice(),
                    sample_weights.map($crate::data::SampleWeights::as_slice),
                    &config,
                    $objective,
                )?;
                Ok(Self {
                    params,
                    core: model::RegressorCore {
                        n_features_in: data.columns(),
                        trees,
                    },
                })
            }

            /// Predicts one regression value for one sample.
            pub fn predict_one(&self, row: &[f32]) -> Result<f32, $crate::api::ModelError> {
                self.core.predict_one(row)
            }

            /// Predicts one value per row, allocating the output vector.
            pub fn predict(
                &self,
                data: &$crate::data::MatrixView<'_>,
            ) -> Result<Vec<f32>, $crate::api::ModelError> {
                <Self as $crate::api::Regressor>::predict(self, data)
            }

            /// Predict every row without allocating. `output.len()` must equal
            /// the number of input rows.
            pub fn predict_into(
                &self,
                data: &$crate::data::MatrixView<'_>,
                output: &mut [f32],
            ) -> Result<(), $crate::api::ModelError> {
                self.core.predict_into(data, output)
            }

            /// Encodes the fitted parameters and canonical logical trees.
            ///
            /// The private packed inference layout is never serialized. Each
            /// tree is expanded into stable logical records first, so the
            /// compact runtime representation stays free to change.
            pub fn to_artifact(
                &self,
                schema: [u8; 32],
            ) -> Result<Vec<u8>, $crate::artifact::ArtifactError> {
                $crate::ensemble::forest::codec::encode_regressor(
                    $kind,
                    &self.params.artifact_fields(self.core.n_features_in),
                    &self.core.trees,
                    schema,
                )
            }

            /// Decodes and revalidates logical trees before building runtime
            /// state.
            ///
            /// Counts and parameters are checked before any tree is read, and
            /// each decoded tree is rebuilt through the same topology validator
            /// that fitting uses, so the encoded bytes are never trusted.
            pub fn from_artifact(
                bytes: &[u8],
                schema: [u8; 32],
            ) -> Result<Self, $crate::artifact::ArtifactError> {
                let (fields, trees) =
                    $crate::ensemble::forest::codec::decode_regressor($kind, bytes, schema)?;
                Ok(Self {
                    params: $params::from_artifact_fields(&fields),
                    core: $crate::ensemble::forest::model::RegressorCore {
                        n_features_in: fields.n_features_in,
                        trees,
                    },
                })
            }
        }

        impl $crate::api::Estimator for $name {
            fn n_features_in(&self) -> usize {
                self.core.n_features_in
            }
        }

        impl $crate::api::Regressor for $name {
            fn predict_into(
                &self,
                data: &$crate::data::MatrixView<'_>,
                output: &mut [f32],
            ) -> Result<(), $crate::api::ModelError> {
                self.core.predict_into(data, output)
            }
        }

        impl $crate::api::HasParams for $name {
            type Params = $params;

            fn get_params(&self) -> &Self::Params {
                &self.params
            }
        }

        impl $crate::api::HasCapabilities for $name {
            const CAPABILITIES: $crate::api::Capabilities = $crate::api::Capabilities::NONE
                .with_sample_weights(true)
                .with_artifact(true);
        }
    };
}

pub(crate) use {forest_classifier, forest_regressor};
