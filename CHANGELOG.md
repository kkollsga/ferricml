# Changelog

All notable changes to FerricML are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Validated sample weights, weighted logistic fitting, and allocation-free raw
  logistic decision scores.
- Bounded version-2 model artifacts with independent payload/component
  versions while retaining strict legacy logistic decoding.
- Dense weighted ordinary least-squares regression with deterministic
  minimum-norm SVD solutions and schema-bound artifacts.

### Fixed

- Honor `LogisticRegressionParams::with_fit_intercept(false)` without silently
  centering features or fitting a folded intercept.

## [0.1.0] - 2026-07-22

### Added

- Validated dense `f32` matrices and classification/regression targets.
- Deterministic random-forest classifiers and regressors with allocation-free
  batch inference.
- Deterministic logistic regression with scikit-style parameters and
  allocation-free prediction.
- Versioned, checksummed logistic-regression artifacts with strict feature
  schema verification.
- Scikit-learn-compatible estimator and prediction semantics for the supported
  API subset.
- Generic static-dispatch pipeline and transformer growth seams.
- Frozen correctness fixtures and an on-demand Rust implementation benchmark.

[Unreleased]: https://github.com/kkollsga/ferricml/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kkollsga/ferricml/releases/tag/v0.1.0
