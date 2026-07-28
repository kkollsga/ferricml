//! Synthetic dataset recipes, deterministic sources, and ground truth.
//!
//! FerricML generates the data it measures itself against. Every dataset this
//! module produces is a pure function of a [`Recipe`]: the same recipe gives the
//! same bytes in this process, in the next one, and on another machine, because
//! no source here evaluates a transcendental function and none of them consults
//! anything outside the recipe.
//!
//! ```
//! use ferricml::datasets::Recipe;
//!
//! let recipe = Recipe::seeded(256, 12, 11)?;
//! let dataset = recipe.generate();
//!
//! assert_eq!(dataset.features().rows(), 256);
//! // Regenerating gives the same bytes, and the digest says which recipe they
//! // came from without holding the data itself.
//! assert_eq!(dataset.features(), recipe.generate().features());
//! assert_eq!(dataset.spec_digest(), recipe.spec_digest());
//! # Ok::<(), ferricml::datasets::DatasetError>(())
//! ```
//!
//! # Behind the non-default `datasets` feature
//!
//! Generation is not part of the estimator surface, and `default = []` is a
//! product boundary in this crate: a consumer fitting models pays nothing for a
//! generator it never calls, and docs.rs renders the estimator vocabulary rather
//! than this one. The feature carries no dependency of its own — the streams
//! come from the crate's existing generator kernels and the digest from the
//! `sha2` already in the graph — so switching it on changes what is visible,
//! not what is built.
//!
//! # Why a seed here is not a seed an estimator sees
//!
//! [`Recipe::seeded`] mixes a caller's number into a stream state that is
//! disjoint from every stream an estimator seeded with the same number draws
//! from. Without that, a design matrix generated from seed `s` and a forest
//! fitted with seed `s` would walk one sequence, and the data would be
//! correlated with the model's own randomness. The derivation lives in
//! `src/numeric/rng.rs` with the generator it derives from, and the disjointness
//! is asserted there against pinned probes rather than argued.
//!
//! A recipe that has to reproduce an *already recorded* stream names the raw
//! state instead, through [`Source::Sampled`]. Both spellings exist because a
//! frozen fixture and a new experiment want opposite things from the same
//! number: the fixture wants the stream it was recorded against, and the
//! experiment wants one nothing else is using.
//!
//! # Ground truth is the point
//!
//! Every generator FerricML replaced could say where two implementations
//! disagree and none could say which was closer to right, because none of them
//! kept what the answer should have been. [`Truth`] is that record. It is
//! `DesignOnly` until a task family draws targets over the design, and it says
//! so with a variant rather than with an empty coefficient vector, because "no
//! correct answer exists" and "the correct answer is zero" are different claims.
//!
//! The absorbed lanes are the counter-example that makes the distinction worth
//! carrying: they draw real targets and still have no recorded truth, because
//! they were written to compare two implementations against each other rather
//! than against a right answer. They report `Truth::Unrecorded`, which is a
//! third statement again.
//!
//! A [`Task`] family records what it actually knows and nothing more. A linear
//! family knows its coefficients; a nonlinear one knows only its conditional
//! mean, because no coefficient vector produces it; a binary family knows the
//! Bayes probability behind every label it drew. None of them reaches for
//! [`Truth::Unrecorded`], which stays what it was: a statement about the
//! absorbed lanes.
//!
//! # Two determinism envelopes, declared rather than assumed
//!
//! Every source in this module is transcendental-free and therefore bit-exact
//! on every target, which is what protects the frozen fixtures. Most task
//! families are not: a Bayes probability is a logistic function, a log-link mean
//! is an exponential, a requested condition number is a real power, and no libm
//! rounds any of those correctly. [`Portability`] is that distinction as a value
//! rather than a paragraph — [`Task::portability`] and [`Recipe::portability`]
//! report which of the two statements a caller is entitled to, and the families
//! are held to matching evidence: a bit-exact family is pinned by literal
//! values, a per-runner one by properties and tolerances.
//!
//! # Contamination is orthogonal to the task
//!
//! [`Contamination`] carries label noise, outliers, heavy tails,
//! heteroscedasticity, duplicated rows, constant columns, collinear pairs and a
//! per-column scale spread. It composes with every family, so a robustness sweep
//! holds the task fixed and moves the contamination. A knob the current task
//! cannot carry is refused at the constructor with a typed error rather than
//! silently ignored, because a sweep that reported a model robust to a
//! contamination it never received would be worse than a build failure.
//!
//! # Frozen presets
//!
//! [`ReferenceQuality`] reproduces the design matrices and targets FerricML's
//! conformance fixtures were recorded against. Those are not recipes a caller
//! would choose — they are transcriptions, pinned by value, and the module
//! documentation in `presets.rs` says which parts of the arithmetic are
//! load-bearing.

mod benchmarks;
mod contamination;
mod dataset;
mod error;
mod presets;
mod recipe;
mod source;
mod task;

#[cfg(test)]
mod family_tests;
#[cfg(test)]
mod tests;

pub use benchmarks::{BenchmarkFixture, BenchmarkLane};
pub use contamination::{Contamination, WeightPattern};
pub use dataset::{Dataset, Target, Truth};
pub use error::{DatasetError, Parameter};
pub use presets::{ReferenceLane, ReferenceQuality};
pub use recipe::{Recipe, Source};
pub use task::{BinaryKind, GlmLink, NonlinearKind, Portability, Task};
