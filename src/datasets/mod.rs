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
//! # Frozen presets
//!
//! [`ReferenceQuality`] reproduces the design matrices and targets FerricML's
//! conformance fixtures were recorded against. Those are not recipes a caller
//! would choose — they are transcriptions, pinned by value, and the module
//! documentation in `presets.rs` says which parts of the arithmetic are
//! load-bearing.

mod benchmarks;
mod dataset;
mod error;
mod presets;
mod recipe;
mod source;

#[cfg(test)]
mod tests;

pub use benchmarks::{BenchmarkFixture, BenchmarkLane};
pub use dataset::{Dataset, Target, Truth};
pub use error::DatasetError;
pub use presets::{ReferenceLane, ReferenceQuality};
pub use recipe::{Recipe, Source};
