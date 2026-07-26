//! Lean, pure-Rust classical machine learning.
//!
//! FerricML focuses first on linear models and random forests, with stable
//! estimator semantics and fast, allocation-conscious inference.

pub mod api;
pub mod artifact;
pub mod calibration;
pub mod data;
pub mod dummy;
pub mod ensemble;
pub mod inspection;
pub mod linear_model;
mod loss;
pub mod metrics;
pub mod model_selection;
mod numeric;
mod optimize;
pub mod pipeline;
pub mod preprocessing;
pub mod ranking;
pub mod tree;

/// The narrative documentation pages, compiled as doctests.
///
/// Every Rust sample on FerricML's documentation site is a real, executing
/// test. The pages under `docs/` are that site's source, and each one is
/// included here, so `cargo test` compiles and runs every Rust fence they
/// contain. A sample that stops compiling — or stops producing the value it
/// claims — fails the ordinary gate alongside everything else. A sample that
/// has silently rotted is worse than no sample, so none of them can.
///
/// One consequence constrains how these pages are written: rustdoc compiles an
/// **indented** block as Rust, so a carried page cannot use MkDocs Material's
/// `!!!` admonition syntax, whose body is indented four spaces. Carried pages
/// use blockquotes instead, which also render correctly when the same file is
/// read directly on GitHub. The constraint is self-enforcing — an indented
/// block fails `cargo test` immediately — but the failure names a Rust parse
/// error rather than the real cause, so it is written down here.
///
/// Three properties make this safe to rely on:
///
/// - `cfg(doctest)` is set only while rustdoc collects doctests, so these
///   modules never reach `cargo doc`, docs.rs, or the exact public-API
///   snapshot. Adding a page changes no public surface.
/// - A renamed or deleted page is a compile error, because `include_str!`
///   cannot find it.
/// - A page that is never added here would otherwise be invisible, which is
///   the one failure the compiler cannot catch. `tests/doc_examples.rs` walks
///   `docs/` and fails if any page is missing from the list below.
#[cfg(doctest)]
mod doc_pages {
    #[doc = include_str!("../docs/index.md")]
    mod index {}

    #[doc = include_str!("../docs/guide/quickstart.md")]
    mod guide_quickstart {}

    #[doc = include_str!("../docs/guide/data.md")]
    mod guide_data {}

    #[doc = include_str!("../docs/guide/linear-models.md")]
    mod guide_linear_models {}

    #[doc = include_str!("../docs/api-and-growth.md")]
    mod api_and_growth {}

    #[doc = include_str!("../docs/artifact-envelope.md")]
    mod artifact_envelope {}

    #[doc = include_str!("../docs/determinism.md")]
    mod determinism {}

    #[doc = include_str!("../docs/evaluation-and-model-selection.md")]
    mod evaluation_and_model_selection {}

    #[doc = include_str!("../docs/histogram-gradient-boosting.md")]
    mod histogram_gradient_boosting {}

    #[doc = include_str!("../docs/model-performance.md")]
    mod model_performance {}

    #[doc = include_str!("../docs/pairwise-ranking.md")]
    mod pairwise_ranking {}

    #[doc = include_str!("../docs/reference-semantics.md")]
    mod reference_semantics {}

    #[doc = include_str!("../docs/security.md")]
    mod security {}
}
