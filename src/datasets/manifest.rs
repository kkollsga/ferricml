//! The exchange manifest: one dataset's recipe, digests and array table, as
//! canonical text.
//!
//! # Why this file writes and parses JSON by hand
//!
//! `default = []` is a product boundary in this crate, and the `datasets`
//! feature carries no dependency of its own. There is no `serde` in the graph
//! and no JSON library, so a manifest a Python reader can open with
//! `json.load` has to be written and read here. That is a deliberate cost, and
//! it buys something beyond the dependency count: the reader below accepts
//! **exactly** the schema the writer emits, in exactly its order, so a manifest
//! that is not this crate's own is refused at the first byte that differs
//! rather than coerced into a partially understood recipe.
//!
//! The output is canonical: two materializations of one recipe produce
//! byte-identical text, because every field is written unconditionally, in a
//! fixed order, with a fixed indentation. `manifest_text_is_canonical` in
//! `exchange_tests.rs` is the assertion, not this paragraph.
//!
//! # Floats round-trip exactly, and that is a property rather than a hope
//!
//! Every float here is written with Rust's `Display`, which emits the shortest
//! decimal string that parses back to the *same* `f32`, and read back with
//! `str::parse`, which is correctly rounded. So the text is a faithful
//! rendering of the bits, and `every_float_field_round_trips_through_the_text`
//! pins that against subnormals, both zeros, and the extremes.
//!
//! Recipes refuse a non-finite parameter at their constructor, so no `inf` or
//! `NaN` — neither of which JSON can spell — ever reaches this writer.
//!
//! # The parser cannot reserve from a length field
//!
//! Reservation from an attacker-controlled count is the defect class
//! `src/artifact/` was hardened against, and this parser is built so the
//! question does not arise: a JSON array is self-delimiting, so the array table
//! declares no count to reserve from, and every string is *borrowed* out of the
//! manifest text rather than copied into a fresh allocation. The declared
//! lengths this format does carry are the array table's, and they are spent in
//! `exchange.rs` against an `ExchangeCursor` that clamps every reservation to
//! the bytes actually present.

use super::contamination::{Contamination, WeightPattern};
use super::error::ExchangeError;
use super::recipe::{Recipe, Source};
use super::structural::{ClassBalance, ClassGeometry, GroupPattern};
use super::task::{BinaryKind, Family, GlmLink, NonlinearKind, Portability, Task};

/// Container format version the writer emits and the reader accepts.
///
/// One number for the whole container — manifest schema and array file
/// together — because the two are written as a pair and are meaningless apart.
/// A reader that meets a version it does not know refuses rather than guessing
/// which half changed.
pub(super) const FORMAT_VERSION: u32 = 1;

/// Largest array table this reader will assemble.
///
/// The families produce at most a dozen arrays, so this is a ceiling rather
/// than a budget: it exists so a manifest cannot make the reader build an
/// arbitrarily long table out of arbitrarily many small objects.
pub(super) const MAX_ARRAYS: usize = 64;

/// Longest string this reader accepts inside a manifest.
///
/// Array names are short identifiers and digests are 64 hex characters, so
/// nothing legitimate comes close.
const MAX_STRING_BYTES: usize = 128;

/// Longest number literal this reader accepts.
///
/// A `u64` is 20 digits and an `f32` written by `Display` never exceeds a
/// couple of dozen characters including an exponent.
const MAX_NUMBER_BYTES: usize = 48;

/// One row of the array table.
///
/// The table is what makes the array file readable without this crate: every
/// entry says where an array starts, how long it is, and how to interpret its
/// bytes, so a NumPy reader maps the file once and slices it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ArrayRecord {
    /// The array's name, unique within one container.
    pub(super) name: String,
    /// How to interpret the array's bytes.
    pub(super) dtype: super::exchange::ArrayDtype,
    /// Rows the array is laid out in, row-major.
    pub(super) rows: usize,
    /// Values per row.
    pub(super) columns: usize,
    /// Byte offset of the array's first element within the array file.
    pub(super) byte_offset: usize,
    /// Number of values, which is `rows * columns`.
    pub(super) len: usize,
}

/// A manifest's whole content.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Manifest {
    /// The recipe the data was generated from.
    pub(super) recipe: Recipe,
    /// The digest the recipe is claimed to have.
    ///
    /// Carried explicitly rather than recomputed on read, because the point of
    /// reading it is to *disagree* with the recipe when the file has been
    /// edited: a manifest whose recipe no longer hashes to this value is
    /// refused.
    pub(super) spec_digest: [u8; 32],
    /// The determinism envelope the recorded data was produced under.
    pub(super) portability: Portability,
    /// Name of the array file beside this manifest.
    pub(super) data_file: String,
    /// Length of the array file.
    pub(super) data_bytes: usize,
    /// SHA-256 of the array file.
    pub(super) data_digest: [u8; 32],
    /// Every array in the file, in the order they are laid out.
    pub(super) arrays: Vec<ArrayRecord>,
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Renders a manifest as canonical text.
pub(super) fn render(manifest: &Manifest) -> String {
    let mut text = String::new();
    let mut root = Object::open(&mut text, 0);
    root.u64("format", u64::from(FORMAT_VERSION));
    root.digest("spec_digest", &manifest.spec_digest);
    root.string("portability", portability_label(manifest.portability));
    root.key("recipe");
    render_recipe(root.text, &manifest.recipe, 1);
    root.key("data");
    {
        let mut data = Object::open(root.text, 1);
        data.string("file", &manifest.data_file);
        data.u64("bytes", manifest.data_bytes as u64);
        data.digest("digest", &manifest.data_digest);
        data.close();
    }
    root.key("arrays");
    render_arrays(root.text, &manifest.arrays, 1);
    root.close();
    text.push('\n');
    text
}

fn render_arrays(text: &mut String, arrays: &[ArrayRecord], depth: usize) {
    if arrays.is_empty() {
        text.push_str("[]");
        return;
    }
    text.push('[');
    for (index, array) in arrays.iter().enumerate() {
        if index > 0 {
            text.push(',');
        }
        text.push('\n');
        indent(text, depth + 1);
        let mut entry = Object::open(text, depth + 1);
        entry.string("name", &array.name);
        entry.string("dtype", array.dtype.label());
        entry.u64("rows", array.rows as u64);
        entry.u64("columns", array.columns as u64);
        entry.u64("byte_offset", array.byte_offset as u64);
        entry.u64("len", array.len as u64);
        entry.close();
    }
    text.push('\n');
    indent(text, depth);
    text.push(']');
}

fn render_recipe(text: &mut String, recipe: &Recipe, depth: usize) {
    let mut object = Object::open(text, depth);
    object.u64("rows", recipe.rows() as u64);
    object.u64("columns", recipe.columns() as u64);
    object.key("source");
    render_source(object.text, recipe.source(), depth + 1);
    object.key("task");
    match recipe.task() {
        None => object.text.push_str("null"),
        Some(task) => render_task(object.text, task, depth + 1),
    }
    object.key("contamination");
    render_contamination(object.text, recipe.contamination(), depth + 1);
    object.key("weights");
    match recipe.weight_pattern() {
        None => object.text.push_str("null"),
        Some(pattern) => render_weights(object.text, pattern, depth + 1),
    }
    object.key("groups");
    match recipe.group_pattern() {
        None => object.text.push_str("null"),
        Some(pattern) => render_groups(object.text, pattern, depth + 1),
    }
    object.close();
}

fn render_source(text: &mut String, source: Source, depth: usize) {
    let mut object = Object::open(text, depth);
    match source {
        Source::Sampled { state } => {
            object.string("kind", "sampled");
            object.u64("state", state);
        }
        Source::Lattice {
            row_stride,
            column_stride,
            modulus,
        } => {
            object.string("kind", "lattice");
            object.u64("row_stride", row_stride);
            object.u64("column_stride", column_stride);
            object.u64("modulus", modulus);
        }
        Source::Xorshift32 { state } => {
            object.string("kind", "xorshift32");
            object.u64("state", u64::from(state));
        }
    }
    object.close();
}

fn render_task(text: &mut String, task: Task, depth: usize) {
    let mut object = Object::open(text, depth);
    // The kind is the family's own recorded label rather than a second
    // spelling of the variant, so a manifest names a problem the same way a
    // benchmark row and an accuracy report do.
    object.string("kind", task.family().label());
    match task {
        Task::LinearRegression {
            informative,
            coefficient_scale,
            intercept,
            noise_scale,
        } => {
            object.u64("informative", informative as u64);
            object.f32("coefficient_scale", coefficient_scale);
            object.f32("intercept", intercept);
            object.f32("noise_scale", noise_scale);
        }
        Task::NonlinearRegression { kind, noise_scale } => {
            object.string("shape", nonlinear_label(kind));
            object.f32("noise_scale", noise_scale);
        }
        Task::GlmRegression {
            link,
            informative,
            coefficient_scale,
            intercept,
            dispersion,
        } => {
            object.string("link", link_label(link));
            object.u64("informative", informative as u64);
            object.f32("coefficient_scale", coefficient_scale);
            object.f32("intercept", intercept);
            object.f32("dispersion", dispersion);
        }
        Task::IllConditioned {
            condition_number,
            rank,
            coefficient_scale,
            noise_scale,
        } => {
            object.f32("condition_number", condition_number);
            object.u64("rank", rank as u64);
            object.f32("coefficient_scale", coefficient_scale);
            object.f32("noise_scale", noise_scale);
        }
        Task::LinearBinary {
            informative,
            separation,
            prevalence,
        } => {
            object.u64("informative", informative as u64);
            object.f32("separation", separation);
            object.f32("prevalence", prevalence);
        }
        Task::NonlinearBinary {
            kind,
            separation,
            prevalence,
        } => {
            object.string("boundary", binary_label(kind));
            object.f32("separation", separation);
            object.f32("prevalence", prevalence);
        }
        Task::Multiclass {
            classes,
            balance,
            geometry,
            separation,
        } => {
            let (balance_label, ratio) = match balance {
                ClassBalance::Balanced => ("balanced", 1.0_f32),
                ClassBalance::Imbalanced { ratio } => ("imbalanced", ratio),
            };
            object.u64("classes", classes as u64);
            object.string("balance", balance_label);
            object.f32("balance_ratio", ratio);
            object.string("geometry", geometry_label(geometry));
            object.f32("separation", separation);
        }
        Task::Clustered { blobs, spread } => {
            object.u64("blobs", blobs as u64);
            object.f32("spread", spread);
        }
        Task::TimeOrdered {
            informative,
            coefficient_scale,
            drift,
            intercept,
            noise_scale,
        } => {
            object.u64("informative", informative as u64);
            object.f32("coefficient_scale", coefficient_scale);
            object.f32("drift", drift);
            object.f32("intercept", intercept);
            object.f32("noise_scale", noise_scale);
        }
        Task::Ranking {
            queries,
            docs_per_query,
            grades,
            informative,
            coefficient_scale,
        } => {
            object.u64("queries", queries as u64);
            object.u64("docs_per_query", docs_per_query as u64);
            object.u64("grades", grades as u64);
            object.u64("informative", informative as u64);
            object.f32("coefficient_scale", coefficient_scale);
        }
    }
    object.close();
}

fn render_contamination(text: &mut String, contamination: Contamination, depth: usize) {
    let mut object = Object::open(text, depth);
    object.f32("label_noise", contamination.label_noise());
    object.f32("outlier_fraction", contamination.outlier_fraction());
    object.f32("heavy_tail", contamination.heavy_tail());
    object.f32("heteroscedastic", contamination.heteroscedastic());
    object.f32("duplicate_rows", contamination.duplicate_rows());
    object.u64("constant_columns", contamination.constant_columns() as u64);
    object.u64("collinear_pairs", contamination.collinear_pairs() as u64);
    object.f32("feature_scale_spread", contamination.feature_scale_spread());
    object.close();
}

fn render_weights(text: &mut String, pattern: WeightPattern, depth: usize) {
    let mut object = Object::open(text, depth);
    match pattern {
        WeightPattern::Uniform => object.string("kind", "uniform"),
        WeightPattern::Ramp { low, high } => {
            object.string("kind", "ramp");
            object.f32("low", low);
            object.f32("high", high);
        }
        WeightPattern::Alternating { first, second } => {
            object.string("kind", "alternating");
            object.f32("first", first);
            object.f32("second", second);
        }
        WeightPattern::ClassBalanced => object.string("kind", "class-balanced"),
    }
    object.close();
}

fn render_groups(text: &mut String, pattern: GroupPattern, depth: usize) {
    let mut object = Object::open(text, depth);
    match pattern {
        GroupPattern::RoundRobin { groups } => {
            object.string("kind", "round-robin");
            object.u64("groups", groups as u64);
        }
        GroupPattern::Contiguous { groups } => {
            object.string("kind", "contiguous");
            object.u64("groups", groups as u64);
        }
        GroupPattern::Unbalanced { groups, ratio } => {
            object.string("kind", "unbalanced");
            object.u64("groups", groups as u64);
            object.f32("ratio", ratio);
        }
    }
    object.close();
}

/// An object under construction, which owns the comma and indentation rules so
/// no call site has to remember which of its fields is the last one.
struct Object<'a> {
    text: &'a mut String,
    depth: usize,
    written: bool,
}

impl<'a> Object<'a> {
    fn open(text: &'a mut String, depth: usize) -> Self {
        text.push('{');
        Self {
            text,
            depth,
            written: false,
        }
    }

    fn key(&mut self, name: &str) {
        if self.written {
            self.text.push(',');
        }
        self.text.push('\n');
        indent(self.text, self.depth + 1);
        self.text.push('"');
        self.text.push_str(name);
        self.text.push_str("\": ");
        self.written = true;
    }

    fn u64(&mut self, name: &str, value: u64) {
        self.key(name);
        self.text.push_str(&value.to_string());
    }

    fn f32(&mut self, name: &str, value: f32) {
        self.key(name);
        self.text.push_str(&value.to_string());
    }

    fn string(&mut self, name: &str, value: &str) {
        self.key(name);
        self.text.push('"');
        self.text.push_str(value);
        self.text.push('"');
    }

    fn digest(&mut self, name: &str, value: &[u8; 32]) {
        self.key(name);
        self.text.push('"');
        for byte in value {
            self.text.push_str(&format!("{byte:02x}"));
        }
        self.text.push('"');
    }

    fn close(self) {
        if self.written {
            self.text.push('\n');
            indent(self.text, self.depth);
        }
        self.text.push('}');
    }
}

fn indent(text: &mut String, depth: usize) {
    for _ in 0..depth {
        text.push_str("  ");
    }
}

const fn portability_label(portability: Portability) -> &'static str {
    match portability {
        Portability::BitExact => "bit-exact",
        Portability::PerRunner => "per-runner",
    }
}

const fn nonlinear_label(kind: NonlinearKind) -> &'static str {
    match kind {
        NonlinearKind::Interaction => "interaction",
        NonlinearKind::Piecewise => "piecewise",
        NonlinearKind::Sinusoid => "sinusoid",
        NonlinearKind::Friedman => "friedman",
    }
}

const fn binary_label(kind: BinaryKind) -> &'static str {
    match kind {
        BinaryKind::Xor => "xor",
        BinaryKind::Moons => "moons",
        BinaryKind::Circles => "circles",
        BinaryKind::Checkerboard => "checkerboard",
    }
}

const fn link_label(link: GlmLink) -> &'static str {
    match link {
        GlmLink::LogCount => "log-count",
        GlmLink::LogPositive => "log-positive",
    }
}

const fn geometry_label(geometry: ClassGeometry) -> &'static str {
    match geometry {
        ClassGeometry::Blob => "blob",
        ClassGeometry::Hierarchical => "hierarchical",
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Parses a manifest, checking the recorded digest against the recipe it
/// describes.
///
/// The digest check is the whole reason the recipe is stored in full rather
/// than as a digest alone: an edited recipe still hashes to something, and what
/// makes the edit visible is that it no longer hashes to the value written
/// beside it.
pub(super) fn parse(text: &str) -> Result<Manifest, ExchangeError> {
    let mut cursor = Cursor::new(text);
    cursor.expect(b'{')?;

    cursor.key("format")?;
    let format = cursor.u64()?;
    if format != u64::from(FORMAT_VERSION) {
        return Err(ExchangeError::UnsupportedFormat { found: format });
    }
    cursor.comma()?;

    cursor.key("spec_digest")?;
    let spec_digest = cursor.digest()?;
    cursor.comma()?;

    cursor.key("portability")?;
    let portability = match cursor.string()? {
        "bit-exact" => Portability::BitExact,
        "per-runner" => Portability::PerRunner,
        _ => return Err(cursor.fault()),
    };
    cursor.comma()?;

    cursor.key("recipe")?;
    let recipe = cursor.recipe()?;
    cursor.comma()?;

    cursor.key("data")?;
    cursor.expect(b'{')?;
    cursor.key("file")?;
    let data_file = cursor.string()?.to_owned();
    cursor.comma()?;
    cursor.key("bytes")?;
    let data_bytes = cursor.usize()?;
    cursor.comma()?;
    cursor.key("digest")?;
    let data_digest = cursor.digest()?;
    cursor.expect(b'}')?;
    cursor.comma()?;

    cursor.key("arrays")?;
    let arrays = cursor.arrays()?;
    cursor.expect(b'}')?;
    cursor.end()?;

    if recipe.spec_digest() != spec_digest {
        return Err(ExchangeError::SpecDigestMismatch);
    }
    // The recorded envelope has to be the one the recipe actually produces.
    // Otherwise a manifest could promise bit-exact bytes for a family that
    // evaluates a transcendental, and a harness comparing two machines would
    // read the promise rather than the recipe.
    if recipe.portability() != portability {
        return Err(ExchangeError::SpecDigestMismatch);
    }

    Ok(Manifest {
        recipe,
        spec_digest,
        portability,
        data_file,
        data_bytes,
        data_digest,
        arrays,
    })
}

/// A position in the manifest text.
///
/// Whitespace-insensitive and structure-strict: it skips blanks between tokens
/// and refuses anything the writer above would not have produced. Every string
/// it yields is borrowed out of the input, so reading a manifest allocates
/// nothing per field.
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    const fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            offset: 0,
        }
    }

    /// The refusal this cursor reports, naming where it stopped.
    const fn fault(&self) -> ExchangeError {
        ExchangeError::MalformedManifest {
            offset: self.offset,
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(&byte) = self.bytes.get(self.offset) {
            if byte == b' ' || byte == b'\n' || byte == b'\t' || byte == b'\r' {
                self.offset += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_whitespace();
        self.bytes.get(self.offset).copied()
    }

    fn expect(&mut self, byte: u8) -> Result<(), ExchangeError> {
        if self.peek() != Some(byte) {
            return Err(self.fault());
        }
        self.offset += 1;
        Ok(())
    }

    fn comma(&mut self) -> Result<(), ExchangeError> {
        self.expect(b',')
    }

    /// Consumes `"name":`, refusing any other key.
    ///
    /// Refusing rather than searching is what makes the field order part of the
    /// format: a manifest with the right fields in another order is not one
    /// this writer produced.
    fn key(&mut self, name: &str) -> Result<(), ExchangeError> {
        let found = self.string()?;
        if found != name {
            return Err(self.fault());
        }
        self.expect(b':')
    }

    fn string(&mut self) -> Result<&'a str, ExchangeError> {
        self.expect(b'"')?;
        let start = self.offset;
        loop {
            let Some(&byte) = self.bytes.get(self.offset) else {
                return Err(self.fault());
            };
            if byte == b'"' {
                break;
            }
            // No escape is ever written — every string in this format is an
            // identifier, a label, or hex — so a backslash is a foreign
            // manifest rather than a string this reader has to unescape. A
            // control byte is refused for the same reason JSON refuses it.
            if byte == b'\\' || byte < 0x20 || self.offset - start >= MAX_STRING_BYTES {
                return Err(self.fault());
            }
            self.offset += 1;
        }
        let text =
            std::str::from_utf8(&self.bytes[start..self.offset]).map_err(|_| self.fault())?;
        self.offset += 1;
        Ok(text)
    }

    fn digest(&mut self) -> Result<[u8; 32], ExchangeError> {
        let text = self.string()?;
        if text.len() != 64 {
            return Err(self.fault());
        }
        let mut digest = [0_u8; 32];
        for (index, byte) in digest.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16)
                .map_err(|_| self.fault())?;
        }
        Ok(digest)
    }

    fn number(&mut self) -> Result<&'a str, ExchangeError> {
        self.skip_whitespace();
        let start = self.offset;
        while let Some(&byte) = self.bytes.get(self.offset) {
            if byte.is_ascii_digit() || matches!(byte, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.offset += 1;
            } else {
                break;
            }
        }
        if self.offset == start || self.offset - start > MAX_NUMBER_BYTES {
            return Err(self.fault());
        }
        std::str::from_utf8(&self.bytes[start..self.offset]).map_err(|_| self.fault())
    }

    fn u64(&mut self) -> Result<u64, ExchangeError> {
        let text = self.number()?;
        text.parse().map_err(|_| self.fault())
    }

    fn u32(&mut self) -> Result<u32, ExchangeError> {
        let text = self.number()?;
        text.parse().map_err(|_| self.fault())
    }

    fn usize(&mut self) -> Result<usize, ExchangeError> {
        let text = self.number()?;
        text.parse().map_err(|_| self.fault())
    }

    fn f32(&mut self) -> Result<f32, ExchangeError> {
        let text = self.number()?;
        let value: f32 = text.parse().map_err(|_| self.fault())?;
        // `parse` accepts `inf` and `NaN` spellings this writer never emits,
        // and a recipe refuses them anyway — but refusing here keeps the
        // refusal at the format boundary rather than several frames inside a
        // constructor.
        if !value.is_finite() {
            return Err(self.fault());
        }
        Ok(value)
    }

    /// Consumes `null`, or reports that an object follows.
    fn is_null(&mut self) -> Result<bool, ExchangeError> {
        if self.peek() != Some(b'n') {
            return Ok(false);
        }
        if self.bytes.get(self.offset..self.offset + 4) != Some(b"null".as_slice()) {
            return Err(self.fault());
        }
        self.offset += 4;
        Ok(true)
    }

    fn end(&mut self) -> Result<(), ExchangeError> {
        self.skip_whitespace();
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(self.fault())
        }
    }

    fn arrays(&mut self) -> Result<Vec<ArrayRecord>, ExchangeError> {
        self.expect(b'[')?;
        // No count precedes this list, so there is nothing to reserve from:
        // every entry costs real manifest bytes to declare, and `MAX_ARRAYS`
        // caps the table above what any family produces.
        let mut records: Vec<ArrayRecord> = Vec::new();
        if self.peek() == Some(b']') {
            self.offset += 1;
            return Ok(records);
        }
        loop {
            if records.len() == MAX_ARRAYS {
                return Err(ExchangeError::InvalidArrayTable);
            }
            self.expect(b'{')?;
            self.key("name")?;
            let name = self.string()?;
            if !is_array_name(name) {
                return Err(self.fault());
            }
            self.comma()?;
            self.key("dtype")?;
            let dtype = match self.string()? {
                "f32" => super::exchange::ArrayDtype::F32,
                "u8" => super::exchange::ArrayDtype::U8,
                "u64" => super::exchange::ArrayDtype::U64,
                _ => return Err(self.fault()),
            };
            self.comma()?;
            self.key("rows")?;
            let rows = self.usize()?;
            self.comma()?;
            self.key("columns")?;
            let columns = self.usize()?;
            self.comma()?;
            self.key("byte_offset")?;
            let byte_offset = self.usize()?;
            self.comma()?;
            self.key("len")?;
            let len = self.usize()?;
            self.expect(b'}')?;
            records.push(ArrayRecord {
                name: name.to_owned(),
                dtype,
                rows,
                columns,
                byte_offset,
                len,
            });
            match self.peek() {
                Some(b',') => self.offset += 1,
                Some(b']') => {
                    self.offset += 1;
                    return Ok(records);
                }
                _ => return Err(self.fault()),
            }
        }
    }

    fn recipe(&mut self) -> Result<Recipe, ExchangeError> {
        self.expect(b'{')?;
        self.key("rows")?;
        let rows = self.usize()?;
        self.comma()?;
        self.key("columns")?;
        let columns = self.usize()?;
        self.comma()?;
        self.key("source")?;
        let source = self.source()?;
        self.comma()?;
        self.key("task")?;
        let task = if self.is_null()? {
            None
        } else {
            Some(self.task()?)
        };
        self.comma()?;
        self.key("contamination")?;
        let contamination = self.contamination()?;
        self.comma()?;
        self.key("weights")?;
        let weights = if self.is_null()? {
            None
        } else {
            Some(self.weights()?)
        };
        self.comma()?;
        self.key("groups")?;
        let groups = if self.is_null()? {
            None
        } else {
            Some(self.groups()?)
        };
        self.expect(b'}')?;

        // Every field goes back through the public constructors, so a manifest
        // cannot describe a recipe a caller could not have written: an
        // impossible shape, an out-of-range knob, or a contamination the task
        // cannot carry is refused here rather than generated from.
        let mut recipe =
            Recipe::new(rows, columns, source).map_err(ExchangeError::InvalidRecipe)?;
        if let Some(task) = task {
            recipe = recipe
                .with_task(task)
                .map_err(ExchangeError::InvalidRecipe)?;
        }
        recipe = recipe
            .with_contamination(contamination)
            .map_err(ExchangeError::InvalidRecipe)?;
        if let Some(weights) = weights {
            recipe = recipe
                .with_weights(weights)
                .map_err(ExchangeError::InvalidRecipe)?;
        }
        if let Some(groups) = groups {
            recipe = recipe
                .with_groups(groups)
                .map_err(ExchangeError::InvalidRecipe)?;
        }
        Ok(recipe)
    }

    fn source(&mut self) -> Result<Source, ExchangeError> {
        self.expect(b'{')?;
        self.key("kind")?;
        let source = match self.string()? {
            "sampled" => {
                self.comma()?;
                self.key("state")?;
                Source::Sampled { state: self.u64()? }
            }
            "lattice" => {
                self.comma()?;
                self.key("row_stride")?;
                let row_stride = self.u64()?;
                self.comma()?;
                self.key("column_stride")?;
                let column_stride = self.u64()?;
                self.comma()?;
                self.key("modulus")?;
                let modulus = self.u64()?;
                Source::Lattice {
                    row_stride,
                    column_stride,
                    modulus,
                }
            }
            "xorshift32" => {
                self.comma()?;
                self.key("state")?;
                Source::Xorshift32 { state: self.u32()? }
            }
            _ => return Err(self.fault()),
        };
        self.expect(b'}')?;
        Ok(source)
    }

    fn task(&mut self) -> Result<Task, ExchangeError> {
        self.expect(b'{')?;
        self.key("kind")?;
        let kind = self.string()?;
        let family = Family::ALL
            .into_iter()
            .find(|family| family.label() == kind)
            .ok_or_else(|| self.fault())?;
        let task = match family {
            Family::LinearRegression => {
                self.comma()?;
                self.key("informative")?;
                let informative = self.usize()?;
                self.comma()?;
                self.key("coefficient_scale")?;
                let coefficient_scale = self.f32()?;
                self.comma()?;
                self.key("intercept")?;
                let intercept = self.f32()?;
                self.comma()?;
                self.key("noise_scale")?;
                Task::LinearRegression {
                    informative,
                    coefficient_scale,
                    intercept,
                    noise_scale: self.f32()?,
                }
            }
            Family::NonlinearRegression => {
                self.comma()?;
                self.key("shape")?;
                let kind = match self.string()? {
                    "interaction" => NonlinearKind::Interaction,
                    "piecewise" => NonlinearKind::Piecewise,
                    "sinusoid" => NonlinearKind::Sinusoid,
                    "friedman" => NonlinearKind::Friedman,
                    _ => return Err(self.fault()),
                };
                self.comma()?;
                self.key("noise_scale")?;
                Task::NonlinearRegression {
                    kind,
                    noise_scale: self.f32()?,
                }
            }
            Family::GlmRegression => {
                self.comma()?;
                self.key("link")?;
                let link = match self.string()? {
                    "log-count" => GlmLink::LogCount,
                    "log-positive" => GlmLink::LogPositive,
                    _ => return Err(self.fault()),
                };
                self.comma()?;
                self.key("informative")?;
                let informative = self.usize()?;
                self.comma()?;
                self.key("coefficient_scale")?;
                let coefficient_scale = self.f32()?;
                self.comma()?;
                self.key("intercept")?;
                let intercept = self.f32()?;
                self.comma()?;
                self.key("dispersion")?;
                Task::GlmRegression {
                    link,
                    informative,
                    coefficient_scale,
                    intercept,
                    dispersion: self.f32()?,
                }
            }
            Family::IllConditioned => {
                self.comma()?;
                self.key("condition_number")?;
                let condition_number = self.f32()?;
                self.comma()?;
                self.key("rank")?;
                let rank = self.usize()?;
                self.comma()?;
                self.key("coefficient_scale")?;
                let coefficient_scale = self.f32()?;
                self.comma()?;
                self.key("noise_scale")?;
                Task::IllConditioned {
                    condition_number,
                    rank,
                    coefficient_scale,
                    noise_scale: self.f32()?,
                }
            }
            Family::LinearBinary => {
                self.comma()?;
                self.key("informative")?;
                let informative = self.usize()?;
                self.comma()?;
                self.key("separation")?;
                let separation = self.f32()?;
                self.comma()?;
                self.key("prevalence")?;
                Task::LinearBinary {
                    informative,
                    separation,
                    prevalence: self.f32()?,
                }
            }
            Family::NonlinearBinary => {
                self.comma()?;
                self.key("boundary")?;
                let kind = match self.string()? {
                    "xor" => BinaryKind::Xor,
                    "moons" => BinaryKind::Moons,
                    "circles" => BinaryKind::Circles,
                    "checkerboard" => BinaryKind::Checkerboard,
                    _ => return Err(self.fault()),
                };
                self.comma()?;
                self.key("separation")?;
                let separation = self.f32()?;
                self.comma()?;
                self.key("prevalence")?;
                Task::NonlinearBinary {
                    kind,
                    separation,
                    prevalence: self.f32()?,
                }
            }
            Family::Multiclass => {
                self.comma()?;
                self.key("classes")?;
                let classes = self.usize()?;
                self.comma()?;
                self.key("balance")?;
                let balanced = match self.string()? {
                    "balanced" => true,
                    "imbalanced" => false,
                    _ => return Err(self.fault()),
                };
                self.comma()?;
                self.key("balance_ratio")?;
                let ratio = self.f32()?;
                let balance = if balanced {
                    ClassBalance::Balanced
                } else {
                    ClassBalance::Imbalanced { ratio }
                };
                self.comma()?;
                self.key("geometry")?;
                let geometry = match self.string()? {
                    "blob" => ClassGeometry::Blob,
                    "hierarchical" => ClassGeometry::Hierarchical,
                    _ => return Err(self.fault()),
                };
                self.comma()?;
                self.key("separation")?;
                Task::Multiclass {
                    classes,
                    balance,
                    geometry,
                    separation: self.f32()?,
                }
            }
            Family::Clustered => {
                self.comma()?;
                self.key("blobs")?;
                let blobs = self.usize()?;
                self.comma()?;
                self.key("spread")?;
                Task::Clustered {
                    blobs,
                    spread: self.f32()?,
                }
            }
            Family::TimeOrdered => {
                self.comma()?;
                self.key("informative")?;
                let informative = self.usize()?;
                self.comma()?;
                self.key("coefficient_scale")?;
                let coefficient_scale = self.f32()?;
                self.comma()?;
                self.key("drift")?;
                let drift = self.f32()?;
                self.comma()?;
                self.key("intercept")?;
                let intercept = self.f32()?;
                self.comma()?;
                self.key("noise_scale")?;
                Task::TimeOrdered {
                    informative,
                    coefficient_scale,
                    drift,
                    intercept,
                    noise_scale: self.f32()?,
                }
            }
            Family::Ranking => {
                self.comma()?;
                self.key("queries")?;
                let queries = self.usize()?;
                self.comma()?;
                self.key("docs_per_query")?;
                let docs_per_query = self.usize()?;
                self.comma()?;
                self.key("grades")?;
                let grades = self.usize()?;
                self.comma()?;
                self.key("informative")?;
                let informative = self.usize()?;
                self.comma()?;
                self.key("coefficient_scale")?;
                Task::Ranking {
                    queries,
                    docs_per_query,
                    grades,
                    informative,
                    coefficient_scale: self.f32()?,
                }
            }
        };
        self.expect(b'}')?;
        Ok(task)
    }

    fn contamination(&mut self) -> Result<Contamination, ExchangeError> {
        self.expect(b'{')?;
        self.key("label_noise")?;
        let label_noise = self.f32()?;
        self.comma()?;
        self.key("outlier_fraction")?;
        let outlier_fraction = self.f32()?;
        self.comma()?;
        self.key("heavy_tail")?;
        let heavy_tail = self.f32()?;
        self.comma()?;
        self.key("heteroscedastic")?;
        let heteroscedastic = self.f32()?;
        self.comma()?;
        self.key("duplicate_rows")?;
        let duplicate_rows = self.f32()?;
        self.comma()?;
        self.key("constant_columns")?;
        let constant_columns = self.usize()?;
        self.comma()?;
        self.key("collinear_pairs")?;
        let collinear_pairs = self.usize()?;
        self.comma()?;
        self.key("feature_scale_spread")?;
        let feature_scale_spread = self.f32()?;
        self.expect(b'}')?;
        Ok(Contamination::none()
            .with_label_noise(label_noise)
            .with_outlier_fraction(outlier_fraction)
            .with_heavy_tail(heavy_tail)
            .with_heteroscedastic(heteroscedastic)
            .with_duplicate_rows(duplicate_rows)
            .with_constant_columns(constant_columns)
            .with_collinear_pairs(collinear_pairs)
            .with_feature_scale_spread(feature_scale_spread))
    }

    fn weights(&mut self) -> Result<WeightPattern, ExchangeError> {
        self.expect(b'{')?;
        self.key("kind")?;
        let pattern = match self.string()? {
            "uniform" => WeightPattern::Uniform,
            "ramp" => {
                self.comma()?;
                self.key("low")?;
                let low = self.f32()?;
                self.comma()?;
                self.key("high")?;
                WeightPattern::Ramp {
                    low,
                    high: self.f32()?,
                }
            }
            "alternating" => {
                self.comma()?;
                self.key("first")?;
                let first = self.f32()?;
                self.comma()?;
                self.key("second")?;
                WeightPattern::Alternating {
                    first,
                    second: self.f32()?,
                }
            }
            "class-balanced" => WeightPattern::ClassBalanced,
            _ => return Err(self.fault()),
        };
        self.expect(b'}')?;
        Ok(pattern)
    }

    fn groups(&mut self) -> Result<GroupPattern, ExchangeError> {
        self.expect(b'{')?;
        self.key("kind")?;
        let pattern = match self.string()? {
            "round-robin" => {
                self.comma()?;
                self.key("groups")?;
                GroupPattern::RoundRobin {
                    groups: self.usize()?,
                }
            }
            "contiguous" => {
                self.comma()?;
                self.key("groups")?;
                GroupPattern::Contiguous {
                    groups: self.usize()?,
                }
            }
            "unbalanced" => {
                self.comma()?;
                self.key("groups")?;
                let groups = self.usize()?;
                self.comma()?;
                self.key("ratio")?;
                GroupPattern::Unbalanced {
                    groups,
                    ratio: self.f32()?,
                }
            }
            _ => return Err(self.fault()),
        };
        self.expect(b'}')?;
        Ok(pattern)
    }
}

/// Whether a name is one this format will carry.
///
/// Lower-case ASCII, digits and underscores, non-empty. The set is deliberately
/// narrower than what a filesystem or a JSON string allows, because an array
/// name is also a Python attribute and a dictionary key on the other side of
/// the exchange.
fn is_array_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_STRING_BYTES
        && name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}
