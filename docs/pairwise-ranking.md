# Pairwise linear ranking

`PairwiseLinearRanker` fits one raw linear item scorer from checked pairs over
a single dense item matrix. Pair outcomes are typed as `LeftPreferred`,
`RightPreferred`, or `Tie`; pair weights are finite and non-negative, with a
positive total required at fit time.

The objective expands each decisive observation into two mirrored difference
rows. Each row receives half of the pair weight. A tie expands into both
targets for both difference orientations, yielding four rows at one quarter
of the pair weight each. The resulting deterministic weighted logistic model
always disables its intercept. Canonical pair orientation and ordering make
equivalent input permutations and orientation reversals fit identically.

Inference exposes item scores and the raw pair margin
`score(left) - score(right)`. An inclusive finite non-negative threshold maps
`abs(margin) <= threshold` to `Tie`; positive and negative margins otherwise
select the left and right item. Scores and margins are objective values, not
calibrated probabilities.

The ranking metric helpers keep denominator failures explicit. Decisive
accuracy excludes expected ties, three-way accuracy includes them, Spearman
correlation uses average ranks for exact ties, and Kendall tau-b adjusts its
denominator for ties in either ordering. Empty usable sets, constant orderings,
length mismatches, and non-finite scores return `RankingMetricError`.
