# Recommender rules

Recommender rules decide which branch to return after the tree exploration is
complete.

- `recommender_rule.py` defines `RecommenderRule`, `BranchPolicy`, and concrete
  implementations such as `AlmostEqualLogistic` and `SoftmaxRule`.

These rules are used by `TreeExploration` to produce the final
`BranchRecommendation`.
