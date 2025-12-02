"""
Income-rank-dependent climate damage distribution.

This module implements climate damage that varies by income level, with
lower-income populations experiencing proportionally greater losses.
This captures both aggregate damage effects and impacts on inequality.

Mathematical Foundation
-----------------------
For a Pareto income distribution with Gini index G and corresponding parameter a,
this module computes:

1. Aggregate damage Ω: fraction of total GDP lost to climate damage
2. Post-damage Gini G_climate: inequality after climate damage is applied
"""

