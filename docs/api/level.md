# level

PR1: the StatsBomb hold-out level gap. The level test (`level_z`: Σ xG − goals against the
Poisson-binomial SD, within `LEVEL_SD`), the segment decomposition (`segments`), the adoption
rule (`adopt`) and the xg-gbm card section (`level_lines`). What is fitted (`models.POOLED`,
`models.LEVEL_FIXES`, `models.ADOPTED`) stays in the hashed `models.py`; this module fits
nothing and is orchestration-only for the prediction cache.

::: football_shot_models.level
