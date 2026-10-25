# cache

The prediction cache: per-spec out-of-fold, in-sample and hold-out predictions under the
git-ignored `data/cache`, keyed by the data digest and a hash of the code, so a rerun reads them
instead of fitting.

::: football_shot_models.cache
