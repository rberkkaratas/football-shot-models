"""Themed shot charts (CH1): the team shot chart and the player shot profile.

Render-only and outside the prediction cache's hash: nothing under ``cache.ROOTS`` imports
this package, so a chart edit never makes a fit cold. :mod:`~football_shot_charts.shots` draws
from frames (no IO); :mod:`~football_shot_charts.render` reads the ``xg-gbm`` hold-out
predictions from the warm cache (never fitting) and writes the git-ignored examples
(``make charts``). The maintenance round folds both into ``football_shot_models`` and
``cache.RENDER_ONLY`` (PM E3).
"""
