"""Themed CH1 charts: shot charts, the keeper (GSAx) chart and the model-card visuals.

:mod:`~football_shot_charts.shots` draws the team shot chart and the player shot profile,
:mod:`~football_shot_charts.keepers` the keeper chart, :mod:`~football_shot_charts.diagnostics`
the calibration and model-comparison charts, all on :mod:`~football_shot_charts.rows` (the
shared row panel and the one chart writer, ``rows.write_all``). :mod:`~football_shot_charts.render`
reads fit outputs from the warm cache (never refitting) and writes the git-ignored examples
(``make charts``).

The package sits outside ``cache.ROOTS``, so a chart edit never makes a fit cold. One module
under the hash imports it: ``gsax_cards`` (in ``cache.RENDER_ONLY``) imports ``keepers`` only
(keeper spec §1, pinned by ``test_outside_hash_and_one_way``). Folding any of these modules
into ``football_shot_models`` must add it to ``RENDER_ONLY`` in the same edit, or every
styling edit makes the cache cold (PM E3: the maintenance round).
"""
