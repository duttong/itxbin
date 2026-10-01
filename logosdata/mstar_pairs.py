"""M-system (M1/M3/M4) flask pair means that follow Montzka's pair rule.

A pair mean is kept only when at least two distinct flasks contribute a value.
hats.ng_pair_avg_view drops rejected rows before grouping, so when one flask
of a pair is rejected (or has no mole fraction) the view still reports a
"pair" mean built from the surviving flask alone.  Its ``n`` column counts
analyses, not flasks (M4 runs each flask twice), so ``n`` cannot be used to
catch these: a lone flask run twice shows n=2.

MSTAR_PAIR_AVG_SQL is a drop-in replacement for ``hats.ng_pair_avg_view`` in
M* queries -- same grouping and columns, restricted to M1/M3/M4, with the
single-flask pairs removed.  Use it as ``FROM {MSTAR_PAIR_AVG_SQL} v``.
The view itself is left unchanged so FE3/OTTO users are unaffected.
MariaDB pushes outer WHERE filters into the derived table, so it runs as fast
as the view.
"""

MSTAR_INST_IDS = ("M1", "M3", "M4")

MSTAR_PAIR_AVG_SQL = """(
    SELECT d.site, d.site_num, d.sample_datetime, d.inst_num, d.inst_id,
           d.pair_id_num, d.parameter_num, d.parameter,
           d.wind_speed AS Wind_Speed, d.wind_direction AS Wind_Direction,
           d.channel,
           MIN(d.analysis_datetime) AS analysis_datetime,
           GROUP_CONCAT(d.analysis_num ORDER BY d.analysis_num SEPARATOR '|') AS analysis_num,
           GROUP_CONCAT(d.sample_id ORDER BY d.sample_id SEPARATOR '|') AS sample_id,
           AVG(d.value) AS pair_avg,
           COUNT(d.value) AS n,
           STD(d.value) AS pair_stdv
    FROM hats.ng_data_view d
    WHERE d.rejected = 0
      AND d.test_num = 0
      AND d.run_type_num <> 10
      AND (d.pair_id_num > 0 OR d.ccgg_event_num > 0)
      AND d.inst_id IN ('M1', 'M3', 'M4')
    GROUP BY d.site, d.site_num, d.sample_datetime, d.inst_num, d.inst_id,
             d.pair_id_num, d.parameter_num, d.parameter, d.wind_speed,
             d.wind_direction, d.channel
    HAVING COUNT(DISTINCT CASE WHEN d.value IS NOT NULL THEN d.sample_id END) >= 2
)"""
