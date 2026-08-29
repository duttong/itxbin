#! /usr/bin/env python
"""Independent (non-Matlab) computation of Perseus (PR1/PR2) ratios and
mole fractions, writing to hats.prs_analysis / hats.prs_mole_fractions.

This is a parallel computation for demonstration -- it does not read or
write anything the legacy Matlab pipeline touches (hats.mole_fractions,
hats.interp_std_response, hats.normal_response are left untouched), and it
does not depend on pr1_gcwerks2db.py being changed in any way. It only
reads hats.analysis / hats.raw_data / hats.flags_internal (already written
by pr1_gcwerks2db.py) plus hats.scale_assignments_view.

v1 scope: CFC-11 (parameter_num 29) only. Known simplifications vs. the
legacy chain:
  - Standard-response smoothing is point-to-point linear interpolation
    between consecutive std analyses within one S-tag segment, not the
    Lowess order the Matlab pipeline actually applies (that selection is
    baked into hats.interp_std_response with no visible formula -- there is
    nothing to replicate without the Matlab source).
  - No pressure correction (hats.PR1_adsorbed_air polynomial) -- raw
    peak_area/peak_height are used directly. PR2 doesn't have adsorbed_air
    rows yet anyway.
  - Blank/std bracketing (nearest sample_type='blank' or 'std' analysis
    before/after) is computed here in Python rather than trusted from
    hats.raw_data.pre_standard_analysis_num/post_standard_analysis_num/
    pre_blank_analysis_num/post_blank_analysis_num -- those columns are
    populated by some other external, lagged process, not by
    pr1_gcwerks2db.py, and were found stale (PR1's last row never got a
    post_standard_analysis_num backfilled once PR1 stopped producing data).

PR1 (inst_num=58) and PR2 (inst_num=238) are different physical mass
spectrometers sharing one GCwerks pipeline -- every step here (bracketing,
segment breaks, reference tank resolution) is done independently per
inst_num and never pools the two.
"""

import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd

from logos_instruments import Perseus_Instrument

RUN_TIME_GAP_MINUTES = 60  # instrument-wide idle-gap threshold for prs_analysis.run_time

# Sample types injected at a pressure well below std/HATS/tank (confirmed:
# PFP responses run roughly half the std response, vs. HATS flasks which
# stay within ~3% of it). The legacy chain applies a response-nonlinearity
# correction somewhere in the Matlab code that isn't visible in any DB view
# or table -- verified missing by comparing our normalized_resp (which DOES
# match the legacy prs_corrected_response_view.normalized_response closely,
# confirming the ratio itself is right) against legacy C_reported: HATS/std
# match well without any further correction, PFP does not. mole_fraction is
# still computed and stored for these types (not nulled), but flagged via
# nonlinearity_uncorrected so a comparison view or the GUI can call it out
# rather than presenting it as equally trustworthy.
NONLINEARITY_UNCORRECTED_TYPES = {'PFP', 'CCGG', 'cal', 'burn', 'test'}


class PRS_batch(Perseus_Instrument):

    def __init__(self):
        super().__init__()
        self.t0 = time.time()

    # ------------------------------------------------------------------
    # prs_analysis backfill
    # ------------------------------------------------------------------

    def _assign_run_time_gap(self, dt: pd.Series) -> pd.Series:
        """Group a sorted datetime Series into runs separated by gaps over
        RUN_TIME_GAP_MINUTES. Mirrors IE3_GCwerks2DB._assign_run_time's
        shape (ie3_gcwerks2db.py) but with a threshold tuned to Perseus's
        own ~24min steady injection cadence rather than IE3's 15min."""
        gap = dt.diff() > pd.Timedelta(minutes=RUN_TIME_GAP_MINUTES)
        segment = gap.cumsum()
        return dt.groupby(segment).transform('first')

    def load_analysis_rows(self, start_date=None, end_date=None, verbose=True):
        """Load hats.analysis rows for PR1+PR2, one row per physical
        injection (not per parameter)."""
        date_filter = ""
        if start_date is not None and end_date is not None:
            date_filter = f"AND analysis_datetime BETWEEN '{start_date}' AND '{end_date}'"

        sql = f"""
            SELECT num AS legacy_analysis_num, analysis_datetime, inst_num,
                   sample_ID AS sample_id, sample_type, port, std_serial_num AS standard_serial_num,
                   site_num, event_num
            FROM hats.analysis
            WHERE inst_num IN ({self.inst_nums[0]}, {self.inst_nums[1]})
                {date_filter}
            ORDER BY inst_num, analysis_datetime;
        """
        if verbose:
            print(f"Loading hats.analysis rows{' from ' + str(start_date) + ' to ' + str(end_date) if date_filter else ' (full history)'}...")
        df = pd.DataFrame(self.doquery(sql))
        if df.empty:
            return df

        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'])
        df['pair_id_num'] = pd.NA  # not resolved yet; hatsflask_event linkage is a future addition

        df['run_time'] = pd.NaT
        for inst_num, idx in df.groupby('inst_num').groups.items():
            df.loc[idx, 'run_time'] = self._assign_run_time_gap(df.loc[idx, 'analysis_datetime'])

        return df

    def upsert_prs_analysis(self, df, verbose=True):
        """Insert/update hats.prs_analysis from a load_analysis_rows() frame."""
        if df.empty:
            return 0

        sql = """
            INSERT INTO hats.prs_analysis (
                legacy_analysis_num, analysis_datetime, run_time, inst_num,
                sample_id, sample_type, port, standard_serial_num,
                site_num, pair_id_num, event_num
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                legacy_analysis_num = VALUES(legacy_analysis_num),
                run_time = VALUES(run_time),
                sample_id = VALUES(sample_id),
                sample_type = VALUES(sample_type),
                port = VALUES(port),
                standard_serial_num = VALUES(standard_serial_num),
                site_num = VALUES(site_num),
                pair_id_num = VALUES(pair_id_num),
                event_num = VALUES(event_num);
        """
        params = []
        n = 0
        for row in df.itertuples(index=False):
            pair_id = None if pd.isna(row.pair_id_num) else int(row.pair_id_num)
            site_num = None if pd.isna(row.site_num) else int(row.site_num)
            params.append((
                int(row.legacy_analysis_num),
                row.analysis_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                row.run_time.strftime('%Y-%m-%d %H:%M:%S'),
                int(row.inst_num),
                str(row.sample_id),
                str(row.sample_type),
                int(row.port),
                None if pd.isna(row.standard_serial_num) else str(row.standard_serial_num),
                site_num,
                pair_id,
                int(row.event_num),
            ))
            n += 1
            if self.db.doMultiInsert(sql, params):
                params = []
        if params:
            self.db.doMultiInsert(sql, params, all=True)
        if verbose:
            print(f"Upserted {n} rows into hats.prs_analysis.")
        return n

    def analysis_num_map(self, start_date=None, end_date=None):
        """Return {legacy_analysis_num: prs_analysis.num} for rows in range."""
        date_filter = ""
        if start_date is not None and end_date is not None:
            date_filter = f"AND analysis_datetime BETWEEN '{start_date}' AND '{end_date}'"
        sql = f"""
            SELECT num, legacy_analysis_num, analysis_datetime, inst_num, sample_type, sample_id
            FROM hats.prs_analysis
            WHERE inst_num IN ({self.inst_nums[0]}, {self.inst_nums[1]})
                {date_filter}
            ORDER BY inst_num, analysis_datetime;
        """
        return pd.DataFrame(self.doquery(sql))

    # ------------------------------------------------------------------
    # prs_mole_fractions compute
    # ------------------------------------------------------------------

    def _peak_response_column(self, pnum, inst_num):
        """Return 'peak_area' or 'peak_height' per hats.PR1_peak_response,
        windowed by (inst_num, parameter_num, start_date). Falls back to
        'area' when no row exists for this inst_num (e.g. PR2 currently has
        no PR1_peak_response row for most parameters, including CFC-11)."""
        rows = self.doquery(
            """
            SELECT response FROM hats.PR1_peak_response
            WHERE inst_num = %s AND parameter_num = %s
            ORDER BY start_date DESC LIMIT 1
            """,
            [inst_num, pnum],
        )
        if rows:
            return 'peak_area' if rows[0]['response'] == 'area' else 'peak_height'
        print(f"  WARNING: no PR1_peak_response row for inst_num={inst_num} pnum={pnum}; defaulting to peak_area.")
        return 'peak_area'

    def _blank_correction_enabled(self, pnum, inst_num, when):
        """PR1_blank_correction.blank flag windowed by start_datetime."""
        rows = self.doquery(
            """
            SELECT blank FROM hats.PR1_blank_correction
            WHERE inst_num = %s AND parameter_num = %s AND start_datetime <= %s
            ORDER BY start_datetime DESC LIMIT 1
            """,
            [inst_num, pnum, when],
        )
        return bool(rows[0]['blank']) if rows else False

    @staticmethod
    def _bracket(dt: np.ndarray, mask: np.ndarray):
        """For every position in dt, return (pre_idx, post_idx) into dt of
        the nearest True position in mask strictly before / at-or-after it.
        -1 means no bracket exists on that side. dt must be sorted."""
        n = len(dt)
        true_positions = np.flatnonzero(mask)
        if true_positions.size == 0:
            return np.full(n, -1), np.full(n, -1)

        # For each row, position in true_positions of the first True at/after it
        post_slot = np.searchsorted(true_positions, np.arange(n), side='left')
        pre_slot = post_slot - 1

        pre_idx = np.where(pre_slot >= 0, true_positions[np.clip(pre_slot, 0, None)], -1)
        # If the row itself is True, "post" bracket is itself
        post_idx = np.where(
            post_slot < true_positions.size,
            true_positions[np.clip(post_slot, 0, true_positions.size - 1)],
            -1,
        )
        return pre_idx, post_idx

    @staticmethod
    def _time_weighted(pre_val, pre_dt, post_val, post_dt, at_dt):
        """(pre*secs(at,post) + post*secs(pre,at)) / secs(pre,post) --
        matches prs_intermediate_calcs_response_view's interpolation exactly."""
        total = (post_dt - pre_dt).astype('timedelta64[s]').astype(float)
        w_pre = (post_dt - at_dt).astype('timedelta64[s]').astype(float)
        w_post = (at_dt - pre_dt).astype('timedelta64[s]').astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = (pre_val * w_pre + post_val * w_post) / total
        return out

    def compute_mole_fractions(self, pnum, start_date=None, end_date=None, verbose=True):
        """Compute ratio + mole fraction for one parameter, independently
        per inst_num. Returns a DataFrame ready for upsert_prs_mole_fractions()."""
        # The rolling local-median outlier check (below) needs enough
        # surrounding context to work -- a narrow start/end window sitting
        # right at the edge of a multi-hour outage would otherwise have no
        # "before" or "after" good readings to compare against. Pad the
        # query by 10 days on each side purely for that context; rows
        # outside the originally-requested range are dropped again below
        # before this function returns, so they are never written to
        # prs_mole_fractions.
        CONTEXT_PAD_DAYS = 10
        query_start, query_end = start_date, end_date
        if start_date is not None and end_date is not None:
            query_start = (pd.Timestamp(start_date) - pd.Timedelta(days=CONTEXT_PAD_DAYS)).strftime('%Y-%m-%d %H:%M:%S')
            query_end = (pd.Timestamp(end_date) + pd.Timedelta(days=CONTEXT_PAD_DAYS)).strftime('%Y-%m-%d %H:%M:%S')

        prs_a = self.analysis_num_map(query_start, query_end)
        if prs_a.empty:
            if verbose:
                print("No prs_analysis rows in range -- run the analysis backfill first.")
            return pd.DataFrame()

        date_filter = ""
        if query_start is not None and query_end is not None:
            date_filter = f"AND a.analysis_datetime BETWEEN '{query_start}' AND '{query_end}'"
        # rejected_other_than_preliminary (excludes tag 318, "Preliminary
        # data, not ready for release" -- same rationale as
        # Perseus_Instrument.load_data()) is carried along for potential
        # future use but is NOT used below to gate std/blank bracketing --
        # see the plausible_resp comment further down for why a response-
        # value plausibility check is used instead of the rejection tag.
        raw_sql = f"""
            SELECT a.num AS legacy_analysis_num, r.peak_area, r.peak_height,
                   v.rejected_other_than_preliminary AS rejected
            FROM hats.analysis a
            JOIN hats.raw_data r ON r.analysis_num = a.num
            LEFT JOIN hats.prs_data_view v ON v.analysis_num = a.num AND v.parameter_num = r.parameter_num
            WHERE a.inst_num IN ({self.inst_nums[0]}, {self.inst_nums[1]})
                AND r.parameter_num = {pnum}
                {date_filter}
        """
        raw = pd.DataFrame(self.doquery(raw_sql))
        if raw.empty:
            if verbose:
                print(f"No raw_data rows for parameter {pnum} in range.")
            return pd.DataFrame()

        df = prs_a.merge(raw, on='legacy_analysis_num', how='inner')
        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'])

        out_frames = []
        for inst_num, grp in df.groupby('inst_num'):
            grp = grp.sort_values('analysis_datetime').reset_index(drop=True)

            resp_col = self._peak_response_column(pnum, int(inst_num))
            grp['raw_response'] = grp[resp_col].astype(float)

            blank_on = self._blank_correction_enabled(
                pnum, int(inst_num), grp['analysis_datetime'].iat[-1].strftime('%Y-%m-%d %H:%M:%S')
            )

            dt = grp['analysis_datetime'].to_numpy()

            # --- S-tag segment breaks (already instrument-partitioned) ---
            # Computed early: the local-median outlier check just below and
            # the std-bracketing further down both need segment membership,
            # since neither should reach across a real sensitivity-level
            # change (confirmed necessary against a real 2016-10-15 event on
            # PR1 where std responses shifted from ~1.0M to a completely
            # different ~340K-960K regime within hours -- a median/bracket
            # computed over both regimes at once is meaningless).
            breaks = self._segment_breaks(
                pnum, int(inst_num),
                grp['analysis_datetime'].iat[0].strftime('%Y-%m-%d %H:%M:%S'),
                grp['analysis_datetime'].iat[-1].strftime('%Y-%m-%d %H:%M:%S'),
            )
            if breaks.empty:
                seg_start = pd.Series(grp['analysis_datetime'].iat[0], index=grp.index)
            else:
                positions = np.searchsorted(breaks.values, dt, side='right') - 1
                seg_start = pd.Series(pd.NaT, index=grp.index, dtype='datetime64[ns]')
                found = positions >= 0
                seg_start.loc[grp.index[found]] = breaks[positions[found]].tz_localize(None)
                seg_start.loc[grp.index[~found]] = grp['analysis_datetime'].iat[0]
            grp['segment_start'] = seg_start

            # A std/blank run with an implausible raw response must not serve
            # as a reference point for OTHER rows' interpolation (it would
            # otherwise corrupt every ratio bracketed against it) --
            # confirmed against a real instrument glitch on PR1 2020-02-21
            # where two consecutive std runs dropped to 0, and a sentinel
            # value of exactly 1e8 seen on other dates. A GRADUAL/SUSTAINED
            # failure (same 2020-02-21 event: std responses ramped
            # 218K -> 15728 -> 0 -> 0 -> ~370, staying near 370 for eleven
            # hours -- ~10 consecutive std runs) is not caught by any global
            # bound, since 15728/370 both look individually plausible; a
            # rolling LOCAL median comparison catches it instead, and needs
            # to be wide enough (101 points, ~4-5 days of std cadence) that
            # a multi-hour outage can't out-vote its own window. Filtering
            # on the rejection tag alone was tried first and rejected: a
            # single "known measurement problem" tag can span hours of
            # otherwise-normal std runs (confirmed 2019-02-15) -- excluding
            # all of them left nothing nearby to interpolate against.
            #
            # std and blank populations sit on completely different scales
            # (std ~O(1e5), blank ~O(1e2)), so they need independent local
            # baselines -- an absolute bound calibrated for one is either a
            # no-op or excludes 100% of the other (confirmed: a std-sized
            # floor of 1000 excluded every real blank, since blanks are
            # ~200-300 by design; the sole "blank" that passed was a genuine
            # contamination event reading 273850, which then became the ONLY
            # candidate for the entire surrounding month, corrupting every
            # row bracketed against it. Root cause found by reproducing the
            # bug on a narrowed 2018-Q1/Q2 window).
            #
            # The std check runs PER SEGMENT (blank is unaffected by
            # sensitivity-level changes, so it stays global) -- confirmed
            # necessary against the same 2016-10-15 event: a median spanning
            # both the ~1.0M and ~340K-960K regimes was unstable right at the
            # transition and let the 1e8 sentinel rows through as
            # "plausible" relative to a confused blended median.
            def _local_median_mask(resp: pd.Series) -> pd.Series:
                if len(resp) < 2:
                    return pd.Series(True, index=resp.index)
                med = resp.rolling(101, center=True, min_periods=15).median()
                return (resp >= 0.2 * med) & (resp <= 5 * med)

            blank_idx = grp.index[grp['sample_type'] == 'blank']
            local_median_ok = pd.Series(True, index=grp.index)
            local_median_ok.loc[blank_idx] = _local_median_mask(grp.loc[blank_idx, 'raw_response'])

            std_mask_raw = grp['sample_type'] == 'std'
            for _seg, seg_idx in grp.loc[std_mask_raw].groupby('segment_start').groups.items():
                local_median_ok.loc[seg_idx] = _local_median_mask(grp.loc[seg_idx, 'raw_response'])

            is_blank = (grp['sample_type'] == 'blank').to_numpy() & local_median_ok.to_numpy()
            is_std = std_mask_raw.to_numpy() & local_median_ok.to_numpy()

            # A bracket point that IS otherwise "plausible" can still be too
            # far away in time to mean anything (real blanks can be sparse --
            # 90th-percentile gap ~8h, 99th ~25h -- so excluding even a
            # handful near a given row can force a bracket days away; std's
            # own gaps are normally much tighter, 99th percentile ~3.4h, but
            # the same cap keeps std/blank handling uniform). Beyond this gap
            # treat that side as missing entirely (falls back to the
            # existing missing-side handling, matching the legacy view's own
            # ifnull()-to-0-then-sum behavior) rather than silently
            # extrapolating across a span that no longer represents "the
            # same run." 24h comfortably covers ~99th-percentile gaps for
            # both types without being so long it papers over a real outage.
            MAX_BRACKET_GAP = pd.Timedelta(hours=24)

            def _apply_gap_cap(b_idx, at_dt):
                """Null out a bracket side whose target is farther than
                MAX_BRACKET_GAP from at_dt. b_idx/at_dt are aligned arrays;
                returns b_idx with out-of-range entries set to -1."""
                b_idx = b_idx.copy()
                valid = b_idx >= 0
                if valid.any():
                    gap = np.abs(dt[np.clip(b_idx, 0, None)].astype('datetime64[s]') - at_dt.astype('datetime64[s]'))
                    too_far = valid & (gap > np.timedelta64(MAX_BRACKET_GAP))
                    b_idx[too_far] = -1
                return b_idx

            # --- blank correction ---
            if blank_on and is_blank.any():
                pre_b, post_b = self._bracket(dt, is_blank)
                pre_b = _apply_gap_cap(pre_b, dt)
                post_b = _apply_gap_cap(post_b, dt)
                pre_resp = np.where(pre_b >= 0, grp['raw_response'].to_numpy()[np.clip(pre_b, 0, None)], np.nan)
                post_resp = np.where(post_b >= 0, grp['raw_response'].to_numpy()[np.clip(post_b, 0, None)], np.nan)
                pre_dt = np.where(pre_b >= 0, dt[np.clip(pre_b, 0, None)], np.datetime64('NaT'))
                post_dt = np.where(post_b >= 0, dt[np.clip(post_b, 0, None)], np.datetime64('NaT'))

                both = ~np.isnan(pre_resp) & ~np.isnan(post_resp)
                blank_corr = np.where(np.isnan(pre_resp), 0.0, pre_resp) + np.where(np.isnan(post_resp), 0.0, post_resp)
                weighted = self._time_weighted(pre_resp, pre_dt, post_resp, post_dt, dt)
                blank_corr = np.where(both, weighted, blank_corr)
            else:
                blank_corr = np.zeros(len(grp))

            grp['blank_correction'] = blank_corr
            grp['blank_corrected_response'] = grp['raw_response'] - grp['blank_correction']

            # --- point-to-point std-response smoothing (v1 simplification) ---
            # Bracketing must not cross an S-tag segment boundary -- that is
            # the entire point of the segment concept (mirrors the legacy
            # view's own hats_interpolation-tag check: a post-std run
            # carrying an S-tag means "don't blend across this break").
            # Confirmed necessary against a real 2016-10-15 sensitivity-level
            # change on PR1 where std responses shifted from ~1.0M to a
            # completely different ~340K-960K regime within hours; without
            # per-segment bracketing, and without this fix the surrounding
            # rolling-median outlier check (which also doesn't know about
            # segments) got confused by the regime change itself. Doing the
            # bracket search per-segment fixes both: no cross-segment blend,
            # and the median check only ever sees one regime at a time.
            std_resp = np.where(is_std, grp['blank_corrected_response'].to_numpy(), np.nan)
            pre_s = np.full(len(grp), -1)
            post_s = np.full(len(grp), -1)
            for _seg, seg_idx in grp.groupby('segment_start').groups.items():
                seg_pos = grp.index.get_indexer(seg_idx)
                seg_dt = dt[seg_pos]
                seg_is_std = is_std[seg_pos]
                seg_pre, seg_post = self._bracket(seg_dt, seg_is_std)
                valid_pre = seg_pre >= 0
                valid_post = seg_post >= 0
                pre_s[seg_pos[valid_pre]] = seg_pos[seg_pre[valid_pre]]
                post_s[seg_pos[valid_post]] = seg_pos[seg_post[valid_post]]
            pre_s = _apply_gap_cap(pre_s, dt)
            post_s = _apply_gap_cap(post_s, dt)
            pre_std_val = np.where(pre_s >= 0, std_resp[np.clip(pre_s, 0, None)], np.nan)
            post_std_val = np.where(post_s >= 0, std_resp[np.clip(post_s, 0, None)], np.nan)
            pre_std_dt = np.where(pre_s >= 0, dt[np.clip(pre_s, 0, None)], np.datetime64('NaT'))
            post_std_dt = np.where(post_s >= 0, dt[np.clip(post_s, 0, None)], np.datetime64('NaT'))

            # own value if this row IS a std
            self_val = np.where(is_std, grp['blank_corrected_response'].to_numpy(), np.nan)

            both_std = ~np.isnan(pre_std_val) & ~np.isnan(post_std_val)
            smoothed = np.where(np.isnan(pre_std_val), 0.0, pre_std_val) + np.where(np.isnan(post_std_val), 0.0, post_std_val)
            weighted_std = self._time_weighted(pre_std_val, pre_std_dt, post_std_val, post_std_dt, dt)
            smoothed = np.where(both_std, weighted_std, smoothed)
            smoothed = np.where(is_std, self_val, smoothed)
            grp['smoothed_response'] = smoothed

            grp['normalized_resp'] = grp['blank_corrected_response'] / grp['smoothed_response']

            # --- reference tank resolution: the std sample_id bracketing this row ---
            ref_sid = np.where(pre_s >= 0, grp['sample_id'].to_numpy()[np.clip(pre_s, 0, None)], None)
            ref_sid = np.where(is_std, grp['sample_id'].to_numpy(), ref_sid)
            grp['ref_tank_serial'] = ref_sid

            coef0_cache = {}
            coef0_vals = []
            for sid, rt, adt in zip(grp['ref_tank_serial'], grp['segment_start'], grp['analysis_datetime']):
                if sid is None:
                    coef0_vals.append(np.nan)
                    continue
                key = (sid, pnum, adt.date())
                if key not in coef0_cache:
                    assign = self.scale_assignments(sid, pnum, run_date=adt)
                    coef0_cache[key] = float(assign['coef0']) if assign and assign.get('coef0') not in (None, 0) else np.nan
                coef0_vals.append(coef0_cache[key])
            grp['ref_tank_coef0'] = coef0_vals

            n_zero_or_missing = pd.isna(grp['ref_tank_coef0']).sum()
            if n_zero_or_missing and verbose:
                print(f"  inst_num={inst_num}: {n_zero_or_missing} rows with missing/zero ref_tank_coef0 (unassigned tank) -- mole_fraction will be NULL for these.")

            grp['mole_fraction'] = grp['normalized_resp'] * grp['ref_tank_coef0']
            grp['nonlinearity_uncorrected'] = grp['sample_type'].isin(NONLINEARITY_UNCORRECTED_TYPES)

            out_frames.append(grp)

        result = pd.concat(out_frames, ignore_index=True)

        # Drop the context-padding rows now that they've served their
        # purpose (giving the rolling local-median check enough surrounding
        # data) -- only the originally-requested range gets written.
        if start_date is not None and end_date is not None:
            in_range = (result['analysis_datetime'] >= pd.Timestamp(start_date)) & \
                       (result['analysis_datetime'] <= pd.Timestamp(end_date))
            result = result.loc[in_range].reset_index(drop=True)
            if result.empty:
                return result

        # legacy comparison snapshot
        legacy_sql = f"""
            SELECT analysis_num AS legacy_analysis_num, C_reported AS legacy_mole_fraction
            FROM hats.mole_fractions
            WHERE parameter_num = {pnum}
                AND analysis_num IN ({",".join(str(int(x)) for x in result['legacy_analysis_num'])})
        """ if len(result) else None
        if legacy_sql:
            legacy = pd.DataFrame(self.doquery(legacy_sql))
            if not legacy.empty:
                legacy['legacy_mole_fraction'] = pd.to_numeric(legacy['legacy_mole_fraction'], errors='coerce')
                legacy.loc[legacy['legacy_mole_fraction'] <= -99, 'legacy_mole_fraction'] = np.nan
                result = result.merge(legacy, on='legacy_analysis_num', how='left')
        if 'legacy_mole_fraction' not in result.columns:
            result['legacy_mole_fraction'] = np.nan

        result['parameter_num'] = pnum
        return result

    def upsert_prs_mole_fractions(self, df, verbose=True):
        if df.empty:
            return 0

        sql = """
            INSERT INTO hats.prs_mole_fractions (
                analysis_num, parameter_num, segment_start, raw_response,
                blank_correction, blank_corrected_response, smoothed_response,
                normalized_resp, ref_tank_serial, ref_tank_coef0, mole_fraction,
                nonlinearity_uncorrected, legacy_mole_fraction
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                segment_start = VALUES(segment_start),
                raw_response = VALUES(raw_response),
                blank_correction = VALUES(blank_correction),
                blank_corrected_response = VALUES(blank_corrected_response),
                smoothed_response = VALUES(smoothed_response),
                normalized_resp = VALUES(normalized_resp),
                ref_tank_serial = VALUES(ref_tank_serial),
                ref_tank_coef0 = VALUES(ref_tank_coef0),
                mole_fraction = VALUES(mole_fraction),
                nonlinearity_uncorrected = VALUES(nonlinearity_uncorrected),
                legacy_mole_fraction = VALUES(legacy_mole_fraction);
        """

        def clean(v):
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return None
            return v

        params = []
        n = 0
        for row in df.itertuples(index=False):
            params.append((
                int(row.num),
                int(row.parameter_num),
                row.segment_start.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row.segment_start) else None,
                clean(row.raw_response),
                clean(row.blank_correction),
                clean(row.blank_corrected_response),
                clean(row.smoothed_response),
                clean(row.normalized_resp),
                row.ref_tank_serial,
                clean(row.ref_tank_coef0),
                clean(row.mole_fraction),
                bool(row.nonlinearity_uncorrected),
                clean(row.legacy_mole_fraction),
            ))
            n += 1
            if self.db.doMultiInsert(sql, params):
                params = []
        if params:
            self.db.doMultiInsert(sql, params, all=True)
        if verbose:
            print(f"Upserted {n} rows into hats.prs_mole_fractions.")
        return n

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    def main(self):
        parser = argparse.ArgumentParser(
            description="Independent (non-Matlab) Perseus PR1/PR2 ratio + mole-fraction computation. "
                        "Writes hats.prs_analysis / hats.prs_mole_fractions."
        )
        parser.add_argument('-p', '--parameter-num', type=int, default=29,
                             help="Parameter number to process (default: 29, CFC-11 -- the only supported analyte in v1)")
        parser.add_argument('-s', '--start-date', type=str, default=None,
                             help="Start date YYYY-MM-DD (default: full history)")
        parser.add_argument('-e', '--end-date', type=str, default=None,
                             help="End date YYYY-MM-DD (default: today)")
        parser.add_argument('-i', '--insert', action='store_true',
                             help="Write results to the database. Without this flag, dry-run only.")
        parser.add_argument('--skip-analysis', action='store_true',
                             help="Skip the prs_analysis backfill step (use existing rows).")
        args = parser.parse_args()

        start_date = args.start_date
        end_date = args.end_date or datetime.today().strftime('%Y-%m-%d')

        if not args.skip_analysis:
            adf = self.load_analysis_rows(start_date=start_date, end_date=end_date)
            print(f"Loaded {len(adf)} hats.analysis rows.")
            if args.insert and not adf.empty:
                self.upsert_prs_analysis(adf)
            elif not adf.empty:
                print("Dry run -- pass -i to write to hats.prs_analysis.")

        mf = self.compute_mole_fractions(args.parameter_num, start_date=start_date, end_date=end_date)
        print(f"Computed {len(mf)} mole-fraction rows for parameter {args.parameter_num}.")
        if not mf.empty:
            n_mf = mf['mole_fraction'].notna().sum()
            print(f"  {n_mf}/{len(mf)} rows have a non-null mole_fraction.")
        if args.insert and not mf.empty:
            self.upsert_prs_mole_fractions(mf)
        elif not mf.empty:
            print("Dry run -- pass -i to write to hats.prs_mole_fractions.")

        print(f"Total time: {time.time() - self.t0:.2f} seconds")


if __name__ == "__main__":
    prs = PRS_batch()
    prs.main()
