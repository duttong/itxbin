#!/usr/bin/env python3
"""IE3 batch calibration fit and mole fraction pipeline.

Analogous to fe3_batch.py. Handles two sequential operations:

  update_fits  -- compute weekly 2-point (or single-point) cal fits and
                  upsert them into hats.ng_response.
  update_runs  -- apply stored ng_response fits to IE3 air-port rows and
                  upsert mole fractions + unc + ng_response_id back to DB.

Usage examples:

  # Dry-run: compute fits and mole fractions, print summary, no DB writes
  python3 ie3_batch.py -p 22 -c b --site smo -s 2025-01

  # Compute fits + mole fractions and write to DB for one analyte
  python3 ie3_batch.py -p 22 -c b --site smo -s 2025-01 -i --fits

  # Process all analytes from instrument start
  python3 ie3_batch.py -p all --site smo -s start -i --fits
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from logos_instruments import IE3_Instrument
from ie3_cal_test import (
    cal_tank_coefs,
    cal_tank_serials,
    filter_tanks,
    weekly_aggregate,
    weekly_cal_fits,
)

CCL4_PNUM = 37   # uses method 3 (single-point through origin, port 9)
REF_PORT = 5     # STANDARD_PORT_NUM — normalization reference

# IE3 weekly cal fits are never computed earlier than this date. A start_date
# before the floor is raised to it; the GUI run list only offers (Cal) weeks
# that exist in hats.ng_response, so this is the single authoritative guard.
CAL_MIN_DATE = "2026-03-01"


def _clamp_cal_start(start_date):
    """Floor a calibration-fit start_date at CAL_MIN_DATE.

    Accepts None (recent default), a YYMM string, or YYYY-MM-DD. None and
    unparseable values pass through unchanged; anything earlier than the floor
    is raised to CAL_MIN_DATE (with a printed notice).
    """
    if start_date is None:
        return start_date
    if isinstance(start_date, str) and len(start_date) == 4:
        ts = pd.to_datetime(start_date, format="%y%m", errors="coerce")
    else:
        ts = pd.to_datetime(start_date, errors="coerce")
    if pd.isna(ts):
        return start_date
    if ts < pd.Timestamp(CAL_MIN_DATE):
        print(f"  NOTE: start_date {start_date} is before the calibration "
              f"floor {CAL_MIN_DATE}; clamping to {CAL_MIN_DATE}")
        return CAL_MIN_DATE
    return start_date


class IE3_batch(IE3_Instrument):

    def __init__(self, site: str = "smo"):
        super().__init__(site=site)
        self.t0 = time.time()

    # ------------------------------------------------------------------
    def _resolve_scale_num(self, pnum: int):
        rows = self.db.doquery(
            f"SELECT idx FROM reftank.scales WHERE parameter_num={pnum} AND current=1"
        )
        return int(rows[0]['idx']) if rows else None

    def _resolve_ref_serial(self):
        pc = self.port_config
        if pc is None:
            return None
        mask = (pc['site_num'] == self.site_num) & (pc['port_num'] == REF_PORT)
        rows = pc.loc[mask, 'label']
        return rows.iat[0] if not rows.empty else None

    # ------------------------------------------------------------------
    def update_fits(
        self,
        pnum: int,
        channel=None,
        start_date=None,
        end_date=None,
        verbose: bool = False,
    ):
        """Compute weekly cal fits and return (fits_df, scale_num, ref_serial, channel_str).

        Returns empty DataFrame on failure.
        """
        start_date = _clamp_cal_start(start_date)

        if verbose:
            print(f"update_fits: pnum={pnum} channel={channel} site={self.site} "
                  f"start={start_date} end={end_date} (per-week method)")

        df = self.load_data(
            pnum=pnum,
            channel=channel,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
        )
        if df.empty:
            if verbose:
                print("No data loaded; skipping fits.")
            return pd.DataFrame(), None, None, None

        tanks = filter_tanks(df)
        if tanks.empty:
            if verbose:
                print("No unflagged cal/ref port data; skipping fits.")
            return pd.DataFrame(), None, None, None

        weekly = weekly_aggregate(tanks)

        serials = cal_tank_serials(self)
        if not serials:
            print(f"  WARNING: no cal tank serials found for site={self.site}")
            return pd.DataFrame(), None, None, None

        coefs = cal_tank_coefs(self, pnum, serials)
        if not coefs:
            print("  WARNING: no scale_assignments found for cal tanks")
            return pd.DataFrame(), None, None, None

        # Each week's calibration method is recorded per-analyte on its air rows
        # (get_week_mf_method). Fit each week according to its own method so a
        # --fits run never clobbers a method chosen in the GUI for that week.
        fit_rows = []
        for week_start, wk in weekly.groupby('week_start'):
            method = self.get_week_mf_method(pnum, channel, week_start)
            if not self.uses_ng_response_fit(method):
                # method 1 (ref): mole fractions come from the ref-tank coef0
                # directly in update_runs; no weekly ng_response fit to store.
                if verbose:
                    print(f"  week {week_start.date()}: method "
                          f"{self.MF_METHOD_LABELS.get(method, method)} stores no "
                          f"fit; skipping (handled by update_runs)")
                continue
            force_zero, single_port = self.fit_params_for_method(method)
            if force_zero:
                if single_port not in coefs:
                    if verbose:
                        print(f"  week {week_start.date()}: method "
                              f"{self.MF_METHOD_LABELS.get(method, method)} needs "
                              f"port {single_port} coefs; skipping")
                    continue
                coefs_week = {single_port: coefs[single_port]}
            else:
                if len(coefs) < 2:
                    if verbose:
                        print(f"  week {week_start.date()}: cal12 needs both cal "
                              f"tanks; skipping")
                    continue
                coefs_week = coefs
            wkfit = weekly_cal_fits(wk, coefs_week, force_zero=force_zero)
            if not wkfit.empty:
                wkfit['method_num'] = method
                fit_rows.append(wkfit)

        if not fit_rows:
            if verbose:
                print("No weekly fits computed (too few points).")
            return pd.DataFrame(), None, None, None

        fits = pd.concat(fit_rows, ignore_index=True)

        # sigma_ref: weekly std of port-5 normalized_resp from raw data
        ref_raw = tanks[tanks['port'] == REF_PORT][
            ['analysis_datetime', 'normalized_resp']
        ].copy()
        ref_raw['week_start'] = (
            ref_raw['analysis_datetime']
            .dt.tz_localize(None)
            .dt.to_period('W-SUN')
            .dt.start_time
        )
        ref_weekly_std = (
            ref_raw.groupby('week_start')['normalized_resp']
            .std()
            .rename('sigma_ref_weekly')
        )
        fits = fits.join(ref_weekly_std, on='week_start')
        fits['sigma_ref_weekly'] = fits['sigma_ref_weekly'].fillna(0.0)

        scale_num = self._resolve_scale_num(pnum)
        if scale_num is None:
            print(f"  WARNING: no current scale for pnum={pnum}; can't upsert ng_response")
            return fits, None, None, None

        ref_serial = self._resolve_ref_serial() or ''
        channel_str = channel or ''

        if verbose:
            print(f"  scale_num={scale_num} ref_serial={ref_serial} "
                  f"computed {len(fits)} weekly fits")

        return fits, scale_num, ref_serial, channel_str

    def _upsert_fits(
        self,
        fits: pd.DataFrame,
        scale_num: int,
        ref_serial: str,
        channel_str: str,
        verbose: bool = False,
    ) -> None:
        """Upsert each row of a fits DataFrame into hats.ng_response."""
        for _, row in fits.iterrows():
            coef0 = float(row['intercept'])
            coef1 = float(row['slope'])
            unc_fit = float(row['unc_ref_pred']) if pd.notna(row.get('unc_ref_pred')) else 0.0
            sigma_ref = float(row.get('sigma_ref_weekly', 0.0) or 0.0)
            row_id = self.upsert_ng_response(
                inst_num=self.inst_num,
                site=self.site,
                run_date=row['week_start'],
                channel=channel_str,
                scale_num=scale_num,
                coef0=coef0,
                coef1=coef1,
                unc_fit=unc_fit,
                sigma_ref=sigma_ref,
                serial_number=ref_serial,
            )
            if verbose:
                print(f"  upserted id={row_id} week={row['week_start']} "
                      f"coef0={coef0:.4g} coef1={coef1:.4g} "
                      f"unc_fit={unc_fit:.4g} sigma_ref={sigma_ref:.4g}")

    # ------------------------------------------------------------------
    def update_runs(
        self,
        pnum: int,
        channel=None,
        start_date=None,
        end_date=None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Apply stored ng_response fits to IE3 air-port rows.

        Returns df with mole_fraction, unc, and ng_response_id populated.
        """
        start_date = _clamp_cal_start(start_date)

        if verbose:
            print(f"update_runs: pnum={pnum} channel={channel} site={self.site} "
                  f"start={start_date} end={end_date}")

        df = self.load_data(
            pnum=pnum,
            channel=channel,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
        )
        if df.empty:
            return pd.DataFrame()

        df = df.loc[df['port'].isin(self.AIR_PORTS)].copy()
        if df.empty:
            if verbose:
                print("No air-port rows in loaded data.")
            return pd.DataFrame()

        df = self.calc_mole_fraction(df)
        df.loc[df['height'] == 0, 'mole_fraction'] = 0.0

        if verbose:
            n_ok = df['mole_fraction'].notna().sum()
            print(f"  Mole fractions: {n_ok}/{len(df)} rows "
                  f"(elapsed {time.time() - self.t0:.1f}s)")

        return df

    # ------------------------------------------------------------------
    def main(self):
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            '-p', '--parameter-num', type=str, required=True,
            help="Parameter number or 'all' to process all analytes.",
        )
        parser.add_argument(
            '-c', '--channel', type=str, default=None,
            help="GC channel (a, b, c).",
        )
        parser.add_argument(
            '-s', '--start-date', type=str, default=None,
            help="Start date: YYMM, YYYY-MM-DD, or 'start' for instrument start.",
        )
        parser.add_argument(
            '-e', '--end-date', type=str, default=None,
            help="End date: YYMM, YYYY-MM-DD, or 'end' for today.",
        )
        parser.add_argument(
            '-i', '--insert', action='store_true',
            help="Write results to DB (ng_response and ng_insitu_mole_fractions).",
        )
        parser.add_argument(
            '--site', type=str, default=self.site,
            help=f"IE3 site code (default: {self.site}).",
        )
        parser.add_argument(
            '--fits', action='store_true',
            help="Run update_fits() before update_runs() to (re)compute weekly cal fits.",
        )
        parser.add_argument(
            '-v', '--verbose', action='store_true',
            help="Print progress detail.",
        )
        args = parser.parse_args()

        # Re-init with correct site if overridden on command line
        if args.site != self.site:
            self.__init__(site=args.site)

        # Resolve date keywords
        if args.start_date and args.start_date.lower() == 'start':
            s = self.start_date
            args.start_date = f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s) == 8 else s
            print(f"Using instrument start date: {args.start_date}")
        if args.end_date and args.end_date.lower() == 'end':
            args.end_date = None
            print("Using end date: today")

        if args.parameter_num.lower() == 'all':
            sql = (f"SELECT param_num, channel, display_name "
                   f"FROM hats.analyte_list WHERE inst_num = {self.inst_num}")
            adf = pd.DataFrame(self.db.doquery(sql))
            for r in adf.itertuples(index=False):
                pnum = int(r.param_num)
                ch = r.channel or None
                print(f"\n=== {r.display_name} (pnum={pnum} channel={ch}) ===")
                self._process_one(pnum, ch, args)
        else:
            pnum = int(args.parameter_num)
            ch = args.channel
            self._process_one(pnum, ch, args)

        print(f"\nDone. Total elapsed: {time.time() - self.t0:.1f}s")

    def _process_one(self, pnum: int, channel, args) -> None:
        if args.fits:
            fits, scale_num, ref_serial, channel_str = self.update_fits(
                pnum, channel=channel,
                start_date=args.start_date, end_date=args.end_date,
                verbose=args.verbose,
            )
            # No fits is normal when every week uses method 1 (ref); still run
            # update_runs so those weeks' mole fractions get (re)computed.
            if fits.empty:
                print(f"  No weekly fits to upsert for pnum={pnum} "
                      "(ref-only or insufficient cal data).")
            else:
                print(f"  Computed {len(fits)} weekly fits for pnum={pnum}")
                if args.insert and scale_num is not None:
                    self._upsert_fits(
                        fits, scale_num, ref_serial, channel_str,
                        verbose=args.verbose,
                    )
                    print(f"  Upserted {len(fits)} fits into ng_response.")

        df = self.update_runs(
            pnum, channel=channel,
            start_date=args.start_date, end_date=args.end_date,
            verbose=args.verbose,
        )
        if df.empty:
            print(f"  No air-port rows for pnum={pnum}.")
            return

        n_ok = df['mole_fraction'].notna().sum()
        print(f"  Mole fractions: {n_ok}/{len(df)} rows for pnum={pnum}")

        if args.insert:
            self.upsert_mole_fractions(df)
            print(f"  Upserted {n_ok} rows into ng_insitu_mole_fractions.")


if __name__ == '__main__':
    ie3 = IE3_batch()
    ie3.main()
