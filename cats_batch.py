#!/usr/bin/env python3
"""CATS batch calibration fit and mole fraction pipeline.

Analogous to ie3_batch.py. Handles two sequential operations:

  update_fits  -- compute weekly 2-point (or single-point) cal fits and
                  upsert them into hats.ng_response.
  update_runs  -- apply stored ng_response fits to CATS air and tank rows and
                  upsert mole fractions + unc + ng_response_id back to DB.

Reuses the fill-aware, uncertainty-propagating fit math from ie3_cal_test.py
(filter_tanks, cal_tank_serials, cal_tank_coefs, weekly_aggregate,
weekly_cal_fits), passing CATS's port layout explicitly instead of relying
on that module's IE3-specific defaults.

Usage examples:

  # Dry-run: compute fits and mole fractions, print summary, no DB writes
  python3 cats_batch.py -p 5 -c q --site brw -s 2025-01

  # Compute fits + mole fractions and write to DB for one analyte
  python3 cats_batch.py -p 5 -c q --site brw -s 2025-01 -i --fits

  # Process all analytes from instrument start
  python3 cats_batch.py -p all --site brw -s start -i --fits
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from logos_instruments import CATS_Instrument
from ie3_cal_test import (
    filter_tanks,
    weekly_cal_fits,
)


class CATS_batch(CATS_Instrument):

    def __init__(self, site: str = "brw"):
        super().__init__(site=site)
        self.t0 = time.time()

    # ------------------------------------------------------------------
    def _cal_ports(self) -> tuple:
        return (self.CAL1_PORT, self.CAL2_PORT)

    def _plot_ports(self) -> tuple:
        return tuple(sorted(set((self.STANDARD_PORT_NUM,) + self._cal_ports())))

    def _measurement_ports(self) -> tuple:
        """CATS ports containing air or calibration/reference measurements."""
        return tuple(sorted(set(self.AIR_PORTS + list(self._plot_ports()))))

    def _apply_week_methods_to_tanks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use each week's air-row calibration method for its tank rows.

        ``mf_method_num`` is selected and recorded on air rows. Tank rows may
        still be NULL in the database (and therefore loaded with the default
        method), so copy the modal air-row method for each week before applying
        the response fit to every measurement port.
        """
        if df.empty:
            return df.copy()

        out = df.copy()
        analysis_dt = pd.to_datetime(out['analysis_datetime'], utc=True)
        out['_week_start'] = (
            analysis_dt.dt.tz_localize(None).dt.to_period('W-SUN').dt.start_time
        )

        air = out.loc[out['port'].isin(self.AIR_PORTS)]
        week_methods = (
            air.groupby('_week_start')['mf_method_num']
            .agg(lambda values: int(values.mode().iat[0]))
        )
        mapped_methods = out['_week_start'].map(week_methods)
        has_week_method = mapped_methods.notna()
        out.loc[has_week_method, 'mf_method_num'] = mapped_methods[has_week_method]
        out['mf_method_num'] = out['mf_method_num'].astype(int)
        return out.drop(columns='_week_start')

    def _resolve_scale_num(self, pnum: int):
        rows = self.db.doquery(
            f"SELECT idx FROM reftank.scales WHERE parameter_num={pnum} AND current=1"
        )
        return int(rows[0]['idx']) if rows else None

    def _resolve_ref_serial(self, when=None):
        return self.tank_serial_for_port(self.STANDARD_PORT_NUM, when=when)

    @staticmethod
    def _apply_qc_flags(df: pd.DataFrame, qc_flags: pd.DataFrame | None) -> pd.DataFrame:
        """Apply a QC CSV's timestamp flags as additional rejected rows."""
        if df.empty or qc_flags is None or qc_flags.empty:
            return df.copy()
        if 'flags' not in qc_flags.columns or not ({'time', 'run_time'} & set(qc_flags.columns)):
            raise ValueError("QC flags must contain 'flags' and either 'time' or 'run_time'")
        qc = qc_flags.copy()
        if 'time' in qc:
            qc['time'] = pd.to_datetime(qc['time'], errors='coerce', utc=True)
        if 'run_time' in qc:
            qc['run_time'] = pd.to_datetime(qc['run_time'], errors='coerce', utc=True)
        qc = qc.loc[qc['flags'].fillna('').astype(str).str.strip().ne('')
                   & qc['flags'].fillna('').astype(str).str.strip().str.lower().ne('ok')]
        if qc.empty:
            return df.copy()
        by_time = qc.groupby('time')['flags'].agg(
            lambda values: ';'.join(dict.fromkeys(';'.join(values).split(';')))
        ) if 'time' in qc else pd.Series(dtype='object')
        by_run = qc.groupby('run_time')['flags'].agg(
            lambda values: ';'.join(dict.fromkeys(';'.join(values).split(';')))
        ) if 'run_time' in qc else pd.Series(dtype='object')
        out = df.copy()
        times = pd.to_datetime(out['analysis_datetime'], errors='coerce', utc=True)
        runs = pd.to_datetime(out['run_time'], errors='coerce', utc=True)
        out['qc_flags'] = times.map(by_time).fillna('')
        out['qc_flags'] = out['qc_flags'].where(out['qc_flags'].ne(''), runs.map(by_run).fillna(''))
        # A ratio flag identifies the calibration pair, not every air sample
        # sharing that run timestamp. Keep it on calibration ports so those
        # injections are excluded from fitting without painting the whole air
        # sequence as QC-flagged.
        ratio_mask = out['qc_flags'].str.contains('ratio_', na=False)
        if 'port' in out.columns:
            out.loc[ratio_mask & ~out['port'].isin((2, 6)), 'qc_flags'] = ''
        out['rejected'] = out['rejected'].fillna(0).astype(int)
        out.loc[out['qc_flags'].ne(''), 'rejected'] = 1
        return out

    def calc_mole_fraction_from_fits(self, df: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
        """Calculate mole fractions from an in-memory replacement fit table."""
        out = df.copy()
        out['mole_fraction'] = np.nan
        out['unc'] = np.nan
        out['ng_response_id'] = None
        if out.empty:
            return out

        methods = out['mf_method_num'].astype(int)
        direct_mask = methods.eq(self.MF_METHOD_REF)
        if direct_mask.any():
            direct = self.calc_mole_fraction_scale_simple(out.loc[direct_mask])
            out.loc[direct.index, 'mole_fraction'] = direct['mole_fraction']

        fit_table = fits.copy() if fits is not None else pd.DataFrame()
        if fit_table.empty:
            return out
        fit_table['week_start'] = pd.to_datetime(fit_table['week_start'], utc=True)
        fit_table = fit_table.sort_values('week_start')
        for idx, row in out.loc[~direct_mask].iterrows():
            resp = row.get('normalized_resp')
            if pd.isna(resp):
                continue
            analysis_dt = pd.to_datetime(row['analysis_datetime'], utc=True)
            applicable = fit_table.loc[fit_table['week_start'] <= analysis_dt]
            if applicable.empty:
                continue
            fit = applicable.iloc[-1]
            slope = float(fit['slope'])
            intercept = float(fit['intercept'])
            method = int(row['mf_method_num'])
            out.at[idx, 'mole_fraction'] = slope * float(resp) + intercept if method == self.MF_METHOD_CAL12 else slope * float(resp)
            unc_fit = float(fit.get('unc_ref_pred', 0.0) or 0.0)
            sigma_ref = float(fit.get('sigma_ref_weekly', 0.0) or 0.0)
            out.at[idx, 'unc'] = np.sqrt(unc_fit ** 2 + (slope * sigma_ref) ** 2)
        return out

    def _weekly_tank_data(self, tanks: pd.DataFrame) -> pd.DataFrame:
        """Aggregate tank responses by week and the tank installed per row."""
        tanks = tanks.copy()
        tanks['tank_serial'] = None
        for port in tanks['port'].dropna().unique():
            mask = tanks['port'].eq(port)
            tanks.loc[mask, 'tank_serial'] = self.tank_serials_for_dates(
                int(port), tanks.loc[mask, 'analysis_datetime']
            )
        tanks['week_start'] = (
            pd.to_datetime(tanks['analysis_datetime'], utc=True)
            .dt.tz_localize(None).dt.to_period('W-SUN').dt.start_time
        )
        return (
            tanks.dropna(subset=['tank_serial'])
            .groupby(['port', 'tank_serial', 'week_start'])['normalized_resp']
            .agg(mean='mean', std='std', count='count')
            .reset_index()
        )

    # ------------------------------------------------------------------
    def update_fits(
        self,
        pnum: int,
        channel=None,
        start_date=None,
        end_date=None,
        verbose: bool = False,
        qc_flags: pd.DataFrame | None = None,
        method_override: int | None = None,
    ):
        """Compute weekly cal fits and return (fits_df, scale_num, ref_serial, channel_str).

        Returns empty DataFrame on failure.
        """
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
        df = self._apply_qc_flags(df, qc_flags)

        cal_ports = self._cal_ports()
        tanks = filter_tanks(df, plot_ports=self._plot_ports())
        if tanks.empty:
            if verbose:
                print("No unflagged cal/ref port data; skipping fits.")
            return pd.DataFrame(), None, None, None

        weekly = self._weekly_tank_data(tanks)
        if weekly.empty:
            if verbose:
                print("No dated cal/ref tank configuration found; skipping fits.")
            return pd.DataFrame(), None, None, None

        # Each week's calibration method is recorded per-analyte on its air rows
        # (get_week_mf_method). Fit each week according to its own method so a
        # --fits run never clobbers a method chosen in the GUI for that week.
        fit_rows = []
        warned = set()
        for week_start, wk in weekly.groupby('week_start'):
            method = (int(method_override) if method_override is not None
                      else self.get_week_mf_method(pnum, channel, week_start))
            if not self.uses_ng_response_fit(method):
                # method 1 (ref): mole fractions come from the ref-tank coef0
                # directly in update_runs; no weekly ng_response fit to store.
                if verbose:
                    print(f"  week {week_start.date()}: method "
                          f"{self.MF_METHOD_LABELS.get(method, method)} stores no "
                          f"fit; skipping (handled by update_runs)")
                continue
            force_zero, single_port = self.fit_params_for_method(method)
            required_ports = (single_port,) if force_zero else cal_ports
            coefs_week = {}
            transition = False
            for port in required_ports:
                port_rows = wk.loc[wk['port'].eq(port)]
                serials = port_rows['tank_serial'].dropna().unique()
                if len(serials) != 1:
                    transition = len(serials) > 1
                    break
                serial = serials[0]
                fills = self.scale_assignment_history(serial, pnum)
                if not fills:
                    warning_key = (serial, pnum)
                    if warning_key not in warned:
                        print(f"  WARNING: no scale_assignments for tank {serial} "
                              f"(port {port}), pnum {pnum}")
                        warned.add(warning_key)
                    break
                coefs_week[port] = fills
            if len(coefs_week) != len(required_ports):
                if verbose:
                    reason = "tank changed during week" if transition else "missing tank data/assignment"
                    print(f"  week {week_start.date()}: {reason}; skipping fit")
                continue
            wkfit = weekly_cal_fits(wk, coefs_week, force_zero=force_zero)
            if not wkfit.empty:
                wkfit['method_num'] = method
                ref_rows = wk.loc[wk['port'].eq(self.STANDARD_PORT_NUM), 'tank_serial']
                wkfit['ref_serial'] = (
                    ref_rows.iat[-1] if not ref_rows.empty
                    else self._resolve_ref_serial(week_start)
                )
                fit_rows.append(wkfit)

        if not fit_rows:
            if verbose:
                print("No weekly fits computed (too few points).")
            return pd.DataFrame(), None, None, None

        fits = pd.concat(fit_rows, ignore_index=True)

        # sigma_ref: weekly std of ref-port normalized_resp from raw data
        ref_raw = tanks[tanks['port'] == self.STANDARD_PORT_NUM][
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
                serial_number=row.get('ref_serial') or ref_serial,
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
        qc_flags: pd.DataFrame | None = None,
        fits_override: pd.DataFrame | None = None,
        method_override: int | None = None,
    ) -> pd.DataFrame:
        """Apply stored ng_response fits to CATS air and tank rows.

        Returns df with mole_fraction, unc, and ng_response_id populated.
        """
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

        df = self._apply_qc_flags(df, qc_flags)

        df = df.loc[df['port'].isin(self._measurement_ports())].copy()
        if df.empty:
            if verbose:
                print("No air or tank measurement rows in loaded data.")
            return pd.DataFrame()

        df = self._apply_week_methods_to_tanks(df)
        if method_override is not None:
            df['mf_method_num'] = int(method_override)
        if fits_override is None:
            df = self.calc_mole_fraction(df)
        else:
            df = self.calc_mole_fraction_from_fits(df, fits_override)
        # A zero-height/zero-RT record is an empty integration, not a zero
        # mole fraction. Keep it in dry-run output for QC review, but do not
        # create/update a mole-fraction value for it.
        empty_peak = pd.Series(False, index=df.index)
        if {'height', 'retention_time'}.issubset(df.columns):
            empty_peak = (pd.to_numeric(df['height'], errors='coerce').eq(0)
                          & pd.to_numeric(df['retention_time'], errors='coerce').eq(0))
        df.loc[empty_peak, ['mole_fraction', 'unc']] = np.nan
        if 'ng_response_id' in df.columns:
            df.loc[empty_peak, 'ng_response_id'] = None

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
            help="GC channel (q, f, cc, a...).",
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
            help=f"CATS site code (default: {self.site}). Valid: "
                 f"{sorted(self.INST_NUM_BY_SITE)}.",
        )
        parser.add_argument(
            '--fits', action='store_true',
            help="Run update_fits() before update_runs() to (re)compute weekly cal fits.",
        )
        parser.add_argument(
            '--qc-flags', type=Path, default=None,
            help="CSV from cats_peak_qc.py; flagged timestamps are rejected for fit inputs.",
        )
        parser.add_argument(
            '-v', '--verbose', action='store_true',
            help="Print progress detail.",
        )
        args = parser.parse_args()
        args.qc_flags_df = None
        if args.qc_flags is not None:
            if not args.qc_flags.is_file():
                parser.error(f"QC flag CSV does not exist: {args.qc_flags}")
            args.qc_flags_df = pd.read_csv(args.qc_flags)

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
        fits_override = None
        if args.fits:
            fits, scale_num, ref_serial, channel_str = self.update_fits(
                pnum, channel=channel,
                start_date=args.start_date, end_date=args.end_date,
                verbose=args.verbose, qc_flags=args.qc_flags_df,
            )
            # No fits is normal when every week uses method 1 (ref); still run
            # update_runs so those weeks' mole fractions get (re)computed.
            if fits.empty:
                print(f"  No weekly fits to upsert for pnum={pnum} "
                      "(ref-only or insufficient cal data).")
            else:
                print(f"  Computed {len(fits)} weekly fits for pnum={pnum}")
                fits_override = fits
                if args.insert and scale_num is not None:
                    self._upsert_fits(
                        fits, scale_num, ref_serial, channel_str,
                        verbose=args.verbose,
                    )
                    print(f"  Upserted {len(fits)} fits into ng_response.")

        df = self.update_runs(
            pnum, channel=channel,
            start_date=args.start_date, end_date=args.end_date,
            verbose=args.verbose, qc_flags=args.qc_flags_df,
            fits_override=fits_override,
        )
        if df.empty:
            print(f"  No air or tank measurement rows for pnum={pnum}.")
            return

        n_ok = df['mole_fraction'].notna().sum()
        print(f"  Mole fractions: {n_ok}/{len(df)} rows for pnum={pnum}")

        if args.insert:
            empty_peak = (pd.to_numeric(df['height'], errors='coerce').eq(0)
                          & pd.to_numeric(df['retention_time'], errors='coerce').eq(0))
            write_df = df.loc[~empty_peak].copy()
            self.upsert_mole_fractions(write_df)
            print(f"  Upserted {len(write_df)} rows into ng_insitu_mole_fractions "
                  f"(skipped {int(empty_peak.sum())} empty integrations).")


if __name__ == '__main__':
    cats = CATS_batch()
    cats.main()
