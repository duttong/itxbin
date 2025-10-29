#! /usr/bin/env python

import json
import argparse
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pathlib import Path
from warnings import warn
from concurrent.futures import ThreadPoolExecutor

from logos_instruments import FE3_Instrument as fe3_inst

class FE3_Prepare(fe3_inst):
    """ Class for preparing data for the HATS DB. This class is used to prepare data
        from the FE3 instrument for the HATS DB, including running queries and
        inserting data into the database. """

    def __init__(self):
        super().__init__()
        self.current_year = pd.Timestamp.now().year

    @staticmethod
    def run_type(runseq):
        # use upper to find 'f' and 'F'
        type = 'flask' if runseq.upper().find('F') > 0 else 'other'
        return type

    @staticmethod
    def return_datetime(metafile):
        """ Returns a datatime from the metafile name.
            for example: meta_20200207-183526.json
        """
        m = metafile.name
        dt = pd.to_datetime(m[5:20])
        return dt

    def two_digit_year(self, year) -> str:
        """Return the last two digits of a year as a zero-padded string.

        Args:
            year (int or str): A four-digit year (e.g. 2025) or a two-digit year (e.g. "25" or 25).

        Returns:
            str: Two-digit year, zero-padded (e.g. "05", "25", "99").

        Raises:
            ValueError: If `year` isn’t a digit, or is outside 0–9999.
        """
        try:
            y = int(year)
        except (TypeError, ValueError):
            raise ValueError(f"Year must be an integer or digit string, got {year!r}")

        # Validate plausible range
        if not (0 <= y <= 9999):
            raise ValueError(f"Year {y!r} out of range 0–9999")

        # Format last two digits, zero-padded
        return f"{y % 100:02d}"

    def runs_df_yy(self, generate=True, year=2025):
        """ Read meta_* files for run information including sequence and port assignments """
        yy = self.two_digit_year(year)
        incoming = self.gc_dir / yy / 'incoming'
        out_pqt = self.gc_dir / yy / 'fe3_runs.parquet'
        engine = 'pyarrow'

        # Read save parquet file if they exist
        if not generate and out_pqt.exists():
            return pd.read_parquet(out_pqt, engine=engine)

        # stale-check - read parquet data if the meta data files are older
        if out_pqt.exists():
            pkl_mtime   = out_pqt.stat().st_mtime
            latest_meta = max(p.stat().st_mtime for p in incoming.rglob('meta_*.json'))
            if pkl_mtime >= latest_meta:
                return pd.read_parquet(out_pqt, engine=engine)

        # parallel parse
        def parse_meta(path):
            try:
                data = json.loads(path.read_text())
            except Exception as e:
                warn(f"Failed to parse {path!r}: {e}")
                return None
            return {
                'time':    self.return_datetime(path),
                'dir':     path.name[5:20],
                'seq':     data[0],
                'flasks':  list(data[1].values()),
                'ports':   list(data[2].values()),
                'type':    data[3] if len(data)>3 else self.run_type(data[0]),
            }

        paths = list(incoming.rglob('meta_*.json'))
        with ThreadPoolExecutor() as ex:
            records = [r for r in ex.map(parse_meta, paths) if r]

        # assemble & persist to parquet files
        df = (pd.DataFrame.from_records(records)
                .set_index('time')
                .sort_index())
        df.to_parquet(out_pqt, index=True, engine=engine)
        return df

    def runs_df(self):
        """Grab all two-digit-year subdirs, load old years cached parquet files,
           regenerate current year, and concat into one DF."""
        # find exactly two-digit numeric dirs
        years = sorted(
            int(d.name) for d in self.gc_dir.iterdir()
            if d.is_dir() and d.name.isdigit() and len(d.name)==2
        )
        dfs = []
        *past, current = years
        for yy in past:
            dfs.append(self.runs_df_yy(generate=False, year=f"{yy:02d}"))
        dfs.append(self.runs_df_yy(generate=True,  year=f"{current:02d}"))
        return pd.concat(dfs, axis=0)

    @staticmethod
    def _seq2list(df, drop_initial=None):
        """Turn the raw seq string into two lists (descriptions, flask_indices)
        of exactly len(df) items, dropping a leading header char only if needed."""

        # how many points to fill
        n = len(df)

        # raw values
        raw_seq    = df['seq'].iat[0]
        raw_ports  = df['ports'].iat[0]
        raw_flasks = df['flasks'].iat[0]

        # auto‐decide whether to drop the first char by comparing lengths
        if drop_initial is None:
            # if seq is exactly one longer than rows, assume header char
            drop_initial = (len(raw_seq) == n + 1)

        seq = raw_seq[1:] if drop_initial else raw_seq

        # build stable lists from your dicts (if they’re dicts)
        ports  = list(raw_ports.values())  if isinstance(raw_ports, dict)  else raw_ports
        flasks = list(raw_flasks.values()) if isinstance(raw_flasks, dict) else raw_flasks

        des = []   # “port name or flask id”
        idx = []   # flask index (or '' for port)
        fcnt = 0   # how many flasks consumed

        # only iterate exactly n events
        for ch in seq[:n]:
            if ch == 'F':          # new flask
                des.append(flasks[fcnt])
                idx .append(fcnt)
                fcnt += 1
            elif ch == 'f':        # repeat last flask
                des.append(flasks[fcnt-1])
                idx .append(fcnt-1)
            elif ch.isdigit():     # a port hit
                p = int(ch)
                des.append(ports[p])
                idx .append('')
            else:
                raise ValueError(f"Bad seq char {ch!r} in run {df.iloc[0].dir}")

        # safety check
        if len(des) != n or len(idx) != n:
            if df.iloc[0].time.strftime('%Y-%m-%d %H:%M:%S') != '2019-12-20 18:20:00':
                print(f'Unexpected length mismatch in _seq2list {df.iloc[0].time}')
                raise ValueError("Length of des/idx does not match df length")
            return None, None

        return des, idx

    @staticmethod
    def split_pairid_flaskid(id_str: str):
        """ method splits a pairid-flaskid string. 
            If a serial number of a tank is in this field it with leave it as the pairid variable. 
        """
        if str(id_str).find('-') > 0:
            pairid, flaskid = id_str.split('-')
            try:
                test = int(pairid)
                return pairid, flaskid, 'flask'
            except ValueError:
                pass
        return None, None, id_str
    
    def fe3_merged_data(self, duration=2):
        """ Method to merge GCwerks results with data contained in the run meta_* files.
            The meta files have the run sequence and port number definitions.
        
        Set duration = 'all' for all of the fe3 data.
            Otherwise duration represents months since the most recent GCwerks data. """
            
        gcw = FE3_GCwerks()
        w_df = gcw.gcwerks_df()
        r_df = self.runs_df()
        
        if duration != 'all':
            last_date = w_df.index.max()
            w_start_date = last_date - DateOffset(months=duration) - DateOffset(days=1)
            start_date = last_date - DateOffset(months=duration)
            w_df = w_df[w_start_date:]
            r_df = r_df[start_date:]

        df = pd.merge_asof(w_df, r_df,
            on='time', direction='backward', tolerance=pd.Timedelta('24h'))
        
        #flask_runs = df.loc[df.type == 'flask'].dir.unique()
        runs = df.dir.unique()

        def assign_seq_cols(grp):
            if grp.name not in runs:
                return grp

            port_id, flask_port = self._seq2list(grp)
            return grp.assign(
                port_id    = port_id,
                flask_port = flask_port,
            )

        clean = df.groupby('dir', group_keys=False) \
                .apply(assign_seq_cols, include_groups=False)
        clean['dir'] = df['dir']   # add dir back to the df
        clean['run_time'] = pd.to_datetime(df['dir'])
        
        t = self.run_type_num()
        clean['run_type_num'] = clean['type'].astype(str).map(t).astype(int)
    
        splits = clean['port_id'].apply(self.split_pairid_flaskid)

        # turn the Series of 3-tuples into a DataFrame with the same index
        split_df = pd.DataFrame(
            splits.tolist(),
            index=clean.index,
            columns=['pair_id_num','flask_id','port_info']
        )
        split_df = split_df.fillna({'pair_id_num': 0, 'flask_id': 0})

        clean = pd.concat([clean, split_df], axis=1)
        clean['pair_id_num'] = pd.to_numeric(clean['pair_id_num'], errors='coerce').astype('Int64')
        clean['flask_id']    = pd.to_numeric(clean['flask_id'],    errors='coerce').astype('Int64')

        return clean

    def gcwerks_2_hatsdb(self, df: pd.DataFrame, batch_size: int = 500):
        """
        Bulk upsert a GCwerks DataFrame into hats.ng_analysis and hats.ng_mole_fractions,
        inserting new rows and updating any changed fields.
        Assumes df has columns:
          time, run_time, run_type_num, port, port_info,
          flask_port, pair_id_num, flask_id,
        plus for each molecule fields: <mol>_ht, <mol>_area, <mol>_rt.
        """
        # 1) Prepare DataFrame and format datetime fields as strings
        df = df.copy()
        # format datetime fields
        df['analysis_time_str'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['run_time_str']      = df['run_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # replace NaN or empty strings with defaults for numeric columns
        df['port_info']    = df['port_info'].fillna('').astype(str)
        df['flask_port']   = pd.to_numeric(df['flask_port'], errors='coerce').fillna(0).astype(int)
        df['pair_id_num']  = pd.to_numeric(df['pair_id_num'], errors='coerce').fillna(0).astype(int)
        df['flask_id']     = pd.to_numeric(df['flask_id'], errors='coerce').fillna(0).astype(int)
        
        # 2) Bulk upsert ng_analysis (insert new, update existing)
        analysis_sql = """
            INSERT INTO hats.ng_analysis (
                analysis_time,
                inst_num,
                run_time,
                run_type_num,
                port,
                port_info,
                flask_port,
                pair_id_num,
                flask_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                run_time      = VALUES(run_time),
                run_type_num  = VALUES(run_type_num),
                port          = VALUES(port),
                port_info     = VALUES(port_info),
                flask_port    = VALUES(flask_port),
                pair_id_num   = VALUES(pair_id_num),
                flask_id      = VALUES(flask_id)
        """
        analysis_params = [
            (
                r.analysis_time_str,
                self.inst_num,
                r.run_time_str,
                int(r.run_type_num),
                int(r.port),
                r.port_info,
                r.flask_port,
                int(r.pair_id_num),
                int(r.flask_id)
            )
            for r in df.itertuples(index=False)
        ]
        for i in range(0, len(analysis_params), batch_size):
            batch = analysis_params[i : i + batch_size]
            self.db.doMultiInsert(analysis_sql, batch, all=True)

        # 3) Fetch mapping of analysis_time -> num after upsert
        unique_times = df['analysis_time_str'].unique().tolist()
        placeholders = ','.join(['%s'] * len(unique_times))
        select_sql = f"""
            SELECT analysis_time, num
              FROM hats.ng_analysis
             WHERE inst_num = %s
               AND analysis_time IN ({placeholders})
        """
        rows = self.db.doquery(select_sql, [self.inst_num] + unique_times)
        existing = {
            r['analysis_time'].strftime('%Y-%m-%d %H:%M:%S'): r['num']
            for r in rows
        }

        # 4) Bulk upsert ng_mole_fractions (insert new, update existing)
        # I need to add qc status and flags to this insert
        mole_sql = """
            INSERT INTO hats.ng_mole_fractions (
              analysis_num,
              parameter_num,
              channel,
              height,
              area,
              retention_time
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              height         = VALUES(height),
              area           = VALUES(area),
              retention_time = VALUES(retention_time)
        """
        mole_params = []
        params = self.query_analytes()
        for r in df.itertuples(index=False):
            analysis_num = existing[r.analysis_time_str]
            for ch, mol_list in self.gc_channels.items():
                for mol in mol_list:
                    param = params[mol]
                    # adjust molecule key for channel-specific naming
                    key = f"{mol}{ch}" if mol in ('CFC11', 'CFC113') else ('MC' if mol == 'CH3CCl3' else mol)
                    height = getattr(r, f"{key}_ht")
                    area   = getattr(r, f"{key}_area")
                    rt     = getattr(r, f"{key}_rt")
                    mole_params.append((analysis_num, param, ch, height, area, rt))

        for i in range(0, len(mole_params), batch_size):
            batch = mole_params[i : i + batch_size]
            self.db.doMultiInsert(mole_sql, batch, all=True)
               
    def upsert_ng_analysis(df_slice):
        """ Use this method manually if there is a problem with ng_analysis """
        
        sql = """
        INSERT INTO hats.ng_analysis (
            analysis_time,
            inst_num,
            run_time,
            run_type_num,
            port,
            port_info,
            flask_port,
            pair_id_num,
            flask_id
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            run_time      = VALUES(run_time),
            run_type_num  = VALUES(run_type_num),
            port          = VALUES(port),
            port_info     = VALUES(port_info),
            flask_port    = VALUES(flask_port),
            pair_id_num   = VALUES(pair_id_num),
            flask_id      = VALUES(flask_id)
        ;
        """

        args = []
        for row in df_slice.itertuples(index=False):
            args.append((
                # these nine must line up with your INSERT columns:
                row.time,              # analysis_time
                fe3.inst_num,          # inst_num
                row.run_time    or None,
                int(row.run_type_num) if pd.notnull(row.run_type_num) else None,
                row.port        or None,
                row.port_id     or None,
                row.flask_port  or None,
                row.pair_id_num or 0,
                row.flask_id    or 0,
            ))
            if fe3.db.doMultiInsert(sql, args):
                args = []
                
        r = fe3.db.doMultiInsert(sql, args, all=True)

class FE3_GCwerks(fe3_inst):
    """ Class and methods for reading GCwerks result file.
        Currently export GCwerks results into a single file for all years,
        may need to break into individual years for loading performance. """

    def __init__(self):
        super().__init__()
        self.gcwerksexport = self.gc_dir / 'results' / 'fe3_gcwerks_all.csv'

    def gcwerks_df(self):
        df = pd.read_csv(self.gcwerksexport,
            index_col=0, skipinitialspace=True, parse_dates=True)

        # every so often there is a duplicate time row returned from gcwerks        
        df = df.reset_index().drop_duplicates('time', keep='last')
        df = df.set_index('time')
        return df


if __name__ == '__main__':
    import time

    opt = argparse.ArgumentParser(
        description = """Load GCwerks results for FE3 into the HATS DB.
            The default behaviour is to work on the last two months of the exported data stored in
            /hats/gc/fe3/results/fe3_gcwerks_all.csv file which is created and updated by fe3_export.py """
    )
    opt.add_argument("-a", "--all", action="store_true",
                     dest="allyears", help="process all of the data (all years)")
    opt.add_argument("-y", "--year", action="store", 
                     dest="yyyy", help=f"operate on a years worth of GCwerks results.")
    
    options = opt.parse_args()
    
    t0 = time.time()
    fe3 = FE3_Prepare()
    
    if options.yyyy:
        df = fe3.fe3_merged_data(duration='all')
        df = df.loc[df['time'].dt.year == int(options.yyyy)]
    elif options.allyears:
        df = fe3.fe3_merged_data(duration='all')
    else:
        df = fe3.fe3_merged_data(duration=2)
    
    fe3.gcwerks_2_hatsdb(df)
    print(f'Execution time {time.time()-t0:3.3f} seconds on {df.shape[0]} records.')
    