#! /usr/bin/env python

import sys
import math
import pandas as pd
import numpy as np
from pathlib import Path


# use itxbin in home directory
repo = Path.home() / 'itxbin'
sys.path.insert(0, str(repo))
from logos_instruments import FE3_Instrument, M4_Instrument, BLD1_Instrument

# use coredata in /hats/gc/itxbin/coredata
cdpath = Path('/hats/gc/itxbin/coredata')
sys.path.insert(0, str(cdpath))
from core_merge import Stratcore_GCwerks, Stratcore_runs

gw = Stratcore_GCwerks()
runs = Stratcore_runs()
bld1 = BLD1_Instrument()

mols = bld1.analytes
attribs = {'flag': False,
            'methdet': 'lowess',
            'methcal': 'quadratic',
            'value': np.nan,
            'unc': np.nan}


def _port_id(row):
    """ assign port_id by cal SSV position where row is from a gcwerks dataframe """
    ssv = row['port']
    
    # ssv == 1 used to be ssv == 11 for cores. changed 220805
    try:
        if ssv == 1:
            pname = 'core'
        else:
            pname = row['ports'][ssv-10]   # changed from -1 to -11, 220805, -10 on 220823
    except TypeError:
        pname = ''
    return pname

def merge_gcwerks_and_metadata():
    """ Load two streams of data. Meta data from Stratcore and GCwerks results, then
        merges into a single file (self.dbfile) """

    core_runs = runs.runs_df()
    gcwerks = gw.gcwerks_df()
    
    # merge the two streams of data
    # if runs are longer than 24 hours, this may not work.
    df = pd.merge_asof(gcwerks, core_runs,
        on='time', direction='backward', tolerance=pd.Timedelta('24h'))

    # add port_id to dataframe
    df['port_id'] = df.apply(_port_id, axis=1)
    
    # create extra attribute columns
    for mol in bld1.analytes.keys():
        for k in attribs:
            col = f'{mol}_{k}'
            df[col] = attribs[k]

    df = cleanup_db_df(df, dropcols=False)
    return df

def cleanup_db_df(df, dropcols=True):
    """ Sorts columns by molecule name but leaves the main columns at the front of the df """

    df = df.reset_index()
    if dropcols:
        # drop area columns to save space.
        droplist = [f'{mol}_area' for mol in self.mols]
        #droplist += ['ports', 'seq']
        cols = df.columns.difference(droplist)
        df = df[cols]

    # rearrange DataFrame
    cols = list(df.columns)
    first = ['time', 'port', 'port_id', 'type', 'dir']
    for item in first:
        cols.remove(item)
    cols = first + sorted(cols, key=str.casefold)
    df = df[cols]

    #df = df.loc[~((df.type != 'core') & (df.type != 'other'))]
    df = df.set_index('time')
    try:
        df = df.drop(['index'], axis=1)
    except KeyError:
        pass
    df = df.tz_localize('utc')  # set time zone

    return df

def piviot_gcwerks_df(df):
    df_melted = (
        df.melt(
            id_vars=['time', 'port', 'port_id', 'type', 'dir', 'ports', 'seq'],
            var_name='variable',
            value_name='value_raw'
        )
    )

    # extract gas and measurement field
    df_melted[['gas', 'field']] = df_melted['variable'].str.extract(r'(\w+)_(\w+)$')

    # pivot so each gas has one row per port
    df_tidy = (
        df_melted
        .pivot_table(
            index=['time', 'port', 'port_id', 'type', 'dir', 'gas'],
            columns='field',
            values='value_raw',
            aggfunc='first'
        )
        .reset_index()
    )

    # add pnum, inst_num, and run_time columns
    df_tidy['pnum'] = df_tidy['gas'].map(bld1.analytes)
    df_tidy['run_time'] = pd.to_datetime(df_tidy['dir'], errors='coerce')
    df_tidy['inst_num'] = 220

    # reorder columns as desired
    df_tidy = df_tidy[['time', 'run_time', 'port', 'port_id', 'type', 'dir', 'gas', 'pnum',
                    'ht', 'area', 'rt', 'inst_num']]

    # insert run_time after 'type'
    cols = list(df_tidy.columns)
    insert_pos = cols.index('time') + 1
    cols.insert(insert_pos, cols.pop(cols.index('run_time')))
    df_tidy = df_tidy[cols]

    df_tidy = df_tidy.drop('dir', axis=1)
    df_tidy.rename(columns={'time':'analysis_time', 'port_id':'port_info'}, inplace=True)

    df_tidy['channel'] = df_tidy['gas'].map(bld1.chan_map)
    
    # Build a mapping dict from runtype.name → runtype.num
    runtype = pd.DataFrame(bld1.doquery('SELECT * FROM hats.ng_run_types;'))
    runtype_map = dict(zip(runtype['name'].str.lower(), runtype['num']))

    # Add special handling for 'test' → 'Other'
    runtype_map['test'] = runtype_map.get('other')
    runtype_map['cal'] = runtype_map.get('calibration')

    # Map the type column to run_type_num
    df_tidy['run_type_num'] = df_tidy['type'].str.lower().map(runtype_map)

    # reorder columns to place run_type_num after inst_num
    cols = list(df_tidy.columns)

    # remove the columns you want to move
    cols.remove('inst_num')
    cols.remove('run_type_num')

    # find insertion point (right after 'run_time')
    insert_pos = cols.index('run_time') + 1

    # insert them in the desired order
    cols[insert_pos:insert_pos] = ['inst_num', 'run_type_num']
    df_tidy = df_tidy[cols]
    
    return df_tidy
    
def create_analysis_df(df_tidy):
    # create a cleaned dataframe with appropriate data types for insertion into database (ng_analysis)
    df_tidy['run_time'] = pd.to_datetime(df_tidy['run_time'])
    df_tidy['analysis_time'] = pd.to_datetime(df_tidy['analysis_time'])
    df_tidy['inst_num'] = df_tidy['inst_num'].astype(int)
    df_tidy['run_type_num'] = df_tidy['run_type_num'].astype(int)
    df_tidy['port'] = df_tidy['port'].astype(int)

    cols_analysis = [
        'analysis_time', 'run_time', 'run_type_num',
        'inst_num', 'port', 'port_info'
    ]
    df_analysis = df_tidy[cols_analysis].drop_duplicates()
    return df_analysis

def stratcore_to_hats_analysis(df_analysis: pd.DataFrame, batch_size: int = 500):
    """
    Bulk upsert a Stratcore DataFrame into hats.ng_analysis.

    Parameters
    ----------
    db : database connection wrapper
        Must implement .doMultiInsert(sql, params, all=True) and .doquery(sql, params).
    df_analysis : pd.DataFrame
        DataFrame with columns:
            analysis_time, run_time, run_type_num, inst_num, port, port_info
    batch_size : int
        Number of rows to insert per batch.
    """
    inst_num = 220  # Stratcore instrument number

    # --- 1) Copy and clean dataframe
    df = df_analysis.copy()

    # Format datetime fields as strings for MySQL
    df['analysis_time_str'] = pd.to_datetime(df['analysis_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df['run_time_str']      = pd.to_datetime(df['run_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Ensure correct dtypes
    df['port_info']    = df['port_info'].fillna('').astype(str)
    df['run_type_num'] = pd.to_numeric(df['run_type_num'], errors='coerce').fillna(0).astype(int)
    df['port']         = pd.to_numeric(df['port'], errors='coerce').fillna(0).astype(int)
    df['inst_num']     = inst_num

    # --- 2) Prepare SQL for hats.ng_analysis
    analysis_sql = """
        INSERT INTO hats.ng_analysis (
            analysis_time,
            inst_num,
            run_time,
            run_type_num,
            port,
            port_info
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            #run_time     = VALUES(run_time),
            run_type_num = VALUES(run_type_num),
            port_info    = VALUES(port_info);
    """

    # --- 3) Build parameter list
    analysis_params = [
        (
            row.analysis_time_str,
            inst_num,
            row.run_time_str,
            int(row.run_type_num),
            int(row.port),
            row.port_info,
        )
        for row in df.itertuples(index=False)
    ]

    # --- 4) Execute in batches
    for i in range(0, len(analysis_params), batch_size):
        batch = analysis_params[i : i + batch_size]
        bld1.db.doMultiInsert(analysis_sql, batch, all=True)

    # --- 5) Verify insert / fetch record IDs
    unique_times = df['analysis_time_str'].unique().tolist()
    placeholders = ','.join(['%s'] * len(unique_times))
    select_sql = f"""
        SELECT analysis_time, num
          FROM hats.ng_analysis
         WHERE inst_num = %s
           AND analysis_time IN ({placeholders})
    """
    rows = bld1.db.doquery(select_sql, [inst_num] + unique_times)
    mapping = {
        r['analysis_time'].strftime('%Y-%m-%d %H:%M:%S'): r['num']
        for r in rows
    }

    print(f"✅ Upsert complete for Stratcore ({inst_num}). {len(mapping)} rows confirmed in hats.ng_analysis.")
    return mapping

def stratcore_to_hats_molefractions(df_tidy: pd.DataFrame, analysis_map: dict, batch_size: int = 500):
    """
    Bulk upsert Stratcore gas-level data into hats.ng_mole_fractions.

    NaN values in numeric columns (height, retention_time, mole_fraction)
    are safely converted to None before insertion.
    """

    df = df_tidy.copy()
    df['analysis_time_str'] = pd.to_datetime(df['analysis_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    mole_sql = """
        INSERT INTO hats.ng_mole_fractions (
            analysis_num,
            parameter_num,
            channel,
            detrend_method_num,
            height,
            retention_time,
            area
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            height             = VALUES(height),
            retention_time     = VALUES(retention_time),
            area               = VALUES(area),
            detrend_method_num = VALUES(detrend_method_num);
    """

    def safe(x):
        """Convert NaN or inf to None for SQL insert."""
        if pd.isna(x) or (isinstance(x, (float, int)) and math.isinf(x)):
            return None
        return x

    mole_params = []
    for r in df.itertuples(index=False):
        analysis_num = analysis_map.get(r.analysis_time.strftime('%Y-%m-%d %H:%M:%S'))
        if analysis_num is None:
            continue  # skip unmatched analysis rows

        mole_params.append((
            analysis_num,                               # analysis_num
            int(r.pnum),                                # parameter_num
            r.channel,                                  # channel ('a'/'b')
            2,                                          # detrend_method_num (2 = lowess)
            safe(r.ht),                                 # height
            safe(r.rt),                                 # retention_time
            safe(r.area),                               # area
        ))

    for i in range(0, len(mole_params), batch_size):
        batch = mole_params[i:i + batch_size]
        bld1.db.doMultiInsert(mole_sql, batch, all=True)

    print(f"✅ Upserted {len(mole_params)} rows into hats.ng_mole_fractions.")


def main(year=None):
    df = merge_gcwerks_and_metadata()
    df.reset_index(inplace=True)    
    df_tidy = piviot_gcwerks_df(df)

    if year is not None:
        df_tidy = df_tidy[df_tidy['analysis_time'].dt.year == year]
    
    df_analysis = create_analysis_df(df_tidy)
    
    analysis_nums = stratcore_to_hats_analysis(df_analysis)
    stratcore_to_hats_molefractions(df_tidy=df_tidy, analysis_map=analysis_nums)
    

if __name__ == '__main__':
    current_year = pd.Timestamp.now().year
    main(current_year)
    