#! /usr/bin/env python

import io
from typing import Optional
from pathlib import Path
from contextlib import redirect_stdout

import pandas as pd
import typer

from gcwerks_export import GCwerks_export

app = typer.Typer(add_completion=False)


class IE3_Process(GCwerks_export):
    """Class hardcoded for the IE3 instrument."""

    def __init__(self, site='smo', prefix=None, flagged=False):
        if prefix is None:
            prefix = f'ie3_{site}'
        super().__init__(site, prefix)
        self.flagged = flagged
        suffix = '_flagged' if flagged else ''
        self.results_file = f'/hats/gc/{site}/results/ie3_{site}_gcwerks_all{suffix}.csv'

    def export_onefile(self, csv=True, report='/hats/gc/itxbin/report.conf'):
        """Exports all data to a single file, optionally including flagged data."""
        years = self.gcwerks_years()
        mindate = f'{years[0]}0101'
        maxdate = f'{str(pd.Timestamp.today().year)[2:4]}1231.2359'
        self.gcwerks_export('all', mindate=mindate, maxdate=maxdate, csv=csv, flagged=self.flagged, report=report)

    def export_years(self, mol, start_year=1998, end_year=pd.Timestamp.today().year):
        """Export yearly files, optionally including flagged data."""
        years = self.gcwerks_years()
        for year in range(int(start_year), int(end_year) + 1):
            if str(year)[2:4] in years:
                mindate = f'{str(year)[2:4]}0101'
                maxdate = f'{str(year)[2:4]}1231.2359'
                self.gcwerks_export(mol, mindate, maxdate, flagged=self.flagged)

    def gcwerks_export(self, mol, mindate=False, maxdate=False, csv=True, flagged=False, mk2yrfile=False, report='/hats/gc/itxbin/report.conf'):
        """Use the flagged IE3 filename for all-data exports."""
        if mol == 'all' and csv and flagged:
            results_file = self.export_path / f'{self.prefix}_gcwerks_all_flagged.csv'
            print(f'Exporting {mol} to {results_file}')
            default_results = self.export_path / f'{self.prefix}_gcwerks_all.csv'
            with redirect_stdout(io.StringIO()):
                exported = super().gcwerks_export(
                    mol,
                    mindate=mindate,
                    maxdate=maxdate,
                    csv=csv,
                    flagged=flagged,
                    mk2yrfile=mk2yrfile,
                    report=report,
                )
            if exported and default_results.exists():
                default_results.rename(results_file)
            return exported

        return super().gcwerks_export(
            mol,
            mindate=mindate,
            maxdate=maxdate,
            csv=csv,
            flagged=flagged,
            mk2yrfile=mk2yrfile,
            report=report,
        )

    def read_results(self, year='all'):
        df = pd.read_csv(self.results_file, skipinitialspace=True, parse_dates=[0])
        df.set_index(df.time, inplace=True)
        if year == 'all':
            return df
        return df[str(year)]


@app.command()
def export(
    mol: str = typer.Argument(
        'all',
        help='Select a single molecule to export or default to "all".',
    ),
    site: str = typer.Option(
        'smo',
        '--site',
        help='Station code for the IE3 instrument.',
    ),
    year: Optional[int] = typer.Option(
        None,
        '--year',
        help="Export this year's data.",
    ),
    flagged: bool = typer.Option(
        False,
        '--flagged',
        help='Include flagged data and write the flagged output filename.',
    ),
):
    valid_sites = {'smo', 'mlo', 'spo', 'brw'}
    if site not in valid_sites:
        typer.secho(
            f'Invalid site "{site}". Valid sites: {sorted(valid_sites)}',
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    ie3 = IE3_Process(site=site, flagged=flagged)
    valid_mols = sorted(ie3.gcwerks_peaks())
    if mol != 'all' and mol not in valid_mols:
        typer.secho(
            f'Invalid molecule "{mol}". Valid mol variables: {valid_mols}',
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if year:
        ie3.export_years(mol, start_year=year, end_year=year)
        raise typer.Exit()

    ie3.export_onefile()


if __name__ == '__main__':
    app()
