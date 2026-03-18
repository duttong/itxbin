#! /usr/bin/env python

from typing import Optional

import pandas as pd
import typer

from gcwerks_export import GCwerks_export

app = typer.Typer(add_completion=False)


class IE3_Process(GCwerks_export):
    """Class hardcoded for the IE3 instrument."""

    def __init__(self, site='smo', prefix=None):
        if prefix is None:
            prefix = f'ie3_{site}'
        super().__init__(site, prefix)
        self.results_file = f'/hats/gc/{site}/results/ie3_{site}_gcwerks_all.csv'

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
):
    valid_sites = {'smo', 'mlo', 'spo', 'brw'}
    if site not in valid_sites:
        typer.secho(
            f'Invalid site "{site}". Valid sites: {sorted(valid_sites)}',
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    ie3 = IE3_Process(site=site)
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
