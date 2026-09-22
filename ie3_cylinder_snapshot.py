#!/usr/bin/env python3
"""
Publish a public, self-contained snapshot of the IE3 field-cylinder dashboard.

Renders the live omi page (omi:/var/www/html/hats/logos_field_cylinders.php)
for one site and writes a standalone HTML file (Bootstrap and Dygraph inlined,
no omi links, no reftank fill notes since they name people) to the public aftp
area, e.g. https://gml.noaa.gov/aftp/hats/smo-ie3-cylinders.html.  /aftp is
synced to the public web server periodically.

omi's database accounts only work under Apache, so the page is run with PHP
CLI on omi against a record/replay database object: each pass records the
queries the page asks for, this script runs them against the HATS DB, and the
page is re-rendered until nothing is missing.  The page's own SQL and logic are
used unchanged, so the snapshot always matches the live dashboard.

Nothing is published if any step fails; the output file is replaced atomically.

Usage:
    ie3_cylinder_snapshot.py [--site smo] [--output PATH] [--dry-run]
"""

import datetime as dt
import decimal
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import typer

OMI = 'omi'
PAGE = '/var/www/html/hats/logos_field_cylinders.php'
AFTP_DIR = Path('/aftp/hats')
MAX_PASSES = 6
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30']
SCP = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30']

app = typer.Typer(add_completion=False)

# Stand-in for the dbutils global DB object: answers from replay.json, records
# anything it can't answer to missing.json.  Keys must match replay_key() below.
REPLAY_DB_PHP = r'''<?php
class IE3SnapshotReplayDB {
    private $replay; private $missing = []; private $dir;
    function __construct($dir) {
        $this->dir = $dir;
        $f = "$dir/replay.json";
        $this->replay = is_file($f) ? json_decode(file_get_contents($f), true) : [];
        register_shutdown_function(function () {
            file_put_contents("{$this->dir}/missing.json", json_encode((object)$this->missing));
        });
    }
    function doquery($sql = '', $numRows = -1, $params = []) {
        if (preg_match('/^\s*SET\s/i', $sql)) return true;
        $params = array_map('strval', array_values($params ?? []));
        $key = sha1($sql . "\x1f" . json_encode($params));
        if (array_key_exists($key, $this->replay)) return $this->replay[$key];
        $this->missing[$key] = ['sql' => $sql, 'params' => $params];
        return [];
    }
}
$dbu_dbutils_global_obj = new IE3SnapshotReplayDB(__DIR__);
'''

FRAME = r'''$bs_adjContent = ob_get_clean();
$inc = fn($p) => file_get_contents('/var/www/html/inc/dbutils' . $p);
$css = $inc('/bs/src/bootstrap-5.3.2-dist/css/bootstrap.min.css') . "\n" . $inc('/graphing/dygraph/v2.2.1/dygraph.min.css');
$js = $inc('/graphing/dygraph/v2.2.1/dygraph.min.js');
$generated = gmdate('Y-m-d H:i');
?><!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__SITE__ IE3 Cylinders</title>
<style><?= $css ?></style>
<style>
body { background: #fff; }
.fc-banner { background: linear-gradient(to right, #0b3d91, #000 320px); color: #fff; padding: .6rem 1rem; }
.fc-banner h1 { font-size: 1.25rem; margin: 0; }
</style>
<script><?= $js ?></script>
</head>
<body>
<div class="fc-banner d-flex justify-content-between align-items-baseline flex-wrap gap-2">
  <h1>NOAA GML &middot; __SITE__ IE3 cylinders</h1>
  <span class="small">Snapshot generated <?= $generated ?> UTC</span>
</div>
<?= $bs_adjContent ?>
</body>
</html>
'''


def replay_key(sql: str, params: list[str]) -> str:
    import hashlib
    # Match PHP json_encode of a list of strings (escapes "/" and non-ASCII by default).
    enc = json.dumps(params, separators=(',', ':'), ensure_ascii=True).replace('/', '\\/')
    return hashlib.sha1((sql + '\x1f' + enc).encode()).hexdigest()


def build_variant(src: str, site: str) -> str:
    """Turn the live page into a one-site, public, standalone snapshot page."""
    def sub(old: str, new: str) -> None:
        nonlocal src
        if src.count(old) != 1:
            raise RuntimeError(f'page layout changed; cannot find: {old[:70]!r}')
        src = src.replace(old, new)

    sub("db_connect_ro('hats');", "require __DIR__ . '/replay_db.php';")
    sub("require_once(WWW_ROOT . UTILS_ROOT_RELPATH . '/dbutils.php');",
        f"$FC_INSTRUMENTS = array_values(array_filter($FC_INSTRUMENTS, fn($i) => $i['site'] === '{site}'));\n"
        "require_once(WWW_ROOT . UTILS_ROOT_RELPATH . '/dbutils.php');")
    # Public copy: fill code + date only (fill notes carry people's names).
    src, n = re.subn(r'''\$title = trim\("Filled \{\$f\['date'\]\}\. " \. .*?\);''',
                     '''$title = "Filled {$f['date']}";''', src)
    if n != 1:
        raise RuntimeError('page layout changed; cannot find fill-notes tooltip')
    sub('Fill = reftank.fill code for the gas in the tank (hover for fill date and notes).',
        'Fill = fill code for the gas in the tank (hover for fill date).')
    i = src.find('$bs_adjContent = ob_get_clean();')
    if i < 0:
        raise RuntimeError('page layout changed; cannot find page frame')
    return src[:i] + FRAME.replace('__SITE__', site)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        # Drop omi's login banner (lines of '*') and X11 noise from the error.
        err = [l for l in p.stderr.splitlines() if l.strip() and not l.startswith('*') and 'X11' not in l]
        raise RuntimeError(f'{" ".join(cmd[-1:])[:120]} failed (exit {p.returncode}): ' + ' | '.join(err[-5:]))
    return p


def jsonable(v):
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat(sep=' ') if isinstance(v, dt.datetime) else v.isoformat()
    if isinstance(v, decimal.Decimal):
        return str(v)
    return str(v)


@app.command()
def main(
    site: str = typer.Option('smo', '--site', help='Site code to publish (must be on the dashboard).'),
    output: Path = typer.Option(None, '--output', help='Output file (default /aftp/hats/{site}-ie3-cylinders.html).'),
    dry_run: bool = typer.Option(False, '--dry-run', help='Write to a local temp file instead of publishing.'),
):
    """Render the omi field-cylinder dashboard for one site and publish it to aftp."""
    site_uc = site.upper()
    out = output or AFTP_DIR / f'{site.lower()}-ie3-cylinders.html'
    started = dt.datetime.now()

    sys.path.append('/ccg/src/db/')
    import db_utils.db_conn as db_conn  # type: ignore
    db = db_conn.HATS_ng()
    db.doquery("SET time_zone = '+00:00'")   # the page's UNIX_TIMESTAMP()s assume UTC

    page = run(SSH + [OMI, f'cat {PAGE}']).stdout
    variant = build_variant(page, site_uc)
    rdir = run(SSH + [OMI, 'mktemp -d /tmp/ie3_cyl_snapshot.XXXXXX']).stdout.strip()
    replay: dict = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'page.php').write_text(variant)
            (tmp / 'replay_db.php').write_text(REPLAY_DB_PHP)
            run(SCP + [str(tmp / 'page.php'), str(tmp / 'replay_db.php'), f'{OMI}:{rdir}/'])
            for npass in range(1, MAX_PASSES + 1):
                (tmp / 'replay.json').write_text(json.dumps(replay, default=jsonable))
                run(SCP + [str(tmp / 'replay.json'), f'{OMI}:{rdir}/'])
                run(SSH + [OMI, f'cd {rdir} && php page.php > page.html 2> php_err.txt'])
                run(SCP + [f'{OMI}:{rdir}/missing.json', f'{OMI}:{rdir}/page.html', str(tmp)])
                missing = json.loads((tmp / 'missing.json').read_text())
                if not missing:
                    break
                for key, q in missing.items():
                    # PHP uses ? placeholders; MySQLdb uses %s and needs literal %
                    # doubled, but only when parameters are passed.
                    sql, params = q['sql'], q['params'] or None
                    if params:
                        sql = sql.replace('%', '%%').replace('?', '%s')
                    replay[key] = db.doquery(sql, params) or []
                    if replay_key(q['sql'], q['params']) != key:
                        raise RuntimeError('replay key mismatch between PHP and Python')
                print(f'pass {npass}: ran {len(missing)} queries')
            else:
                raise RuntimeError(f'page still issuing new queries after {MAX_PASSES} passes')
            html = (tmp / 'page.html').read_text()
    finally:
        subprocess.run(SSH + [OMI, f'rm -rf {rdir}'], capture_output=True)

    for marker in (f'{site_uc} IE3 cylinders', 'window.FC_CONFIG', f'{site_uc} cylinder pressure'):
        if marker not in html:
            raise RuntimeError(f'rendered page is missing {marker!r}; not publishing')

    if dry_run:
        out = Path(tempfile.gettempdir()) / f'{site.lower()}-ie3-cylinders.dry-run.html'
    tmp_out = out.with_name('.' + out.name + '.tmp')
    tmp_out.write_text(html)
    os.chmod(tmp_out, 0o644)
    os.replace(tmp_out, out)
    secs = (dt.datetime.now() - started).total_seconds()
    print(f'{"wrote (dry run)" if dry_run else "published"} {out} ({len(html.encode()):,} bytes, {npass} passes, {secs:.0f}s)')


if __name__ == '__main__':
    app()
