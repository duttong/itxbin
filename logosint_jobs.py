#!/usr/bin/env python3
"""Worker for the logos_integ page's "Reintegrate" button.

The page queues a row in hats.ng_integ_jobs (DDL: sql/ng_integ_jobs.sql). This script, run every
minute from the logos_user crontab, claims the oldest queued job of an instrument that has none
running, runs it, and records status and log in the row. It loops until nothing is claimable, then exits.

A job (channel N, start_time .. end_time, UTC, end exclusive / NULL = to now) is:
  1. logosint_export.py SITE --channels N --start YYMM --end YYMM   the months covering the range
  2. cats_gcwerks2db.py SITE --source logosint --no-flagged --start .. --end .. --channel L
  3. if run_batch: cats_batch.py --site SITE --gas <GAS_L> -i --fits -s .. -e .. for each gas on that
     channel, then insitu_monthly_means_batch.py cats SITE -i
The whole job holds the site's logosint lock (logosint_lock.py), shared with the daily
`cats_ingest.py SITE --source logosint`.

The CATS instruments can run jobs; which source each uses, and from when, is hats.ng_integ_source
(integ_source.py). action (ng_integ_jobs.action):
  reintegrate  as above, for an instrument on logosint, clipped to its logosint_from
  switch_on    the same for a channel's whole logosint range, queued by the page's Admin menu when it
               turns logosint on (the source row is set first)
  switch_off   back to GCwerks' values from start_time on (the source row already says gcwerks):
               cats_gcwerks2db.py SITE --flagged --start .., cats_batch.py -p all, monthly means

  logosint_jobs.py                          # claim and run queued jobs (cron)
  logosint_jobs.py --dry-run                # show the commands for the next claimable job; write nothing
  logosint_jobs.py --dry-run --job-json '{"inst_num":239,"channel":2,"start_time":"2026-09-29 00:54","end_time":null,"run_batch":1}'
  logosint_jobs.py --job-json '{...}'       # run that job now, without the jobs table (log to stdout only)
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path[:0] = ["/ccg/src/db", "/ccg/src/db/db_utils"]

BIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BIN_DIR))
import integ_source  # noqa: E402
import logosint_lock  # noqa: E402

LOGOSINT_RESULTS = Path("/hats/gc/cats_results/logosint")
TABLE = "hats.ng_integ_jobs"
LOG_CAP = 60_000          # bytes of log kept in the row (the tail)
LOG_FLUSH_S = 3.0         # how often streaming output is written to the row

# Channels: integ_source.CATS_CHANNELS, logosint channel number -> CATS channel letter (the letter in
# the export's column names; BRW and SPO peak ID files: channel0 q, channel1 a, channel2 f). Channel 3
# (cc) is not integrated by logosint. Job starts are clipped to the site's logosint_from.
def job_inst(inst_num):
    """site and channels of a CATS instrument, or None."""
    from cats_gcwerks2db import CATS_GCwerks2DB as L   # class attributes only: no DB connection
    site = {v: k for k, v in L.INST_NUM_BY_SITE.items()}.get(int(inst_num))
    return dict(site=site, channels=integ_source.CATS_CHANNELS) if site else None

_t0 = time.time()


def say(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- job log
class JobLog:
    """Accumulates output; mirrors it to stdout and, for a DB job, to the row's `log` (tail, capped)."""

    def __init__(self, db=None, num=None):
        self.db, self.num = db, num
        self.text = ""
        self._last = 0.0

    def add(self, s, flush=False):
        if not s:
            return
        sys.stdout.write(s)
        sys.stdout.flush()
        self.text += s
        if len(self.text) > LOG_CAP:
            self.text = "...[earlier output cut]\n" + self.text[-LOG_CAP:]
        if flush or time.time() - self._last >= LOG_FLUSH_S:
            self.flush()

    def line(self, s, flush=True):
        self.add(s.rstrip("\n") + "\n", flush=flush)

    def flush(self):
        self._last = time.time()
        if self.db is not None and self.num is not None:
            try:
                self.db.doquery(f"UPDATE {TABLE} SET log = %s WHERE num = %s", [self.text, self.num])
            except Exception as e:  # a failed log write must not kill the job
                sys.stdout.write(f"(could not write job log: {e})\n")


# ---------------------------------------------------------------- planning
def utc_naive(v):
    """DB datetime / ISO string -> naive pandas Timestamp (UTC), or None."""
    if v is None or v == "":
        return None
    t = pd.Timestamp(v)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def channel_gases(site, letter):
    """Gases on a channel, from the export's CSV headers (CFC11_f_rt -> 'CFC11_f' for channel f)."""
    from cats_gcwerks2db import CATS_GCwerks2DB as L   # classmethods only: no DB connection

    gases = []
    for f in sorted(LOGOSINT_RESULTS.glob(f"{site}_*.csv")):
        cols = [c.strip() for c in pd.read_csv(f, skipinitialspace=True, nrows=0).columns]
        for (mol, ch) in L._parse_measurement_columns(cols):
            if ch == letter and mol in L.UPLOAD_MOLS:
                gases.append((mol, f"{mol}_{ch}"))
    return gases


def listed_channels(inst_num):
    """{(display_name, channel)} in hats.analyte_list for an instrument: what cats_batch.py --gas accepts."""
    rows = connect().doquery("SELECT display_name, channel FROM hats.analyte_list WHERE inst_num = %s",
                             [int(inst_num)]) or []
    return {(r["display_name"], (r["channel"] or "").lower()) for r in rows}


def plan(job):
    """Resolve a job to (commands, notes) or raise ValueError with a message for the page.
    commands: list of (label, argv)."""
    cfg = job_inst(job["inst_num"])
    if cfg is None:
        raise ValueError(f"inst_num {job['inst_num']} is not a CATS instrument; jobs are CATS only so far")
    site = cfg["site"]
    action = job.get("action") or "reintegrate"
    src = integ_source.current(job["inst_num"], db=connect())
    if action == "switch_off":
        return plan_switch_off(job, site, src)
    if not src.logosint:
        raise ValueError(f"{site} is on {src.source}, not logosint (hats.ng_integ_source)")
    if job.get("site") and str(job["site"]).lower() != site:
        raise ValueError(f"job says site {job['site']!r} but inst_num {job['inst_num']} is {site!r}")
    channel = int(job["channel"])
    letter = cfg["channels"].get(channel)
    if letter is None:
        raise ValueError(f"channel {channel} of {site} is not integrated by logosint "
                         f"(known: {sorted(cfg['channels'])})")

    now = pd.Timestamp.now("UTC").tz_localize(None)
    start = utc_naive(job["start_time"])
    end = utc_naive(job.get("end_time"))
    notes = []
    floor = src.logosint_from if src.logosint_from is not None else pd.Timestamp("1998-01-01")
    if start < floor:
        notes.append(f"start {start} clipped to {floor} (first logosint date for {site})")
        start = floor
    if end is not None and end > now:
        end = None   # the future is "to now"
    if end is not None and end <= start:
        raise ValueError(f"nothing to do: range {start} .. {end} is empty after clipping to {floor}")
    if start >= now:
        raise ValueError(f"start {start} is in the future")

    # export months: the month of start .. the month after the last instant (end exclusive)
    last = (end - pd.Timedelta(seconds=1)) if end is not None else now
    m0 = pd.Timestamp(start.year, start.month, 1)
    m1 = pd.Timestamp(last.year, last.month, 1) + pd.offsets.MonthBegin(1)
    ym = lambda t: t.strftime("%y%m")
    py = sys.executable

    cmds = [
        ("export", [py, str(BIN_DIR / "logosint_export.py"), site, "--channels", str(channel),
                    "--start", ym(m0), "--end", ym(m1)]),
    ]
    load = [py, str(BIN_DIR / "cats_gcwerks2db.py"), site, "--source", "logosint", "--no-flagged",
            "--start", f"{start:%Y-%m-%d %H:%M}", "--channel", letter]
    if end is not None:
        load += ["--end", f"{end:%Y-%m-%d %H:%M}"]
    else:
        load += ["--end", f"{m1:%Y-%m-%d}"]   # nothing exported past the covered months anyway
    cmds.append(("load", load))

    if int(job.get("run_batch", 1)):
        from cats_gcwerks2db import CATS_GCwerks2DB as L
        # cats_batch works on dates: from the start's day through the day after the end (its end bound is
        # inclusive at midnight), so every analysis in the range is recomputed.
        b_start = f"{start:%Y-%m-%d}"
        b_end = f"{(end - pd.Timedelta(seconds=1)).normalize() + pd.Timedelta(days=1):%Y-%m-%d}" if end is not None else None
        gases = channel_gases(site, letter)
        if not gases:
            notes.append(f"no gases found for channel {letter} in {LOGOSINT_RESULTS}: batch step skipped")
        listed = listed_channels(job["inst_num"]) if gases else set()
        for mol, gas in gases:
            if (mol, letter) not in listed:
                notes.append(f"{gas}: not in hats.analyte_list for inst_num {job['inst_num']} "
                             f"(cats_batch.py can't run it), batch skipped; heights are still loaded")
                continue
            stop = L.STOP_AFTER.get((site, mol))
            if stop is not None and start >= pd.Timestamp(stop):
                notes.append(f"{gas}: range starts after its STOP_AFTER ({stop}), batch skipped")
                continue
            c = [py, str(BIN_DIR / "cats_batch.py"), "--site", site, "--gas", gas, "-i", "--fits", "-s", b_start]
            if b_end:
                c += ["-e", b_end]
            cmds.append((f"batch {gas}", c))
        cmds.append(("monthly means", [py, str(BIN_DIR / "insitu_monthly_means_batch.py"), "cats", site, "-i"]))
    return site, cmds, notes


def plan_switch_off(job, site, src):
    """Back to GCwerks' values from start_time on: load GCwerks' export (all channels, with its flags),
    then fits and mole fractions for every gas, then monthly means."""
    if src.logosint:
        raise ValueError(f"{site} is still on logosint (hats.ng_integ_source): set it to gcwerks first")
    start = utc_naive(job["start_time"])
    py = sys.executable
    cmds = [
        ("load GCwerks", [py, str(BIN_DIR / "cats_gcwerks2db.py"), site, "--flagged", "--start", f"{start:%Y-%m-%d %H:%M}"]),
        ("batch all", [py, str(BIN_DIR / "cats_batch.py"), "--site", site, "-p", "all", "-i", "--fits", "-s", f"{start:%Y-%m-%d}"]),
        ("monthly means", [py, str(BIN_DIR / "insitu_monthly_means_batch.py"), "cats", site, "-i"]),
    ]
    return site, cmds, [f"switch back to GCwerks from {start:%Y-%m-%d}"]


# ---------------------------------------------------------------- running
def run_cmd(argv, log):
    """Run one command, streaming its combined output into the log. Returns the exit code."""
    log.line("$ " + " ".join(argv))
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1, cwd=str(BIN_DIR))
    for ln in p.stdout:
        log.add(ln)
    rc = p.wait()
    log.line(f"(exit {rc})")
    return rc


def run_job(job, log):
    """Plan and run a job under the site lock. Returns (ok, message)."""
    try:
        site, cmds, notes = plan(job)
    except ValueError as e:
        log.line(f"FAILED: {e}")
        return False, str(e)
    for n in notes:
        log.line(f"note: {n}")
    lock = logosint_lock.acquire(site, log=log.line)
    try:
        for label, argv in cmds:
            log.line(f"--- {label} ({time.time() - _t0:.0f}s) ---")
            rc = run_cmd(argv, log)
            if rc != 0:
                log.line(f"FAILED at step '{label}' (exit {rc})")
                return False, f"step '{label}' exited {rc}"
        log.line(f"--- done ({time.time() - _t0:.0f}s) ---")
        return True, "ok"
    finally:
        lock.close()


# ---------------------------------------------------------------- queue
def connect():
    import db_conn
    return db_conn.HATS_ng()


CLAIMABLE = f"""
    SELECT j.num FROM {TABLE} j
    WHERE j.status = 'queued'
      AND NOT EXISTS (SELECT 1 FROM {TABLE} r WHERE r.inst_num = j.inst_num AND r.status = 'running')
    ORDER BY j.requested_at, j.num LIMIT 1"""


def next_job(db):
    rows = db.doquery(CLAIMABLE)
    if not rows:
        return None
    return db.doquery(f"SELECT * FROM {TABLE} WHERE num = %s", [rows[0]["num"]])[0]


def claim(db, job):
    """queued -> running, only if still queued and nothing of that instrument is running."""
    worker = f"{socket.gethostname()}:{os.getpid()}"
    n = db.doquery(
        f"""UPDATE {TABLE} SET status = 'running', started_at = UTC_TIMESTAMP(), worker = %s
            WHERE num = %s AND status = 'queued'
              AND NOT EXISTS (SELECT 1 FROM (SELECT 1 FROM {TABLE}
                              WHERE inst_num = %s AND status = 'running') x)""",
        [worker, job["num"], job["inst_num"]])
    return bool(n)


def reap_dead(db):
    """A 'running' job whose worker on this host is gone (killed, rebooted) would block its
    instrument for good: mark it failed so the queue moves on."""
    host = socket.gethostname()
    for r in db.doquery(f"SELECT num, worker FROM {TABLE} WHERE status = 'running'") or []:
        h, _, pid = (r["worker"] or "").rpartition(":")
        if h != host or not pid.isdigit() or os.path.exists(f"/proc/{pid}"):
            continue
        db.doquery(f"""UPDATE {TABLE} SET status = 'failed', finished_at = UTC_TIMESTAMP(),
                       log = CONCAT(COALESCE(log, ''), %s) WHERE num = %s AND status = 'running'""",
                   [f"\nFAILED: worker {r['worker']} is no longer running\n", r["num"]])
        say(f"job {r['num']}: worker {r['worker']} gone, marked failed")


def finish(db, job, ok, log):
    db.doquery(f"UPDATE {TABLE} SET status = %s, finished_at = UTC_TIMESTAMP(), log = %s WHERE num = %s",
               ["done" if ok else "failed", log.text, job["num"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="claim and write nothing; print the commands the next job would run")
    ap.add_argument("--job-json", help="a job as JSON (inst_num, channel, start_time, end_time, run_batch; "
                                       "optional site, num): run/print it without the jobs table")
    a = ap.parse_args()

    if a.job_json:
        job = json.loads(a.job_json)
        job.setdefault("run_batch", 1)
        if a.dry_run:
            try:
                site, cmds, notes = plan(job)
            except ValueError as e:
                sys.exit(f"would fail: {e}")
            print(f"site {site}, lock {logosint_lock.LOCK_TEMPLATE.format(site=site)}")
            for n in notes:
                print(f"note: {n}")
            for label, argv in cmds:
                print(f"[{label}] " + " ".join(argv))
            return
        ok, msg = run_job(job, JobLog())
        sys.exit(0 if ok else 1)

    db = connect()
    if a.dry_run:
        try:
            job = next_job(db)
        except Exception as e:
            sys.exit(f"cannot read {TABLE} (DDL not run yet?): {e}")
        if job is None:
            print("no claimable jobs")
            return
        print(f"would claim job {job['num']} (inst {job['inst_num']}, channel {job['channel']}, "
              f"{job['start_time']} .. {job['end_time']}, run_batch {job['run_batch']})")
        try:
            site, cmds, notes = plan(job)
        except ValueError as e:
            print(f"would fail: {e}")
            return
        for n in notes:
            print(f"note: {n}")
        for label, argv in cmds:
            print(f"[{label}] " + " ".join(argv))
        return

    reap_dead(db)
    while True:
        job = next_job(db)
        if job is None:
            return
        if not claim(db, job):
            continue          # someone else got it (or the instrument became busy): look again
        say(f"job {job['num']}: inst {job['inst_num']} channel {job['channel']} "
            f"{job['start_time']} .. {job['end_time']} ({job['kind']} {job['name']}, by {job['requested_by']})")
        log = JobLog(db, job["num"])
        log.line(f"job {job['num']} started {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC on "
                 f"{socket.gethostname()}: {job['kind']} {job['name']} channel {job['channel']}, "
                 f"{job['start_time']} .. {job['end_time'] or 'now'}, run_batch {job['run_batch']}")
        try:
            ok, msg = run_job(job, log)
        except Exception as e:   # never leave a job 'running'
            log.line(f"FAILED: unexpected error: {e!r}")
            ok = False
        finish(db, job, ok, log)
        say(f"job {job['num']} {'done' if ok else 'failed'}")


if __name__ == "__main__":
    main()
