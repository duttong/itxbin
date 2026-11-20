#!/usr/bin/env python3
"""Worker for the LOGOS Load page (omi hats/logos_load.php).

The page queues rows in hats.ng_ingest_jobs through logos_data_service.py (op jobqueue; DDL in
sql/ng_ingest_jobs.sql). This script, run every minute from cron on catsdata, claims the oldest
queued job of an instrument that has none running, runs it, and records status, command and log in
the row. It loops until nothing is claimable, then exits.

A job runs the instrument's own ingest script, the one cron runs daily, with a range and steps:

  fetch    <inst>_ingest.py                                   the daily run, now
  reload   <inst>_ingest.py --steps [update,]load,batch -s START [-e END]   (FE3 adds export, IE3 eng
           and methods; update = gcupdate + gccalc first, when the job asks for it)
  recalc   <inst>_ingest.py --steps batch -s START [-e END]
  means    insitu_monthly_means_batch.py <ie3 | cats SITE> [-p PNUM] -i   (IE3 and CATS: the whole
           history of one analyte, or of all of them, from the Data page's Update monthly means)

The ingest script takes the instrument's lock (logos_lock.py), so a job waits for a cron run of the
same instrument and the other way round.  A means job takes no lock: it only reads mole fractions
and tags, and one running per instrument (the queue's rule) is enough.

  logos_jobs.py                     # claim and run queued jobs (cron)
  logos_jobs.py --dry-run           # show the command for the next claimable job; write nothing
  logos_jobs.py --dry-run --job-json '{"inst":"fe3","action":"reload","start_time":"2026-10-01","end_time":"2026-10-31 23:59:59"}'
  logos_jobs.py --dry-run --job-json '{"inst":"brw","action":"means","parameter_num":22}'
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

BIN_DIR = Path(__file__).resolve().parent
TABLE = "hats.ng_ingest_jobs"
LOG_CAP = 60_000          # bytes of log kept in the row (the tail)
LOG_FLUSH_S = 3.0         # how often streaming output is written to the row

# The page's instrument ids. steps: what a reload runs (update is added in front when asked).
INSTS = {
    "m4":      dict(inst_num=192, script="m4_ingest.py",  args=[],                reload=["load", "batch"]),
    "fe3":     dict(inst_num=193, script="fe3_ingest.py", args=[],                reload=["export", "load", "batch"]),
    "ie3-smo": dict(inst_num=236, script="ie3_ingest.py", args=["--site", "smo"], reload=["load", "eng", "methods", "batch"]),
}
# The in situ instruments whose monthly means (hats.ng_insitu_monthly_means) a 'means' job refreshes, by the page's
# instrument id (logos_data_service PAGE_INSTS): inst_num and the insitu_monthly_means_batch.py arguments.
MEANS_INSTS = {
    "ie3-smo":  dict(inst_num=236, args=["ie3"]),
    "brw":      dict(inst_num=239, args=["cats", "brw"]),
    "sum":      dict(inst_num=240, args=["cats", "sum"]),
    "nwr":      dict(inst_num=241, args=["cats", "nwr"]),
    "mlo":      dict(inst_num=242, args=["cats", "mlo"]),
    "smo_cats": dict(inst_num=243, args=["cats", "smo"]),
    "spo":      dict(inst_num=244, args=["cats", "spo"]),
}
ACTIONS = ("fetch", "reload", "recalc", "means")


def say(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


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


def utc_naive(v):
    """DB datetime / ISO string -> naive Timestamp (UTC), or None."""
    if v is None or v == "":
        return None
    t = pd.Timestamp(v)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def plan(job):
    """A job as the command to run: (argv, description). ValueError with a message for the page."""
    action = job.get("action")
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    if action == "means":
        return plan_means(job)
    cfg = INSTS.get(job.get("inst"))
    if cfg is None:
        raise ValueError(f"unknown instrument {job.get('inst')!r}; jobs run for {', '.join(INSTS)}")
    argv = [sys.executable, str(BIN_DIR / cfg["script"]), *cfg["args"]]
    if action == "fetch":
        return argv, "the daily ingest"
    start, end = utc_naive(job.get("start_time")), utc_naive(job.get("end_time"))
    if start is None:
        raise ValueError(f"{action} needs a start time")
    now = pd.Timestamp.now("UTC").tz_localize(None)
    if start > now:
        raise ValueError(f"start {start} is in the future")
    if end is not None and end < start:
        raise ValueError(f"end {end} is before start {start}")
    if action == "recalc":
        steps = ["batch"]
    else:
        steps = (["update"] if int(job.get("update_gcwerks") or 0) else []) + cfg["reload"]
    argv += ["--steps", ",".join(steps), "-s", f"{start:%Y-%m-%d %H:%M:%S}"]
    if end is not None and end < now:
        argv += ["-e", f"{end:%Y-%m-%d %H:%M:%S}"]
    return argv, f"{action} {start:%Y-%m-%d %H:%M} to {f'{end:%Y-%m-%d %H:%M}' if end is not None and end < now else 'now'}"


def plan_means(job):
    """A 'means' job: the monthly means of one analyte (parameter_num) or, with none, of every analyte."""
    cfg = MEANS_INSTS.get(job.get("inst"))
    if cfg is None:
        raise ValueError(f"monthly means are for the in situ instruments ({', '.join(MEANS_INSTS)}), "
                         f"not {job.get('inst')!r}")
    argv = [sys.executable, str(BIN_DIR / "insitu_monthly_means_batch.py"), *cfg["args"]]
    pnum = job.get("parameter_num")
    if pnum not in (None, "", 0):
        try:
            pnum = int(pnum)
        except (TypeError, ValueError):
            raise ValueError(f"bad parameter_num {pnum!r}") from None
        if pnum <= 0:
            raise ValueError(f"bad parameter_num {pnum}")
        argv += ["-p", str(pnum)]
        what = f"monthly means of parameter {pnum}"
    else:
        what = "monthly means of every analyte"
    return argv + ["-i"], what


def run_cmd(argv, log):
    """Run the command, streaming its combined output into the log. Returns the exit code."""
    log.line("$ " + " ".join(argv))
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1, cwd=str(BIN_DIR), env=env)
    for ln in p.stdout:
        log.add(ln)
    rc = p.wait()
    log.line(f"(exit {rc})")
    return rc


def connect():
    for d in ("/ccg/src/db/db_utils", "/ccg/src/db"):
        if d not in sys.path:
            sys.path.insert(0, d)
    import db_conn
    return db_conn.HATS_ng()


CLAIMABLE = f"""
    SELECT j.num FROM {TABLE} j
    WHERE j.status = 'queued'
      AND NOT EXISTS (SELECT 1 FROM {TABLE} r WHERE r.inst = j.inst AND r.status = 'running')
    ORDER BY j.requested_at, j.num LIMIT 1"""


def next_job(db):
    rows = db.doquery(CLAIMABLE)
    if not rows:
        return None
    return db.doquery(f"SELECT * FROM {TABLE} WHERE num = %s", [rows[0]["num"]])[0]


def claim(db, job, command):
    """queued -> running, only if still queued and nothing of that instrument is running."""
    worker = f"{socket.gethostname()}:{os.getpid()}"
    n = db.doquery(
        f"""UPDATE {TABLE} SET status = 'running', started_at = UTC_TIMESTAMP(), worker = %s, command = %s
            WHERE num = %s AND status = 'queued'
              AND NOT EXISTS (SELECT 1 FROM (SELECT 1 FROM {TABLE}
                              WHERE inst = %s AND status = 'running') x)""",
        [worker, command[:512], job["num"], job["inst"]])
    return bool(n)


def reap_dead(db):
    """A 'running' job whose worker on this host is gone (killed, rebooted) would block its instrument
    for good: mark it failed so the queue moves on."""
    host = socket.gethostname()
    for r in db.doquery(f"SELECT num, worker FROM {TABLE} WHERE status = 'running'") or []:
        h, _, pid = (r["worker"] or "").rpartition(":")
        if h != host or not pid.isdigit() or os.path.exists(f"/proc/{pid}"):
            continue
        db.doquery(f"""UPDATE {TABLE} SET status = 'failed', finished_at = UTC_TIMESTAMP(),
                       log = CONCAT(COALESCE(log, ''), %s) WHERE num = %s AND status = 'running'""",
                   [f"\nFAILED: worker {r['worker']} is no longer running\n", r["num"]])
        say(f"job {r['num']}: worker {r['worker']} gone, marked failed")


def fail_unplannable(db, job, msg):
    db.doquery(f"""UPDATE {TABLE} SET status = 'failed', started_at = UTC_TIMESTAMP(), finished_at = UTC_TIMESTAMP(),
                   log = %s WHERE num = %s AND status = 'queued'""", [f"FAILED: {msg}\n", job["num"]])


def finish(db, job, ok, log):
    db.doquery(f"UPDATE {TABLE} SET status = %s, finished_at = UTC_TIMESTAMP(), log = %s WHERE num = %s",
               ["done" if ok else "failed", log.text, job["num"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="claim and write nothing; print the command the next job would run")
    ap.add_argument("--job-json", help="a job as JSON (inst, action, start_time, end_time, update_gcwerks, parameter_num): "
                                       "run it now without the jobs table, or print it with --dry-run")
    a = ap.parse_args()

    if a.job_json:
        job = json.loads(a.job_json)
        try:
            argv, what = plan(job)
        except ValueError as e:
            sys.exit(f"would fail: {e}")
        if a.dry_run:
            print(f"{job['inst']}: {what}\n" + " ".join(argv))
            return
        sys.exit(0 if run_cmd(argv, JobLog()) == 0 else 1)

    db = connect()
    if a.dry_run:
        try:
            job = next_job(db)
        except Exception as e:
            sys.exit(f"cannot read {TABLE} (DDL not run yet?): {e}")
        if job is None:
            print("no claimable jobs")
            return
        try:
            argv, what = plan(job)
        except ValueError as e:
            print(f"job {job['num']} would fail: {e}")
            return
        print(f"would claim job {job['num']} ({job['inst']}: {what}, by {job['requested_by']})\n" + " ".join(argv))
        return

    try:
        reap_dead(db)
    except Exception as e:
        if "doesn't exist" in str(e):
            return           # the table isn't there yet: nothing to do
        raise
    while True:
        job = next_job(db)
        if job is None:
            return
        try:
            argv, what = plan(job)
        except ValueError as e:
            fail_unplannable(db, job, str(e))
            say(f"job {job['num']}: {e}")
            continue
        if not claim(db, job, " ".join(argv)):
            continue          # someone else got it (or the instrument became busy): look again
        say(f"job {job['num']}: {job['inst']} {what} (by {job['requested_by']})")
        log = JobLog(db, job["num"])
        log.line(f"job {job['num']} started {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC on "
                 f"{socket.gethostname()}: {job['inst']} {what}, requested by {job['requested_by']}")
        t0 = time.time()
        try:
            rc = run_cmd(argv, log)
            ok = rc == 0
            log.line(f"--- {'done' if ok else f'FAILED (exit {rc})'} after {time.time() - t0:.0f} s ---")
        except Exception as e:   # never leave a job 'running'
            log.line(f"FAILED: unexpected error: {e!r}")
            ok = False
        finish(db, job, ok, log)
        say(f"job {job['num']} {'done' if ok else 'failed'}")


if __name__ == "__main__":
    main()
