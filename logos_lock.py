"""One lock per instrument for the ingest scripts (m4_ingest.py, fe3_ingest.py, ie3_ingest.py), so the
daily cron run and a job from the LOGOS Load page (logos_jobs.py runs the same ingest script) never
work on the same GCwerks folder and tables at once.

An flock on /var/tmp/logos-ingest-<name>.lock (mode 666, shared by gdutton and logos_user), held until
the returned file is closed or the process exits; a second taker waits. Same mechanism as
logosint_lock.py (CATS logosint), which keeps its own lock files.
"""

import fcntl
import os
import time

from logosint_lock import _open

LOCK_TEMPLATE = "/var/tmp/logos-ingest-{name}.lock"


def acquire(name, log=print):
    """Block until this process holds the instrument's lock; returns the open file (close to release)."""
    path = LOCK_TEMPLATE.format(name=name)
    fh = os.fdopen(_open(path), "r")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log(f"waiting for {path} (another {name} ingest or load job is running)...")
        t0 = time.time()
        fcntl.flock(fh, fcntl.LOCK_EX)
        log(f"got {path} after {time.time() - t0:.0f}s")
    return fh
