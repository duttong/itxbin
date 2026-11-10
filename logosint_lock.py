"""One lock per CATS site shared by everything that rewrites that site's logosint results:
the daily `cats_ingest.py SITE --source logosint` and the reintegration worker logosint_jobs.py.

The lock is an flock on /var/tmp/logosint-<site>.lock (mode 666, so gdutton and logos_user
share it). It is held until the returned file object is closed or the process exits, and a
second taker blocks until the first lets go.
"""

import errno
import fcntl
import os
import time

LOCK_TEMPLATE = "/var/tmp/logosint-{site}.lock"


def _open(path):
    # Open an existing file first: in a sticky /var/tmp, O_CREAT on a file another user owns can be
    # refused (fs.protected_regular) even when its mode allows writing, and flock needs no write access.
    for _ in range(3):
        for flags in (os.O_RDWR, os.O_RDONLY):
            try:
                return os.open(path, flags)
            except FileNotFoundError:
                break
            except PermissionError:
                continue
        else:
            raise PermissionError(errno.EACCES, f"cannot open lock file {path}")
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue                 # someone created it in between: open it
        try:
            os.chmod(path, 0o666)    # the umask (002) would have left it 664
        except OSError:
            pass
        return fd
    raise OSError(f"cannot open or create lock file {path}")


def acquire(site, log=print):
    """Block until this process holds the site's lock; returns the open file (close it to release)."""
    path = LOCK_TEMPLATE.format(site=site)
    fd = _open(path)
    fh = os.fdopen(fd, "r")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log(f"waiting for {path} (another logosint job or ingest for {site} is running)...")
        t0 = time.time()
        fcntl.flock(fh, fcntl.LOCK_EX)
        log(f"got {path} after {time.time() - t0:.0f}s")
    return fh
