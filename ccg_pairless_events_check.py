#!/usr/bin/env python3
"""Email a list of CCG surface flask events that have no pair id.

om2 gives every CCG event a pair id (sheet check-in, Event Input and the
no-sheet check-in), but events entered with the legacy om pages get none.
This lists ccgg.flask_event rows (strategy 1 flask, project 1 ccg_surface)
since --since with pair_id 0 or NULL, grouped by site, date, time and method,
with an om2 Event Edit link for each group: open it and use Make a pair (admins).

Dry run by default: prints the email.  --send emails it (local sendmail) when
there is anything to report; no events, no email.

  ccg_pairless_events_check.py                  # print
  ccg_pairless_events_check.py --send           # cron, weekdays: 30 7 * * 1-5
"""

import argparse
import subprocess
import sys
from collections import OrderedDict
from email.message import EmailMessage

sys.path.extend(["/ccg/src/db", "/ccg/src/db/db_utils"])
import db_conn  # noqa: E402

EVENT_EDIT = "https://omi.cmdl.noaa.gov/om2/flask_eventedit.php?project=ccg_surface&event={}"
DEFAULT_TO = "geoff.dutton@noaa.gov"


def pairless_groups(db, since: str) -> "OrderedDict":
    rows = db.doquery("""
        SELECT e.num, e.id, e.date, e.time, e.me, s.code
        FROM   ccgg.flask_event e
        JOIN   gmd.site s ON s.num = e.site_num
        WHERE  e.strategy_num = 1 AND e.project_num = 1 AND e.date >= %s
          AND  (e.pair_id IS NULL OR e.pair_id = 0)
        ORDER  BY s.code, e.date, e.time, e.num""", [since]) or []
    groups = OrderedDict()
    for r in rows:
        t = int(r["time"].total_seconds())
        key = (r["code"], str(r["date"]), "%02d:%02d" % (t // 3600, t % 3600 // 60), r["me"])
        groups.setdefault(key, []).append(r)
    return groups


def build(groups, since: str, to: str) -> EmailMessage:
    n = sum(len(v) for v in groups.values())
    lines = [f"{n} CCG surface flask event{'s' if n != 1 else ''} since {since} have no pair id "
             f"({len(groups)} group{'s' if len(groups) != 1 else ''} by site and sample time).",
             "They were probably entered with the legacy om pages.  Open each link in om2 Event Edit,",
             "tick the flasks sampled together in Make a pair, and click Create pair.", ""]
    for (code, date, hhmm, me), evs in groups.items():
        flasks = ", ".join(f"{e['id']} (#{e['num']})" for e in evs)
        lines.append(f"{code} {date} {hhmm} UTC, method {me}: {flasks}")
        lines.append("  " + EVENT_EDIT.format(evs[0]["num"]))
    msg = EmailMessage()
    msg["Subject"] = f"CCG: {n} flask event{'s' if n != 1 else ''} without a pair id"
    msg["From"] = "om2 checks <gdutton@catsdata.cmdl.noaa.gov>"
    msg["To"] = to
    msg.set_content("\n".join(lines) + "\n")
    return msg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--send", action="store_true", help="email the list (default: print it)")
    ap.add_argument("--since", default="2025-01-01", help="first sample date to check (default 2025-01-01)")
    ap.add_argument("--to", default=DEFAULT_TO, help=f"recipient (default {DEFAULT_TO})")
    args = ap.parse_args()

    groups = pairless_groups(db_conn.HATS_ng(), args.since)
    if not groups:
        print(f"No CCG surface events without a pair id since {args.since}.")
        return
    msg = build(groups, args.since, args.to)
    if args.send:
        subprocess.run(["/usr/sbin/sendmail", "-t", "-oi"], input=msg.as_bytes(), check=True)
        print(f"Sent: {msg['Subject']} -> {args.to}")
    else:
        print(msg)


if __name__ == "__main__":
    main()
