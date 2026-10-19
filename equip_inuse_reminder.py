#!/usr/bin/env python3
"""Email shippers to set a unit's In Use date (hats.hatsflask_equip_inuse_reminder).

Daily job on catsdata, weekdays 8 am (see sql/equip_inuse_reminder.md).  For each open
hats.hatsflask_equip_inuse_reminder row it reads the unit's current trip from
ccgg_equip.gen_inv and:

  - closes the reminder if the In Use date is set (inuse_set), the unit is no
    longer checked out (checked_in), or it was checked out again on another
    date (rechecked_out);
  - otherwise, when one or more schedule steps (days after start_date) are due
    and not yet sent, sends ONE email (for the latest due step, so a missed
    day does not send a burst) and records the steps as sent;
  - after the last step, marks the reminder expired; the unit then stays on
    the dashboard's "Waiting for In Use date" list.

Dry run by default: prints what it would send.  --send emails (local
sendmail) and updates the table.  --preview runs on the units checked out in
the last 120 days with no In Use date, without the table, to show the emails
(creator unknown there, so they go to --to).

  equip_inuse_reminder.py                 # dry run on the table
  equip_inuse_reminder.py --send          # cron, weekdays 8 am Boulder: 0 8 * * 1-5
  equip_inuse_reminder.py --preview --to geoff.dutton@noaa.gov [--send]
"""

import argparse
import re
import subprocess
import sys
from datetime import date
from email.message import EmailMessage

sys.path += ['/ccg/src/db', '/ccg/src/db/db_utils']
import db_conn  # noqa: E402

UNIT_URL = 'https://omi.cmdl.noaa.gov/om2/equip_inventory.php?unit={}'
MAX_DAYS = 56   # no reminders more than 8 weeks after start_date, whatever the schedule says


def kind(row):
    if int(row['gen_type_num']) == 2:
        return 'PCP'
    return 'pump' if row['gen_inv_id'].endswith('-P') else 'PSU'


def steps(text):
    return sorted({int(s) for s in re.findall(r'\d+', text or '')})


def first_flasks(db, row):
    """First flask sample at the site on or after date_out, as a hint only:
    events do not record which unit filled them."""
    p, d = int(row['project_num']), row['date_out']
    if p == 6:
        v = db.doquery("""SELECT MIN(DATE(ev_datetime)) FROM hats.hatsflask_event_view
                          WHERE site = %s AND project_num = 6 AND ev_datetime >= %s""",
                       [row['site'], d], numRows=0)
    elif p == 1:
        v = db.doquery("""SELECT MIN(date) FROM ccgg.flask_event
                          WHERE site_num = %s AND project_num = 1 AND strategy_num = 1 AND date >= %s""",
                       [row['site_num'], d], numRows=0)
    else:
        v = None
    return str(v) if v else ''


def recipients(db, row, override):
    if override:
        return [override]
    to = []
    uid = (row.get('creator') or '').split('@')[0].strip()
    if re.fullmatch(r'[A-Za-z0-9._-]+', uid):
        to.append(uid.lower() + '@noaa.gov')
    if row.get('num'):
        for r in db.doquery("""SELECT c.email FROM hats.hatsflask_equip_inuse_reminder_recipient r
                               JOIN ccgg.contact c ON c.num = r.contact_num
                               WHERE r.reminder_num = %s AND c.affiliation = 'GML' AND c.is_active = 1
                                 AND c.email <> ''""", [row['num']]) or []:
            to.append(r['email'].strip())
    return sorted(set(to))


def message(db, row, days, last_step, to):
    k = kind(row)
    hint = first_flasks(db, row)
    weeks = days // 7
    msg = EmailMessage()
    msg['To'] = ', '.join(to)
    msg['Subject'] = (f"Set the In Use date: {k} {row['gen_inv_id']} at {row['site']} "
                      f"(shipped {weeks} week{'s' if weeks != 1 else ''} ago)")
    lines = [
        f"{k} {row['gen_inv_id']} was checked out to {row['site']} ({row['project']}) on {row['date_out']}, "
        f"{days} days ago, and has no In Use date yet.",
        '',
        "If the site has put it in service, set the In Use date on its card:",
        UNIT_URL.format(row['gen_inv_id']),
        '',
    ]
    if hint:
        lines += [f"First flask sample at {row['site']} after it was shipped: {hint}. "
                  "That may be the date, but the sample may also have been filled with the previous unit.", '']
    lines += ["If it is a spare or a test unit that will not go into service, cancel the reminder on the card.",
              '',
              'This is the last reminder; the unit stays on the dashboard list until the date is set.'
              if last_step else
              'You will be reminded again if the date is still not set.']
    msg.set_content('\n'.join(lines))
    return msg


def send(msg):
    subprocess.run(['/usr/sbin/sendmail', '-t', '-oi'], input=msg.as_bytes(), check=True)


UNIT_SQL = """SELECT g.id AS gen_inv_id, g.gen_type_num, g.site_num, s.code AS site, g.project_num,
                     p.abbr AS project, g.gen_status_num, NULLIF(g.date_out, '0000-00-00') AS cur_out,
                     NULLIF(g.date_inuse, '0000-00-00') AS date_inuse
              FROM ccgg_equip.gen_inv g
              JOIN gmd.site s ON s.num = g.site_num
              LEFT JOIN gmd.project p ON p.num = g.project_num"""


def run_table(db, args, today):
    rems = db.doquery("SELECT * FROM hats.hatsflask_equip_inuse_reminder WHERE status = 'open' ORDER BY date_out") or []
    for r in rems:
        u = db.doquery(UNIT_SQL + " WHERE g.id = %s AND g.gen_type_num = %s",
                       [r['gen_inv_id'], r['gen_type_num']])
        u = u[0] if u else None
        reason = None
        if not u or int(u['gen_status_num']) != 3:
            reason = 'checked_in'
        elif str(u['cur_out']) != str(r['date_out']):
            reason = 'rechecked_out'
        elif u['date_inuse']:
            reason = 'inuse_set'
        if reason:
            print(f"close {r['gen_inv_id']} ({r['date_out']}): {reason}")
            if args.send:
                db.doquery("""UPDATE hats.hatsflask_equip_inuse_reminder SET status = 'done', closed_reason = %s,
                              closed_at = NOW() WHERE num = %s AND status = 'open'""", [reason, r['num']])
            continue
        row = {**u, **r}
        days = (today - r['start_date']).days       # the schedule counts from start_date
        days_out = (today - r['date_out']).days     # the email says how long ago it was shipped
        sched = [s for s in steps(r['schedule']) if s <= MAX_DAYS] or [MAX_DAYS]
        sent = set(steps(r['sent_days']))
        due = [s for s in sched if s <= days and s not in sent]
        if not due:
            continue
        last = max(due) == sched[-1]
        to = recipients(db, row, args.to)
        if not to:
            print(f"skip {r['gen_inv_id']}: no recipient for creator {r['creator']!r}")
            continue
        msg = message(db, row, days_out, last, to)
        print(f"{'send' if args.send else 'would send'} {r['gen_inv_id']} day {days} of reminders (steps {due}) to {msg['To']}")
        if args.verbose:
            print(msg.get_content())
        if args.send:
            send(msg)
            new_sent = ','.join(str(s) for s in sorted(sent | set(due)))
            db.doquery("""UPDATE hats.hatsflask_equip_inuse_reminder SET sent_days = %s, last_sent_at = NOW(),
                          status = IF(%s, 'expired', status), closed_reason = IF(%s, 'schedule_done', closed_reason),
                          closed_at = IF(%s, NOW(), closed_at) WHERE num = %s""",
                       [new_sent, last, last, last, r['num']])


def run_preview(db, args, today):
    units = db.doquery(UNIT_SQL + """ WHERE g.gen_status_num = 3 AND NULLIF(g.date_inuse, '0000-00-00') IS NULL
                                      AND g.date_out >= CURDATE() - INTERVAL 120 DAY ORDER BY g.date_out""") or []
    for u in units:
        row = {**u, 'date_out': u['cur_out'], 'creator': ''}
        days = (today - u['cur_out']).days
        if days < 28 or days > 56:   # default schedule: 4, 6, 8 weeks; nothing after 8
            continue
        msg = message(db, row, days, days >= 56, [args.to])
        print(f"{'send' if args.send else 'would send'} preview {u['gen_inv_id']} day {days} to {args.to}")
        if args.verbose or not args.send:
            print(msg.get_content(), '\n' + '-' * 60)
        if args.send:
            send(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--send', action='store_true', help='email and update the table (default: dry run)')
    ap.add_argument('--preview', action='store_true', help='run on current checked-out units without the table')
    ap.add_argument('--to', help='send every email here instead of the shipper (required with --preview)')
    ap.add_argument('-v', '--verbose', action='store_true', help='print the email bodies')
    args = ap.parse_args()
    if args.preview and not args.to:
        ap.error('--preview needs --to')
    db = db_conn.HATS_ng()
    today = date.today()
    (run_preview if args.preview else run_table)(db, args, today)


if __name__ == '__main__':
    main()
