#!/usr/bin/env python3
"""Copy GCwerks' integration parameter and peak ID files into the hats tables.

Each file under <gcdir>/integrator/channelN/{parameters,peakid}/ becomes one
current 'gcwerks' set in hats.ng_integ_sets, with its lines in
hats.ng_integ_parameters or hats.ng_integ_peakid (DDL: sql/ng_integ_params.sql).
logosint and the logos_integ page read them from there, and the page's edits
are saved as 'logosint' sets in front of them.

Reruns are cheap: a file whose text (sha1) is unchanged is skipped; a changed
file gets a new set and the old one is retired; a file gone from disk has its
set retired.  As gcupdate does, <name>.edited is read in place of <name>.

--overlay also copies the files saved by the logos_integ page before the move to
the database (<gcdir>/logosint/channelN/...) as 'logosint' sets, where none is
current yet.

--proposal DIR --channel N loads a folder of proposed files (DIR/parameters, DIR/peakid, GCwerks'
layout and names, as logosint integrate --params-dir reads them) as 'logosint' sets for one
instrument and channel: the logos_integ page then shows them under Params: logosint, and the
LOGOS Integrator uses them. GCwerks never reads them. A name already current in the logosint
set is replaced (retired, kept in the table) when its text differs; other logosint sets are left.
The logosint set sits in front of GCwerks' name by name, so each GCwerks file dated after the
proposal's first file is masked with a logosint copy of the proposal file that governs its date;
otherwise GCwerks' later files would take over again from their own dates.

  logosint_params_import.py brw --proposal ~/logosint_pilot/brw_ch1/proposal --channel 1      # dry run
  logosint_params_import.py brw --proposal ~/logosint_pilot/brw_ch1/proposal --channel 1 -i

Dry run by default; -i writes.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime

sys.path.extend(["/ccg/src/db", "/ccg/src/db/db_utils"])
from db_utils import db_conn  # noqa: E402

GC_ROOT = "/nfs/hats/gc"
# GCwerks folder -> inst_num.  std (the cal lab GC) has no inst_num yet.
INSTRUMENTS = {
    "brw": 239, "sum": 240, "nwr": 241, "mlo": 242, "smo_cats": 243, "spo": 244,
    "smo": 236,   # IE3
    "fe3": 193, "m4": 192, "bld1": 220, "pr1": 58,
}
GCDIRS = {"pr1": "/nfs/hats/Perseus-1"}
KINDS = ("parameters", "peakid")
NAME_RE = re.compile(r"^(\d\d(\d\d(\d\d(\.\d{4})?)?)?|initial)$")
MEMBER_RE = re.compile(r"^(mass\d+|default)$")
IMPORTER = "gcwerks-import"


def gcdir(inst):
    return GCDIRS.get(inst, os.path.join(GC_ROOT, inst))


def start_time(name):
    """Effective time of a dated name (as logosint's files::effective), else None."""
    m = re.match(r"^(\d\d)(\d\d)?(\d\d)?(?:\.(\d\d)(\d\d))?$", name)
    if not m:
        return None
    yy = int(m.group(1))
    try:
        return datetime(1900 + yy if yy >= 70 else 2000 + yy, int(m.group(2) or 1), int(m.group(3) or 1),
                        int(m.group(4) or 0), int(m.group(5) or 0))
    except ValueError:
        return None


def valid_name(name):
    return name == "initial" or (NAME_RE.match(name) and start_time(name) is not None)


def read_gcupdate(path):
    """<path>.edited wins when it exists, as gcupdate reads it."""
    for p in (path + ".edited", path):
        if os.path.isfile(p):
            with open(p, "rb") as f:
                return f.read().decode("utf-8", "replace")
    return None


def files_in(kind_dir):
    """(name, member, path) for every file in a parameters or peakid folder."""
    out = []
    if not os.path.isdir(kind_dir):
        return out

    def folder(d, prefix):
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if n.endswith(".edited"):
                continue
            if os.path.isfile(p) and valid_name(n):
                out.append((prefix + n, "", p))
            elif os.path.isdir(p) and valid_name(n):   # GC/MS: <date>/mass83, initial/default
                for m in sorted(os.listdir(p)):
                    if MEMBER_RE.match(m) and os.path.isfile(os.path.join(p, m)):
                        out.append((prefix + n, m, os.path.join(p, m)))

    folder(kind_dir, "")
    sp = os.path.join(kind_dir, "special")
    if os.path.isdir(sp):
        for n in sorted(os.listdir(sp)):
            p = os.path.join(sp, n)
            if n.endswith(".edited"):
                continue
            if os.path.isfile(p):
                out.append(("special/" + n, "", p))
            elif os.path.isdir(p):
                for m in sorted(os.listdir(p)):
                    if MEMBER_RE.match(m) and os.path.isfile(os.path.join(p, m)):
                        out.append(("special/" + n, m, os.path.join(p, m)))
    return out


def split_comment(line):
    body, _, comment = line.partition("#")
    comment = comment.strip() if "#" in line else None
    return body.split(), comment


def parse_parameters(text):
    """Rows (t, code, args, comment) and the number of lines kept only as comments."""
    rows, bad = [], 0
    for line in text.splitlines():
        tok, comment = split_comment(line)
        if not tok:
            if comment is not None:
                rows.append((None, None, None, comment))
            continue
        try:
            t = float(tok[0])
            ok = len(tok) >= 2 and re.match(r"^[A-Za-z][A-Za-z0-9]*$", tok[1]) and len(tok[1]) <= 8
        except ValueError:
            ok = False
        if not ok:
            rows.append((None, None, None, line.strip()))
            bad += 1
            continue
        rows.append((t, tok[1], " ".join(tok[2:]) or None, comment))
    return rows, bad


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


def parse_peakid(text):
    """Rows (peak_name, rt, rt_window, mass, extra, comment) and the lines kept only as comments."""
    rows, bad = [], 0
    for line in text.splitlines():
        tok, comment = split_comment(line)
        if not tok:
            if comment is not None:
                rows.append((None, None, None, None, None, comment))
            continue
        if len(tok) < 3 or num(tok[1]) is None or num(tok[2]) is None or len(tok[0]) > 32:
            rows.append((None, None, None, None, None, line.strip()))
            bad += 1
            continue
        rest = tok[3:]
        mass = num(rest[0]) if rest else None
        extra = " ".join(rest[1:] if mass is not None else rest) or None
        rows.append((tok[0], num(tok[1]), num(tok[2]), mass, extra, comment))
    return rows, bad


def insert_set(db, inst_num, channel, kind, param_set, name, member, text, by, comment=None, replaces=None):
    rows, bad = (parse_parameters if kind == "parameters" else parse_peakid)(text)
    if not rows:
        rows = [(None,) * (3 if kind == "parameters" else 5) + ("(empty file)",)]
    sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
    num_ = db.doquery(
        """INSERT INTO hats.ng_integ_sets (inst_num, channel, kind, param_set, name, member, start_time,
               source_sha1, comment, created_by, replaces_num) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (inst_num, channel, kind, param_set, name, member, start_time(name), sha, comment, by, replaces),
        insert=True, commit=False)
    if kind == "parameters":
        sql = "INSERT INTO hats.ng_integ_parameters (set_num, seq, t, code, args, comment) VALUES (%s,%s,%s,%s,%s,%s)"
    else:
        sql = ("INSERT INTO hats.ng_integ_peakid (set_num, seq, peak_name, rt, rt_window, mass, extra, comment)"
               " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)")
    db.doquery(sql, [(num_, i + 1) + r for i, r in enumerate(rows)], multiInsert=True, commit=False)
    return num_, len(rows), bad


def retire(db, set_num, by):
    db.doquery("UPDATE hats.ng_integ_sets SET retired_at = NOW(), retired_by = %s WHERE num = %s AND retired_at IS NULL",
               (by, set_num), commit=False)


def current_sets(db, inst_num, channel, param_set):
    rows = db.doquery(
        """SELECT num, kind, name, member, source_sha1 FROM hats.ng_integ_sets
            WHERE inst_num = %s AND channel = %s AND param_set = %s AND retired_at IS NULL""",
        (inst_num, channel, param_set), commit=False) or []   # commit=True would end a dry run's transaction
    return {(r["kind"], r["name"], r["member"]): (r["num"], r["source_sha1"]) for r in rows}


def overlay_users(root):
    """name -> user of the last save, from the page's edits.log (by channel and kind)."""
    users = {}
    try:
        with open(os.path.join(root, "logosint", "edits.log")) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    users[(r["channel"], r["kind"], r["name"])] = r.get("user") or "logos_integ"
                except (ValueError, KeyError):
                    pass
    except OSError:
        pass
    return users


def import_inst(db, inst, write, overlay, verbose):
    inst_num = INSTRUMENTS[inst]
    root = gcdir(inst)
    integ = os.path.join(root, "integrator")
    if not os.path.isdir(integ):
        print(f"{inst}: no integrator folder in {root}")
        return
    channels = sorted(int(n[7:]) for n in os.listdir(integ) if re.match(r"^channel\d+$", n))
    for c in channels:
        cur = current_sets(db, inst_num, c, "gcwerks")
        seen = set()
        stats = {"new": 0, "changed": 0, "same": 0, "gone": 0, "lines": 0, "bad": 0}
        for kind in KINDS:
            for name, member, path in files_in(os.path.join(integ, f"channel{c}", kind)):
                key = (kind, name, member)
                seen.add(key)
                text = read_gcupdate(path)
                if text is None:
                    continue
                sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
                old = cur.get(key)
                if old and old[1] == sha:
                    stats["same"] += 1
                    continue
                stats["changed" if old else "new"] += 1
                if old:
                    retire(db, old[0], IMPORTER)
                _, n, bad = insert_set(db, inst_num, c, kind, "gcwerks", name, member, text, IMPORTER,
                                       replaces=old[0] if old else None)
                stats["lines"] += n
                stats["bad"] += bad
                if bad and verbose:
                    print(f"  {inst} channel{c} {kind}/{name}{'/' + member if member else ''}: {bad} line(s) kept as comments")
        for key, (num_, _) in cur.items():
            if key not in seen:
                retire(db, num_, IMPORTER)
                stats["gone"] += 1

        over = []
        if overlay:
            lcur = current_sets(db, inst_num, c, "logosint")
            users = overlay_users(root)
            for kind in KINDS:
                for name, member, path in files_in(os.path.join(root, "logosint", f"channel{c}", kind)):
                    if member or (kind, name, member) in lcur:
                        continue
                    text = read_gcupdate(path)
                    by = users.get((c, kind, name), "logos_integ")
                    insert_set(db, inst_num, c, kind, "logosint", name, member, text, by,
                               comment="copied from " + path)
                    over.append(f"{kind}/{name} ({by})")

        if write:
            db.doquery("COMMIT", commit=False)
        else:
            db.doquery("ROLLBACK", commit=False)
        print(f"{inst} ({inst_num}) channel{c}: {stats['new']} new, {stats['changed']} changed, {stats['same']} unchanged,"
              f" {stats['gone']} gone; {stats['lines']} lines" + (f", {stats['bad']} kept as comments" if stats["bad"] else "")
              + (f"; logosint copies: {', '.join(over)}" if over else ""))


def import_proposal(db, inst, channel, folder, by, write):
    """Load a proposal folder's files as current 'logosint' sets for one instrument and channel."""
    inst_num = INSTRUMENTS[inst]
    folder = os.path.abspath(os.path.expanduser(folder))
    if not any(os.path.isdir(os.path.join(folder, k)) for k in KINDS):
        sys.exit(f"{folder}: no parameters/ or peakid/ folder")
    cur = current_sets(db, inst_num, channel, "logosint")
    gcw = current_sets(db, inst_num, channel, "gcwerks")
    done = []
    for kind in KINDS:
        files = []
        for name, member, path in files_in(os.path.join(folder, kind)):
            if member:
                continue
            with open(path) as f:
                files.append((name, path, f.read()))
        dated = sorted((start_time(n) or datetime.min, n, p, t) for n, p, t in files)
        first = dated[0][0] if dated else None
        # GCwerks files after the proposal starts: mask each with the proposal file in force then.
        masks = []
        if first is not None:
            for (k, gname, gmember) in gcw:
                st = start_time(gname)
                if k != kind or gmember or st is None or st <= first or any(n == gname for _, n, _, _ in dated):
                    continue
                gov = [d for d in dated if d[0] <= st][-1]
                masks.append((gname, gov[2] + f" (masks GCwerks {gname})", gov[3]))
        for name, path, text in [(n, p, t) for _, n, p, t in dated] + masks:
            member = ""
            key = (kind, name, member)
            sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
            old = cur.get(key)
            if old and old[1] == sha:
                done.append(f"{kind}/{name} unchanged")
                continue
            if old:
                retire(db, old[0], by)
            num_, n, bad = insert_set(db, inst_num, channel, kind, "logosint", name, member, text, by,
                                      comment="proposal " + path, replaces=old[0] if old else None)
            done.append(f"{kind}/{name} {'replaced' if old else 'new'} ({n} lines" + (f", {bad} kept as comments" if bad else "") + ")")
    db.doquery("COMMIT" if write else "ROLLBACK", commit=False)
    print(f"{inst} ({inst_num}) channel{channel} logosint set from {folder}: " + ("; ".join(done) or "no files"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insts", nargs="*", help="GCwerks folders (default all): " + " ".join(INSTRUMENTS))
    ap.add_argument("-i", "--insert", action="store_true", help="write to the database (default: dry run)")
    ap.add_argument("--overlay", action="store_true", help="also copy the page's old logosint/ files as logosint sets")
    ap.add_argument("-v", "--verbose", action="store_true", help="name files with lines that did not parse")
    ap.add_argument("--proposal", metavar="DIR", help="load DIR/parameters and DIR/peakid as logosint sets (needs one instrument and --channel)")
    ap.add_argument("--channel", type=int, help="the channel for --proposal")
    ap.add_argument("--by", default=None, help="created_by for --proposal (default: agent-proposal/<your login>)")
    a = ap.parse_args()
    if a.proposal:
        if len(a.insts) != 1 or a.insts[0] not in INSTRUMENTS or a.channel is None:
            ap.error("--proposal needs exactly one instrument folder and --channel")
        by = a.by or f"agent-proposal/{os.environ.get('USER', 'unknown')}"
        import_proposal(db_conn.HATS_ng(), a.insts[0], a.channel, a.proposal, by, a.insert)
        if not a.insert:
            print("dry run: nothing written (use -i)")
        return
    insts = a.insts or list(INSTRUMENTS)
    bad = [i for i in insts if i not in INSTRUMENTS]
    if bad:
        ap.error("unknown instrument folder: " + ", ".join(bad))
    db = db_conn.HATS_ng()
    for inst in insts:
        import_inst(db, inst, a.insert, a.overlay, a.verbose)
    if not a.insert:
        print("dry run: nothing written (use -i)")


if __name__ == "__main__":
    main()
