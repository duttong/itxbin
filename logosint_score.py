#! /usr/bin/env python
"""
Score logosint integrations of a CATS instrument without GCwerks as the answer key.

Reads one or more `logosint integrate` outputs (TSV, one row per gas per run) and
measures how steady the integration is, the way the data are used: every height is
normalized to the smoothed cal2 (port 6) reference with logos_data's own Normalizing
(LOWESS, about 10 points per run_time segment, detrend method 5), falling back to
cal1 (port 2) where a segment has too few cal2 points.  Lower is better for every
noise, step and flip column; higher for found.

  cal2%    scatter of cal2 heights about their own smoothed curve (robust sd, %)
  cal1%    short-term scatter of cal1/cal2: robust sd of successive differences / sqrt 2
  air%     the same for each air port's normalized response, averaged over the ports
  x%       gases on two channels: short-term scatter of the air ratio between them
  drift%   cal1/cal2 should be flat for a tank pair: robust sd of its monthly medians
           about a straight line per tank pair (needs 3+ months)
  step%    cal1/cal2 change across each parameter file change inside a tank pair
           (median of 3 days after vs before), largest and median
  flip%    injections whose baseline codes differ from the month's usual codes
  found%   injections with a peak (air, cal1 and cal2 together), leaving out dead runs
  dead%    runs with no usable signal: no peak ID window on the channel found a peak
           (integrate's `named` column is 0; older files: every gas in the file missing)
  bnd s    robust sd of start - rt and end - rt on cal2, in seconds

Only the sample ports are scored (2, 4, 6, 8), as cats_gcwerks2db loads; odd ports are
valve faults.  Cal1 statistics leave out the first day after a tank change on port 2 or 6. DB
rejection tags are not applied; the statistics are robust (median absolute
deviation) so the odd bad run doesn't dominate, and every candidate sees the
same runs.

  logosint integrate --site brw --start 2601 --end 2611 --channel 2 --set gcwerks --out gcw.tsv
  logosint integrate --site brw --start 2601 --end 2611 --channel 2 --params-dir cand1 --out cand1.tsv
  logosint_score.py gcw.tsv cand1.tsv [--gas cfc11] [--monthly out.csv] [--plot DIR]
  logosint_score.py export:brw cand1.tsv    # export:SITE = GCwerks' own heights in /hats/gc/cats_results,
                                            # over the time span of the other files (no boundaries or codes)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.extend(["/ccg/src/db", "/ccg/src/db/db_utils", str(Path(__file__).resolve().parent)])

CAL1, CAL2, AIR = 2, 6, (4, 8)       # CATS port layout, every site
SAMPLE_PORTS = (CAL1, CAL2) + AIR
RUN_TIME_GAP = pd.Timedelta(minutes=90)
SETTLE = pd.Timedelta(hours=24)       # after a cal tank change
STEP_WINDOW = pd.Timedelta(days=3)
FOLDER_INST = {"brw": 239, "sum": 240, "nwr": 241, "mlo": 242, "smo_cats": 243, "spo": 244}


def rsd(x):
    """Robust standard deviation (1.4826 x median absolute deviation)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def short_term(x):
    """Robust scatter of successive differences / sqrt 2: noise without slow drift."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return rsd(np.diff(x)) / np.sqrt(2) if len(x) > 3 else np.nan


def read_tsv(path):
    head = []
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            head.append(line.rstrip())
    df = pd.read_csv(path, sep="\t", comment="#", dtype={"run": str, "params": str, "peakid": str, "codes": str})
    folder = None
    if head:
        m = re.search(r" integrate (\S+) ", head[0])
        folder = Path(m.group(1)).name if m else None
    df = df[df["status"] != "refused"].copy()
    df["time"] = pd.to_datetime(df["unix_time"], unit="s")
    df["port"] = pd.to_numeric(df["port"], errors="coerce")
    df = df[df["port"].isin(SAMPLE_PORTS)].copy()
    df["height"] = pd.to_numeric(df["height"], errors="coerce").fillna(0.0)
    df["dead"] = dead_runs(df)
    return df, folder, head


def dead_runs(df):
    """True where the run had no usable signal on that channel: integrate's `named` is 0,
    or, without that column, every gas in the file missed on that run and channel."""
    if "named" in df.columns and df["named"].notna().any():
        return pd.to_numeric(df["named"], errors="coerce").fillna(1).eq(0)
    # Without it, a run counts as dead only on a channel with two or more gases in the file;
    # with one gas, a miss and a dead run look the same, so nothing is marked dead.
    hit = (df["status"].eq("ok") & df["height"].gt(0)).groupby([df["run"], df["channel"]]).transform("any")
    gases = df.groupby("channel")["gas"].transform("nunique")
    one = sorted(df.loc[gases < 2, "gas"].unique())
    if one:
        print(f"no named column (older logosint) and one gas on its channel: {', '.join(one)}; "
              "dead runs can't be told from misses there, so found% counts them", file=sys.stderr)
    return ~hit & (gases >= 2)


CHANNEL_OF = {"q": 0, "a": 1, "f": 2, "cc": 3}
EXPORT_DIR = Path("/hats/gc/cats_results")


def read_export(site, t0, t1):
    """GCwerks' exported heights (cats_export.py CSVs) between t0 and t1, in the integrate layout."""
    frames = []
    for f in sorted(EXPORT_DIR.glob(f"{site}_*.csv")):
        if f.stem.endswith("_2yr") or "gcwerks_all" in f.stem:
            continue
        d = pd.read_csv(f, skipinitialspace=True, dtype=str)
        hts = [c for c in d.columns if c.endswith("_ht")]
        if not hts or "time" not in d.columns:
            continue
        d["time"] = pd.to_datetime(d["time"], errors="coerce")
        d = d[(d["time"] >= t0) & (d["time"] < t1)]
        for hc in hts:
            gas = hc[:-3]
            ch = CHANNEL_OF.get(gas.rsplit("_", 1)[-1]) if "_" in gas else None
            val = lambda c: pd.to_numeric(d[c].str.rstrip("F*BO") if c in d else np.nan, errors="coerce")
            frames.append(pd.DataFrame({
                "time": d["time"], "port": pd.to_numeric(d["port"], errors="coerce"), "gas": gas,
                "channel": ch, "rt": val(f"{gas}_rt"), "height": val(hc), "area": val(f"{gas}_area"),
                "flag": d[hc].str.extract(r"([F*BO])$")[0],
            }))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["time", "port", "gas"])
    df = df[df["port"].isin(SAMPLE_PORTS)].copy()
    df["run"] = df["time"].dt.strftime("%y%m%d.%H%M.") + df["port"].astype("Int64").astype(str)
    df["unix_time"] = df["time"].astype("int64") // 10**9
    df["height"] = df["height"].fillna(0.0)
    df["status"] = np.where(df["height"] > 0, "ok", "missing")
    for c in ("start", "end", "codes", "params"):
        df[c] = np.nan
    df["dead"] = dead_runs(df)
    return df


def tank_changes(inst_num):
    """Times when the cal1 or cal2 tank changed, from hats.ng_port_info."""
    try:
        from db_utils import db_conn
        rows = db_conn.HATS_ng().doquery(
            "SELECT start_datetime FROM hats.ng_port_info WHERE inst_num = %s AND port_num IN (%s, %s) ORDER BY start_datetime",
            [inst_num, CAL1, CAL2]) or []
        return sorted(pd.Timestamp(r["start_datetime"]) for r in rows if r["start_datetime"] is not None)
    except Exception as e:  # scoring still works, without tank pairs
        print(f"no tank history ({e}); drift and steps use the whole range as one tank pair", file=sys.stderr)
        return []


def normalize(g):
    """height / smoothed cal2 (else cal1) for one gas, with logos_data's Normalizing."""
    from logosdata.logos_instruments_core import Normalizing
    g = g.sort_values("time").copy()
    g["run_time"] = g["time"].where(g["time"].diff() > RUN_TIME_GAP).ffill().fillna(g["time"].iloc[0])
    g["analysis_datetime"] = g["time"]
    g["detrend_method_num"] = 5
    g["rejected"] = 0
    out = Normalizing("cats", CAL2, "port", "height").merge_smoothed_data(g)
    out["ref_port"] = np.where(out["normalized_resp"].notna(), CAL2, 0)
    need = out["normalized_resp"].isna()
    if need.any():
        alt = Normalizing("cats", CAL1, "port", "height").merge_smoothed_data(g)
        alt = alt.set_index(["run", "gas"])["normalized_resp"]
        fill = out.loc[need].set_index(["run", "gas"]).index.map(alt.get)
        out.loc[need, "normalized_resp"] = np.asarray(fill, dtype=float)
        out.loc[need & out["normalized_resp"].notna(), "ref_port"] = CAL1
    return out.reset_index(drop=True)


def pair_of(t, changes):
    """Index of the tank pair (period between cal tank changes) for each time, and whether it is settling."""
    ch = np.array(changes, dtype="datetime64[ns]") if changes else np.array([], dtype="datetime64[ns]")
    tv = t.to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(ch, tv, side="right")
    settling = np.zeros(len(tv), dtype=bool)
    if len(ch):
        prev = ch[np.clip(idx - 1, 0, len(ch) - 1)]
        settling = (idx > 0) & ((tv - prev) < np.timedelta64(SETTLE))
    return idx, settling


def gas_metrics(g, changes):
    """Metrics for one gas over the rows given (already normalized)."""
    m = {}
    ok = g["status"].eq("ok") & g["height"].gt(0)
    live = ~g["dead"]
    m["found%"] = 100.0 * ok[live].mean() if live.any() else np.nan
    m["dead%"] = 100.0 * g["dead"].mean() if len(g) else np.nan
    c2 = g[(g["port"] == CAL2) & ok & (g["ref_port"] == CAL2)]
    m["cal2%"] = 100.0 * rsd(c2["height"] / c2["smoothed"] - 1.0) if len(c2) else np.nan
    pair, settling = pair_of(g["time"], changes)
    g = g.assign(pair=pair, settling=settling)
    c1 = g[(g["port"] == CAL1) & ok & (g["ref_port"] == CAL2) & ~g["settling"]]
    per_pair = [short_term(s["normalized_resp"]) for _, s in c1.groupby("pair") if len(s) > 3]
    per_pair = [v for v in per_pair if np.isfinite(v)]
    m["cal1%"] = 100.0 * float(np.median(per_pair)) if per_pair else np.nan   # none: a gap month
    air = [short_term(g.loc[(g["port"] == p) & ok, "normalized_resp"]) for p in AIR]
    m["air%"] = 100.0 * np.nanmean(air) if np.isfinite(air).any() else np.nan
    # drift: monthly medians of cal1/cal2 about a line per tank pair
    resid = []
    for _, s in c1.groupby("pair"):
        mon = s.groupby(s["time"].dt.to_period("M"))["normalized_resp"].median()
        if len(mon) >= 3:
            x = np.arange(len(mon), dtype=float)
            y = mon.to_numpy() / np.median(mon.to_numpy())
            resid.extend(y - np.polyval(np.polyfit(x, y, 1), x))
    m["drift%"] = 100.0 * rsd(resid) if len(resid) >= 3 else np.nan
    # steps at parameter file changes inside a tank pair
    steps = []
    gg = g.sort_values("time")
    change_rows = gg[gg["params"].ne(gg["params"].shift()) & gg["params"].shift().notna()]
    for _, r in change_rows.iterrows():
        s = c1[c1["pair"] == r["pair"]]
        before = s.loc[(s["time"] < r["time"]) & (s["time"] >= r["time"] - STEP_WINDOW), "normalized_resp"]
        after = s.loc[(s["time"] >= r["time"]) & (s["time"] < r["time"] + STEP_WINDOW), "normalized_resp"]
        if len(before) >= 3 and len(after) >= 3:
            steps.append(100.0 * abs(after.median() / before.median() - 1.0))
    m["step%max"] = max(steps) if steps else np.nan
    m["step%med"] = float(np.median(steps)) if steps else np.nan
    m["steps"] = len(steps)
    # baseline codes: share differing from the usual codes for that port role and month
    role = np.where(g["port"].isin(AIR), "air", np.where(g["port"] == CAL1, "cal1", "cal2"))
    key = pd.Series(role, index=g.index) + g["time"].dt.to_period("M").astype(str)
    okc = g.loc[ok, "codes"]
    if okc.notna().any():
        usual = okc.groupby(key[ok]).agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
        m["flip%"] = 100.0 * (okc != key[ok].map(usual)).mean()
    else:
        m["flip%"] = np.nan   # GCwerks' export has no codes
    m["bnd s"] = np.nanmean([rsd(c2["start"] - c2["rt"]), rsd(c2["end"] - c2["rt"])]) if len(c2) else np.nan
    m["n"] = int(len(g))
    return m


def cross_metrics(norm, changes):
    """Short-term scatter of the air ratio for gases on two channels (CFC11_a / CFC11_f ...)."""
    out = {}
    base = {}
    for gas in norm:
        base.setdefault(gas.split("_")[0].upper(), []).append(gas)
    for b, gases in base.items():
        if len(gases) < 2:
            continue
        gases = sorted(gases)
        for i in range(len(gases)):
            for j in range(i + 1, len(gases)):
                a, c = norm[gases[i]], norm[gases[j]]
                keep = ["run", "port", "normalized_resp", "time"]
                m = a[keep].merge(c[["run", "normalized_resp"]], on="run", suffixes=("_a", "_b"))
                m = m[m["port"].isin(AIR)]
                ratio = m["normalized_resp_a"] / m["normalized_resp_b"]
                vals = [short_term(ratio[m["port"] == p]) for p in AIR]
                x = 100.0 * np.nanmean(vals) if np.isfinite(vals).any() else np.nan
                for gname in (gases[i], gases[j]):
                    out.setdefault(gname, []).append(x)
    return {g: float(np.nanmean(v)) for g, v in out.items()}


def score(path, want, changes, span=None, gases=None):
    if path.startswith("export:"):
        df, folder = read_export(path.split(":", 1)[1], *span), path.split(":", 1)[1]
        if gases is not None:
            df = df[df["gas"].isin(gases)]
    else:
        df, folder, head = read_tsv(path)
    if want:
        df = df[df["gas"].str.lower().str.startswith(want.lower())]
    norm = {gas: normalize(g) for gas, g in df.groupby("gas")}
    x = cross_metrics(norm, changes)
    rows, monthly = [], []
    for gas, g in sorted(norm.items()):
        m = gas_metrics(g, changes)
        m["x%"] = x.get(gas, np.nan)
        rows.append({"gas": gas, **m})
        for mon, s in g.groupby(g["time"].dt.to_period("M")):
            mm = gas_metrics(s, changes)
            monthly.append({"gas": gas, "month": str(mon), **mm})
    return pd.DataFrame(rows).set_index("gas"), pd.DataFrame(monthly), folder, norm


def plot(norms, labels, outdir, changes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    gases = sorted(set().union(*[n.keys() for n in norms]))
    for gas in gases:
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for n, lab in zip(norms, labels):
            if gas not in n:
                continue
            g = n[gas]
            ok = g["status"].eq("ok") & g["height"].gt(0)
            c1 = g[(g["port"] == CAL1) & ok]
            ax[0].plot(c1["time"], c1["normalized_resp"], ".", ms=2, label=lab)
            air = g[g["port"].isin(AIR) & ok]
            ax[1].plot(air["time"], air["normalized_resp"], ".", ms=1.5, label=lab)
        times = pd.concat([n[gas]["time"] for n in norms if gas in n])
        t0, t1 = times.min(), times.max()
        for t in changes:
            if t0 <= t <= t1:
                for a in ax:
                    a.axvline(t, color="0.7", lw=0.6)
        ax[1].set_xlim(t0, t1)
        ax[0].set_ylabel("cal1 / cal2")
        ax[1].set_ylabel("air / cal2")
        ax[0].set_title(gas)
        ax[0].legend(markerscale=6)
        lo, hi = ax[0].get_xlim()
        for a, port in ((ax[0], CAL1), (ax[1], None)):
            vals = np.concatenate([n[gas].loc[(n[gas]["port"] == port) if port else n[gas]["port"].isin(AIR), "normalized_resp"].dropna().to_numpy() for n in norms if gas in n])
            if len(vals):
                q1, q9 = np.nanpercentile(vals, [1, 99])
                pad = 0.15 * (q9 - q1)
                a.set_ylim(q1 - pad, q9 + pad)
        fig.tight_layout()
        fig.savefig(outdir / f"{gas}.png", dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv", nargs="+", help="logosint integrate outputs; the first is the baseline when there are several")
    ap.add_argument("--gas", help="only gases starting with this (case ignored)")
    ap.add_argument("--monthly", help="write per-month metrics for every file to this CSV")
    ap.add_argument("--plot", help="write one PNG per gas (cal1/cal2 and air/cal2 for every file) to this folder")
    ap.add_argument("--inst", type=int, help="inst_num for the tank history (default from the folder in the TSV header)")
    a = ap.parse_args()

    results, labels, norms, monthly = [], [], [], []
    files = [p for p in a.tsv if not p.startswith("export:")]
    if not files:
        ap.error("give at least one integrate output; export:SITE takes its time span from them")
    spans = [read_tsv(p) for p in files]
    folder = spans[0][1]
    span = (min(d["time"].min() for d, _, _ in spans), max(d["time"].max() for d, _, _ in spans) + pd.Timedelta(seconds=1))
    inst = a.inst or FOLDER_INST.get(folder or "")
    changes = tank_changes(inst) if inst else []
    gases = set().union(*[set(d["gas"]) for d, _, _ in spans])
    for p in a.tsv:
        r, mon, _, norm = score(p, a.gas, changes, span, gases)
        lab = p.replace(":", "_") if p.startswith("export:") else Path(p).stem
        results.append(r)
        labels.append(lab)
        norms.append(norm)
        mon.insert(0, "file", lab)
        monthly.append(mon)

    cols = ["found%", "dead%", "cal2%", "cal1%", "air%", "x%", "drift%", "step%max", "step%med", "flip%", "bnd s"]
    pd.set_option("display.width", 200)
    for lab, r in zip(labels, results):
        print(f"\n== {lab}")
        print(r[cols + ["steps", "n"]].round(3).to_string())
    if len(results) > 1:
        base = results[0]
        for lab, r in zip(labels[1:], results[1:]):
            d = (r[cols] - base[cols]).reindex(base.index)
            print(f"\n== {lab} minus {labels[0]} (negative is better, except found%)")
            print(d.round(3).to_string())
    if a.monthly:
        pd.concat(monthly).to_csv(a.monthly, index=False)
        print(f"\nmonthly metrics to {a.monthly}")
    if a.plot:
        plot(norms, labels, Path(a.plot), changes)
        print(f"plots to {a.plot}")


if __name__ == "__main__":
    main()
