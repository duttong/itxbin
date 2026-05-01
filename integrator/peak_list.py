"""
Parse GCwerks peak.list config file.

File format (whitespace-delimited, # comments):
    Gas Name    Channel Number    Gas Report Name

Example:
    CFC11a    0    CFC11a
    N2O       1    N2O
    CFC12     2    CFC12
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Compound:
    """One entry from the peak list."""
    name: str            # internal GCwerks name
    channel: int         # detector channel (0-based)
    report_name: str     # display/report name
    nominal_rt: float | None = None    # seconds, discovered from data
    rt_window_lo: float = 15.0         # seconds before RT to start integration window
    rt_window_hi: float = 15.0         # seconds after  RT to end   integration window
    ref: bool = False                  # use as RT alignment reference for this channel
    meth: int = 1                      # 1=fixed window, 2=tangent skim


class PeakList:
    """
    Parsed peak.list configuration.

    Usage
    -----
    pl = PeakList.from_file('/hats/gc/fe3/config/peak.list')
    pl.compounds           # all compounds
    pl.by_channel(0)       # compounds on channel 0, in elution order
    pl.get('CFC11a')       # look up by name
    """

    def __init__(self, compounds: list[Compound]):
        self.compounds = compounds

    @classmethod
    def from_file(cls, path: Path | str) -> PeakList:
        """
        Parse a peak list file.

        Accepts these column formats (auto-detected by column count):

        3-column  (original GCwerks peak.list):
            name   channel   report_name

        5-column:
            name   channel   report_name   nominal_rt   win_lo
            (symmetric window: win_hi = win_lo)

        6-column (old extended format):
            name   channel   report_name   nominal_rt   win   ref
            (symmetric window; detected when col[5] is 0 or 1)

        8-column (current fe3_peaks.conf format):
            name   channel   report_name   nominal_rt   win_lo   win_hi   ref   meth
        """
        compounds = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                name = parts[0]
                try:
                    channel = int(parts[1])
                except ValueError:
                    continue
                report_name = parts[2] if len(parts) >= 3 else name
                nominal_rt  = float(parts[3]) if len(parts) >= 4 else None
                rt_window_lo = float(parts[4]) if len(parts) >= 5 else 15.0

                # Distinguish 6-column old format (col[5] is ref=0/1)
                # from new asymmetric format (col[5] is win_hi > 1).
                if len(parts) == 6 and parts[5] in ('0', '1'):
                    rt_window_hi = rt_window_lo
                    ref  = bool(int(parts[5]))
                    meth = 1
                else:
                    rt_window_hi = float(parts[5]) if len(parts) >= 6 else rt_window_lo
                    ref  = bool(int(parts[6])) if len(parts) >= 7 else False
                    meth = int(parts[7])        if len(parts) >= 8 else 1

                compounds.append(Compound(
                    name=name,
                    channel=channel,
                    report_name=report_name,
                    nominal_rt=nominal_rt,
                    rt_window_lo=rt_window_lo,
                    rt_window_hi=rt_window_hi,
                    ref=ref,
                    meth=meth,
                ))
        return cls(compounds)

    def by_channel(self, channel: int) -> list[Compound]:
        """Return compounds on *channel*, preserving file order (= elution order)."""
        return [c for c in self.compounds if c.channel == channel]

    def channels(self) -> list[int]:
        """Unique channel numbers, sorted."""
        return sorted({c.channel for c in self.compounds})

    def get(self, name: str) -> Compound | None:
        """Look up a compound by name (case-insensitive)."""
        name_lo = name.lower()
        for c in self.compounds:
            if c.name.lower() == name_lo or c.report_name.lower() == name_lo:
                return c
        return None

    def reference_peak(self, channel: int) -> Compound | None:
        """Return the ref=True compound for *channel*, or None if none marked."""
        refs = [c for c in self.by_channel(channel) if c.ref]
        return refs[0] if refs else None

    def set_nominal_rts(self, rt_map: dict[str, float]) -> None:
        """Assign nominal RTs from a dict of {name: rt_seconds}."""
        for name, rt in rt_map.items():
            c = self.get(name)
            if c is not None:
                c.nominal_rt = rt

    def summary(self) -> str:
        lines = ['Compound       Ch   RT(s)   -Lo  +Hi  Ref  Meth']
        lines.append('-' * 50)
        for c in self.compounds:
            rt  = f'{c.nominal_rt:.1f}' if c.nominal_rt is not None else '  —  '
            ref = ' *' if c.ref else ''
            lines.append(
                f'{c.report_name:<14} {c.channel}   {rt:>6}'
                f'   {c.rt_window_lo:>3.0f}  {c.rt_window_hi:>3.0f}'
                f'   {int(c.ref)}    {c.meth}{ref}'
            )
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f'PeakList({len(self.compounds)} compounds, channels={self.channels()})'
