*CATS Experiment Notes*

- CATS stands for Chromatograph for Atmospheric Trace Species
- Developed in 1990s
- Installed from 1998 to 2005 at the sites: brw, sum, nwr, mlo, smo, spo
- Four channel Electron Capture Detector (ECD) gas chromatograph (GC)
- Nominally 30 minute chromatograms on each channel
- All four channels injected at the same time
- A typical run sequence controlled by a multiposition stream selection valve (SSV) is cal1, air1, cal2, air2, ...
- Port numbers: cal1 is port 2, air1 is port 4, cal2 is port 6, air2 is port 8
- cal1 is the low calibration standard. Diluted ambient air about 90% ambient (ambient relative to the year the cylinder was filled)
- cal2 is the high calibration standard. Near ambient.
- cal2 also serves as the normalization/reference tank, since it's nearest to ambient
- air1 and air2 are air sample tubes on a tower at the site. Both are independent and have separate air pumps
- cal and air samples are shared and injected at the same time on all four ECD channels
- Channel naming: q is N2O/SF6, f is the halocarbon solvents (CFC12, CFC11, CFC113, H1211, CCl4, CH3CCl3, CHCl3)
- As the instrumentation has aged, we have focused on a couple analytes as top priority. N2O, SF6, CFC11, CFC12, and CCl4. This means lesser priority channels have been scavenged to keep channels q and f running.

** Data processing **
- GCwerks is used to measure the chromatogram peak heights. The height is a measure of response that scales to the mole fraction of the analyte measured.
- GCwerks integration results are exported to /hats/gc/<sitecode>/results and /hats/gc/cats_results
- The analytes/gases/peaks are defined in the GCwerks file /hats/gc/<sitecode>/config/peak.list
- Originally the CATS data integration results were further processed via Igor Pro custom software
- in 2026, CATS data has been transitioned to the HATS database tables and a python data processing framework
- Mole fractions are computed weekly, normalizing air/cal response against a fit through the cal tank(s). Four methods: cal12 (fit through both cal1 and cal2, capturing gain and offset), cal1 or cal2 (fit through a single tank, forced through the origin), ref (scale directly off a reference tank's assigned value, no weekly fit)
- cal12 is the preferred/default method for every analyte except CCl4, which defaults to cal1
- A bad or offline cal tank breaks cal12 for that stretch, since it needs both tanks. Switching to the single-tank method (cal1 or cal2) for just that window routes around it

** HATS database tables **
- This is not a comprehensive list of tables, rather the important one
- The raw GCwerks results are imported into hats.ng_insitu_analysis and hats.ng_insitu_mole_fractions
- Tagging/flagging individual peaks are located in hats.ng_insitu_mole_fraction_tags
- Which cal method (cal12/cal1/cal2/ref) is used for a given week is recorded per analyte on the air-port rows in hats.ng_insitu_mole_fractions, column mf_method_num
- hats.scale_assignments_view holds each cal tank's assigned mole fraction per fill, keyed on tank serial number and parameter_num. This is what a weekly cal fit is fit against
- Current plans on how to update the initial CATS import into the hats tables are in CATS-processing.md
