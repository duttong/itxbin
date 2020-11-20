# FE3data

<h3>Data processing software for the FE3 instrument</h3>

<h4>fe3_data.py</h4>
<p>This is the main program but most of the code resides in other python modules</p>
<p>There are two methods for running fe3_data.py, interactive or batch mode. To run interatively
   simply call fe3_data.py. For batch processing without the graphical interface check out
   the help options with fe3_data.py -h</p>

<h4>fe3_gui.py</h4>
<p>Pandas and matplotlib code for all of the figures and tables. This code also interfaces with
   code generated by Qt Designer</p>

<h4>fe3_incoming.py</h4>
<p>Methods for merging GCwerks integration results with the meta data from
flask and calibration runs on the FE3 instrument.</p>

<h4>processing_routines.py</h4>
<p>The numerical processing routines to generate normalized (detrended) data,
calibration curves, and mole fractions.</p>
