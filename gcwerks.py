#!/usr/bin/env python

import sys
import subprocess
from PyQt5 import QtWidgets, QtGui, QtCore

GCWERKS_PATH = '/hats/gc/gcwerks-3/bin/gcwerks'
CATS_SITES = {'brw', 'sum', 'nwr', 'mlo', 'smo', 'spo'}
GOOD_SITES = sorted(CATS_SITES | {'std', 'stdhp', 'bld1', 'fe3', 'pr1', 'm4'})

def launch_gcwerks(site):
    if site == 'pr1':
        command = [GCWERKS_PATH, '-gcdir', '/data/Perseus-1/']
    else:
        command = [GCWERKS_PATH, '-gcdir', f'/hats/gc/{site}']
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}", file=sys.stderr)

def site_selection_ui():
    app = QtWidgets.QApplication(sys.argv)
    
    window = QtWidgets.QWidget()
    window.setWindowTitle("Select GCwerks Site")
    window.setGeometry(100, 100, 300, 150)
    window.setStyleSheet("background-color: mistyrose;")
    
    layout = QtWidgets.QVBoxLayout()
    
    label = QtWidgets.QLabel("Select a site:")
    label.setFont(QtGui.QFont('Helvetica', 16))
    layout.addWidget(label)
    
    site_combo = QtWidgets.QComboBox()
    site_combo.setFont(QtGui.QFont('Helvetica', 16))

    # Insert a placeholder as the first item
    site_combo.addItem("-- Select Site --")

    # Disable the placeholder item so it can’t be selected
    site_combo.model().item(0).setFlags(
        site_combo.model().item(0).flags() & ~QtCore.Qt.ItemIsEnabled
    )

    # Add real sites
    site_combo.addItems(GOOD_SITES)

    site_combo.setStyleSheet("QComboBox { background-color: lightcoral; }")
    layout.addWidget(site_combo)
    
    def on_select():
        selected_site = site_combo.currentText().strip()
        if selected_site and selected_site != "-- Select Site --":
            window.close()
            launch_gcwerks(selected_site)

    site_combo.currentIndexChanged.connect(on_select)
    
    window.setLayout(layout)
    window.show()
    
    sys.exit(app.exec_())

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Launch GCwerks integration software."
    )
    parser.add_argument(
        "site", nargs='?', choices=GOOD_SITES, default=None,
        help=f"Open GCwerks for a specified site. Available sites: {', '.join(GOOD_SITES)}."
    )
    args = parser.parse_args()

    if args.site:
        launch_gcwerks(args.site)
    else:
        site_selection_ui()

if __name__ == "__main__":
    main()
