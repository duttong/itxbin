#!/usr/bin/env python3

import sys
import argparse
from datetime import date
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from PyQt5 import QtCore, QtWidgets

from logos_instruments import HATS_DB_Functions, LOGOS_Instruments


def _normalize_serial(value):
    if value is None:
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    if not text:
        return ""
    match = re.match(r"^([A-Z]*)(\d+)$", text)
    if match:
        prefix, digits = match.groups()
        digits = digits.lstrip("0") or "0"
        return f"{prefix}{digits}"
    if text.isdigit():
        return text.lstrip("0") or "0"
    return text


def _escape_sql(value):
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


class TankHistDB(HATS_DB_Functions):
    def __init__(self, inst_id="fe3"):
        super().__init__(inst_id=inst_id)
        self._fill_df = None
        self._fill_norms = None
        self._tank_uses_df = None
        self._tank_uses_by_desc = None
        self._tank_uses_by_abbr = None

    def load_fill_table(self):
        if self._fill_df is not None:
            return self._fill_df
        sql = "SELECT * FROM reftank.fill;"
        df = pd.DataFrame(self.doquery(sql))
        if df.empty:
            df = pd.DataFrame(columns=["idx", "serial_number", "date", "code"])
        df["serial_number"] = df["serial_number"].astype(str)
        df["serial_norm"] = df["serial_number"].map(_normalize_serial)
        self._fill_df = df
        self._fill_norms = df["serial_norm"].tolist()
        return df

    def load_tank_uses(self):
        if self._tank_uses_df is not None:
            return self._tank_uses_df
        sql = "SELECT * FROM hats.ng_tank_uses;"
        df = pd.DataFrame(self.doquery(sql))
        if df.empty:
            df = pd.DataFrame(columns=["num", "description", "abbr"])
        self._tank_uses_df = df
        self._tank_uses_by_desc = {str(r["description"]).lower(): int(r["num"]) for _, r in df.iterrows()}
        self._tank_uses_by_abbr = {str(r["abbr"]).lower(): int(r["num"]) for _, r in df.iterrows()}
        return df

    def search_serials(self, query, limit=20, min_score=80):
        self.load_fill_table()
        qn = _normalize_serial(query)
        if not qn:
            return []
        matches = process.extract(qn, self._fill_norms, scorer=fuzz.WRatio, limit=limit)
        results = []
        for _, score, idx in matches:
            if score < min_score:
                continue
            serial = self._fill_df.iloc[idx]["serial_number"]
            if serial not in results:
                results.append(serial)
        return results

    def fills_for_serial(self, serial):
        df = self.load_fill_table()
        return df[df["serial_number"].str.upper() == str(serial).upper()].copy()

    def resolve_use_num(self, use_text):
        if use_text is None:
            return None
        self.load_tank_uses()
        key = str(use_text).strip().lower()
        if key.isdigit():
            return int(key)
        if key in self._tank_uses_by_desc:
            return self._tank_uses_by_desc[key]
        if key in self._tank_uses_by_abbr:
            return self._tank_uses_by_abbr[key]
        return None

    def get_use_history(self, fill_idx, inst_num=None):
        sql = "SELECT * FROM hats.ng_tank_use_history WHERE fill_idx = {fill_idx}"
        if inst_num is not None:
            sql += f" AND inst_num = {int(inst_num)}"
        sql += ";"
        rows = self.doquery(sql.format(fill_idx=int(fill_idx)))
        return rows

    def upsert_use_history(
        self,
        fill_idx,
        use_num,
        inst_num=None,
        site_num=None,
        start=None,
        end=None,
        comment=None,
        update_comment=False,
    ):
        if inst_num is None:
            raise ValueError("inst_num is required for tank use history updates.")
        fill_idx = int(fill_idx)
        use_num = int(use_num)
        inst_num = int(inst_num)
        current_rows = self.get_use_history(fill_idx, inst_num)
        current = current_rows[0] if current_rows else None
        inst_num = int(inst_num) if inst_num is not None else None
        site_num = int(site_num) if site_num is not None else None
        start_val = _escape_sql(start) if start else "NULL"
        end_val = _escape_sql(end) if end else "NULL"
        comment_val = _escape_sql(comment) if comment else "NULL"

        if current:
            updates = [f"ng_tank_uses_num = {use_num}"]
            if inst_num is not None:
                updates.append(f"inst_num = {inst_num}")
            if site_num is not None:
                updates.append(f"site_num = {site_num}")
            if start is not None:
                updates.append(f"start = {start_val}")
            if end is not None:
                updates.append(f"end = {end_val}")
            if comment is not None or update_comment:
                updates.append(f"comment = {comment_val}")
            sql = (
                "UPDATE hats.ng_tank_use_history "
                f"SET {', '.join(updates)} "
                f"WHERE fill_idx = {fill_idx} AND inst_num = {inst_num};"
            )
            self.doquery(sql)
            return "updated"

        sql = (
            "INSERT INTO hats.ng_tank_use_history "
            "(fill_idx, ng_tank_uses_num, inst_num, site_num, start, end, comment) "
            f"VALUES ({fill_idx}, {use_num}, "
            f"{inst_num if inst_num is not None else 'NULL'}, "
            f"{site_num if site_num is not None else 'NULL'}, "
            f"{start_val}, {end_val}, {comment_val});"
        )
        self.doquery(sql)
        return "inserted"


class TankHistEditor(QtWidgets.QMainWindow):
    def __init__(self, inst_id="fe3"):
        super().__init__()
        self.setWindowTitle("Tank History Editor")
        self.db = TankHistDB(inst_id=inst_id)
        self.fill_df = self.db.load_fill_table()
        self.tank_uses_df = self.db.load_tank_uses()
        self._checked_serial = None
        self._selected_fill_idx = None
        self._comment_clear_requested = False
        self._comment_dirty = False

        self._build_ui()
        self._populate_instruments(inst_id)
        self._populate_tank_uses()

    def _build_ui(self):
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)

        inst_row = QtWidgets.QHBoxLayout()
        inst_row.addWidget(QtWidgets.QLabel("Instrument"))
        self.inst_combo = QtWidgets.QComboBox()
        self.inst_label = QtWidgets.QLabel("")
        inst_row.addWidget(self.inst_combo)
        inst_row.addWidget(self.inst_label)
        inst_row.addStretch(1)
        layout.addLayout(inst_row)

        search_row = QtWidgets.QHBoxLayout()
        self.serial_input = QtWidgets.QLineEdit()
        self.serial_input.setPlaceholderText("Enter tank serial (best guess)")
        self.serial_input.returnPressed.connect(self.search_serials)
        self.search_btn = QtWidgets.QPushButton("Search")
        self.search_btn.clicked.connect(self.search_serials)
        search_row.addWidget(self.serial_input)
        search_row.addWidget(self.search_btn)
        layout.addLayout(search_row)

        layout.addWidget(QtWidgets.QLabel("Matching serials"))
        self.serial_list = QtWidgets.QListWidget()
        self.serial_list.itemChanged.connect(self._serial_checked)
        layout.addWidget(self.serial_list)

        layout.addWidget(QtWidgets.QLabel("Fill records"))
        self.fill_table = QtWidgets.QTableWidget(0, 7)
        self.fill_table.setHorizontalHeaderLabels(
            ["Fill idx", "Date", "Code", "Location", "Method", "Type", "Notes"]
        )
        self.fill_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.fill_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.fill_table.itemSelectionChanged.connect(self._fill_selected)
        self.fill_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.fill_table)

        edit_group = QtWidgets.QGroupBox("Editing Tank History")
        edit_group.setStyleSheet(
            "QGroupBox { border: 2px solid #8b0000; margin-top: 12px; "
            "background-color: #fff0f0; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
            "padding: 0 3px 0 3px; color: #8b0000; font-weight: bold; }"
        )
        edit_layout = QtWidgets.QVBoxLayout(edit_group)

        edit_layout.addWidget(QtWidgets.QLabel("Tank use history"))
        self.history_table = QtWidgets.QTableWidget(0, 8)
        self.history_table.setHorizontalHeaderLabels(
            ["num", "fill_idx", "ng_tank_uses_num", "inst_num", "site_num", "start", "end", "comment"]
        )
        self.history_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.history_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        edit_layout.addWidget(self.history_table)

        use_row = QtWidgets.QHBoxLayout()
        use_row.addWidget(QtWidgets.QLabel("Tank use"))
        self.use_combo = QtWidgets.QComboBox()
        use_row.addWidget(self.use_combo)
        edit_layout.addLayout(use_row)

        dates_row = QtWidgets.QHBoxLayout()
        self.start_check = QtWidgets.QCheckBox("Start")
        self.start_date = QtWidgets.QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QtCore.QDate.currentDate())
        self.start_date.setEnabled(False)
        self.start_check.toggled.connect(self.start_date.setEnabled)

        self.end_check = QtWidgets.QCheckBox("End")
        self.end_date = QtWidgets.QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QtCore.QDate.currentDate())
        self.end_date.setEnabled(False)
        self.end_check.toggled.connect(self.end_date.setEnabled)

        dates_row.addWidget(self.start_check)
        dates_row.addWidget(self.start_date)
        dates_row.addWidget(self.end_check)
        dates_row.addWidget(self.end_date)
        dates_row.addStretch(1)
        edit_layout.addLayout(dates_row)

        comment_row = QtWidgets.QHBoxLayout()
        comment_row.addWidget(QtWidgets.QLabel("Comment"))
        self.comment_input = QtWidgets.QLineEdit()
        self.comment_input.textChanged.connect(self._comment_text_changed)
        comment_row.addWidget(self.comment_input)
        self.clear_comment_btn = QtWidgets.QPushButton("Clear comment")
        self.clear_comment_btn.clicked.connect(self._clear_comment)
        comment_row.addWidget(self.clear_comment_btn)
        edit_layout.addLayout(comment_row)

        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self.save_use)
        self.save_btn.setStyleSheet(
            "QPushButton { background-color: #c8f7c5; border: 1px solid #6aa96a; }"
        )
        edit_layout.addWidget(self.save_btn)

        layout.addWidget(edit_group)

        self.setCentralWidget(root)

    def _populate_instruments(self, inst_id):
        self.inst_combo.blockSignals(True)
        self.inst_combo.clear()
        for name in LOGOS_Instruments.INSTRUMENTS:
            self.inst_combo.addItem(name, LOGOS_Instruments.INSTRUMENTS[name])
        idx = self.inst_combo.findText(inst_id)
        if idx >= 0:
            self.inst_combo.setCurrentIndex(idx)
        self.inst_combo.currentIndexChanged.connect(self._inst_changed)
        self.inst_combo.blockSignals(False)
        self._inst_changed()

    def _populate_tank_uses(self):
        self.use_combo.clear()
        for _, row in self.tank_uses_df.iterrows():
            label = f"{int(row['num'])}: {row['description']} ({row['abbr']})"
            self.use_combo.addItem(label, int(row["num"]))

    def _inst_changed(self):
        inst_name = self.inst_combo.currentText()
        inst_num = self.inst_combo.currentData()
        self.inst_label.setText(f"inst_num: {inst_num}")
        if inst_name != self.db.inst_id:
            self.db = TankHistDB(inst_id=inst_name)
            self.fill_df = self.db.load_fill_table()
            self.tank_uses_df = self.db.load_tank_uses()
            self._populate_tank_uses()
            self.serial_list.clear()
            self.fill_table.setRowCount(0)
            self.history_table.setRowCount(0)
            self._checked_serial = None
            self._selected_fill_idx = None
            self._comment_clear_requested = False
            self._comment_dirty = False
            self.comment_input.blockSignals(True)
            self.comment_input.clear()
            self.comment_input.blockSignals(False)
            self.save_btn.setText("Save")

    def search_serials(self):
        self.serial_list.blockSignals(True)
        self.serial_list.clear()
        query = self.serial_input.text().strip()
        results = self.db.search_serials(query)
        for serial in results:
            item = QtWidgets.QListWidgetItem(serial)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.serial_list.addItem(item)
        self.serial_list.blockSignals(False)
        self.fill_table.setRowCount(0)
        self.history_table.setRowCount(0)
        self._selected_fill_idx = None
        self._checked_serial = None
        self._comment_clear_requested = False
        self._comment_dirty = False
        self.comment_input.blockSignals(True)
        self.comment_input.clear()
        self.comment_input.blockSignals(False)
        self.save_btn.setText("Save")
        if results:
            first = self.serial_list.item(0)
            if first:
                first.setCheckState(QtCore.Qt.Checked)

    def _serial_checked(self, item):
        if item.checkState() != QtCore.Qt.Checked:
            if self._checked_serial == item.text():
                self._checked_serial = None
                self.fill_table.setRowCount(0)
            return

        for i in range(self.serial_list.count()):
            other = self.serial_list.item(i)
            if other is not item:
                other.setCheckState(QtCore.Qt.Unchecked)
        self._checked_serial = item.text()
        self.fill_table.setRowCount(0)
        self.history_table.setRowCount(0)
        self._selected_fill_idx = None
        self._comment_clear_requested = False
        self._comment_dirty = False
        self.comment_input.blockSignals(True)
        self.comment_input.clear()
        self.comment_input.blockSignals(False)
        self.save_btn.setText("Save")
        self._load_fills(self._checked_serial)

    def _load_fills(self, serial):
        df = self.db.fills_for_serial(serial)
        df = df.sort_values("date", ascending=False)
        self.fill_table.setRowCount(0)
        for _, row in df.iterrows():
            r = self.fill_table.rowCount()
            self.fill_table.insertRow(r)
            self.fill_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(row.get("idx", ""))))
            self.fill_table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(row.get("date", ""))))
            self.fill_table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(row.get("code", ""))))
            self.fill_table.setItem(r, 3, QtWidgets.QTableWidgetItem(str(row.get("location", ""))))
            self.fill_table.setItem(r, 4, QtWidgets.QTableWidgetItem(str(row.get("method", ""))))
            self.fill_table.setItem(r, 5, QtWidgets.QTableWidgetItem(str(row.get("type", ""))))
            self.fill_table.setItem(r, 6, QtWidgets.QTableWidgetItem(str(row.get("notes", ""))))

    def _fill_selected(self):
        items = self.fill_table.selectedItems()
        if not items:
            self._selected_fill_idx = None
            self.save_btn.setText("Save")
            return
        row = items[0].row()
        fill_idx_item = self.fill_table.item(row, 0)
        if not fill_idx_item:
            self._selected_fill_idx = None
            self.save_btn.setText("Save")
            return
        self._selected_fill_idx = int(fill_idx_item.text())
        history_rows = self.db.get_use_history(self._selected_fill_idx)
        self.history_table.setRowCount(0)
        columns = [
            "num",
            "fill_idx",
            "ng_tank_uses_num",
            "inst_num",
            "site_num",
            "start",
            "end",
            "comment",
        ]
        if history_rows:
            current_inst = str(self.db.inst_num)
            current_row_idx = None
            for row_idx, history in enumerate(history_rows):
                self.history_table.insertRow(row_idx)
                for col_idx, key in enumerate(columns):
                    value = history.get(key, "")
                    item = QtWidgets.QTableWidgetItem(str(value))
                    if key == "inst_num" and str(value) != current_inst:
                        item.setBackground(QtCore.Qt.yellow)
                    self.history_table.setItem(row_idx, col_idx, item)
                if str(history.get("inst_num")) == current_inst:
                    current_row_idx = row_idx

            selected = self.db.get_use_history(self._selected_fill_idx, self.db.inst_num)
            selected_row = selected[0] if selected else None
            self.comment_input.blockSignals(True)
            self.comment_input.setText(str(selected_row.get("comment") if selected_row else "" or ""))
            self.comment_input.blockSignals(False)
            if selected_row:
                use_num = selected_row.get("ng_tank_uses_num")
                if use_num is not None:
                    use_idx = self.use_combo.findData(int(use_num))
                    if use_idx >= 0:
                        self.use_combo.setCurrentIndex(use_idx)
                start_value = selected_row.get("start")
                if start_value:
                    qdate = QtCore.QDate.fromString(str(start_value), "yyyy-MM-dd")
                    if qdate.isValid():
                        self.start_date.setDate(qdate)
                        self.start_check.setChecked(True)
            self._comment_clear_requested = False
            self._comment_dirty = False
            if current_row_idx is not None:
                self.history_table.selectRow(current_row_idx)
                self.save_btn.setText("Save updates")
            else:
                self.save_btn.setText("Insert new tank history row")
        else:
            self.comment_input.blockSignals(True)
            self.comment_input.clear()
            self.comment_input.blockSignals(False)
            self._comment_clear_requested = False
            self._comment_dirty = False
            self.use_combo.setCurrentIndex(0)
            self.save_btn.setText("Insert new tank history row")

        fill_date_item = self.fill_table.item(row, 1)
        if fill_date_item:
            qdate = QtCore.QDate.fromString(fill_date_item.text(), "yyyy-MM-dd")
            if qdate.isValid() and (not history_rows or not (selected_row and selected_row.get("start"))):
                self.start_date.setDate(qdate)
                self.start_check.setChecked(False)

    def _selected_use_num(self):
        data = self.use_combo.currentData()
        return int(data) if data is not None else None

    def save_use(self):
        if self._selected_fill_idx is None:
            QtWidgets.QMessageBox.warning(self, "Missing fill", "Select a fill record.")
            return
        use_num = self._selected_use_num()
        if use_num is None:
            QtWidgets.QMessageBox.warning(self, "Missing use", "Select a tank use.")
            return
        start = None
        end = None
        if self.start_check.isChecked():
            start = self.start_date.date().toString("yyyy-MM-dd")
        if self.end_check.isChecked():
            end = self.end_date.date().toString("yyyy-MM-dd")
        raw_comment = self.comment_input.text().strip()
        comment = raw_comment or None
        update_comment = self._comment_dirty or self._comment_clear_requested

        action = self.db.upsert_use_history(
            fill_idx=self._selected_fill_idx,
            use_num=use_num,
            inst_num=self.db.inst_num,
            start=start,
            end=end,
            comment=comment,
            update_comment=update_comment,
        )
        self._fill_selected()

    def _clear_comment(self):
        self._comment_clear_requested = True
        self._comment_dirty = True
        self.comment_input.clear()

    def _comment_text_changed(self):
        self._comment_dirty = True
        if self._comment_clear_requested and self.comment_input.text().strip():
            self._comment_clear_requested = False


def _cli_search(db, args):
    results = db.search_serials(args.serial)
    if not results:
        print("No matches.")
        return
    for serial in results:
        print(serial)


def _cli_list_fills(db, args):
    df = db.fills_for_serial(args.serial)
    if df.empty:
        print("No fills.")
        return
    df = df.sort_values("date", ascending=False)
    cols = ["idx", "date", "code", "location", "method", "type"]
    print(df[cols].to_string(index=False))


def _cli_set_use(db, args):
    use_num = db.resolve_use_num(args.use)
    if use_num is None:
        raise SystemExit(f"Unknown tank use: {args.use}")

    fill_idx = args.fill_idx
    if fill_idx is None:
        df = db.fills_for_serial(args.serial)
        df = df[df["code"].astype(str).str.upper() == str(args.fill_code).upper()]
        if df.empty:
            raise SystemExit("No fill matching serial + code.")
        if len(df) > 1:
            raise SystemExit("Multiple fills matched. Use --fill-idx.")
        fill_idx = int(df.iloc[0]["idx"])

    action = db.upsert_use_history(
        fill_idx=fill_idx,
        use_num=use_num,
        inst_num=db.inst_num,
        site_num=args.site_num,
        start=args.start,
        end=args.end,
        comment=args.comment,
    )
    print(f"{action} fill_idx {fill_idx}")


def build_parser():
    parser = argparse.ArgumentParser(description="Edit hats.ng_tank_use_history")
    sub = parser.add_subparsers(dest="command")

    s = sub.add_parser("search", help="Search for serial numbers")
    s.add_argument("serial", help="Serial number (best guess)")
    s.set_defaults(func=_cli_search)

    l = sub.add_parser("list-fills", help="List fills for a serial")
    l.add_argument("serial", help="Serial number")
    l.set_defaults(func=_cli_list_fills)

    u = sub.add_parser("set-use", help="Insert/update tank use history")
    u.add_argument("--fill-idx", type=int, help="Fill index")
    u.add_argument("--serial", help="Tank serial number")
    u.add_argument("--fill-code", help="Fill code (e.g. A123)")
    u.add_argument("--use", required=True, help="Use description, abbr, or num")
    u.add_argument("--site-num", type=int, help="Site number")
    u.add_argument("--start", help="Start date YYYY-MM-DD")
    u.add_argument("--end", help="End date YYYY-MM-DD")
    u.add_argument("--comment", help="Comment")
    u.set_defaults(func=_cli_set_use)

    return parser


def main():
    parser = build_parser()
    argv = sys.argv[1:]
    subcommands = {"search", "list-fills", "set-use"}
    inst = None
    if argv and argv[0] not in subcommands and not argv[0].startswith("-"):
        inst = argv[0]
        argv = argv[1:]
    args = parser.parse_args(argv)
    args.inst = inst
    if not args.command:
        app = QtWidgets.QApplication(sys.argv)
        inst_id = args.inst
        if not inst_id:
            inst_choices = list(LOGOS_Instruments.INSTRUMENTS.keys())
            inst_id, ok = QtWidgets.QInputDialog.getItem(
                None,
                "Select instrument",
                "Instrument:",
                inst_choices,
                0,
                False,
            )
            if not ok or not inst_id:
                raise SystemExit("Instrument selection required.")
        win = TankHistEditor(inst_id=inst_id)
        win.resize(900, 700)
        win.show()
        sys.exit(app.exec_())

    if not args.inst:
        raise SystemExit("Instrument is required for CLI commands.")
    db = TankHistDB(inst_id=args.inst)
    args.func(db, args)


if __name__ == "__main__":
    main()
