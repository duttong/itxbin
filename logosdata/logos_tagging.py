"""Shared tag-table plumbing: reject/info tag layout, DB CRUD, and the
floating Multi-Tag selection panel.

Split out of logos_data.py so logos_timeseries.py can use it too --
logos_data.py does `from logos_timeseries import TimeseriesWidget` at module
level, so logos_timeseries.py cannot import back from logos_data.py.

MultiTagPanel talks to whatever "host" it's constructed with purely through
a small interface: the seven TagCRUDMixin methods, plus
`_record_pending_tag`, `on_tag_state_changed`, and (optionally)
`update_all_analytes` (its presence toggles the "Copy Tags to all Analytes"
section on/off). It never assumes a single-run `self.run` DataFrame -- any
row-identity semantics belong entirely to the host.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QApplication,
)
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import Qt, QTimer


_TAG_LAYOUT = [
    ("Sampling/Collection Issues", [
        ("B", "Leaky flask valve",                                   168, 223),
        ("F", "Insufficient flushing of sample collection system",     6, 222),
        ("K", "Contamination (Room air)",                              8,   9),
        ("L", "Leak in sample collection system",                     10,  11),
        ("P", "Bad flask pair agreement",                             14,  15),
        ("T", "Test sample collected not used in data analysis",      16, 189),
        ("U", "Unknown sample collection problem",                    17,  18),
        ("V", "Insufficient sample pressure",                         19, 136),
    ]),
    ("Measurement Issues", [
        ("A", "Known measurement problem",                           282, 326),
        ("A", "Valco Valve Problem",                                 143, 164),
        ("C", "Mole fraction falls outside of calibration",          107, 327),
        ("G", "Chromatography issue",                                290, 291),
        ("M", "Agilent (MS or GC) device issue",                     132, 133),
        ("O", "Measurement lab operator error",                       43, 121),
        ("U", "Unknown measurement problem",                         141, 142),
    ]),
    ("Automated Tags", [
        ("C", "Mole fraction falls outside of calibration",          286, 287),
        ("G", "Chromatography issue",                                316,   0),
        ("P", "Bad flask pair agreement",                            167,   0),
        ("V", "Out of range sample pressure",                         32,   0),
        ("W", "Rejected in GCwerks integration",                     324,   0),
        ("S", "Detector cal-response rapid change",                  328, 401),
        ("X", "Abnormal chromatogram",                                329,   0),
    ]),
]

_INFO_TAG_NUMS = frozenset(
    i_tag
    for _, entries in _TAG_LAYOUT
    for _, _, _r, i_tag in entries
    if i_tag
)

_INFO_TAG_DESCRIPTIONS = {
    i_tag: desc
    for _, entries in _TAG_LAYOUT
    for _, desc, _r, i_tag in entries
    if i_tag
}

# Auto tags the user may manually remove (but not add).
# 316: first-reference-run flag set by m4_gcwerks2db; qc_status is moved to
# 'F' at the same time, so removing the tag won't cause it to be reapplied.
_USER_REMOVABLE_AUTO_TAGS = frozenset({316})


class TagCRUDMixin:
    """DB read/write for reject+info tags, keyed purely on mole-fraction
    primary keys (`ng_mole_fraction_num` / `ng_insitu_mole_fraction_num`) and
    `self.instrument` (`.inst_id` for table selection, `.db` for queries).
    No dependency on any particular "current run" dataframe."""

    def _tag_table_info(self):
        if self.instrument.inst_id in ('ie3', 'cats'):
            return (
                'hats.ng_insitu_mole_fraction_tags',
                'ng_insitu_mole_fraction_num',
                'hats.ng_insitu_mole_fractions',
                'mf_num',
            )
        return (
            'hats.ng_mole_fraction_tags',
            'ng_mole_fraction_num',
            'hats.ng_mole_fractions',
            'ng_mole_fraction_num',
        )

    def _insert_tag_for_mf_nums(self, mf_nums: list[int], tag_num: int):
        """Insert a tag directly given mole-fraction primary keys."""
        if not mf_nums:
            return
        tag_table, tag_key, _, _ = self._tag_table_info()
        sql = f"INSERT IGNORE INTO {tag_table} ({tag_key}, tag_num) VALUES (%s, %s);"
        self.instrument.db.doMultiInsert(sql, [(n, tag_num) for n in mf_nums], all=True)

    def _delete_tag_for_mf_nums(self, mf_nums: list[int], tag_num: int):
        """Remove a specific tag from the given mole-fraction primary keys."""
        if not mf_nums:
            return
        tag_table, tag_key, _, _ = self._tag_table_info()
        placeholders = ",".join(["%s"] * len(mf_nums))
        self.instrument.db.doquery(
            f"DELETE FROM {tag_table} WHERE {tag_key} IN ({placeholders}) AND tag_num = %s;",
            list(mf_nums) + [tag_num],
        )

    def _fetch_tag_nums_for_mf_nums(self, mf_nums: list[int]) -> set[int]:
        """Return the set of tag_nums currently applied to the given mf primary keys."""
        if not mf_nums:
            return set()
        tag_table, tag_key, _, _ = self._tag_table_info()
        placeholders = ",".join(["%s"] * len(mf_nums))
        rows = self.instrument.db.doquery(
            f"SELECT DISTINCT tag_num FROM {tag_table} WHERE {tag_key} IN ({placeholders});",
            list(mf_nums),
        )
        return {int(r["tag_num"]) for r in (rows or [])}

    def _fetch_tag_counts_for_mf_nums(self, mf_nums: list[int]) -> dict[int, int]:
        """Return {tag_num: count} for how many of the given mf_nums carry each tag."""
        if not mf_nums:
            return {}
        tag_table, tag_key, _, _ = self._tag_table_info()
        placeholders = ",".join(["%s"] * len(mf_nums))
        rows = self.instrument.db.doquery(
            f"SELECT tag_num, COUNT(*) AS cnt FROM {tag_table} "
            f"WHERE {tag_key} IN ({placeholders}) GROUP BY tag_num;",
            list(mf_nums),
        )
        return {int(r["tag_num"]): int(r["cnt"]) for r in (rows or [])}

    def _fetch_first_comment_for_mf_nums(self, mf_nums: list[int]) -> str:
        if not mf_nums:
            return ""
        tag_table, tag_key, _, _ = self._tag_table_info()
        placeholders = ",".join(["%s"] * len(mf_nums))
        rows = self.instrument.db.doquery(
            f"SELECT comment FROM {tag_table} WHERE {tag_key} IN ({placeholders}) "
            f"AND comment IS NOT NULL AND comment != '' LIMIT 1;",
            list(mf_nums),
        )
        return (rows[0].get("comment") or "") if rows else ""

    def _save_comment_for_mf_nums(self, mf_nums: list[int], comment: str | None):
        if not mf_nums:
            return
        tag_table, tag_key, _, _ = self._tag_table_info()
        placeholders = ",".join(["%s"] * len(mf_nums))
        self.instrument.db.doquery(
            f"UPDATE {tag_table} SET comment = %s WHERE {tag_key} IN ({placeholders});",
            [comment] + list(mf_nums),
        )


class MultiTagPanel(QWidget):
    """Floating panel: R/I tag checkboxes grouped by Sampling, Measurement, and Auto categories.

    `host` must implement the seven TagCRUDMixin methods plus
    `_record_pending_tag(row_idxs, tag_num, applied)` and
    `on_tag_state_changed(row_idxs, mf_nums, tag_num, applied, is_reject)`.
    If `host` also has `update_all_analytes()`, the "Copy Tags to all
    Analytes" section is shown; otherwise it's omitted.
    """

    def __init__(self, host):
        super().__init__(None, Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.host = host
        self._supports_copy_all = hasattr(host, 'update_all_analytes')
        self.setWindowTitle("Multi-Tag")
        self._mf_nums: list[int] = []
        self._row_idxs: list = []
        self._total_points: int = 0
        self._updating = False
        self._rows: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        copy_all_line = (
            "<li><b>Copy Tags to all Analytes</b> copies the tags and recalculates all mole fractions. "
            "Saving in the main window keeps tags on this analyte only.</li>"
            if self._supports_copy_all else ""
        )
        instructions = QLabel(
            "<b>How to tag:</b>"
            "<ul style='-qt-list-indent:0; margin-left:8px; margin-top:2px; margin-bottom:0px;'>"
            "<li>Click a point, or drag a box on the plot, to select.</li>"
            "<li><b>SHIFT+click</b> adds/removes single points; <b>SHIFT+drag</b> adds a region.</li>"
            "<li>Toolbar Pan/Zoom take the mouse while engaged; selection resumes when toggled off.</li>"
            "<li>Check <b>R</b> to reject, <b>I</b> for an info tag &mdash; applied immediately to this analyte.</li>"
            f"{copy_all_line}"
            "<li>Add a note with <b>Save/Update Comment</b> (needs at least one tag).</li>"
            "</ul>"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(
            "background-color: #eef3fb; border: 1px solid #c8d4e8; "
            "border-radius: 6px; padding: 4px 6px; font-size: 11px; color: #1a2a4a;"
        )
        layout.addWidget(instructions)

        self._info_label = QLabel("No point selected")
        self._info_label.setStyleSheet("color: gray; font-style: italic; font-size: 11px;")
        layout.addWidget(self._info_label)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["R", "I", "N", "Description"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.setColumnWidth(0, 30)
        self._table.setColumnWidth(1, 30)
        self._table.setColumnWidth(2, 22)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self._table)

        # Comment section
        _comment_yellow = "#fffde7"
        comment_header = QWidget()
        comment_header.setStyleSheet(f"background-color: {_comment_yellow};")
        ch_layout = QHBoxLayout(comment_header)
        ch_layout.setContentsMargins(6, 3, 4, 3)
        ch_label = QLabel("Tag Comment")
        ch_label.setStyleSheet(f"background-color: {_comment_yellow}; font-weight: bold; font-size: 11px;")
        self._save_comment_btn = QPushButton("Save/Update Comment")
        self._save_comment_btn.setEnabled(False)
        self._save_comment_btn.clicked.connect(self._save_comment)
        ch_layout.addWidget(ch_label)
        ch_layout.addStretch()
        ch_layout.addWidget(self._save_comment_btn)
        layout.addWidget(comment_header)

        self._comment_edit = QTextEdit()
        self._comment_edit.setPlaceholderText("Enter comment…")
        self._comment_edit.setStyleSheet(f"background-color: {_comment_yellow};")
        self._comment_edit.setEnabled(False)
        self._comment_edit.setFixedHeight(62)
        layout.addWidget(self._comment_edit)

        self._copy_tags_btn = None
        if self._supports_copy_all:
            copy_row = QWidget()
            copy_layout = QHBoxLayout(copy_row)
            copy_layout.setContentsMargins(4, 4, 4, 2)
            copy_layout.addStretch()
            self._copy_tags_btn = QPushButton("Copy Tags to all Analytes")
            self._copy_tags_btn.setToolTip(
                "Propagate pending tag changes (reject and info, adds and removals)\n"
                "to all analytes for the same injection, then recalculate and save\n"
                "mole fractions for all analytes.\n"
                "Note: a Save finalizes tags for the current analyte only and\n"
                "removes them from this queue."
            )
            self._copy_tags_btn_default_style = (
                "QPushButton { padding: 4px 8px; border: 1px solid #aaa; border-radius: 4px; }"
            )
            self._copy_tags_btn_busy_style = (
                "QPushButton { padding: 4px 8px; border: 1px solid #c9b458; "
                "border-radius: 4px; background-color: #fff59d; }"
            )
            self._copy_tags_btn.setStyleSheet(self._copy_tags_btn_default_style)
            self._copy_tags_btn.clicked.connect(self._copy_tags_to_all)
            copy_layout.addWidget(self._copy_tags_btn)
            layout.addWidget(copy_row)

        self.setMinimumWidth(420)
        self._build_layout()

    def _build_layout(self):
        self._rows = []
        row_count = sum(1 + len(entries) for _, entries in _TAG_LAYOUT)
        self._table.setRowCount(row_count)

        section_bg = QColor("#c8d4e8")

        table_row = 0
        for section_name, entries in _TAG_LAYOUT:
            is_auto_section = (section_name == "Automated Tags")

            hdr = QTableWidgetItem(f"  {section_name}")
            hdr.setBackground(QBrush(section_bg))
            hdr.setForeground(QBrush(QColor("#1a2a4a")))
            f = hdr.font()
            f.setBold(True)
            hdr.setFont(f)
            hdr.setFlags(Qt.ItemIsEnabled)
            self._table.setItem(table_row, 0, hdr)
            self._table.setSpan(table_row, 0, 1, 4)
            self._table.setRowHeight(table_row, 22)
            table_row += 1

            for letter, desc, r_tag, i_tag in entries:
                r_cb = QCheckBox()
                r_cb.setTristate(True)
                r_cb.setEnabled(False)
                r_wrap = QWidget()
                r_box = QHBoxLayout(r_wrap)
                r_box.addWidget(r_cb)
                r_box.setAlignment(Qt.AlignCenter)
                r_box.setContentsMargins(0, 0, 0, 0)
                self._table.setCellWidget(table_row, 0, r_wrap)
                is_removable = is_auto_section and r_tag in _USER_REMOVABLE_AUTO_TAGS
                if not is_auto_section or is_removable:
                    r_cb.clicked.connect(
                        lambda _c, tnum=r_tag, cb=r_cb: self._on_clicked(cb, tnum, is_reject=True)
                    )

                i_cb = None
                if i_tag:
                    i_cb = QCheckBox()
                    i_cb.setTristate(True)
                    i_cb.setEnabled(False)
                    i_wrap = QWidget()
                    i_box = QHBoxLayout(i_wrap)
                    i_box.addWidget(i_cb)
                    i_box.setAlignment(Qt.AlignCenter)
                    i_box.setContentsMargins(0, 0, 0, 0)
                    self._table.setCellWidget(table_row, 1, i_wrap)
                    if not is_auto_section:
                        i_cb.clicked.connect(
                            lambda _c, tnum=i_tag, cb=i_cb: self._on_clicked(cb, tnum, is_reject=False)
                        )

                n_item = QTableWidgetItem(letter)
                n_item.setTextAlignment(Qt.AlignCenter)
                n_item.setFlags(Qt.ItemIsEnabled)
                fn = n_item.font()
                fn.setBold(True)
                n_item.setFont(fn)
                self._table.setItem(table_row, 2, n_item)

                desc_item = QTableWidgetItem(desc)
                desc_item.setFlags(Qt.ItemIsEnabled)
                self._table.setItem(table_row, 3, desc_item)

                self._rows.append({
                    "table_row": table_row,
                    "r_tag": r_tag,
                    "i_tag": i_tag,
                    "is_auto": is_auto_section,
                    "is_removable": is_removable,
                    "r_cb": r_cb,
                    "i_cb": i_cb,
                })
                table_row += 1

        self._table.resizeRowsToContents()
        self._fit_table_height()

    def _fit_table_height(self):
        header_h = self._table.horizontalHeader().height()
        rows_h = sum(self._table.rowHeight(r) for r in range(self._table.rowCount()))
        self._table.setFixedHeight(header_h + rows_h + 4)
        self.adjustSize()

    def _save_comment(self):
        if not self._mf_nums:
            return
        try:
            self.host._save_comment_for_mf_nums(self._mf_nums, self._comment_edit.toPlainText().strip() or None)
        except Exception as exc:
            print(f"MultiTagPanel save comment error: {exc}")

    def _copy_tags_to_all(self):
        # Flash the button yellow immediately, then defer the (slow, blocking)
        # recalc work to the next event-loop tick so the repaint lands first —
        # same pattern as the plot-legend "Save flags to all gases" button.
        self._copy_tags_btn.setText("Recalculating…")
        self._copy_tags_btn.setEnabled(False)
        self._copy_tags_btn.setStyleSheet(self._copy_tags_btn_busy_style)
        QApplication.processEvents()
        QTimer.singleShot(0, self._run_copy_tags_to_all)

    def _run_copy_tags_to_all(self):
        try:
            self.host.update_all_analytes()
        except Exception as exc:
            print(f"Copy tags error: {exc}")
        finally:
            self._copy_tags_btn.setText("Copy Tags to all Analytes")
            self._copy_tags_btn.setEnabled(True)
            self._copy_tags_btn.setStyleSheet(self._copy_tags_btn_default_style)

    def populate_tags(self, all_tags_ordered: list):
        pass  # layout is static; kept for API compatibility

    def update_for_point(self, row_idxs: list, mf_nums: list[int],
                         tag_counts: dict[int, int], total: int = 1,
                         info_text: str | None = None):
        """`row_idxs` is an opaque, host-defined selection token list (e.g.
        DataFrame index labels for logos_data.py, or (artist, point index)
        pairs for the Timeseries figure) -- this panel never interprets its
        contents, only its length, and passes it back to the host as-is.
        `info_text`, if given, is shown verbatim instead of the generic
        "N points selected" message (used by hosts that can describe the
        single selected point, e.g. its timestamp)."""
        self._row_idxs = row_idxs
        self._mf_nums = mf_nums
        self._total_points = total
        self._updating = True
        try:
            if info_text is not None:
                self._info_label.setText(info_text)
            elif len(row_idxs) == 1:
                self._info_label.setText(f"row {row_idxs[0]}")
            else:
                self._info_label.setText(f"{len(row_idxs)} points selected")

            for entry in self._rows:
                r_tag = entry["r_tag"]
                i_tag = entry["i_tag"]
                is_auto = entry["is_auto"]
                r_cb: QCheckBox = entry["r_cb"]
                i_cb: QCheckBox | None = entry["i_cb"]

                r_count = tag_counts.get(r_tag, 0)
                is_removable = entry.get("is_removable", False)
                if is_removable:
                    # Enabled only when all selected points carry the tag — remove only, no re-add.
                    r_cb.setEnabled(bool(mf_nums) and r_count >= total)
                else:
                    r_cb.setEnabled(bool(mf_nums) and not is_auto)
                if r_count == 0:
                    r_cb.setCheckState(Qt.Unchecked)
                elif r_count >= total:
                    r_cb.setCheckState(Qt.Checked)
                else:
                    r_cb.setCheckState(Qt.PartiallyChecked)

                if i_cb is not None:
                    i_count = tag_counts.get(i_tag, 0)
                    i_cb.setEnabled(bool(mf_nums) and not is_auto)
                    if i_count == 0:
                        i_cb.setCheckState(Qt.Unchecked)
                    elif i_count >= total:
                        i_cb.setCheckState(Qt.Checked)
                    else:
                        i_cb.setCheckState(Qt.PartiallyChecked)

            self._comment_edit.setEnabled(bool(mf_nums))
            self._save_comment_btn.setEnabled(bool(mf_nums) and bool(tag_counts))
            comment = self.host._fetch_first_comment_for_mf_nums(mf_nums) if mf_nums else ""
            self._comment_edit.setPlainText(comment)

        finally:
            self._updating = False

    def _on_clicked(self, cb: QCheckBox, tag_num: int, is_reject: bool):
        try:
            self._on_clicked_inner(cb, tag_num, is_reject)
        except Exception as exc:
            print(f"MultiTagPanel click error: {exc}")

    def _on_clicked_inner(self, cb: QCheckBox, tag_num: int, is_reject: bool):
        if self._updating or not self._mf_nums:
            return
        new_state = cb.checkState()

        if new_state == Qt.PartiallyChecked:
            self._updating = True
            cb.setCheckState(Qt.Checked)
            self._updating = False
            apply = True
        elif new_state == Qt.Checked:
            apply = True
        else:
            apply = False

        if apply:
            self.host._insert_tag_for_mf_nums(self._mf_nums, tag_num)
        else:
            self.host._delete_tag_for_mf_nums(self._mf_nums, tag_num)
        self.host._record_pending_tag(self._row_idxs, tag_num, applied=apply)

        # Tag rows just changed — refresh the comment-button state, which was
        # computed at selection time and goes stale after an add/remove.
        tag_counts = self.host._fetch_tag_counts_for_mf_nums(self._mf_nums)
        self._save_comment_btn.setEnabled(bool(self._mf_nums) and bool(tag_counts))

        self.host.on_tag_state_changed(self._row_idxs, self._mf_nums, tag_num, apply, is_reject)

    def clear_selection(self):
        self._mf_nums = []
        self._row_idxs = []
        self._total_points = 0
        self._info_label.setText("No point selected")
        self._comment_edit.setPlainText("")
        self._comment_edit.setEnabled(False)
        self._save_comment_btn.setEnabled(False)
        self._updating = True
        try:
            for entry in self._rows:
                entry["r_cb"].setEnabled(False)
                entry["r_cb"].setCheckState(Qt.Unchecked)
                if entry["i_cb"] is not None:
                    entry["i_cb"].setEnabled(False)
                    entry["i_cb"].setCheckState(Qt.Unchecked)
        finally:
            self._updating = False
