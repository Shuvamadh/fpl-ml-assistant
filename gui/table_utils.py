"""Helpers for putting a pandas DataFrame into a QTableWidget with correct
numeric sorting, light conditional coloring, and readable styling (row
height, font size, padding, alternating rows, sticky styled header,
per-column widths, right-aligned numerics)."""
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QHeaderView, QTableWidget, QTableWidgetItem,
)

import theme


class NumericItem(QTableWidgetItem):
    def __init__(self, value, text):
        super().__init__(text)
        self._value = value

    def __lt__(self, other):
        if isinstance(other, NumericItem):
            try:
                return float(self._value) < float(other._value)
            except (TypeError, ValueError):
                return str(self._value) < str(other._value)
        return super().__lt__(other)


NUMERIC_KEYWORDS = ("pred", "cost", "price", "fdr", "form", "points", "pct",
                    "ratio", "signal", "profit", "percent", "minutes", "gp",
                    "rank", "total", "swing", "value", "count", "n")

# Column-label substrings (case-insensitive) that get a wide, left-aligned
# "identity" column instead of the default width -- names/teams shouldn't elide.
WIDE_LABEL_KEYWORDS = ("player", "team", "manager", "out", "in", "chip", "signal")

ROW_HEIGHT = 34
HEADER_HEIGHT = 38


def style_table(table: QTableWidget):
    """Apply the shared readability styling to a QTableWidget. Safe to call
    repeatedly (e.g. after every fill_table)."""
    # GUI_PLAN 6.5. Full-grid tables are the most dated element in the app, and
    # Qt cannot draw horizontal-only gridlines from a stylesheet -- so the grid
    # is switched off here and the row separator comes from the ::item
    # border-bottom rule in theme.stylesheet().
    table.setShowGrid(False)
    table.setAlternatingRowColors(False)  # 34px rows + a hairline make zebra noise
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
    table.horizontalHeader().setFixedHeight(HEADER_HEIGHT)
    # Qt bolds the selected column's header by default: dated, and it reflows
    # the column width as selection moves.
    table.horizontalHeader().setHighlightSections(False)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setWordWrap(False)
    table.setFrameShape(QFrame.NoFrame)
    # without mouse tracking, ::item:hover only fires while a button is held
    table.setMouseTracking(True)
    table.viewport().setMouseTracking(True)
    # styling now lives entirely in the app-wide sheet so a theme switch
    # repaints tables too; no per-table stylesheet to drift out of sync.
    table.setStyleSheet("")


def fill_table(table: QTableWidget, df: pd.DataFrame, columns: list[tuple], row_color_col: str | None = None):
    """columns: list of (df_col, header_label, fmt) where fmt(value) -> str."""
    table.clear()
    table.setColumnCount(len(columns))
    table.setRowCount(len(df))
    table.setHorizontalHeaderLabels([c[1] for c in columns])
    table.setSortingEnabled(False)
    style_table(table)

    for r, (_, row) in enumerate(df.iterrows()):
        for c, (col, label, fmt) in enumerate(columns):
            raw = row.get(col, "")
            text = fmt(raw) if fmt else str(raw)
            is_numeric = any(k in col.lower() for k in NUMERIC_KEYWORDS)
            item = NumericItem(raw, text) if is_numeric else QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if is_numeric:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            else:
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            t = theme.current()
            # semantic good/bad, not the chrome accents; and the glyph in the
            # flag column means this isn't communicated by color alone
            if row_color_col and row.get(row_color_col) == "RISE":
                item.setForeground(QColor(t["good"]))
                if col == row_color_col:
                    item.setText(f"▲ {text}")
            elif row_color_col and row.get(row_color_col) == "FALL":
                item.setForeground(QColor(t["bad"]))
                if col == row_color_col:
                    item.setText(f"▼ {text}")
            table.setItem(r, c, item)

    table.setSortingEnabled(True)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeToContents)
    for c, (col, label, fmt) in enumerate(columns):
        is_wide = any(k in label.lower() for k in WIDE_LABEL_KEYWORDS)
        if is_wide:
            header.setSectionResizeMode(c, QHeaderView.Stretch)
    header.setStretchLastSection(False)
    if not any(any(k in label.lower() for k in WIDE_LABEL_KEYWORDS) for _, label, _ in columns):
        header.setStretchLastSection(True)
