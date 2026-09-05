"""A reusable one-chart-at-a-time carousel: prev/next navigation, a category
selector, and a page indicator. Reuses the same ChartCanvas widgets across
navigation (they're created once and redrawn in place by the caller), so
switching pages is instant -- no widget recreation.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QStackedLayout, QVBoxLayout,
    QWidget,
)


class ChartCarousel(QWidget):
    """Usage:
        carousel = ChartCarousel()
        carousel.add_chart("Players", "XI predicted points", chart_xi_canvas)
        carousel.add_chart("Players", "Value hunting", chart_value_scatter_canvas)
        ...
        carousel.finalize()
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries = []  # list of dicts: category, title, canvas
        self._current = 0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        nav_row = QHBoxLayout()
        nav_row.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        nav_row.addWidget(self.category_combo)

        nav_row.addStretch()

        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self.prev)
        nav_row.addWidget(self.prev_btn)

        self.page_label = QLabel("")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setMinimumWidth(260)
        self.page_label.setStyleSheet("font-weight: 600;")
        nav_row.addWidget(self.page_label)

        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.next)
        nav_row.addWidget(self.next_btn)

        layout.addLayout(nav_row)

        self._stack_holder = QWidget()
        self._stack = QStackedLayout(self._stack_holder)
        layout.addWidget(self._stack_holder, stretch=1)

        # keyboard navigation
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.next)

    def add_chart(self, category: str, title: str, canvas: QWidget):
        self._entries.append({"category": category, "title": title, "canvas": canvas})
        self._stack.addWidget(canvas)

    def finalize(self):
        """Call once after all add_chart() calls -- populates the category
        selector and shows the first chart."""
        categories = list(dict.fromkeys(e["category"] for e in self._entries))
        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        self.category_combo.addItems(categories)
        self.category_combo.blockSignals(False)
        self._current = 0
        self._refresh()

    def _refresh(self):
        if not self._entries:
            self.page_label.setText("No charts available")
            return
        entry = self._entries[self._current]
        self._stack.setCurrentWidget(entry["canvas"])

        cat_indices = [i for i, e in enumerate(self._entries) if e["category"] == entry["category"]]
        pos_in_cat = cat_indices.index(self._current) + 1
        self.page_label.setText(f"{entry['category']} — {entry['title']} ({pos_in_cat} / {len(cat_indices)})")

        if self.category_combo.currentText() != entry["category"]:
            self.category_combo.blockSignals(True)
            self.category_combo.setCurrentText(entry["category"])
            self.category_combo.blockSignals(False)

    def next(self):
        if not self._entries:
            return
        self._current = (self._current + 1) % len(self._entries)
        self._refresh()

    def prev(self):
        if not self._entries:
            return
        self._current = (self._current - 1) % len(self._entries)
        self._refresh()

    def _on_category_changed(self, category: str):
        for i, e in enumerate(self._entries):
            if e["category"] == category:
                self._current = i
                self._refresh()
                return

    def all_canvases(self):
        return [e["canvas"] for e in self._entries]
