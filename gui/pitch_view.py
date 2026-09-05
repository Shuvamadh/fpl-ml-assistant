"""FPL-style pitch view: the starting XI laid out by formation on a drawn
pitch, with kit/mugshot images, captain and vice armbands, and the bench in a
strip underneath.

Rendering is plain Qt painting + QLabels rather than matplotlib -- this is
chrome, not a chart, and it needs to stay crisp when the window resizes.
"""
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

import fpl_assets
import theme

# FPL's element_types report the keeper as "GKP"; older/derived frames
# sometimes use "GK". Match both or the goalkeeper silently vanishes.
GK_ALIASES = ("GKP", "GK", "GKP1")
ROW_ORDER = [GK_ALIASES, ("DEF",), ("MID",), ("FWD",)]


def _is_gk(position) -> bool:
    return str(position).strip().upper() in GK_ALIASES


# Kits by default (that's how the real FPL pitch reads, and every club has
# one); mugshots are available too but a handful of recent signings have no
# photo on the CDN, so those fall back to the kit anyway.
USE_PHOTOS = False


def set_use_photos(enabled: bool) -> None:
    global USE_PHOTOS
    USE_PHOTOS = bool(enabled)


class PlayerChip(QWidget):
    """One player: image, name plate, points plate, optional armband."""

    def __init__(self, row: pd.Series, armband: str = "", parent=None):
        super().__init__(parent)
        t = theme.current()
        # Without this the global `QWidget { background-color }` rule fills the
        # chip with theme background, punching a dark rectangle in the grass.
        self.setAttribute(Qt.WA_StyledBackground, False)
        self.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(0)
        lay.setAlignment(Qt.AlignCenter)

        is_gk = _is_gk(row.get("position", ""))
        pm = None
        try:
            pm = fpl_assets.player_image(
                int(row.get("id", 0)), int(row.get("team_id", 0)), is_gk=is_gk,
                height=52, prefer_photo=USE_PHOTOS,
            )
        except (TypeError, ValueError):
            pm = None

        img = QLabel()
        img.setAlignment(Qt.AlignCenter)
        if pm is not None:
            img.setPixmap(pm)
        else:
            # no mugshot and no kit cached -- show the position instead of a hole
            img.setText(str(row.get("position", "?")))
            img.setFixedHeight(52)
            img.setStyleSheet(
                f"color:{t['text_dim']}; font-weight:700; font-size:15px;"
            )
        lay.addWidget(img)

        name = str(row.get("web_name", ""))
        if armband:
            name = f"{name} {armband}"
        name_lbl = QLabel(name)
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setStyleSheet(
            f"background-color:{t['bg']}; color:{t['text']}; font-size:11px;"
            f"font-weight:600; padding:2px 6px; border-top-left-radius:4px;"
            f"border-top-right-radius:4px;"
        )
        lay.addWidget(name_lbl)

        pts = row.get("pred_points_adj", None)
        sub = f"{float(pts):.1f}" if pd.notna(pts) else "-"
        fixture = str(row.get("next_fixture", "") or "")
        pts_lbl = QLabel(f"{sub}  {fixture}".strip())
        pts_lbl.setAlignment(Qt.AlignCenter)
        pts_lbl.setStyleSheet(
            f"background-color:{t['accent2']}; color:#0b0b0b; font-size:10px;"
            f"font-weight:700; padding:2px 6px; border-bottom-left-radius:4px;"
            f"border-bottom-right-radius:4px;"
        )
        lay.addWidget(pts_lbl)

        tip = [f"{row.get('web_name','')} ({row.get('position','')}) - {row.get('name','')}"]
        if pd.notna(pts):
            tip.append(f"Predicted: {float(pts):.2f} pts")
        if row.get("next_fixture"):
            tip.append(f"Next: {row.get('next_fixture')} (FDR {row.get('next_fdr','?')})")
        news = str(row.get("news", "") or "").strip()
        if news:
            tip.append(f"! {news}")
        self.setToolTip("\n".join(tip))


class PitchWidget(QFrame):
    """Draws the pitch and arranges rows of PlayerChips by formation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("Pitch")
        # Opt out of stylesheet-driven background painting. The app-level sheet
        # has a `QMainWindow, QWidget { background-color: ... }` rule which, in
        # practice, beats even a `QFrame#Pitch` ID rule in the same sheet -- so
        # both a widget stylesheet and an ID rule render as flat theme
        # background. With WA_StyledBackground off, paintEvent below owns the
        # background outright and the grass is guaranteed to draw.
        self.setAttribute(Qt.WA_StyledBackground, False)
        self._grid = QVBoxLayout(self)
        self._grid.setContentsMargins(12, 12, 12, 12)
        self._grid.setSpacing(4)
        self._empty = QLabel("No squad loaded yet - hit Refresh.")
        self._empty.setAlignment(Qt.AlignCenter)
        self._grid.addWidget(self._empty)

    def paintEvent(self, ev):  # noqa: N802  (Qt naming)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()

        grad = QLinearGradient(0, r.top(), 0, r.bottom())
        grad.setColorAt(0.0, QColor("#1b7a3d"))
        grad.setColorAt(0.5, QColor("#146733"))
        grad.setColorAt(1.0, QColor("#0f5228"))
        p.fillRect(r, QBrush(grad))

        # mown stripes
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 12))
        bands = 8
        bh = max(1, r.height() // bands)
        for i in range(0, bands, 2):
            p.drawRect(r.left(), r.top() + i * bh, r.width(), bh)

        # markings
        pen = QPen(QColor(255, 255, 255, 70), 2)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        m = 10
        p.drawRect(r.adjusted(m, m, -m, -m))
        cy = r.center().y()
        p.drawLine(r.left() + m, cy, r.right() - m, cy)
        cr = min(r.width(), r.height()) // 8
        p.drawEllipse(r.center(), cr, cr)
        bw, bh2 = int(r.width() * 0.42), int(r.height() * 0.14)
        p.drawRect(r.center().x() - bw // 2, r.top() + m, bw, bh2)
        p.drawRect(r.center().x() - bw // 2, r.bottom() - m - bh2, bw, bh2)
        p.end()

    def set_xi(self, xi: pd.DataFrame, captain_id=None, vice_id=None) -> None:
        """Lay the XI out in formation rows: GK, DEF, MID, FWD."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                # takeAt only detaches from the LAYOUT -- the widget stays a
                # child, keeps its old geometry and keeps painting, and
                # deleteLater is deferred to the next event loop pass. Reparent
                # to None so it stops covering the pitch immediately.
                w.setParent(None)
                w.deleteLater()

        if xi is None or xi.empty:
            lbl = QLabel("No squad loaded yet - hit Refresh.")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:white; font-weight:600;")
            self._grid.addWidget(lbl)
            return

        pos_upper = xi["position"].astype(str).str.strip().str.upper()
        for aliases in ROW_ORDER:
            band = xi[pos_upper.isin(aliases)]
            if band.empty:
                continue
            row_w = QWidget()
            # each formation row spans the full pitch width, so a stylesheet
            # background here would cover the grass almost entirely
            row_w.setAttribute(Qt.WA_StyledBackground, False)
            row_w.setStyleSheet("background: transparent;")
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(6)
            row_l.addStretch()
            for _, r in band.iterrows():
                band_mark = ""
                if captain_id is not None and r.get("id") == captain_id:
                    band_mark = "(C)"
                elif vice_id is not None and r.get("id") == vice_id:
                    band_mark = "(V)"
                row_l.addWidget(PlayerChip(r, armband=band_mark))
            row_l.addStretch()
            self._grid.addWidget(row_w, stretch=1)


class SquadPitchView(QWidget):
    """Pitch + bench strip + formation/points summary."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        top_row = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setObjectName("SubHeader")
        top_row.addWidget(self.summary, stretch=1)
        self.kit_toggle = QCheckBox("Player photos")
        self.kit_toggle.setChecked(USE_PHOTOS)
        self.kit_toggle.setToolTip(
            "Off: club kits (default). On: player mugshots, falling back to the "
            "kit for players the FPL CDN has no photo for."
        )
        self.kit_toggle.toggled.connect(self._on_toggle_photos)
        top_row.addWidget(self.kit_toggle)
        lay.addLayout(top_row)

        self._last = None  # (xi, bench, captain_id, vice_id) for re-render

        self.pitch = PitchWidget()
        lay.addWidget(self.pitch, stretch=1)

        self.bench_label = QLabel("Bench")
        self.bench_label.setObjectName("SubHeader")
        lay.addWidget(self.bench_label)

        self.bench_holder = QWidget()
        self.bench_layout = QHBoxLayout(self.bench_holder)
        self.bench_layout.setContentsMargins(8, 6, 8, 6)
        self.bench_layout.setSpacing(6)
        self.bench_holder.setFixedHeight(112)
        lay.addWidget(self.bench_holder)
        self._restyle_bench()

    def _restyle_bench(self):
        t = theme.current()
        self.bench_holder.setStyleSheet(
            f"background-color:{t['panel']}; border:1px solid {t['border']};"
            f"border-radius:8px;"
        )

    def _on_toggle_photos(self, checked: bool) -> None:
        set_use_photos(checked)
        if self._last is not None:
            self.update_squad(*self._last)

    def update_squad(self, xi: pd.DataFrame, bench: pd.DataFrame,
                     captain_id=None, vice_id=None) -> None:
        self._last = (xi, bench, captain_id, vice_id)
        self._restyle_bench()
        self.pitch.set_xi(xi, captain_id, vice_id)

        while self.bench_layout.count():
            item = self.bench_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)  # see set_xi: takeAt alone leaves it painting
                w.deleteLater()
        self.bench_layout.addStretch()
        if bench is not None and not bench.empty:
            for _, r in bench.iterrows():
                self.bench_layout.addWidget(PlayerChip(r))
        else:
            lbl = QLabel("No bench data")
            lbl.setStyleSheet(f"color:{theme.current()['text_dim']};")
            self.bench_layout.addWidget(lbl)
        self.bench_layout.addStretch()

        if xi is not None and not xi.empty:
            counts = xi["position"].astype(str).str.upper().value_counts()
            formation = "-".join(
                str(int(counts.get(p, 0))) for p in ("DEF", "MID", "FWD")
            )
            total = xi["pred_points_adj"].sum() if "pred_points_adj" in xi else float("nan")
            cap_bonus = 0.0
            if captain_id is not None and "id" in xi:
                cap_row = xi[xi["id"] == captain_id]
                if not cap_row.empty and "pred_points_adj" in cap_row:
                    cap_bonus = float(cap_row.iloc[0]["pred_points_adj"])
            self.summary.setText(
                f"Formation {formation}   ·   projected {total + cap_bonus:.1f} pts "
                f"(XI {total:.1f} + captain {cap_bonus:.1f})"
            )
        else:
            self.summary.setText("")
