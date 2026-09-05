import html
from pathlib import Path

import pandas as pd
import re

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QProgressBar, QPushButton, QSizePolicy, QStatusBar, QTableWidget,
    QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

import charts
import theme
from chart_carousel import ChartCarousel
from config import load_config, save_config
from data_bridge import run_refresh
from llm_worker import ask_llm
from pitch_view import SquadPitchView
from table_utils import fill_table

BACKTEST_CSV = Path(__file__).resolve().parent.parent / "data" / "backtest_results.csv"

XI_COLS = [
    ("web_name", "Player", None),
    ("position", "Pos", None),
    ("name", "Club", None),
    ("now_cost_m", "Cost", lambda v: f"£{v:.1f}m"),
    ("pred_points_adj", "Pred Pts", lambda v: f"{v:.2f}"),
    ("next_fixture", "Next Fixture", None),
    ("next_fdr", "FDR", lambda v: f"{v:.0f}"),
]

VALUE_COLS = [
    ("web_name", "Player", None),
    ("buy_price_m", "Bought", lambda v: f"£{v:.1f}m"),
    ("now_cost_m", "Now", lambda v: f"£{v:.1f}m"),
    ("sell_price_m", "Sell Value", lambda v: f"£{v:.1f}m"),
    ("profit_m", "Profit", lambda v: f"{'+' if v > 0 else ''}£{v:.1f}m"),
]

TRANSFER_COLS = [
    ("out", "Out", None),
    ("out_pos", "Pos", None),
    ("out_sell_price", "Sell For", lambda v: f"£{v:.1f}m"),
    ("out_pred", "Out Pred", lambda v: f"{v:.2f}"),
    ("in", "In", None),
    ("in_team", "Club", None),
    ("in_cost", "Cost", lambda v: f"£{v:.1f}m"),
    ("in_pred", "In Pred", lambda v: f"{v:.2f}"),
    ("pred_gain", "Gain", lambda v: f"+{v:.2f}"),
    ("in_price_flag", "Price", None),
    ("leftover_bank", "Leftover", lambda v: f"£{v:.1f}m"),
]

ALL_PLAYERS_COLS = [
    ("web_name", "Player", None),
    ("position", "Pos", None),
    ("name", "Club", None),
    ("now_cost_m", "Cost", lambda v: f"£{v:.1f}m"),
    ("pred_points_adj", "Pred Pts", lambda v: f"{v:.2f}"),
    ("value_ratio", "Pts/£m", lambda v: f"{v:.2f}"),
    ("form", "Form", None),
    ("selected_by_percent", "Owned %", lambda v: f"{v}%"),
    ("next_fixture", "Next Fixture", None),
    ("next_fdr", "FDR", lambda v: f"{v:.0f}"),
    ("expected_goal_involvements", "xGI", lambda v: f"{v:.2f}" if pd.notna(v) else "-"),
    ("defensive_contribution", "DC", None),
    ("status", "Status", None),
    ("price_flag", "Price Signal", None),
]

PRICE_COLS = [
    ("web_name", "Player", None),
    ("position", "Pos", None),
    ("name", "Club", None),
    ("now_cost_m", "Cost", lambda v: f"£{v:.1f}m"),
    ("selected_by_percent", "Owned %", lambda v: f"{v}%"),
    ("price_flag", "Signal", None),
    ("price_change_percent", "Momentum %", lambda v: f"{v:.1f}%" if pd.notna(v) else "-"),
]

STANDINGS_COLS = [
    ("rank", "Rank", None),
    ("entry_name", "Team", None),
    ("player_name", "Manager", None),
    ("event_total", "GW Pts", None),
    ("total", "Total", None),
]

OWNERSHIP_COLS = [
    ("web_name", "Player", None),
    ("position", "Pos", None),
    ("owned_by_n", "Managers", None),
    ("owned_pct", "Owned %", lambda v: f"{v:.0f}%"),
]

CAPTAINS_COLS = [
    ("web_name", "Player", None),
    ("captained_by_n", "Captained By", None),
]

CHIPS_COLS = [
    ("entry_name", "Team", None),
    ("player_name", "Manager", None),
    ("active_chip", "Chip", None),
]

PROJECTION_COLS = [
    ("projected_rank", "Proj Rank", None),
    ("entry_name", "Team", None),
    ("player_name", "Manager", None),
    ("captain_name", "Captain", None),
    ("projected_gw_points", "Proj GW Pts", lambda v: f"{v:.1f}"),
    ("projected_total", "Proj Total", lambda v: f"{v:.0f}"),
    ("current_rank", "Now", None),
    ("rank_change", "Move", lambda v: ("+" if v > 0 else "") + f"{v:.0f}" if pd.notna(v) else "-"),
]

RIVAL_TRANSFER_COLS = [
    ("entry_name", "Team", None),
    ("out", "Weakest Starter", None),
    ("out_pred", "Their Pred", lambda v: f"{v:.2f}" if pd.notna(v) else "-"),
    ("in", "Best Upgrade", None),
    ("in_pred", "Upgrade Pred", lambda v: f"{v:.2f}" if pd.notna(v) else "-"),
    ("pred_gain", "Gain", lambda v: f"+{v:.2f}" if pd.notna(v) else "-"),
]

TC_COLS = [
    ("web_name", "Player", None),
    ("position", "Pos", None),
    ("name", "Club", None),
    ("pred_points_adj", "Pred Pts", lambda v: f"{v:.2f}"),
    ("is_dgw", "DGW", lambda v: "Yes" if v else "-"),
    ("tc_gain_vs_normal_captain", "Gain vs Normal Capt", lambda v: f"{v:+.2f}"),
]

OPTIMAL_SQUAD_COLS = [
    ("web_name", "Player", None),
    ("position", "Pos", None),
    ("name", "Club", None),
    ("now_cost_m", "Cost", lambda v: f"£{v:.1f}m"),
    ("pred_points_adj", "Pred Pts", lambda v: f"{v:.2f}"),
]

BENCH_POINTS_COLS = [
    ("entry_name", "Team", None),
    ("bench_points", "Bench Pts Lost", None),
]

VALUE_GROWTH_COLS = [
    ("entry_name", "Team", None),
    ("value_growth_m", "Value Growth", lambda v: f"{'+' if v > 0 else ''}£{v:.1f}m"),
]

RANK_SWING_COLS = [
    ("entry_name", "Team", None),
    ("rank_swing", "Biggest GW Rank Swing", lambda v: f"{v:,.0f}" if pd.notna(v) else "-"),
]

EO_COLS = [
    ("web_name", "Player", None),
    ("position", "Pos", None),
    ("owned_n", "Owned By", None),
    ("owned_pct", "Owned %", lambda v: f"{v:.0f}%"),
    ("captained_n", "Captained By", None),
    ("eo_pct", "Effective Own %", lambda v: f"{v:.0f}%"),
    ("template_risk", "Verdict", None),
]

FORM_COLS = [
    ("entry_name", "Team", None),
    ("recent_points", "Last 4 GW Pts", None),
    ("bench_wasted", "Bench Pts Wasted", None),
]

H2H_COLS = [
    ("web_name", "Player", None),
    ("position", "Pos", None),
    ("pred_points_adj", "Pred Pts", lambda v: f"{v:.2f}"),
]

OWNERS_COLS = [
    ("web_name", "Player", None),
    ("entry_name", "Manager", None),
    ("role", "Role", None),
]

TEMPLATE_COLS = [
    ("entry_name", "Team", None),
    ("template_count", "Template Players", None),
    ("template_pct", "Template %", lambda v: f"{v:.0f}%"),
    ("differentials_owned", "Differentials", None),
]


def stat_card(label: str, value: str, small_value: bool = False) -> QFrame:
    """small_value=True for cards holding a name rather than a number -- at
    22pt a long player name clips in a 5-across row (GUI_PLAN 6.8)."""
    card = QFrame()
    card.setObjectName("StatCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 12, 16, 12)
    layout.setSpacing(4)
    val = QLabel(value)
    val.setProperty("class", "StatCardValueSmall" if small_value else "StatCardValue")
    lab = QLabel(label.upper())
    lab.setProperty("class", "StatCardLabel")
    # letter-spacing has no Qt stylesheet equivalent -- it must be set on the
    # font object (GUI_PLAN 6.11)
    lab_font = lab.font()
    lab_font.setLetterSpacing(QFont.AbsoluteSpacing, 0.6)
    lab.setFont(lab_font)
    layout.addWidget(val)
    layout.addWidget(lab)
    card.value_label = val
    card.label_widget = lab
    return card


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPL ML Assistant")
        self.resize(1280, 820)
        self._thread = None
        self._worker = None
        self._data = None
        self._config = load_config()
        theme.set_theme(self._config.get("theme", theme.current_name()))
        self.setStyleSheet(theme.stylesheet())

        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(24)

        # header
        header_row = QHBoxLayout()
        title_box = QVBoxLayout()
        self.title_label = QLabel("FPL ML Assistant")
        self.title_label.setObjectName("Header")
        self.subtitle_label = QLabel("Loading...")
        self.subtitle_label.setObjectName("SubHeader")
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.subtitle_label)
        header_row.addLayout(title_box)
        header_row.addStretch()

        header_row.addWidget(QLabel("Team ID:"))
        self.team_id_input = QLineEdit(str(self._config["team_id"]))
        self.team_id_input.setFixedWidth(100)
        header_row.addWidget(self.team_id_input)

        header_row.addWidget(QLabel("Last event:"))
        self.event_input = QLineEdit(str(self._config["event"]))
        self.event_input.setFixedWidth(50)
        header_row.addWidget(self.event_input)

        header_row.addWidget(QLabel("League ID:"))
        self.league_id_input = QLineEdit(str(self._config["league_id"]))
        self.league_id_input.setFixedWidth(100)
        header_row.addWidget(self.league_id_input)

        header_row.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(theme.names())
        self.theme_combo.setCurrentText(theme.current_name())
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        header_row.addWidget(self.theme_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        header_row.addWidget(self.refresh_btn)
        layout.addLayout(header_row)

        # stat cards
        self.stats_row = QHBoxLayout()
        self.card_bank = stat_card("Bank", "-")
        self.card_value = stat_card("Squad Value", "-")
        self.card_sellable = stat_card("Sellable Value", "-")
        self.card_xi_pts = stat_card("Predicted XI Pts", "-")
        self.card_captain = stat_card("Suggested Captain", "-", small_value=True)
        for c in (self.card_bank, self.card_value, self.card_sellable, self.card_xi_pts, self.card_captain):
            self.stats_row.addWidget(c)
        layout.addLayout(self.stats_row)
        self._restyle_stat_cards()

        # progress
        self.status_label = QLabel("")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setVisible(False)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)

        # tabs
        self.tabs = QTabWidget()
        # 8 tabs at 1280px risks eliding; be explicit rather than
        # inheriting the platform default
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.ElideNone)
        layout.addWidget(self.tabs)

        self.xi_table = QTableWidget()
        self.bench_table = QTableWidget()
        squad_tab = QWidget()
        squad_layout = QVBoxLayout(squad_tab)
        self.squad_subtabs = QTabWidget()
        squad_layout.addWidget(self.squad_subtabs)

        # FPL-style pitch, using the club kits / player mugshots under assets/
        self.pitch_view = SquadPitchView()
        self.squad_subtabs.addTab(self.pitch_view, "Pitch")

        list_page = QWidget()
        list_layout = QVBoxLayout(list_page)
        list_layout.addWidget(QLabel("Starting XI"))
        list_layout.addWidget(self.xi_table, stretch=2)
        list_layout.addWidget(QLabel("Bench"))
        list_layout.addWidget(self.bench_table, stretch=1)
        self.squad_subtabs.addTab(list_page, "List")

        self.tabs.addTab(squad_tab, "My Squad")

        self.value_table = QTableWidget()
        value_tab = QWidget()
        value_layout = QVBoxLayout(value_tab)
        value_layout.addWidget(QLabel("Squad value: buy price -> real sell price (half-profit rule applied)"))
        value_layout.addWidget(self.value_table)
        self.tabs.addTab(value_tab, "Squad Value")

        self.transfer_table = QTableWidget()
        transfer_tab = QWidget()
        transfer_layout = QVBoxLayout(transfer_tab)
        transfer_layout.addWidget(QLabel("Suggested upgrades within your real sellable budget"))
        transfer_layout.addWidget(self.transfer_table)
        self.tabs.addTab(transfer_tab, "Transfers")

        all_players_tab = QWidget()
        all_layout = QVBoxLayout(all_players_tab)
        filter_row = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search player or club...")
        self.search_box.textChanged.connect(self._apply_player_filter)
        self.position_filter = QComboBox()
        self.position_filter.addItems(["All", "GKP", "DEF", "MID", "FWD"])
        self.position_filter.currentTextChanged.connect(self._apply_player_filter)
        filter_row.addWidget(self.search_box, stretch=3)
        filter_row.addWidget(self.position_filter, stretch=1)
        all_layout.addLayout(filter_row)
        self.all_players_table = QTableWidget()
        all_layout.addWidget(self.all_players_table)
        self.tabs.addTab(all_players_tab, "All Players")

        self.price_table = QTableWidget()
        price_tab = QWidget()
        price_layout = QVBoxLayout(price_tab)
        price_layout.addWidget(QLabel("Players FPL's own transfer-momentum model flags as close to a price change"))
        price_layout.addWidget(self.price_table)
        self.tabs.addTab(price_tab, "Price Watch")

        league_tab = QWidget()
        league_layout = QVBoxLayout(league_tab)
        self.league_header = QLabel("Mini League")
        self.league_header.setObjectName("Header")
        league_layout.addWidget(self.league_header)

        # Sub-tabs: previously all five tables shared one screen (three of them
        # squeezed side-by-side in a single row), which is what made them
        # unreadable. Each now gets the full width.
        self.league_subtabs = QTabWidget()
        league_layout.addWidget(self.league_subtabs, stretch=1)

        def _table_page(caption: str) -> QTableWidget:
            page = QWidget()
            lay = QVBoxLayout(page)
            lay.setContentsMargins(0, 8, 0, 0)
            cap = QLabel(caption)
            cap.setObjectName("SubHeader")
            lay.addWidget(cap)
            tbl = QTableWidget()
            tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lay.addWidget(tbl, stretch=1)
            return page, tbl

        page, self.standings_table = _table_page(
            "Full league standings — click any column header to sort")
        self.league_subtabs.addTab(page, "Standings")

        page, self.ownership_table = _table_page(
            "Most-owned players in the league — this is the template you're measured against")
        self.league_subtabs.addTab(page, "Template")

        page, self.captains_table = _table_page(
            "Who the league captained this gameweek — where the rank swings come from")
        # captaincy is the single biggest source of rank movement, so the useful
        # line is the edge against the field, not just the raw split
        self.captaincy_note = QLabel("")
        self.captaincy_note.setWordWrap(True)
        page.layout().insertWidget(1, self.captaincy_note)
        self.league_subtabs.addTab(page, "Captaincy")

        page, self.differentials_table = _table_page(
            "Your differentials — players you own that most of the league doesn't")
        self.league_subtabs.addTab(page, "Differentials")

        page, self.chips_table = _table_page("Chips played this gameweek")
        self.league_subtabs.addTab(page, "Chips")

        # NOTE the caption: FPL hides other managers' picks until the deadline
        # passes, so this is built on their LAST KNOWN squad and is blind to
        # transfers made this week. Saying so is the difference between a
        # forecast and a lie.
        page, self.projection_table = _table_page(
            "Projected finish this gameweek — based on last known squads, "
            "so it cannot see transfers your rivals made this week")
        self.league_subtabs.addTab(page, "Projection")

        page, self.rival_transfers_table = _table_page(
            "Transfers your rivals should make (their weakest starter vs the "
            "best upgrade, assuming no money in the bank)")
        self.league_subtabs.addTab(page, "Rival Moves")

        page, self.eo_table = _table_page(
            "Effective ownership — starting ownership plus an extra copy for every "
            "captain. This, not plain ownership, is what decides whether a pick "
            "gains or loses you rank")
        self.league_subtabs.addTab(page, "Eff. Ownership")

        page, self.form_table = _table_page(
            "Form over the last 4 gameweeks — the cumulative standings hide who is "
            "actually coming for you")
        self.league_subtabs.addTab(page, "Form")

        # ---- head to head ----
        h2h_page = QWidget()
        h2h_layout = QVBoxLayout(h2h_page)
        h2h_layout.setContentsMargins(0, 8, 0, 0)
        h2h_controls = QHBoxLayout()
        h2h_controls.addWidget(QLabel("Compare:"))
        self.h2h_a = QComboBox()
        self.h2h_b = QComboBox()
        for cb in (self.h2h_a, self.h2h_b):
            cb.setMinimumWidth(180)
            cb.currentIndexChanged.connect(self._update_h2h)
        h2h_controls.addWidget(self.h2h_a)
        h2h_controls.addWidget(QLabel("vs"))
        h2h_controls.addWidget(self.h2h_b)
        h2h_controls.addStretch()
        h2h_layout.addLayout(h2h_controls)
        self.h2h_summary = QLabel("Refresh with a League ID set to compare managers.")
        self.h2h_summary.setObjectName("SubHeader")
        self.h2h_summary.setWordWrap(True)
        h2h_layout.addWidget(self.h2h_summary)
        h2h_tables = QHBoxLayout()
        a_box = QVBoxLayout()
        self.h2h_a_label = QLabel("Only in A")
        a_box.addWidget(self.h2h_a_label)
        self.h2h_a_table = QTableWidget()
        a_box.addWidget(self.h2h_a_table)
        h2h_tables.addLayout(a_box)
        b_box = QVBoxLayout()
        self.h2h_b_label = QLabel("Only in B")
        b_box.addWidget(self.h2h_b_label)
        self.h2h_b_table = QTableWidget()
        b_box.addWidget(self.h2h_b_table)
        h2h_tables.addLayout(b_box)
        h2h_layout.addLayout(h2h_tables, stretch=1)
        self.league_subtabs.addTab(h2h_page, "Head to Head")

        # ---- who owns ----
        owns_page = QWidget()
        owns_layout = QVBoxLayout(owns_page)
        owns_layout.setContentsMargins(0, 8, 0, 0)
        owns_row = QHBoxLayout()
        owns_row.addWidget(QLabel("Player:"))
        self.owns_input = QLineEdit()
        self.owns_input.setPlaceholderText("e.g. Haaland — who in the league has him?")
        self.owns_input.textChanged.connect(self._update_owners)
        owns_row.addWidget(self.owns_input, stretch=1)
        owns_layout.addLayout(owns_row)
        self.owners_table = QTableWidget()
        owns_layout.addWidget(self.owners_table, stretch=1)
        self.league_subtabs.addTab(owns_page, "Who Owns")

        fun_page = QWidget()
        fun_layout = QVBoxLayout(fun_page)
        fun_layout.setContentsMargins(0, 8, 0, 0)
        self.fun_subtabs = QTabWidget()
        fun_layout.addWidget(self.fun_subtabs)
        for attr, label, caption in (
            ("bench_points_table", "Bench Points",
             "Points left rotting on the bench, season to date — luck, or bad picks?"),
            ("value_growth_table", "Value Growth",
             "Squad value change since GW1 — who is actually good at price rises"),
            ("rank_swings_table", "Rank Swings",
             "Biggest single-gameweek move in overall FPL rank"),
            ("template_table", "Template",
             "How closely each manager hugs the template, and who owns real differentials"),
        ):
            sub_page, tbl = _table_page(caption)
            setattr(self, attr, tbl)
            self.fun_subtabs.addTab(sub_page, label)
        self.league_subtabs.addTab(fun_page, "Fun Stats")

        # dedicated canvases for the league charts sub-tab (the global Charts
        # tab keeps its own -- a QWidget can only have one parent, so these
        # can't be the same objects)
        league_charts_page = QWidget()
        lc_layout = QVBoxLayout(league_charts_page)
        lc_layout.setContentsMargins(0, 8, 0, 0)
        self.league_carousel = ChartCarousel()
        self.lc_standings = charts.ChartCanvas(8, 6)
        self.lc_rank_progression = charts.ChartCanvas(8, 6)
        self.lc_ownership = charts.ChartCanvas(8, 6)
        self.lc_captaincy = charts.ChartCanvas(8, 6)
        self.league_carousel.add_chart("League", "Rank progression by GW", self.lc_rank_progression)
        self.league_carousel.add_chart("League", "Standings", self.lc_standings)
        self.league_carousel.add_chart("League", "Template ownership", self.lc_ownership)
        self.league_carousel.add_chart("League", "Captaincy split", self.lc_captaincy)
        self.league_carousel.finalize()
        lc_layout.addWidget(self.league_carousel)
        self.league_subtabs.addTab(league_charts_page, "Charts")

        self.tabs.addTab(league_tab, "Mini League")

        charts_tab = QWidget()
        charts_layout = QVBoxLayout(charts_tab)
        charts_layout.setContentsMargins(0, 0, 0, 0)

        # Players
        self.chart_xi = charts.ChartCanvas(8, 6)
        self.chart_value_scatter = charts.ChartCanvas(8, 6)
        self.chart_fixture_heatmap = charts.ChartCanvas(8, 6)

        # Transfers
        self.chart_value = charts.ChartCanvas(8, 6)  # squad value bars
        self.chart_transfer_gains = charts.ChartCanvas(8, 6)
        self.chart_price_momentum = charts.ChartCanvas(8, 6)

        # Mini League
        self.chart_standings = charts.ChartCanvas(8, 6)
        self.chart_ownership = charts.ChartCanvas(8, 6)
        self.chart_captaincy = charts.ChartCanvas(8, 6)
        self.chart_rank_progression = charts.ChartCanvas(8, 6)

        # Model
        self.chart_backtest = charts.ChartCanvas(8, 6)

        self.carousel = ChartCarousel()
        self.carousel.add_chart("Players", "Starting XI predicted points", self.chart_xi)
        self.carousel.add_chart("Players", "Value hunting (cost vs pts)", self.chart_value_scatter)
        self.carousel.add_chart("Players", "Fixture difficulty heatmap", self.chart_fixture_heatmap)

        self.carousel.add_chart("Transfers", "Squad value: bought vs now", self.chart_value)
        self.carousel.add_chart("Transfers", "Best upgrades available", self.chart_transfer_gains)
        self.carousel.add_chart("Transfers", "Price momentum vs ownership", self.chart_price_momentum)

        self.carousel.add_chart("Mini League", "League standings", self.chart_standings)
        self.carousel.add_chart("Mini League", "Most-owned players", self.chart_ownership)
        self.carousel.add_chart("Mini League", "Captaincy choices", self.chart_captaincy)
        self.carousel.add_chart("Mini League", "Rank progression by GW", self.chart_rank_progression)

        self.carousel.add_chart("Model", "Walk-forward validation (MAE)", self.chart_backtest)

        self.carousel.finalize()
        charts_layout.addWidget(self.carousel)
        self.tabs.addTab(charts_tab, "Charts")

        # ---- Chips ----
        chips_tab = QWidget()
        chips_layout = QVBoxLayout(chips_tab)
        chips_layout.setContentsMargins(16, 16, 16, 16)
        self.chip_status = QLabel("Run a refresh to compute chip strategy.")
        self.chip_status.setObjectName("SubHeader")
        self.chip_status.setWordWrap(True)
        chips_layout.addWidget(self.chip_status)

        self.chip_subtabs = QTabWidget()
        chips_layout.addWidget(self.chip_subtabs, stretch=1)

        page, self.wildcard_table = _table_page(
            "Best wildcard squad — optimised over the next 5 gameweeks, because "
            "a wildcard is permanent")
        self.chip_subtabs.addTab(page, "Wildcard")

        page, self.freehit_table = _table_page(
            "Best free-hit squad — optimised for ONE gameweek only; it is close "
            "to wasted outside a blank or double gameweek")
        self.chip_subtabs.addTab(page, "Free Hit")

        page, self.tc_table = _table_page(
            "Triple captain targets — ranked by gain over the captain you'd have "
            "picked anyway, not by raw points")
        self.chip_subtabs.addTab(page, "Triple Captain")

        bb_page = QWidget()
        bb_layout = QVBoxLayout(bb_page)
        bb_layout.setContentsMargins(0, 8, 0, 0)
        self.bench_boost_label = QLabel("-")
        self.bench_boost_label.setWordWrap(True)
        bb_layout.addWidget(self.bench_boost_label)
        bb_layout.addStretch()
        self.chip_subtabs.addTab(bb_page, "Bench Boost")

        page, self.bgw_dgw_table = _table_page(
            "Blank and double gameweeks ahead — this is what makes a Free Hit or "
            "Bench Boost worth playing")
        self.chip_subtabs.addTab(page, "Blanks / Doubles")

        self.tabs.addTab(chips_tab, "Chips")

        ai_tab = QWidget()
        ai_layout = QVBoxLayout(ai_tab)
        ai_layout.addWidget(QLabel("Ask about your squad, transfers, or the mini league "
                                    "(runs locally via Ollama/Qwen, grounded in your latest refreshed data)"))
        self.chat_log = QTextBrowser()
        self.chat_log.setObjectName("ChatLog")
        self.chat_log.setOpenExternalLinks(False)
        ai_layout.addWidget(self.chat_log, stretch=1)

        # strongest "not frozen" signal available; a local model can run 20-60s
        self.chat_progress = QProgressBar()
        self.chat_progress.setRange(0, 0)
        self.chat_progress.setTextVisible(False)
        self.chat_progress.setFixedHeight(3)
        self.chat_progress.setVisible(False)
        ai_layout.addWidget(self.chat_progress)
        chat_input_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setObjectName("ChatInput")
        self.chat_input.setPlaceholderText("e.g. Who should I captain this week and why?")
        self.chat_input.returnPressed.connect(self._send_chat)
        self.chat_send_btn = QPushButton("Ask")
        self.chat_send_btn.clicked.connect(self._send_chat)
        chat_input_row.addWidget(self.chat_input, stretch=1)
        chat_input_row.addWidget(self.chat_send_btn)
        ai_layout.addLayout(chat_input_row)
        self.tabs.addTab(ai_tab, "AI Assistant")

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def refresh(self):
        try:
            team_id = int(self.team_id_input.text())
            event = int(self.event_input.text())
            league_id = int(self.league_id_input.text()) if self.league_id_input.text().strip() else None
        except ValueError:
            self.status_bar.showMessage("Team ID, event, and league ID must be numbers.")
            return

        save_config(team_id, event, league_id or 0, theme.current_name())
        self.refresh_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_label.setText("Starting refresh...")

        self._thread, self._worker = run_refresh(
            team_id, event,
            on_progress=self._on_progress,
            on_done=self._on_done,
            on_error=self._on_error,
            league_id=league_id,
        )

    def _on_progress(self, msg: str):
        self.status_label.setText(msg)
        print(f"[progress] {msg}")

    def _on_error(self, msg: str):
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.status_label.setText("Refresh failed - see status bar.")
        self.status_bar.showMessage(msg[:200])
        print(f"[error] {msg}")

    def _on_done(self, data: dict):
        print("[done] refresh complete")
        self._data = data
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.status_label.setText("Up to date.")

        meta = data["meta"]
        squad, xi, bench, transfers, predictions = (
            data["squad"], data["xi"], data["bench"], data["transfers"], data["predictions"]
        )

        self.title_label.setText(meta["team_name"])
        self.subtitle_label.setText(f"Squad entering GW{meta['event'] + 1}")

        self.card_bank.value_label.setText(f"£{meta['bank']:.1f}m")
        self.card_value.value_label.setText(f"£{meta['value']:.1f}m")
        sellable = squad["sell_price_m"].sum() if "sell_price_m" in squad else meta["value"]
        self.card_sellable.value_label.setText(f"£{sellable:.1f}m")
        self.card_xi_pts.value_label.setText(f"{xi['pred_points_adj'].sum():.1f}")
        cap = xi.iloc[0]
        self.card_captain.value_label.setText(f"{cap['web_name']}")

        fill_table(self.xi_table, xi, XI_COLS)
        fill_table(self.bench_table, bench, XI_COLS)

        # same captain/vice convention as recommend.py: XI is sorted by
        # adjusted predicted points, so [0] is the pick and [1] the backup
        cap_id = xi.iloc[0]["id"] if "id" in xi and len(xi) else None
        vice_id = xi.iloc[1]["id"] if "id" in xi and len(xi) > 1 else None
        self.pitch_view.update_squad(xi, bench, cap_id, vice_id)

        self._populate_chips(data.get("chips"))
        self._populate_league_extras(data)
        fill_table(self.value_table, squad.sort_values("profit_m", ascending=False), VALUE_COLS)

        if transfers is not None and not transfers.empty:
            fill_table(self.transfer_table, transfers, TRANSFER_COLS, row_color_col="in_price_flag")
        else:
            self.transfer_table.clear()
            self.transfer_table.setRowCount(0)
            self.transfer_table.setColumnCount(0)

        self._all_players_df = predictions
        self._apply_player_filter()

        price_watch = predictions[predictions["price_flag"] != ""].sort_values(
            "price_signal", key=lambda s: s.abs(), ascending=False
        )
        fill_table(self.price_table, price_watch, PRICE_COLS, row_color_col="price_flag")

        self._redraw_charts(data)

    def _redraw_charts(self, data: dict):
        """Draws every chart in the carousel from the last-fetched data. Also
        called on a theme switch (without re-fetching) so colors update
        immediately."""
        meta = data["meta"]
        squad, xi, transfers, predictions = (
            data["squad"], data["xi"], data["transfers"], data["predictions"]
        )

        if data.get("standings") is not None:
            standings = data["standings"]
            insights = data["insights"]
            self.league_header.setText(data["league_name"] or "Mini League")
            fill_table(self.standings_table, standings, STANDINGS_COLS)
            fill_table(self.ownership_table, insights["ownership"].head(20), OWNERSHIP_COLS)
            fill_table(self.captains_table, insights["captains"], CAPTAINS_COLS)
            fill_table(self.differentials_table, insights["differentials"], OWNERSHIP_COLS)
            if not insights["chips_active_this_gw"].empty:
                fill_table(self.chips_table, insights["chips_active_this_gw"], CHIPS_COLS)
            else:
                self.chips_table.clear()
                self.chips_table.setRowCount(0)
                self.chips_table.setColumnCount(0)

            for canvas in (self.chart_standings, self.lc_standings):
                charts.draw_league_standings_bar(canvas, standings, meta["team_name"])
            for canvas in (self.chart_ownership, self.lc_ownership):
                charts.draw_ownership_chart(canvas, insights["ownership"])
            if not insights["captains"].empty:
                for canvas in (self.chart_captaincy, self.lc_captaincy):
                    charts.draw_captaincy_pie(canvas, insights["captains"])

            rank_progression = data.get("rank_progression")
            if rank_progression is not None and not rank_progression.empty:
                for canvas in (self.chart_rank_progression, self.lc_rank_progression):
                    charts.draw_rank_progression(canvas, rank_progression, meta["team_name"])
            else:
                for canvas in (self.chart_rank_progression, self.lc_rank_progression):
                    self._placeholder_chart(canvas, "Rank progression unavailable")
        else:
            msg = "Set a League ID to see mini-league charts"
            for canvas in (self.chart_standings, self.chart_ownership, self.chart_captaincy,
                           self.chart_rank_progression, self.lc_standings, self.lc_ownership,
                           self.lc_captaincy, self.lc_rank_progression):
                self._placeholder_chart(canvas, msg)

        # Players
        charts.draw_xi_points_bar(self.chart_xi, xi)
        charts.draw_value_scatter(self.chart_value_scatter, predictions)

        fixture_matrix = data.get("fixture_matrix")
        if fixture_matrix is not None and not fixture_matrix.empty:
            charts.draw_fixture_heatmap(self.chart_fixture_heatmap, fixture_matrix)
        else:
            self._placeholder_chart(self.chart_fixture_heatmap, "Fixture data unavailable")

        # Transfers
        charts.draw_squad_value_bars(self.chart_value, squad)
        charts.draw_transfer_gains(self.chart_transfer_gains, transfers)
        charts.draw_price_momentum_scatter(self.chart_price_momentum, predictions)

        # Model
        self._draw_backtest_chart()

    def _placeholder_chart(self, canvas, message: str):
        fig = canvas.figure
        fig.clear()
        t = theme.current()
        fig.set_facecolor(t["panel"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(t["panel"])
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", color=t["text_dim"], fontsize=11, wrap=True)
        canvas.draw()

    def _draw_backtest_chart(self):
        if BACKTEST_CSV.exists():
            try:
                backtest_df = pd.read_csv(BACKTEST_CSV)
                charts.draw_backtest_mae(self.chart_backtest, backtest_df)
                return
            except Exception:
                pass
        self._placeholder_chart(self.chart_backtest, "Backtest not yet run")

    def _restyle_stat_cards(self):
        """Sizes/colors now come from the app sheet via the "class" property.
        Qt does not re-evaluate property selectors on its own, so each label
        must be unpolished and re-polished after the sheet changes, or the
        cards keep their old look until something else forces a repaint."""
        for c in (self.card_bank, self.card_value, self.card_sellable,
                  self.card_xi_pts, self.card_captain):
            for w in (c.value_label, c.label_widget):
                w.setStyleSheet("")
                w.style().unpolish(w)
                w.style().polish(w)
                w.update()

    def _on_theme_changed(self, name: str):
        theme.set_theme(name)
        self.setStyleSheet(theme.stylesheet())
        self._restyle_stat_cards()
        save_config(
            int(self.team_id_input.text() or self._config["team_id"]),
            int(self.event_input.text() or self._config["event"]),
            int(self.league_id_input.text()) if self.league_id_input.text().strip() else 0,
            name,
        )
        # redraw every currently-populated chart canvas so colors update immediately
        if self._data is not None:
            self._redraw_charts(self._data)
        else:
            for canvas in self.carousel.all_canvases():
                self._placeholder_chart(canvas, "Loading...")

    def _apply_player_filter(self):
        if not hasattr(self, "_all_players_df") or self._all_players_df is None:
            return
        df = self._all_players_df
        text = self.search_box.text().strip().lower()
        pos = self.position_filter.currentText()
        if pos != "All":
            df = df[df["position"] == pos]
        if text:
            df = df[
                df["web_name"].str.lower().str.contains(text, na=False)
                | df["name"].str.lower().str.contains(text, na=False)
            ]
        fill_table(self.all_players_table, df.head(300), ALL_PLAYERS_COLS, row_color_col="price_flag")

    # ----------------------------------------------------------- chips ---

    @staticmethod
    def _clear_table(table):
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)

    def _populate_chips(self, chips_data):
        if not chips_data:
            self.chip_status.setText("Chip strategy unavailable.")
            return
        if "error" in chips_data:
            # most likely the ILP solver dependency is missing
            self.chip_status.setText(
                f"Chip optimisation unavailable: {chips_data['error']}\n"
                "The squad optimiser needs an ILP solver (pip install pulp)."
            )
            for t in (self.wildcard_table, self.freehit_table, self.tc_table,
                      self.bgw_dgw_table):
                self._clear_table(t)
            return

        used = chips_data.get("used") or []
        self.chip_status.setText(
            "Chips already played this season: "
            + (", ".join(used) if used else "none")
            + ".  Wildcard is optimised over 5 gameweeks; Free Hit over one."
        )

        for key, table in (("wildcard", self.wildcard_table),
                           ("free_hit", self.freehit_table)):
            res = chips_data.get(key) or {}
            squad = res.get("squad")
            if squad is not None and not squad.empty:
                fill_table(table, squad.sort_values("pred_points_adj", ascending=False),
                           OPTIMAL_SQUAD_COLS)
            else:
                self._clear_table(table)

        tc = chips_data.get("triple_captain")
        if tc is not None and not tc.empty:
            fill_table(self.tc_table, tc.head(15), TC_COLS)
        else:
            self._clear_table(self.tc_table)

        bb = chips_data.get("bench_boost") or {}
        risky = bb.get("risky_bench_players") or []
        msg = f"Bench Boost is worth about {bb.get('bench_boost_value', 0):.1f} predicted points."
        if risky:
            msg += ("\n\nRotation risk on the bench: " + ", ".join(risky) +
                    " — a bench boost is worthless if they don't play.")
        self.bench_boost_label.setText(msg)

        bd = chips_data.get("blank_double")
        if bd is not None and not bd.empty:
            cols = [(c, c.replace("_", " ").title(), None) for c in bd.columns]
            fill_table(self.bgw_dgw_table, bd, cols)
        else:
            self._clear_table(self.bgw_dgw_table)

    def _update_h2h(self):
        """Recompute the head-to-head whenever either manager selection
        changes. Shared players cancel out between two squads, so only the
        differences are shown -- they are the only thing that can move the gap.
        """
        data = self._data or {}
        squads = data.get("league_squads")
        preds = data.get("predictions")
        if squads is None or preds is None:
            return
        a_id = self.h2h_a.currentData()
        b_id = self.h2h_b.currentData()
        if a_id is None or b_id is None or a_id == b_id:
            self.h2h_summary.setText("Pick two different managers to compare.")
            self._clear_table(self.h2h_a_table)
            self._clear_table(self.h2h_b_table)
            return
        import league_extras
        h = league_extras.head_to_head(squads, preds, a_id, b_id)
        if "error" in h:
            self.h2h_summary.setText(h["error"])
            return
        diff = h["projected_a"] - h["projected_b"]
        lead = h["name_a"] if diff >= 0 else h["name_b"]
        self.h2h_summary.setText(
            f"{h['name_a']} vs {h['name_b']} — {h['shared_n']} of {h['squad_size']} "
            f"players shared ({h['overlap_pct']:.0f}% overlap). "
            f"Captains: {h['captain_a'] or '-'} vs {h['captain_b'] or '-'}. "
            f"Projected this GW: {h['projected_a']:.1f} vs {h['projected_b']:.1f} "
            f"— {lead} by {abs(diff):.1f}."
        )
        self.h2h_a_label.setText(f"Only {h['name_a']} has ({h['unique_pred_a']:.1f} pred pts)")
        self.h2h_b_label.setText(f"Only {h['name_b']} has ({h['unique_pred_b']:.1f} pred pts)")
        fill_table(self.h2h_a_table, h["only_a"], H2H_COLS)
        fill_table(self.h2h_b_table, h["only_b"], H2H_COLS)

    def _update_owners(self):
        data = self._data or {}
        squads = data.get("league_squads")
        if squads is None:
            return
        import league_extras
        text = self.owns_input.text().strip()
        if len(text) < 2:
            self._clear_table(self.owners_table)
            return
        owners = league_extras.player_owners(squads, text)
        if owners.empty:
            self._clear_table(self.owners_table)
            return
        fill_table(self.owners_table, owners, OWNERS_COLS)

    def _populate_league_extras(self, data: dict):
        proj = data.get("projection")
        if proj is not None and not proj.empty:
            fill_table(self.projection_table, proj, PROJECTION_COLS)
        else:
            self._clear_table(self.projection_table)

        rivals = data.get("rival_transfers")
        if rivals is not None and not rivals.empty:
            cols = [c for c in RIVAL_TRANSFER_COLS if c[0] in rivals.columns]
            fill_table(self.rival_transfers_table, rivals, cols or
                       [(c, c.replace("_", " ").title(), None) for c in rivals.columns])
        else:
            self._clear_table(self.rival_transfers_table)

        eo = data.get("eff_ownership")
        if eo is not None and not eo.empty:
            fill_table(self.eo_table, eo.head(40), EO_COLS)
        else:
            self._clear_table(self.eo_table)

        form = data.get("league_form")
        if form is not None and not form.empty:
            fill_table(self.form_table, form, FORM_COLS)
        else:
            self._clear_table(self.form_table)

        # captaincy edge shown above the existing captaincy table
        ci = data.get("captaincy_impact") or {}
        if ci and "error" not in ci:
            edge = ci.get("edge_vs_league", 0.0)
            verdict = ("ahead of" if edge > 0 else "behind" if edge < 0 else "level with")
            self.captaincy_note.setText(
                f"Your captain: {ci.get('my_captain')} ({ci.get('my_captain_pred'):.2f} pred). "
                f"League average captain: {ci.get('league_avg_captain_pred'):.2f}. "
                f"That puts you {abs(edge):.2f} pts {verdict} the field once doubled "
                f"— most popular pick is {ci.get('most_popular')}."
            )
        else:
            self.captaincy_note.setText("")

        # populate the head-to-head selectors once, defaulting to me vs the leader
        squads = data.get("league_squads")
        if squads is not None and not squads.empty and self.h2h_a.count() == 0:
            managers = (squads[["entry", "entry_name"]].drop_duplicates()
                        .sort_values("entry_name"))
            for cb in (self.h2h_a, self.h2h_b):
                cb.blockSignals(True)
                cb.clear()
                for _, r in managers.iterrows():
                    cb.addItem(str(r["entry_name"]), int(r["entry"]))
                cb.blockSignals(False)
            # recommend.load_squad's meta carries no entry id, so use the team
            # ID the user actually typed
            try:
                my_id = int(self.team_id_input.text())
            except (TypeError, ValueError):
                my_id = None
            if my_id is not None:
                idx = self.h2h_a.findData(my_id)
                if idx >= 0:
                    self.h2h_a.setCurrentIndex(idx)
            self.h2h_b.setCurrentIndex(1 if self.h2h_b.currentIndex() == 0 else 0)
            self._update_h2h()

        fun = data.get("fun_stats") or {}
        for key, table, cols in (
            ("bench_points", self.bench_points_table, BENCH_POINTS_COLS),
            ("value_growth", self.value_growth_table, VALUE_GROWTH_COLS),
            ("rank_swings", self.rank_swings_table, RANK_SWING_COLS),
            ("template_adherence", self.template_table, TEMPLATE_COLS),
        ):
            df = fun.get(key)
            if df is not None and not df.empty:
                fill_table(table, df, cols)
            else:
                self._clear_table(table)

    # ------------------------------------------------------------ chat ---

    @staticmethod
    def _md_to_html(raw: str) -> str:
        """Escape FIRST, then apply a small markdown transform over the escaped
        text. Never the other way round, and never trust a markdown library's
        output here -- Qt's rich-text parser is lenient and will happily render
        tags smuggled in through the model's reply.
        """
        esc = html.escape(raw)
        # fenced code first, so its contents aren't touched by the inline rules
        esc = re.sub(r"```(.*?)```", r"<pre>\1</pre>", esc, flags=re.S)
        esc = re.sub(r"`([^`]+)`", r"<code>\1</code>", esc)
        esc = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", esc)

        out = []
        in_list = False
        for ln in esc.split("\n"):
            if re.match(r"[ \t]*[-*][ \t]+", ln):
                if not in_list:
                    out.append("<ul>")
                    in_list = True
                out.append("<li>" + re.sub(r"^[ \t]*[-*][ \t]+", "", ln) + "</li>")
            else:
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append(ln + "<br>" if ln.strip() else "<br>")
        if in_list:
            out.append("</ul>")
        return "".join(out)

    def _chat_append(self, html_block: str) -> None:
        self.chat_log.append(html_block)
        # append() does NOT scroll once the user has scrolled up, so a long
        # answer can land entirely off-screen and look like nothing happened
        bar = self.chat_log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _send_chat(self):
        question = self.chat_input.text().strip()
        if not question:
            return
        if not self._data:
            self._chat_append("<i>No data yet -- wait for the first refresh to finish.</i>")
            return
        t = theme.current()
        self._chat_append(
            f"<div style='color:{t['text_dim']}'><b>You</b></div>"
            f"<div>{html.escape(question)}</div>"
        )
        self.chat_input.clear()
        self.chat_input.setEnabled(False)
        self.chat_send_btn.setEnabled(False)
        self.chat_send_btn.setText("Thinking...")
        self.chat_progress.setVisible(True)

        # animated ellipsis + elapsed seconds: with no progress signal from the
        # model, motion and elapsed time are the only honest feedback there is
        self._chat_elapsed = 0
        self._chat_dots = 0
        self._chat_append("<i>Thinking</i>")
        self._chat_timer = QTimer(self)
        self._chat_timer.setInterval(400)
        self._chat_timer.timeout.connect(self._tick_thinking)
        self._chat_timer.start()

        self._chat_thread, self._chat_worker = ask_llm(
            question, self._data, on_answer=self._on_chat_answer, on_error=self._on_chat_error
        )

    def _tick_thinking(self):
        self._chat_dots = (self._chat_dots + 1) % 4
        self._chat_elapsed += 0.4
        dots = "." * self._chat_dots
        secs = f" ({int(self._chat_elapsed)}s)" if self._chat_elapsed >= 3 else ""
        self._remove_thinking_line()
        self._chat_append(f"<i>Thinking{dots}{secs}</i>")

    def _stop_thinking(self):
        timer = getattr(self, "_chat_timer", None)
        if timer is not None:
            timer.stop()
            self._chat_timer = None
        self._remove_thinking_line()
        self.chat_progress.setVisible(False)
        self.chat_send_btn.setText("Ask")
        self.chat_input.setEnabled(True)
        self.chat_send_btn.setEnabled(True)

    def _remove_thinking_line(self):
        cursor = self.chat_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        self.chat_log.setTextCursor(cursor)

    def _on_chat_answer(self, answer: str):
        self._stop_thinking()
        t = theme.current()
        self._chat_append(
            f"<div style='color:{t['accent2']}'><b>Assistant</b></div>"
            f"<div>{self._md_to_html(answer)}</div>"
        )
        self.chat_input.setFocus()

    def _on_chat_error(self, msg: str):
        self._stop_thinking()
        t = theme.current()
        # themed critical, plus the literal word "Error" so it is not
        # communicated by color alone
        self._chat_append(
            f"<div style='color:{t['critical']}'><b>Error</b></div>"
            f"<div style='color:{t['critical']}'>{html.escape(msg)}</div>"
        )

    def _restyle_stat_cards(self):
        """Sizes/colors now come from the app sheet via the "class" property.
        Qt does not re-evaluate property selectors on its own, so each label
        must be unpolished and re-polished after the sheet changes, or the
        cards keep their old look until something else forces a repaint."""
        for c in (self.card_bank, self.card_value, self.card_sellable,
                  self.card_xi_pts, self.card_captain):
            for w in (c.value_label, c.label_widget):
                w.setStyleSheet("")
                w.style().unpolish(w)
                w.style().polish(w)
                w.update()

    def _on_theme_changed(self, name: str):
        theme.set_theme(name)
        self.setStyleSheet(theme.stylesheet())
        self._restyle_stat_cards()
        save_config(
            int(self.team_id_input.text() or self._config["team_id"]),
            int(self.event_input.text() or self._config["event"]),
            int(self.league_id_input.text()) if self.league_id_input.text().strip() else 0,
            name,
        )
        # redraw every currently-populated chart canvas so colors update immediately
        if self._data is not None:
            self._redraw_charts(self._data)
        else:
            for canvas in self.carousel.all_canvases():
                self._placeholder_chart(canvas, "Loading...")

    def _apply_player_filter(self):
        if not hasattr(self, "_all_players_df") or self._all_players_df is None:
            return
        df = self._all_players_df
        text = self.search_box.text().strip().lower()
        pos = self.position_filter.currentText()
        if pos != "All":
            df = df[df["position"] == pos]
        if text:
            df = df[
                df["web_name"].str.lower().str.contains(text, na=False)
                | df["name"].str.lower().str.contains(text, na=False)
            ]
        fill_table(self.all_players_table, df.head(300), ALL_PLAYERS_COLS, row_color_col="price_flag")

    def _send_chat(self):
        question = self.chat_input.text().strip()
        if not question:
            return
        if not self._data:
            self.chat_log.append("<i>No data yet -- wait for the first refresh to finish.</i>")
            return
        # escape: chat_log is a rich-text widget, so a raw '<' in the question
        # (or in the model's answer below) silently corrupts the document
        self.chat_log.append(f"<b>You:</b> {html.escape(question)}")
        self.chat_input.clear()
        self.chat_input.setEnabled(False)
        self.chat_send_btn.setEnabled(False)
        self.chat_log.append("<i>Thinking...</i>")
        self._chat_thread, self._chat_worker = ask_llm(
            question, self._data, on_answer=self._on_chat_answer, on_error=self._on_chat_error
        )

    def _remove_thinking_line(self):
        cursor = self.chat_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        self.chat_log.setTextCursor(cursor)

    def _on_chat_answer(self, answer: str):
        self._remove_thinking_line()
        safe = html.escape(answer).replace("\n", "<br>")
        self.chat_log.append(f"<b>Assistant:</b> {safe}")
        self.chat_input.setEnabled(True)
        self.chat_send_btn.setEnabled(True)
        self.chat_input.setFocus()

    def _on_chat_error(self, msg: str):
        self._remove_thinking_line()
        t = theme.current()
        self.chat_log.append(
            f"<b>Assistant:</b> <span style='color:{t['accent']}'>Error: {html.escape(msg)}</span>"
        )
        self.chat_input.setEnabled(True)
        self.chat_send_btn.setEnabled(True)
