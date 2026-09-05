"""Multiple selectable themes. Charts read colors from `current()` at DRAW
time (not import time) so switching themes and redrawing updates everything."""

THEMES = {
    "FPL Purple": {
        "bg": "#1c0524", "bg2": "#37003c", "bg3": "#4b0a52", "panel": "#2b0730",
        "accent": "#e90052", "accent2": "#00ff87", "text": "#f2eef3",
        "text_dim": "#b9a9bd", "border": "#54155c",
    },
    "Midnight Slate": {
        "bg": "#0f1115", "bg2": "#1a1d24", "bg3": "#252a34", "panel": "#181b22",
        "accent": "#3b82f6", "accent2": "#22d3ee", "text": "#e6e8ec",
        "text_dim": "#8b93a3", "border": "#2e3340",
    },
    "Forest": {
        "bg": "#0d1b12", "bg2": "#132a1c", "bg3": "#1c3a26", "panel": "#112015",
        "accent": "#f59e0b", "accent2": "#4ade80", "text": "#eaf5ee",
        "text_dim": "#8fae9a", "border": "#274a32",
    },
    "Ocean": {
        "bg": "#071a2c", "bg2": "#0c2842", "bg3": "#123a5e", "panel": "#0a2238",
        "accent": "#f97316", "accent2": "#38bdf8", "text": "#eaf2fb",
        "text_dim": "#7fa0c0", "border": "#1c4a72",
    },
    "Light": {
        "bg": "#f4f5f7", "bg2": "#ffffff", "bg3": "#e9ebf0", "panel": "#ffffff",
        "accent": "#e90052", "accent2": "#059669", "text": "#1a1a2e",
        "text_dim": "#5b5f6b", "border": "#d8dae0",
    },
    # Default per GUI_PLAN 6.1: neutral surfaces, brand accents kept for
    # CHROME ONLY. Chart series come from series_* -- never from accent.
    "Graphite": {
        "bg": "#0f1116", "bg2": "#171a26", "bg3": "#1e2230", "panel": "#171a26",
        "accent": "#e0407a", "accent2": "#00d977", "text": "#e8eaf0",
        "text_dim": "#9aa3b5", "border": "#272c3a",
        "text_mute": "#6b7488", "focus": "#3987e5",
        "series_1": "#3987e5", "series_2": "#d95926", "series_3": "#199e70",
        "series_4": "#c98500", "series_mute": "#3f4657",
        "div_pos": "#3987e5", "div_neg": "#e0603c", "div_mid": "#6b7280",
        "good": "#199e70", "warn": "#c98500", "bad": "#d95926",
        "critical": "#e34948",
    },
}

# GUI_PLAN 6.1: the extended keys must exist on EVERY theme or stylesheet()
# and the charts blow up when a user switches back to an older theme. Fill any
# theme that predates them, rather than gating every lookup with .get().
_EXTRA_DEFAULTS = {
    "text_mute": "text_dim", "focus": "series_1",
    "series_1": "#3987e5", "series_2": "#d95926", "series_3": "#199e70",
    "series_4": "#c98500", "series_mute": "border",
    "div_pos": "#3987e5", "div_neg": "#e0603c", "div_mid": "#6b7280",
    "good": "#199e70", "warn": "#c98500", "bad": "#d95926",
    "critical": "#e34948",
}
for _name, _t in THEMES.items():
    for _k, _v in _EXTRA_DEFAULTS.items():
        if _k not in _t:
            _t[_k] = _t[_v] if _v in _t else _v

_current_name = "Graphite"


def shade(hex_color: str, factor: float) -> str:
    """Lighten (factor>1) or darken (factor<1) a #rrggbb color.

    Qt stylesheets have no `opacity` property -- a rule like
    `opacity: 0.9` is silently dropped and the widget gets no hover
    feedback at all -- so hover/pressed states need a real computed color.
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, min(255, round(c * factor))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def names() -> list[str]:
    return list(THEMES.keys())


def set_theme(name: str):
    global _current_name
    if name in THEMES:
        _current_name = name


def current() -> dict:
    return THEMES[_current_name]


def current_name() -> str:
    return _current_name


def stylesheet(name: str | None = None) -> str:
    """App-wide sheet. Sizes are in pt (not px) so Windows DPI scaling is
    respected -- GUI_PLAN 6.3."""
    t = THEMES[name] if name else current()
    inset = t["bg"]          # inputs sit INSET (darker) on a raised panel
    row_line = shade(t["border"], 0.75)
    return f"""
QMainWindow, QWidget {{
    background-color: {t['bg']};
    color: {t['text']};
    font-family: 'Segoe UI Variable Text', 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 10pt;
}}

QLabel#Header {{ font-size: 17pt; font-weight: 700; color: {t['text']}; }}
QLabel#SubHeader {{ color: {t['text_dim']}; font-size: 9pt; }}
QLabel#SectionTitle {{ font-size: 13pt; font-weight: 600; color: {t['text']}; }}

QFrame#StatCard {{
    background-color: {t['panel']};
    border: 1px solid {t['border']};
    border-radius: 10px;
}}
QFrame#StatCard:hover {{ border-color: {t['series_mute']}; }}
QLabel[class="StatCardValue"] {{ font-size: 22pt; font-weight: 600; color: {t['text']}; }}
QLabel[class="StatCardValueSmall"] {{ font-size: 14pt; font-weight: 600; color: {t['text']}; }}
QLabel[class="StatCardLabel"] {{ font-size: 8.5pt; font-weight: 600; color: {t['text_mute']}; }}

QPushButton {{
    background-color: {t['accent']};
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 9px 18px;
    font-size: 10pt;
    font-weight: 600;
}}
QPushButton:hover {{ background-color: {shade(t['accent'], 1.15)}; }}
QPushButton:pressed {{ background-color: {shade(t['accent'], 0.85)}; padding-top: 10px; padding-bottom: 8px; }}
QPushButton:disabled {{ background-color: {t['border']}; color: {t['text_mute']}; }}
QPushButton:focus {{ outline: none; border: 2px solid {t['focus']}; padding: 7px 16px; }}

QPushButton#Secondary {{
    background-color: transparent;
    border: 1px solid {t['border']};
    color: {t['text']};
}}
QPushButton#Secondary:hover {{ background-color: {t['bg3']}; }}
QPushButton#Secondary:pressed {{ background-color: {t['border']}; }}

QLineEdit, QComboBox {{
    background-color: {inset};
    border: 1px solid {t['border']};
    border-radius: 8px;
    padding: 8px 12px;
    color: {t['text']};
    font-size: 10pt;
    selection-background-color: {t['focus']};
}}
QLineEdit:hover, QComboBox:hover {{ border-color: {t['series_mute']}; }}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {t['focus']}; background-color: {t['panel']}; }}
QLineEdit:disabled {{ background-color: {shade(t['bg'], 0.9)}; color: {t['text_mute']}; border-color: {t['bg3']}; }}
QComboBox QAbstractItemView {{
    background-color: {t['panel']};
    color: {t['text']};
    border: 1px solid {t['border']};
    selection-background-color: {t['focus']};
}}
QCheckBox {{ color: {t['text_dim']}; font-size: 9pt; }}

QTabWidget::pane {{
    border: 1px solid {t['border']};
    border-radius: 10px;
    top: -1px;
    background-color: {t['panel']};
}}
QTabBar::tab {{
    background: transparent;
    color: {t['text_dim']};
    padding: 11px 16px;
    margin-right: 2px;
    font-size: 10pt;
    font-weight: 600;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{
    color: {t['text']};
    background-color: {t['panel']};
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{ color: {t['text']}; border-bottom: 2px solid {t['accent2']}; }}

/* Vertical gridlines are killed in code via setShowGrid(False); the row
   separator below is the only rule left. See table_utils.style_table. */
QTableWidget {{
    background-color: {t['panel']};
    border: none;
    selection-background-color: transparent;
    font-size: 10pt;
}}
QTableWidget::item {{
    border-bottom: 1px solid {row_line};
    padding: 0px 12px;
    color: {t['text']};
}}
QTableWidget::item:hover {{ background-color: {t['bg3']}; }}
QTableWidget::item:selected {{ background-color: {t['bg3']}; color: {t['text']}; }}
QHeaderView::section {{
    background-color: {t['bg2']};
    color: {t['text_dim']};
    font-size: 9pt;
    font-weight: 600;
    padding: 10px 12px;
    border: none;
    border-bottom: 1px solid {t['border']};
}}
QTableCornerButton::section {{ background-color: {t['bg2']}; border: none; }}

QProgressBar {{
    background-color: {t['panel']};
    border: none;
    border-radius: 4px;
    height: 4px;
    text-align: center;
    color: {t['text_dim']};
    font-size: 9pt;
}}
QProgressBar::chunk {{ background-color: {t['accent2']}; border-radius: 4px; }}

QTextEdit, QTextBrowser {{
    background-color: {t['panel']};
    border: 1px solid {t['border']};
    border-radius: 10px;
    padding: 8px;
    color: {t['text']};
    selection-background-color: {t['focus']};
}}

QToolTip {{
    background-color: {t['bg2']};
    color: {t['text']};
    border: 1px solid {t['border']};
    padding: 6px 8px;
}}

QStatusBar {{ color: {t['text_dim']}; }}

/* add-line/sub-line/add-page are REQUIRED -- without them Qt falls back to
   native arrow buttons and a gray trough that ignore the handle styling. */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 5px; min-height: 32px; }}
QScrollBar::handle:vertical:hover {{ background: {t['series_mute']}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {t['border']}; border-radius: 5px; min-width: 32px; }}
QScrollBar::handle:horizontal:hover {{ background: {t['series_mute']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""


# --- backwards-compatible module-level constants (some older code imports these directly) ---
def __getattr__(name):
    mapping = {
        "PURPLE_DARK": "bg", "PURPLE": "bg2", "PURPLE_MID": "bg3", "PANEL": "panel",
        "PINK": "accent", "GREEN": "accent2", "TEXT": "text", "TEXT_DIM": "text_dim",
        "BORDER": "border", "STYLESHEET": None,
    }
    if name == "STYLESHEET":
        return stylesheet()
    if name in mapping:
        return current()[mapping[name]]
    raise AttributeError(name)
