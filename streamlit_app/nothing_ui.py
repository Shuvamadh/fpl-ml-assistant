"""Nothing OS-inspired design system for the Streamlit frontend.

Design language, from Nothing's own products:
  - Monochrome base. Colour is information, never decoration. Exactly one
    accent, used only where something needs a decision or a warning.
  - Display type for headers: Cinzel Decorative from Google Fonts -- an
    ornate gothic/deco display face in the register the user asked for
    ("Batman Forever"-style), properly licensed for web redistribution
    (the actual movie logotype is a fan-recreated font with no such
    license). Body copy stays a clean grotesk/mono pair so the gothic
    face reads as a deliberate accent, not the whole page.
  - Heavy negative space and a visible grid. Low clutter.
  - Squares with generously rounded corners, hairline borders, flat fills.
  - Transparency as a motif: layered panels, no drop shadows.

Colourways mirror what Nothing actually ships: the hardware is Black or White,
and the accent is their red. Extra accents are offered because the user asked
for choice, but red is the default and the brand-true one.

Everything here is presentation only -- no data access, no FPL logic. Charts
come from src/charts_core.py; this module supplies the palette they're drawn
with so the figures match the page.
"""

# ---------------------------------------------------------------- palettes ---

THEMES = {
    "Black": {
        "bg": "#000000",
        "panel": "#0A0A0A",
        "panel_2": "#141414",
        "border": "#2A2A2A",
        "border_bright": "#3D3D3D",
        "text": "#FFFFFF",
        "text_dim": "#8A8A8A",
        "text_faint": "#5A5A5A",
        "grid": "#1C1C1C",
    },
    "White": {
        "bg": "#FFFFFF",
        "panel": "#FAFAFA",
        "panel_2": "#F2F2F2",
        "border": "#E0E0E0",
        "border_bright": "#C8C8C8",
        "text": "#000000",
        "text_dim": "#6A6A6A",
        "text_faint": "#9A9A9A",
        "grid": "#EEEEEE",
    },
    # Grey/Olive mirror CMF by Nothing's hardware colourways (Watch Pro 2 /
    # Buds ship in Light Grey and Light Green) rather than Nothing's own
    # Black/White phones -- kept as light themes like White, just tinted,
    # so text contrast stays solid instead of trying to run dark text on a
    # literal mid-grey/olive panel.
    "Grey": {
        "bg": "#C9C7C0",
        "panel": "#D6D4CD",
        "panel_2": "#DEDCD5",
        "border": "#B0AEA6",
        "border_bright": "#98968E",
        "text": "#141414",
        "text_dim": "#5A5854",
        "text_faint": "#89877F",
        "grid": "#BDBBB3",
    },
    "Olive": {
        "bg": "#DDE7A8",
        "panel": "#E4EDB8",
        "panel_2": "#EAF1C6",
        "border": "#C4D189",
        "border_bright": "#AABE68",
        "text": "#20260F",
        "text_dim": "#4E5A2A",
        "text_faint": "#7C8956",
        "grid": "#CBD696",
    },
}

# Nothing red is the default and the only brand-true one. The rest exist
# because the user asked for options; each is a single flat hue, never a
# gradient, to stay inside the design language. Cyan and Orange extend the
# same idea into CMF by Nothing's own accent range.
ACCENTS = {
    "Nothing Red": "#D71921",
    "Glyph White": "#F5F5F5",
    "Signal Yellow": "#FFE600",
    "Terminal Green": "#00E676",
    "Deep Blue": "#2979FF",
    "Glyph Cyan": "#00E1FF",
    "CMF Orange": "#FF6B00",
}

# Semantic colours are kept separate from the accent so that changing the
# accent never makes "good" and "bad" ambiguous.
GOOD = "#00C853"
BAD = "#FF3B30"
WARN = "#FFB300"


def palette(theme_name: str = "Black", accent_name: str = "Nothing Red") -> dict:
    """Full token set, including the keys src/charts_core.py expects, so a
    chart drawn with this palette matches the page it sits on."""
    t = THEMES.get(theme_name, THEMES["Black"])
    accent = ACCENTS.get(accent_name, ACCENTS["Nothing Red"])
    return {
        **t,
        "accent": accent,
        "accent2": t["text"],      # charts_core's secondary = high-contrast mono
        "good": GOOD,
        "bad": BAD,
        "warn": WARN,
        "series_1": t["text_dim"],
        "series_mute": t["border_bright"],
    }


def chart_palette(theme_name: str = "Black", accent_name: str = "Nothing Red") -> dict:
    """Palette handed to charts_core. Panel is transparent so figures sit on
    the page background instead of punching a coloured rectangle into it."""
    p = palette(theme_name, accent_name)
    return {**p, "panel": "none"}


# --------------------------------------------------------------------- css ---

def css(theme_name: str = "Black", accent_name: str = "Nothing Red") -> str:
    p = palette(theme_name, accent_name)
    a = p["accent"]
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700;900&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {{
  --n-bg: {p['bg']};
  --n-panel: {p['panel']};
  --n-panel-2: {p['panel_2']};
  --n-border: {p['border']};
  --n-border-bright: {p['border_bright']};
  --n-text: {p['text']};
  --n-dim: {p['text_dim']};
  --n-faint: {p['text_faint']};
  --n-grid: {p['grid']};
  --n-accent: {a};
  --n-good: {GOOD};
  --n-bad: {BAD};
  --n-warn: {WARN};
}}

.stApp {{ background: var(--n-bg); }}
html, body, [class*="css"] {{
  font-family: 'Inter', -apple-system, sans-serif;
  color: var(--n-text);
}}

/* Dot-matrix display type, reserved for headers -- as on Nothing OS, where
   NDot is used for titles rather than body copy. */
.n-dot {{
  font-family: 'Cinzel Decorative', serif; font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--n-text);
}}

.n-h1 {{ font-size: 30px; line-height: 1.1; margin: 0; }}
.n-h2 {{ font-size: 15px; color: var(--n-dim); margin: 0 0 10px 0; }}

/* Section rule: label, then a hairline running to the edge. The grid should
   be visible, not implied. */
.n-rule {{
  display: flex; align-items: center; gap: 12px;
  margin: 26px 0 12px 0;
}}
.n-rule .line {{ flex: 1; height: 1px; background: var(--n-border); }}
.n-rule .lbl {{
  font-family: 'Cinzel Decorative', serif; font-weight: 700;
  font-size: 13px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--n-dim); white-space: nowrap;
}}

/* Metric tiles: flat fill, hairline border, generous rounding -- the quick
   settings tile shape from Nothing OS 2.5 onward. */
.n-tiles {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.n-tile {{
  flex: 1 1 130px; min-width: 130px;
  background: var(--n-panel); border: 1px solid var(--n-border);
  border-radius: 18px; padding: 14px 16px;
}}
.n-tile .k {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--n-faint); margin-bottom: 8px;
}}
.n-tile .v {{
  font-family: 'Cinzel Decorative', serif; font-weight: 700;
  font-size: 19px; line-height: 1.15; color: var(--n-text);
  white-space: nowrap;
}}
.n-tile .s {{ font-size: 10.5px; color: var(--n-dim); margin-top: 6px; }}
.n-tile.accent {{ border-color: var(--n-accent); }}
.n-tile.accent .v {{ color: var(--n-accent); }}

/* Decision cards: the app's whole point is a recommendation, so it gets a
   card, not a table row. */
.n-card {{
  background: var(--n-panel); border: 1px solid var(--n-border);
  border-radius: 20px; padding: 16px 18px; margin-bottom: 8px;
}}
.n-card.hot {{ border-color: var(--n-accent); }}
.n-card .lbl {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--n-faint);
}}
.n-card .big {{
  font-family: 'Cinzel Decorative', serif; font-weight: 700; font-size: 21px;
  margin: 7px 0 3px 0; color: var(--n-text);
}}
.n-card .sub {{ font-size: 12px; color: var(--n-dim); line-height: 1.5; }}

/* Live indicator -- the Glyph reference. Pulses, doesn't blink. */
.n-live {{
  display: inline-flex; align-items: center; gap: 7px;
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--n-accent);
}}
.n-live .dot {{
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--n-accent); animation: nglow 1.9s ease-in-out infinite;
}}
@keyframes nglow {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.28; }} }}

/* Pills */
.n-pill {{
  display: inline-block; padding: 3px 10px; border-radius: 100px;
  font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
  letter-spacing: 0.1em; text-transform: uppercase;
  border: 1px solid var(--n-border-bright); color: var(--n-dim);
}}
.n-pill.good {{ border-color: var(--n-good); color: var(--n-good); }}
.n-pill.bad  {{ border-color: var(--n-bad);  color: var(--n-bad); }}
.n-pill.warn {{ border-color: var(--n-warn); color: var(--n-warn); }}

/* ---- Pitch view ---------------------------------------------------------
   The desktop app has a real pitch (gui/pitch_view.py) but it's a Qt widget
   and can't cross over, so this is a CSS rebuild. Flexbox with wrap rather
   than st.columns: st.columns collapses to ONE stacked column below
   Streamlit's mobile breakpoint no matter how many are requested, which turns
   a pitch into a vertical list of giant cards on a phone. Flex reflows. */
.n-pitch {{
  background:
    repeating-linear-gradient(0deg, var(--n-grid) 0 1px, transparent 1px 46px),
    repeating-linear-gradient(90deg, var(--n-grid) 0 1px, transparent 1px 46px),
    var(--n-panel);
  border: 1px solid var(--n-border); border-radius: 22px;
  padding: 20px 10px; position: relative;
}}
.n-pitch .halfway {{
  position: absolute; left: 8%; right: 8%; top: 50%;
  height: 1px; background: var(--n-border);
}}
.n-pitch .circle {{
  position: absolute; left: 50%; top: 50%;
  width: 88px; height: 88px; margin: -44px 0 0 -44px;
  border: 1px solid var(--n-border); border-radius: 50%;
}}
.n-row {{
  display: flex; justify-content: center; flex-wrap: wrap;
  gap: 4px; margin: 12px 0; position: relative; z-index: 2;
}}
.n-player {{ flex: 0 0 78px; text-align: center; }}
.n-player .shirt {{ width: 46px; height: auto; }}
.n-player .nm {{
  font-family: 'JetBrains Mono', monospace; font-size: 10.5px; font-weight: 500;
  margin-top: 4px; color: var(--n-text);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.n-player .mt {{ font-size: 9px; color: var(--n-faint); margin-top: 1px; }}
.n-player .pts {{
  font-family: 'Cinzel Decorative', serif; font-weight: 700; font-size: 13px;
  color: var(--n-text); margin-top: 2px;
}}
.n-player .arm {{
  display: inline-block; background: var(--n-accent); color: #fff;
  font-family: 'JetBrains Mono', monospace; font-size: 8px; font-weight: 700;
  padding: 1px 4px; border-radius: 3px; margin-left: 3px;
}}
.n-player .arm.v {{ background: var(--n-border-bright); color: var(--n-text); }}
.n-player .risk {{
  width: 5px; height: 5px; border-radius: 50%;
  display: inline-block; margin-right: 3px; vertical-align: middle;
}}

/* Streamlit chrome, pulled toward the design language */
[data-testid="stSidebar"] {{
  background: var(--n-panel); border-right: 1px solid var(--n-border);
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid var(--n-border); }}
.stTabs [data-baseweb="tab"] {{
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--n-dim); background: transparent; border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
  color: var(--n-text); border-bottom: 2px solid var(--n-accent);
}}
.stButton > button {{
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  letter-spacing: 0.1em; text-transform: uppercase;
  background: transparent; color: var(--n-text);
  border: 1px solid var(--n-border-bright); border-radius: 100px;
}}
.stButton > button:hover {{ border-color: var(--n-accent); color: var(--n-accent); }}
#MainMenu, footer {{ visibility: hidden; }}

@media (max-width: 640px) {{
  .n-h1 {{ font-size: 22px; }}
  .n-tile {{ flex: 1 1 104px; min-width: 104px; padding: 11px 12px; }}
  .n-tile .v {{ font-size: 16px; }}
  .n-player {{ flex: 0 0 62px; }}
  .n-player .shirt {{ width: 36px; }}
}}
</style>
"""


# ----------------------------------------------------------- html builders ---
# All of these return single-line HTML with no leading whitespace on any line.
# Markdown treats 4+ spaces of indentation as a code block, which silently
# renders indented HTML as literal text -- it breaks everything after the
# first element rather than failing loudly.

def rule(label: str) -> str:
    return f'<div class="n-rule"><span class="lbl">{label}</span><span class="line"></span></div>'


def tile(key: str, value, sub: str = "", accent: bool = False) -> str:
    cls = "n-tile accent" if accent else "n-tile"
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    return f'<div class="{cls}"><div class="k">{key}</div><div class="v">{value}</div>{sub_html}</div>'


def tiles(items) -> str:
    return f'<div class="n-tiles">{"".join(items)}</div>'


def card(label: str, big: str, sub: str = "", hot: bool = False) -> str:
    cls = "n-card hot" if hot else "n-card"
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="{cls}"><div class="lbl">{label}</div><div class="big">{big}</div>{sub_html}</div>'


def live_badge(text: str = "Live") -> str:
    return f'<span class="n-live"><span class="dot"></span>{text}</span>'


def pill(text: str, kind: str = "") -> str:
    return f'<span class="n-pill {kind}">{text}</span>'


def risk_dot(start_prob) -> str:
    """Rotation risk as a colour dot. start_probability is the minutes
    classifier's P(minutes>=60) at AUC 0.95 -- a far better rotation signal
    than raw past minutes, and it was buried in a table before."""
    try:
        p = float(start_prob)
    except (TypeError, ValueError):
        return f'<span class="risk" style="background:{THEMES["Black"]["border_bright"]}"></span>'
    colour = GOOD if p >= 0.9 else (WARN if p >= 0.5 else BAD)
    return f'<span class="risk" style="background:{colour}"></span>'


def player_chip(name: str, meta: str, pts, shirt_b64: str | None = None,
                is_captain: bool = False, is_vice: bool = False,
                start_prob=None) -> str:
    shirt = (f'<img class="shirt" src="data:image/png;base64,{shirt_b64}">'
             if shirt_b64 else '<div style="height:46px"></div>')
    arm = ""
    if is_captain:
        arm = '<span class="arm">C</span>'
    elif is_vice:
        arm = '<span class="arm v">V</span>'
    try:
        pts_s = f"{float(pts):.1f}"
    except (TypeError, ValueError):
        pts_s = "-"
    return (f'<div class="n-player">{shirt}'
            f'<div class="nm">{risk_dot(start_prob)}{name}{arm}</div>'
            f'<div class="mt">{meta}</div>'
            f'<div class="pts">{pts_s}</div></div>')


def pitch(rows) -> str:
    """rows: list of lists of player_chip() strings, back line first."""
    body = "".join(f'<div class="n-row">{"".join(r)}</div>' for r in rows)
    return (f'<div class="n-pitch"><div class="halfway"></div>'
            f'<div class="circle"></div>{body}</div>')
