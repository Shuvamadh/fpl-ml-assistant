"""Local LLM (Ollama + Qwen) as a targeted assistant layer on top of the
numeric pipeline -- NOT a replacement for the GBDT points model (see
README for why: this is tabular regression, trees win; the LLM's job here
is natural-language reasoning over the numeric outputs, and reading the
free-text `news` field the model can't use directly).
"""
import json

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"


def is_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def chat(prompt: str, system: str | None = None, think: bool = False, timeout: int = 120) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "stream": False, "think": think},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


ASSISTANT_SYSTEM_PROMPT = (
    "You are a concise Fantasy Premier League (FPL) assistant embedded in a "
    "desktop app. You are given real, freshly-pulled data about the user's "
    "squad, model-predicted points, transfer suggestions, and mini-league "
    "context as JSON/table context below the question. Answer using ONLY "
    "that context plus general FPL rules knowledge. Be direct and specific "
    "(name players, numbers). Keep answers under 150 words unless asked for "
    "detail. If the context doesn't contain what's needed to answer, say so "
    "rather than guessing."
)


def build_context_summary(data: dict) -> str:
    """Compact text summary of the app's current refreshed data, small enough
    to fit comfortably in the model's context window alongside a question."""
    parts = []
    meta = data.get("meta", {})
    parts.append(f"Team: {meta.get('team_name')}, bank £{meta.get('bank', 0):.1f}m, "
                 f"squad value £{meta.get('value', 0):.1f}m, gameweek {meta.get('event', '?') }.")

    xi = data.get("xi")
    if xi is not None and not xi.empty:
        lines = [f"{row['web_name']} ({row['position']}, {row['name']}, £{row['now_cost_m']:.1f}m, "
                 f"pred {row['pred_points_adj']:.1f}, next: {row.get('next_fixture', '?')})"
                 for _, row in xi.iterrows()]
        parts.append("Starting XI: " + "; ".join(lines))

    bench = data.get("bench")
    if bench is not None and not bench.empty:
        parts.append("Bench: " + ", ".join(bench["web_name"].tolist()))

    transfers = data.get("transfers")
    if transfers is not None and not transfers.empty:
        top = transfers.head(5)
        lines = [f"{row['out']} -> {row['in']} (+{row['pred_gain']:.1f} pred pts, "
                 f"leftover £{row['leftover_bank']:.1f}m)"
                 for _, row in top.iterrows()]
        parts.append("Top transfer suggestions: " + "; ".join(lines))

    standings = data.get("standings")
    if standings is not None and not standings.empty:
        top5 = standings.sort_values("rank").head(5)
        lines = [f"{r.rank}. {r.entry_name} ({r.player_name}) - {r.total} pts" for r in top5.itertuples()]
        parts.append(f"Mini-league top 5: " + "; ".join(lines))
        mine = standings[standings["entry_name"] == meta.get("team_name")]
        if not mine.empty:
            m = mine.iloc[0]
            parts.append(f"Your league rank: {m['rank']} of {len(standings)}, {m['total']} pts.")

    insights = data.get("insights")
    if insights:
        if insights.get("differentials") is not None and not insights["differentials"].empty:
            diffs = ", ".join(insights["differentials"]["web_name"].tolist())
            parts.append(f"Your differentials (low ownership in league): {diffs}")
        if insights.get("captains") is not None and not insights["captains"].empty:
            caps = insights["captains"].sort_values("captained_by_n", ascending=False).head(3)
            parts.append("Most-captained in league: " + ", ".join(
                f"{r.web_name} ({r.captained_by_n})" for r in caps.itertuples()
            ))

    return "\n".join(parts)


def ask(question: str, data: dict) -> str:
    context = build_context_summary(data)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    return chat(prompt, system=ASSISTANT_SYSTEM_PROMPT, think=False)


def summarise_news(player_name: str, news_text: str) -> str:
    if not news_text or not isinstance(news_text, str):
        return ""
    prompt = (
        f"FPL player {player_name} has this official status note: \"{news_text}\". "
        "In under 15 words, say what it means for their chance of playing next gameweek."
    )
    return chat(prompt, think=False, timeout=30)


if __name__ == "__main__":
    print("Ollama available:", is_available())
    if is_available():
        print(chat("Say hello in exactly five words.", think=False))
