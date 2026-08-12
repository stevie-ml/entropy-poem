"""Server-side usage tracking for s2lab.

Counts real Streamlit sessions in SQLite. No third-party service, no JavaScript,
so ad blockers cannot suppress it. Client details come from the request headers
Streamlit already has, and the visitor id is a salted hash, so no raw IP is stored.

Set S2LAB_ANALYTICS_DB to the database path. On Render this should live on a
mounted disk, otherwise the file is wiped on every deploy.
"""

import hashlib
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import streamlit as st

DEFAULT_DB = "/var/data/analytics.db"
FALLBACK_DB = "/tmp/analytics.db"

# Stable across restarts so a returning visitor hashes to the same id. Override in
# the environment if you ever want to rotate and orphan the old hashes.
SALT = os.environ.get("S2LAB_ANALYTICS_SALT", "s2lab-v1")

BOT_PATTERN = re.compile(
    r"bot|crawl|spider|slurp|scrape|curl|wget|python-requests|httpx|okhttp|"
    r"headless|phantom|puppeteer|playwright|lighthouse|monitor|uptime|preview|"
    r"facebookexternalhit|bingpreview|feedfetcher|duckduckbot|semrush|ahrefs",
    re.I,
)

_init_lock = threading.Lock()
_initialised = False


def _db_path():
    configured = os.environ.get("S2LAB_ANALYTICS_DB")
    if configured:
        return configured
    parent = os.path.dirname(DEFAULT_DB)
    if os.path.isdir(parent) and os.access(parent, os.W_OK):
        return DEFAULT_DB
    return FALLBACK_DB


def storage_is_persistent():
    """False when we fell back to ephemeral disk, i.e. counts reset on deploy."""
    return _db_path() not in (FALLBACK_DB,)


@contextmanager
def _connect():
    conn = sqlite3.connect(_db_path(), timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    global _initialised
    if _initialised:
        return
    with _init_lock:
        if _initialised:
            return
        with _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts         TEXT NOT NULL,
                    day        TEXT NOT NULL,
                    visitor    TEXT,
                    referrer   TEXT,
                    user_agent TEXT,
                    language   TEXT,
                    is_bot     INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS events (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts      TEXT NOT NULL,
                    day     TEXT NOT NULL,
                    visitor TEXT,
                    name    TEXT NOT NULL,
                    detail  TEXT,
                    is_bot  INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_day ON sessions(day);
                CREATE INDEX IF NOT EXISTS idx_events_day   ON events(day);
                """
            )
        _initialised = True


def _headers():
    try:
        return dict(st.context.headers or {})
    except Exception:
        # Older Streamlit, or no request context (e.g. `streamlit run` at import time).
        return {}


def _client():
    h = {k.lower(): v for k, v in _headers().items()}
    forwarded = h.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else h.get("x-real-ip", "")
    ua = h.get("user-agent", "")
    referrer = h.get("referer") or h.get("referrer") or ""
    visitor = hashlib.sha256(f"{SALT}|{ip}|{ua}".encode()).hexdigest()[:16] if (ip or ua) else None
    return visitor, referrer[:500], ua[:300], bool(ua and BOT_PATTERN.search(ua))


def track_session(language=None):
    """Record one visit. Safe to call on every rerun; only the first counts."""
    try:
        if st.session_state.get("_analytics_logged"):
            return
        st.session_state["_analytics_logged"] = True

        init()
        visitor, referrer, ua, is_bot = _client()
        st.session_state["_analytics_visitor"] = visitor
        st.session_state["_analytics_is_bot"] = is_bot
        now = datetime.now(timezone.utc)
        with _connect() as conn:
            conn.execute(
                "INSERT INTO sessions (ts, day, visitor, referrer, user_agent, language, is_bot)"
                " VALUES (?,?,?,?,?,?,?)",
                (now.isoformat(), now.strftime("%Y-%m-%d"), visitor, referrer, ua,
                 language, int(is_bot)),
            )
    except Exception:
        # Analytics must never take the app down.
        pass


def track_event(name, detail=None):
    try:
        init()
        now = datetime.now(timezone.utc)
        with _connect() as conn:
            conn.execute(
                "INSERT INTO events (ts, day, visitor, name, detail, is_bot) VALUES (?,?,?,?,?,?)",
                (now.isoformat(), now.strftime("%Y-%m-%d"),
                 st.session_state.get("_analytics_visitor"), name,
                 (detail or "")[:200], int(bool(st.session_state.get("_analytics_is_bot")))),
            )
    except Exception:
        pass


def stats(days=30, include_bots=False):
    """Return a summary dict for the last `days` days."""
    init()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    bot_clause = "" if include_bots else " AND is_bot = 0"
    with _connect() as conn:
        row = conn.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT visitor) FROM sessions"
            f" WHERE day >= ?{bot_clause}", (since,)
        ).fetchone()
        by_day = conn.execute(
            f"SELECT day, COUNT(*), COUNT(DISTINCT visitor) FROM sessions"
            f" WHERE day >= ?{bot_clause} GROUP BY day ORDER BY day DESC", (since,)
        ).fetchall()
        referrers = conn.execute(
            f"SELECT CASE WHEN referrer = '' THEN '(direct)' ELSE referrer END, COUNT(*)"
            f" FROM sessions WHERE day >= ?{bot_clause}"
            f" GROUP BY 1 ORDER BY 2 DESC LIMIT 15", (since,)
        ).fetchall()
        events = conn.execute(
            f"SELECT name, COUNT(*) FROM events WHERE day >= ?{bot_clause}"
            f" GROUP BY name ORDER BY 2 DESC", (since,)
        ).fetchall()
        langs = conn.execute(
            f"SELECT COALESCE(language,'?'), COUNT(*) FROM sessions WHERE day >= ?{bot_clause}"
            f" GROUP BY 1 ORDER BY 2 DESC", (since,)
        ).fetchall()
        bots = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE day >= ? AND is_bot = 1", (since,)
        ).fetchone()[0]
    return {
        "days": days,
        "sessions": row[0],
        "visitors": row[1],
        "bot_sessions": bots,
        "by_day": by_day,
        "referrers": referrers,
        "events": events,
        "languages": langs,
        "db_path": _db_path(),
        "persistent": storage_is_persistent(),
    }


def render_dashboard():
    """Render the stats page. Returns True if it handled the request.

    Gated on ?stats=<S2LAB_ANALYTICS_TOKEN>. Without the env var set, the
    dashboard is unreachable.
    """
    token = os.environ.get("S2LAB_ANALYTICS_TOKEN")
    if not token:
        return False
    try:
        supplied = st.query_params.get("stats")
    except Exception:
        return False
    if supplied != token:
        return False

    import pandas as pd

    st.title("s2lab usage")
    days = st.selectbox("Window", [7, 30, 90, 365], index=1, format_func=lambda d: f"last {d} days")
    show_bots = st.checkbox("Include bots", value=False)
    s = stats(days=days, include_bots=show_bots)

    if not s["persistent"]:
        st.warning(
            f"Storage is ephemeral ({s['db_path']}). Counts reset on every deploy. "
            "Attach a Render disk and set S2LAB_ANALYTICS_DB to a path on it."
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Sessions", f"{s['sessions']:,}")
    c2.metric("Unique visitors", f"{s['visitors']:,}")
    c3.metric("Bot sessions filtered", f"{s['bot_sessions']:,}")

    if s["by_day"]:
        df = pd.DataFrame(s["by_day"], columns=["day", "sessions", "visitors"])
        st.subheader("Per day")
        st.line_chart(df.set_index("day").sort_index())
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("No sessions recorded yet in this window.")

    if s["referrers"]:
        st.subheader("Referrers")
        st.dataframe(pd.DataFrame(s["referrers"], columns=["referrer", "sessions"]),
                     width="stretch", hide_index=True)
    if s["events"]:
        st.subheader("Feature use")
        st.dataframe(pd.DataFrame(s["events"], columns=["event", "count"]),
                     width="stretch", hide_index=True)
    if s["languages"]:
        st.subheader("Language")
        st.dataframe(pd.DataFrame(s["languages"], columns=["language", "sessions"]),
                     width="stretch", hide_index=True)
    return True
