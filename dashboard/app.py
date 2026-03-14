"""
dashboard/app.py
----------------
Streamlit research dashboard for UltimateDocResearcher.

Run with:
    streamlit run dashboard/app.py

Features:
  - Recent runs table with topic/date/score/status + download links
  - Metrics charts: avg_score over time, pass_rate trend
  - Run explorer: filter by topic, drill into per-iteration metrics
  - New run trigger: topic input → similarity check → launch research
  - Cache stats sidebar
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add project root to path so memory/ and autoresearch/ are importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory.memory import RunMemory, topic_similarity
from memory.cache import PromptCache

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UltimateDocResearcher",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared state ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_memory() -> RunMemory:
    return RunMemory(ROOT / "dashboard" / "runs.db")


@st.cache_resource
def get_cache() -> PromptCache:
    return PromptCache(ROOT / "dashboard" / "cache")


mem = get_memory()
cache = get_cache()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 UltimateDocResearcher")
    st.caption("Research dashboard")

    st.divider()
    stats = mem.stats()
    col1, col2 = st.columns(2)
    col1.metric("Total runs", stats.get("total_runs", 0))
    col2.metric("Completed", stats.get("completed", 0))

    avg = stats.get("avg_score")
    st.metric("Avg score", f"{avg:.3f}" if avg else "—")

    st.divider()
    cache_stats = cache.stats()
    st.caption(f"🗃 Cache: {cache_stats['total_entries']} entries · {cache_stats['total_hits']} hits")

    st.divider()
    page = st.radio(
        "View",
        ["Recent Runs", "Run Explorer", "Metrics", "New Run"],
        label_visibility="collapsed",
    )

# ── Helper ────────────────────────────────────────────────────────────────────

def _score_badge(score: float | None) -> str:
    if score is None:
        return "—"
    color = "🟢" if score >= 0.75 else "🟡" if score >= 0.5 else "🔴"
    return f"{color} {score:.3f}"


def _status_badge(status: str) -> str:
    return {"completed": "✅", "running": "⏳", "failed": "❌"}.get(status, status)


def _fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts or "—"


# ── Page: Recent Runs ─────────────────────────────────────────────────────────

if page == "Recent Runs":
    st.header("Recent Runs")

    runs = mem.recent_runs(limit=50)

    if not runs:
        st.info("No runs yet. Use the 'New Run' tab to start your first research cycle.")
    else:
        # Build display table
        rows = []
        for r in runs:
            rows.append({
                "ID": r["id"],
                "Topic": r["topic"],
                "Date": _fmt_ts(r["timestamp"]),
                "Status": _status_badge(r["status"]),
                "Score": _score_badge(r.get("avg_score")),
                "Pass rate": f"{r['pass_rate']:.0%}" if r.get("pass_rate") else "—",
                "Eval": f"{r['weighted_eval']:.2f}" if r.get("weighted_eval") else "—",
                "Iters": r.get("iterations") or "—",
            })

        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download buttons for selected run
        st.subheader("Download outputs")
        selected_id = st.selectbox("Select run ID", [r["id"] for r in runs])
        selected = next((r for r in runs if r["id"] == selected_id), None)

        if selected:
            col1, col2, col3 = st.columns(3)
            for col, key, label in [
                (col1, "suggestions_path", "📄 Code suggestions"),
                (col2, "eval_path",        "📊 Eval report"),
                (col3, "results_path",     "📈 results.tsv"),
            ]:
                path = selected.get(key)
                if path and Path(ROOT / path).exists():
                    content = Path(ROOT / path).read_bytes()
                    col.download_button(label, content, file_name=Path(path).name)
                else:
                    col.button(label, disabled=True)


# ── Page: Run Explorer ────────────────────────────────────────────────────────

elif page == "Run Explorer":
    st.header("Run Explorer")

    runs = mem.recent_runs(limit=100)
    if not runs:
        st.info("No runs yet.")
    else:
        topics = ["All"] + sorted({r["topic"] for r in runs})
        filter_topic = st.selectbox("Filter by topic", topics)
        filter_status = st.multiselect(
            "Status", ["completed", "running", "failed"],
            default=["completed", "running"]
        )

        filtered = [
            r for r in runs
            if (filter_topic == "All" or r["topic"] == filter_topic)
            and r["status"] in filter_status
        ]

        if not filtered:
            st.info("No matching runs.")
        else:
            selected_run = st.selectbox(
                "Select run",
                filtered,
                format_func=lambda r: f"#{r['id']} — {r['topic'][:50]} ({_fmt_ts(r['timestamp'])})"
            )

            if selected_run:
                st.subheader(f"Run #{selected_run['id']}: {selected_run['topic']}")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Status",    _status_badge(selected_run["status"]))
                col2.metric("Avg score", f"{selected_run['avg_score']:.3f}" if selected_run.get("avg_score") else "—")
                col3.metric("Pass rate", f"{selected_run['pass_rate']:.0%}" if selected_run.get("pass_rate") else "—")
                col4.metric("Eval score", f"{selected_run['weighted_eval']:.2f}" if selected_run.get("weighted_eval") else "—")

                # Per-iteration metrics
                metrics = mem.get_metrics(selected_run["id"])
                if metrics:
                    import pandas as pd
                    df = pd.DataFrame(metrics)[
                        ["iteration", "val_score", "train_loss", "val_loss",
                         "judge_avg_score", "judge_pass_rate"]
                    ]
                    st.subheader("Iteration metrics")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    if "val_score" in df.columns:
                        st.line_chart(df.set_index("iteration")["val_score"])
                else:
                    st.caption("No per-iteration metrics logged for this run.")

                # Show notes / file paths
                if selected_run.get("notes"):
                    st.info(selected_run["notes"])


# ── Page: Metrics ─────────────────────────────────────────────────────────────

elif page == "Metrics":
    st.header("Metrics overview")

    runs = mem.recent_runs(limit=100)
    completed = [r for r in runs if r["status"] == "completed"]

    if not completed:
        st.info("No completed runs yet.")
    else:
        try:
            import pandas as pd

            df = pd.DataFrame(completed)
            df["date"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("date")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Avg score over time")
                chart_data = df[df["avg_score"].notna()].set_index("date")["avg_score"]
                if not chart_data.empty:
                    st.line_chart(chart_data)
                else:
                    st.caption("No score data yet.")

            with col2:
                st.subheader("Pass rate over time")
                chart_data = df[df["pass_rate"].notna()].set_index("date")["pass_rate"]
                if not chart_data.empty:
                    st.line_chart(chart_data)
                else:
                    st.caption("No pass rate data yet.")

            st.subheader("Weighted eval score over time")
            chart_data = df[df["weighted_eval"].notna()].set_index("date")["weighted_eval"]
            if not chart_data.empty:
                st.line_chart(chart_data)

            st.subheader("Score by topic (top 10)")
            topic_avg = (
                df[df["avg_score"].notna()]
                .groupby("topic")["avg_score"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            if not topic_avg.empty:
                st.bar_chart(topic_avg)

        except ImportError:
            st.error("pandas is required for the Metrics view. Run: pip install pandas")


# ── Page: New Run ─────────────────────────────────────────────────────────────

elif page == "New Run":
    st.header("Start a new research run")

    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Claude tool use patterns for agentic workflows",
    )

    col1, col2, col3 = st.columns(3)
    judge = col1.selectbox("Judge model", ["ollama:llama3.2", "claude-3-5-haiku-20241022", "gpt-4o-mini"])
    iterations = col2.number_input("Iterations", min_value=1, max_value=20, value=3)
    threshold = col3.number_input("Similarity threshold", min_value=0.5, max_value=1.0, value=0.8, step=0.05)

    if topic:
        # Check for similar past runs
        similar = mem.find_similar(topic, threshold=threshold)
        if similar:
            st.warning(
                f"⚠️ Found {len(similar)} similar past run(s). "
                f"Most similar: **'{similar[0]['topic']}'** "
                f"(similarity={similar[0]['similarity']:.0%}, "
                f"score={similar[0].get('avg_score') or '?'})"
            )
            if st.checkbox("Show similar runs"):
                for s in similar[:5]:
                    st.markdown(
                        f"- **#{s['id']}** {s['topic']} — "
                        f"score={_score_badge(s.get('avg_score'))} — "
                        f"{_fmt_ts(s['timestamp'])}"
                    )
            proceed = st.radio(
                "What would you like to do?",
                ["Start new run anyway", "Skip — use cached results"],
                index=1,
            )
            if proceed == "Skip — use cached results":
                best = similar[0]
                st.success(f"Using results from run #{best['id']}: {best['topic']}")
                if best.get("suggestions_path") and Path(ROOT / best["suggestions_path"]).exists():
                    content = Path(ROOT / best["suggestions_path"]).read_text(encoding="utf-8")
                    st.code(content, language="markdown")
                st.stop()
        else:
            st.success("✅ No similar runs found — this is a fresh topic.")

        if st.button("🚀 Start research", type="primary"):
            with st.spinner("Launching research pipeline…"):
                run_id = mem.start_run(topic, judge_model=judge)
                st.info(f"Run #{run_id} started. Check the terminal for live output.")

                # Launch as subprocess so the dashboard stays responsive
                cmd = [
                    sys.executable, "-m", "autoresearch.train",
                    "--topic", topic,
                    "--iterations", str(iterations),
                    "--judge-model", judge,
                    "--run-id", str(run_id),   # train.py reads this to log to DB
                ]
                st.code(" ".join(cmd), language="bash")
                st.caption("Copy the command above to run in your terminal, or run it from a separate process.")

                # Store run_id in session state so user can track it
                st.session_state["last_run_id"] = run_id
                st.success(f"Run #{run_id} registered. Refresh 'Recent Runs' to track progress.")
