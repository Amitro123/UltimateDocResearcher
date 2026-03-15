"""
chat/app.py
-----------
Research Chat interface for UltimateDocResearcher.

Run with:
    streamlit run chat/app.py

Features:
  - Topic input + PDF/ZIP upload + URL list
  - Memory check: warns if a similar run already exists
  - Live progress via st.status (each pipeline step shown in real time)
  - Result tabs: Code Suggestions | Q&A Pairs | Logs | Dashboard link
  - "Start New Research" rerun button to reset after a completed run
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) ─────────────────────────────────

st.set_page_config(
    page_title="Research Chat",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ─────────────────────────────────────────────────────────

defaults = {
    "pipeline_result": None,
    "pipeline_logs": [],
    "confirm_mode": False,   # waiting for user to confirm running despite similar run
    "pending": {},           # form inputs saved while confirm_mode is active
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_badge(score) -> str:
    if score is None:
        return "—"
    color = "🟢" if score >= 0.75 else "🟡" if score >= 0.5 else "🔴"
    return f"{color} {score:.2f}"


def _get_memory_stats() -> dict:
    try:
        from memory.memory import RunMemory
        mem = RunMemory(ROOT / "dashboard" / "runs.db")
        return mem.stats()
    except Exception:
        return {}


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Research Chat")
    st.caption("UltimateDocResearcher")
    st.divider()

    stats = _get_memory_stats()
    if stats:
        c1, c2 = st.columns(2)
        c1.metric("Total runs", stats.get("total_runs", 0))
        c2.metric("Completed", stats.get("completed", 0))
        avg = stats.get("avg_score")
        st.metric("Avg score", f"{avg:.3f}" if avg else "—")
    else:
        st.caption("No run history yet")

    st.divider()
    st.markdown("**Dashboard**")
    st.code("streamlit run dashboard/app.py\n--server.port 8502", language="bash")
    st.caption("Run in a separate terminal to view charts and run history.")


# ── Main header ────────────────────────────────────────────────────────────────

st.header("🔬 Research Pipeline")
st.caption(
    "Enter a topic, optionally upload PDFs, add URLs — then run the full pipeline: "
    "collect → analyze → Q&A → code suggestions."
)

# ── Rerun button (shown when a result exists) ──────────────────────────────────

if st.session_state["pipeline_result"]:
    if st.button("🔄 Start New Research", type="secondary"):
        st.session_state["pipeline_result"] = None
        st.session_state["pipeline_logs"] = []
        st.session_state["confirm_mode"] = False
        st.session_state["pending"] = {}
        st.rerun()
    st.divider()


# ── Input form (hidden after a result is ready) ───────────────────────────────

if not st.session_state["pipeline_result"]:

    with st.form("research_form", clear_on_submit=False):
        topic = st.text_input(
            "Research topic",
            placeholder="e.g. LLM evaluation frameworks and best practices",
            help="Be specific — this drives both collection queries and suggestion focus.",
        )

        col_left, col_right = st.columns(2)

        with col_left:
            uploaded_files = st.file_uploader(
                "Upload PDFs / ZIPs",
                type=["pdf", "zip"],
                accept_multiple_files=True,
                help="Papers or zipped document collections to include in the corpus.",
            )

        with col_right:
            urls_text = st.text_area(
                "URLs (one per line)",
                placeholder=(
                    "https://arxiv.org/abs/2404.01234\n"
                    "https://github.com/owner/repo\n"
                    "https://some-blog.com/relevant-post"
                ),
                height=120,
                help="Web pages, arXiv links, or GitHub repos. GitHub URLs are auto-detected.",
            )

        col_a, col_b, col_c = st.columns(3)
        n_suggestions = col_a.slider("Code suggestions", 3, 8, 5)
        sim_threshold = col_b.slider(
            "Similar-run threshold", 0.5, 1.0, 0.75, 0.05,
            help="Warn if a past run is at least this similar.",
        )

        submitted = st.form_submit_button(
            "🚀 Run Research Pipeline",
            type="primary",
            use_container_width=True,
        )

    # Save inputs to session state on submit so confirm flow can access them
    if submitted and topic:
        st.session_state["pending"] = {
            "topic": topic,
            "uploaded_files": uploaded_files,
            "urls_text": urls_text,
            "n_suggestions": n_suggestions,
            "sim_threshold": sim_threshold,
        }

        # Memory check
        from chat.chat_handler import find_similar_runs
        similar = find_similar_runs(topic, threshold=sim_threshold)
        if similar:
            st.session_state["confirm_mode"] = True
            st.session_state["similar_runs"] = similar
        else:
            st.session_state["confirm_mode"] = False
            st.session_state["similar_runs"] = []


# ── Confirm dialog (similar run found) ────────────────────────────────────────

if st.session_state.get("confirm_mode") and not st.session_state["pipeline_result"]:
    similar = st.session_state.get("similar_runs", [])
    best = similar[0]

    st.warning(
        f"⚠️ Found {len(similar)} similar past run(s). "
        f"Most similar: **'{best['topic']}'** "
        f"({best.get('similarity', 0):.0%} match · "
        f"score {_score_badge(best.get('avg_score'))})"
    )

    if st.checkbox("Show similar runs", key="show_similar"):
        for s in similar[:5]:
            st.markdown(
                f"- **#{s['id']}** {s['topic']} — "
                f"score {_score_badge(s.get('avg_score'))} — "
                f"{s.get('timestamp', '')[:16]}"
            )

    col_yes, col_no = st.columns(2)
    run_anyway = col_yes.button("🔄 Run fresh pipeline", type="primary", use_container_width=True)
    use_cached = col_no.button("📂 Use cached results", use_container_width=True)

    if use_cached:
        suggestions_path = best.get("suggestions_path")
        if suggestions_path and (ROOT / suggestions_path).exists():
            content = (ROOT / suggestions_path).read_text(encoding="utf-8")
            st.session_state["pipeline_result"] = {
                "topic": best["topic"],
                "code_suggestions": content,
                "qa_pairs": [],
                "_from_cache": True,
            }
            st.session_state["pipeline_logs"] = ["Loaded from cached run #" + str(best["id"])]
            st.session_state["confirm_mode"] = False
            st.rerun()
        else:
            st.error("Cached results not found on disk. Run fresh pipeline instead.")

    if not run_anyway:
        st.stop()

    # User clicked "Run fresh" — fall through to pipeline below
    st.session_state["confirm_mode"] = False


# ── Run pipeline ───────────────────────────────────────────────────────────────

pending = st.session_state.get("pending", {})
should_run = (
    pending.get("topic")
    and not st.session_state["confirm_mode"]
    and not st.session_state["pipeline_result"]
    and (
        # submitted fresh (no similar found) OR user just clicked "run anyway"
        submitted or st.session_state.get("_run_triggered")
    )
)

# Also trigger if user just clicked "Run fresh" in confirm dialog
if not should_run and pending.get("topic") and not st.session_state["confirm_mode"] and not st.session_state["pipeline_result"]:
    # Check if we just exited confirm mode (run_anyway was clicked this frame)
    if "run_anyway" in st.session_state and st.session_state.get("run_anyway"):
        should_run = True
        st.session_state["run_anyway"] = False

if should_run and pending.get("topic"):
    from chat.chat_handler import run_pipeline

    # Parse URLs
    raw_urls = [u.strip() for u in pending.get("urls_text", "").splitlines() if u.strip()]
    github_repos = [
        u.replace("https://github.com/", "").replace("http://github.com/", "")
        for u in raw_urls if "github.com" in u
    ]
    extra_urls = [u for u in raw_urls if "github.com" not in u]

    logs: list[str] = []

    with st.status("Running research pipeline…", expanded=True) as status_box:
        try:
            for event_type, payload in run_pipeline(
                topic=pending["topic"],
                pdf_files=pending.get("uploaded_files") or None,
                extra_urls=extra_urls or None,
                github_repos=github_repos or None,
                n_suggestions=pending.get("n_suggestions", 5),
            ):
                if event_type == "status":
                    status_box.update(label=payload)
                    st.write(f"**{payload}**")
                elif event_type == "log":
                    st.write(f"`{payload}`")
                    logs.append(payload)
                elif event_type == "done":
                    st.write(f"✅ {payload}")
                elif event_type == "error":
                    st.error(payload)
                    status_box.update(label=f"❌ {payload}", state="error")
                    logs.append(f"ERROR: {payload}")
                    break
                elif event_type == "result":
                    st.session_state["pipeline_result"] = payload
                    st.session_state["pipeline_logs"] = logs

            if st.session_state["pipeline_result"]:
                status_box.update(
                    label="✅ Pipeline complete!", state="complete", expanded=False
                )

        except Exception as exc:
            status_box.update(label=f"❌ {exc}", state="error")
            st.error(f"Pipeline error: {exc}")
            logs.append(f"FATAL: {exc}")
            st.session_state["pipeline_logs"] = logs

    st.session_state["pending"] = {}
    st.rerun()


# ── Results tabs ───────────────────────────────────────────────────────────────

if st.session_state["pipeline_result"]:
    result = st.session_state["pipeline_result"]
    from_cache = result.get("_from_cache", False)

    st.subheader(f"Results: {result['topic']}")
    if from_cache:
        st.info("ℹ️ Showing cached results from a previous run.")

    tab_code, tab_qa, tab_logs, tab_dash = st.tabs(
        ["💡 Code Suggestions", "❓ Q&A Pairs", "📋 Logs", "🔗 Dashboard"]
    )

    with tab_code:
        suggestions = result.get("code_suggestions", "")
        if suggestions:
            st.markdown(suggestions)
            st.download_button(
                "⬇️ Download code_suggestions.md",
                data=suggestions,
                file_name="code_suggestions.md",
                mime="text/markdown",
            )
        else:
            st.info("No code suggestions were generated.")

    with tab_qa:
        qa_pairs = result.get("qa_pairs", [])
        if qa_pairs:
            st.caption(f"{len(qa_pairs)} Q&A pairs shown (first 20)")
            for i, pair in enumerate(qa_pairs, 1):
                q = pair.get("question") or pair.get("prompt") or f"Pair {i}"
                a = pair.get("answer") or pair.get("completion") or "—"
                with st.expander(f"Q{i}: {q[:90]}{'…' if len(q) > 90 else ''}"):
                    st.markdown(f"**Q:** {q}")
                    st.markdown(f"**A:** {a}")
        else:
            st.info("No Q&A pairs available.")

    with tab_logs:
        logs = st.session_state.get("pipeline_logs", [])
        if logs:
            st.code("\n".join(logs), language="text")
        else:
            st.info("No logs captured.")

    with tab_dash:
        st.markdown(
            "The full dashboard shows run history, score trends, and iteration metrics."
        )
        st.code("streamlit run dashboard/app.py --server.port 8502", language="bash")
        st.link_button("Open Dashboard (localhost:8502)", "http://localhost:8502")
