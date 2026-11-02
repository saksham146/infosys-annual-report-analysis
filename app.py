import streamlit as st

from src.rag_pipeline import (
    COLLECTION_NAME,
    DB_PATH,
    get_api_key,
    get_collection,
    ask_annual_report_stream,
)

st.set_page_config(
    page_title="Infosys Report Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { color-scheme: dark; }
    .stApp { background: radial-gradient(ellipse at 50% 0%, #132538 0%, #080d16 56%); color: #e8eef7; font-family: Manrope, sans-serif; }
    [data-testid="stSidebar"] { background: rgba(10, 18, 30, .96); border-right: 1px solid rgba(86, 176, 255, .16); }
    .agent-hero { padding: 25px 30px; margin: 8px 0 22px; border: 1px solid rgba(88, 182, 255, .28); border-radius: 20px; background: linear-gradient(130deg, rgba(23, 45, 69, .92), rgba(15, 23, 42, .82)); box-shadow: 0 18px 55px rgba(0,0,0,.25); }
    .agent-kicker { color: #6ee7c0; font: 500 11px 'DM Mono', monospace; letter-spacing: 1.8px; text-transform: uppercase; }
    .agent-hero h1 { margin: 9px 0 5px; font-size: clamp(26px, 4vw, 42px); letter-spacing: -1.5px; color: #f2f7ff; }
    .agent-hero p { margin: 0; color: #a9bbd0; }
    div[data-testid="stChatMessage"] { border: 1px solid rgba(151, 181, 218, .12); border-radius: 16px; background: rgba(18, 30, 48, .72); }
    div[data-testid="stChatInput"] { border: 1px solid rgba(86, 176, 255, .35); border-radius: 16px; background: rgba(12, 22, 37, .94); }
    .source-note { color: #9bb4d1; font: 12px 'DM Mono', monospace; padding-top: 7px; }
    </style>
    <section class="agent-hero">
      <div class="agent-kicker">● Document intelligence · Infosys FY 2023–24</div>
      <h1>Infosys Report Analyst</h1>
      <p>Ask a question. The agent searches the report, checks relevant passages, and streams an answer with page references.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Agent controls")
    top_k = st.slider("Evidence passages", 3, 12, 5, help="How many report passages to retrieve before answering.")
    st.divider()
    try:
        collection = get_collection()
        passage_count = collection.count()
        db_ready = passage_count > 0
    except Exception:
        passage_count = 0
        db_ready = False
    key_ready = bool(get_api_key())
    st.markdown("### System status")
    st.write(f"{'🟢' if db_ready else '🔴'} Report index · {passage_count:,} passages" if db_ready else "🔴 Report index · unavailable")
    st.write("🟢 Gemini key · configured" if key_ready else "🟠 Gemini key · missing")
    st.caption(f"Index path: {DB_PATH.name} · {COLLECTION_NAME}")
    st.divider()
    st.markdown("### Try a question")
    prompts = [
        "How did revenue and net profit change year over year?",
        "What risks did the auditors highlight?",
        "What were employee count and attrition?",
    ]
    for index, suggestion in enumerate(prompts):
        if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            st.session_state.pending_prompt = suggestion
            st.rerun()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown(f"<div class='source-note'>📄 Report pages: {', '.join(map(str, message['sources']))}</div>", unsafe_allow_html=True)

prompt = st.chat_input("Ask about financial, people, strategy, or risk disclosures…")
if not prompt:
    prompt = st.session_state.pop("pending_prompt", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer_text = ""
    source_pages = []
    with st.chat_message("assistant"):
        try:
            with st.status("Agent is reviewing the report…", expanded=True) as status:
                st.write("🔎 Searching the indexed report passages")
                answer_stream, source_pages = ask_annual_report_stream(prompt, top_k=top_k)
                st.write(f"🧾 Evidence found on report pages: {', '.join(map(str, source_pages)) or 'none'}")
                st.write("✍️ Drafting a cited answer")
                answer_text = st.write_stream(answer_stream)
                status.update(label="Analysis complete", state="complete", expanded=False)
            if source_pages:
                st.markdown(f"<div class='source-note'>📄 Report pages: {', '.join(map(str, source_pages))}</div>", unsafe_allow_html=True)
        except Exception as exc:
            # Keep provider details and any credentials out of the chat transcript.
            st.error("The agent could not complete this request. Check the API key, report index, and network connection, then try again.")
            st.caption(f"Diagnostic: {type(exc).__name__}")
            answer_text = "I couldn't complete that analysis. Please check the agent status in the sidebar and try again."
    st.session_state.messages.append({"role": "assistant", "content": answer_text, "sources": source_pages})
