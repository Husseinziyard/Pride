"""
Pride — AI-Powered Data Science Assistant
"""

import os
import glob
import uuid
import streamlit as st
import pandas as pd
from agent import create_agent, get_agent_response
from tools import set_dataset, get_dataset

# ── Page Config ──
st.set_page_config(
    page_title="Pride",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="auto",
)

# ── Dark Claude-like CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* ── Dark theme globals ── */
    :root {
        --bg-main: #1a1a1a;
        --bg-chat: #262626;
        --bg-sidebar: #141414;
        --bg-input: #333333;
        --bg-hover: #2a2a2a;
        --text-primary: #e8e8e8;
        --text-secondary: #a0a0a0;
        --text-muted: #707070;
        --accent: #d4a574;
        --accent-soft: rgba(212, 165, 116, 0.1);
        --border: #333333;
        --border-light: #3a3a3a;
    }

    /* Force dark background everywhere */
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }

    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        background-color: var(--bg-main) !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: var(--bg-main) !important;
        border-bottom: none !important;
    }

    /* ── Sidebar — dark like Claude ── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text-secondary) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary) !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 16px 0 20px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 16px;
    }
    .sidebar-brand-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, var(--accent), #c4915a);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }
    .sidebar-brand-text {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary) !important;
    }

    .sidebar-new-chat {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: var(--text-secondary) !important;
        padding: 8px 12px;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.15s;
        margin-bottom: 4px;
    }
    .sidebar-new-chat:hover {
        background: var(--bg-hover);
    }

    .history-item {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        color: var(--text-muted) !important;
        padding: 6px 12px;
        border-radius: 6px;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .history-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 12px 12px 4px 12px;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 8px 12px !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--bg-hover) !important;
        border-color: var(--border-light) !important;
    }

    /* ── Main chat area ── */
    .chat-container {
        max-width: 720px;
        margin: 0 auto;
        padding: 0 20px;
    }

    /* Header in main area */
    .main-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 16px 0 12px 0;
    }
    .dataset-pill {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 3px 10px;
        border-radius: 6px;
    }
    .dataset-pill.loaded {
        background: rgba(74, 222, 128, 0.1);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.2);
    }
    .dataset-pill.empty {
        background: rgba(251, 191, 36, 0.1);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.2);
    }

    /* ── Welcome screen ── */
    .welcome-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 100px 24px 40px 24px;
        text-align: center;
    }
    .welcome-logo {
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, var(--accent), #c4915a);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin-bottom: 20px;
    }
    .welcome-h {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 8px 0;
    }
    .welcome-p {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        color: var(--text-muted);
        margin: 0 0 32px 0;
        max-width: 420px;
    }

    /* ── Suggestion buttons ── */
    .sug-grid .stButton > button {
        background: var(--bg-chat) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        width: 100% !important;
        text-align: left !important;
        transition: all 0.15s ease !important;
    }
    .sug-grid .stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: var(--accent-soft) !important;
    }

    /* ── Chat messages dark theme ── */
    .stChatMessage {
        font-family: 'DM Sans', sans-serif !important;
        max-width: 720px;
        margin: 0 auto;
        background: transparent !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] span {
        color: var(--text-primary) !important;
    }
    [data-testid="stChatMessageContent"] strong {
        color: var(--accent) !important;
    }
    [data-testid="stChatMessageContent"] code {
        background: var(--bg-input) !important;
        color: var(--accent) !important;
        border-radius: 4px;
        padding: 1px 5px;
    }
    [data-testid="stChatMessageContent"] pre {
        background: var(--bg-sidebar) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* ── Chat input dark ── */
    .stChatInput {
        max-width: 720px;
        margin: 0 auto;
    }
    .stChatInput > div {
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
        background: var(--bg-input) !important;
        box-shadow: none !important;
    }
    .stChatInput > div:focus-within {
        border-color: var(--accent) !important;
    }
    .stChatInput textarea {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        color: var(--text-primary) !important;
        background: transparent !important;
    }
    .stChatInput textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* ── File uploader dark ── */
    [data-testid="stFileUploader"] {
        max-width: 400px;
        margin: 0 auto;
    }
    [data-testid="stFileUploader"] > div {
        background: var(--bg-chat) !important;
        border: 1px dashed var(--border-light) !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p {
        color: var(--text-secondary) !important;
    }
    [data-testid="stFileUploader"] button {
        background: var(--bg-input) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }

    /* Warning messages */
    .stAlert {
        max-width: 720px;
        margin: 0 auto;
    }

    /* Spinner */
    .stSpinner > div > div {
        border-top-color: var(--accent) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-main);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 3px;
    }

    /* Sidebar collapse/expand button */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        color: var(--text-secondary) !important;
    }
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {
        color: var(--text-secondary) !important;
        background: transparent !important;
    }
    [data-testid="collapsedControl"] {
        background: var(--bg-main) !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""
if "dataset_shape" not in st.session_state:
    st.session_state.dataset_shape = ""
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "agent_error" not in st.session_state:
    st.session_state.agent_error = ""
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"id": str, "title": str, "messages": list}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
# FIX 1 — stable thread ID that persists for the whole session
if "session_thread_id" not in st.session_state:
    st.session_state.session_thread_id = str(uuid.uuid4())[:8]
# FIX 2 — store the DataFrame in session state so it survives Streamlit reruns
if "stored_df" not in st.session_state:
    st.session_state.stored_df = None


# ── Auto-initialize agent ──
if not st.session_state.agent_ready and not st.session_state.agent_error:
    try:
        st.session_state.agent = create_agent()
        st.session_state.agent_ready = True
    except Exception as e:
        st.session_state.agent_error = str(e)

# FIX 3 — Re-inject the dataset into the tool store on every rerun
# Streamlit reruns wipe in-memory globals; this restores the df transparently
if st.session_state.dataset_loaded and st.session_state.stored_df is not None:
    if get_dataset() is None:
        set_dataset(st.session_state.stored_df)


# ── Helper: save current chat to history ──
def save_current_chat():
    if st.session_state.messages and len(st.session_state.messages) > 0:
        # Get title from first user message
        first_msg = next(
            (m["content"][:50] for m in st.session_state.messages if m["role"] == "user"),
            "New chat"
        )
        if st.session_state.active_chat_id:
            # Update existing
            for chat in st.session_state.chat_history:
                if chat["id"] == st.session_state.active_chat_id:
                    chat["messages"] = st.session_state.messages.copy()
                    chat["title"] = first_msg
                    return
        # Create new
        chat_id = str(uuid.uuid4())[:8]
        st.session_state.chat_history.insert(0, {
            "id": chat_id,
            "title": first_msg,
            "messages": st.session_state.messages.copy(),
        })
        st.session_state.active_chat_id = chat_id


# ── Sidebar ──
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">⚡</div>
        <span class="sidebar-brand-text">Pride</span>
    </div>
    """, unsafe_allow_html=True)

    # New chat button
    if st.button("＋  New chat", use_container_width=True, key="new_chat"):
        save_current_chat()
        st.session_state.messages = []
        st.session_state.active_chat_id = None
        st.session_state.dataset_loaded = False
        st.session_state.dataset_name = ""
        st.session_state.dataset_shape = ""
        st.session_state.stored_df = None
        # FIX 4 — reset thread ID on new chat so memory is fresh
        st.session_state.session_thread_id = str(uuid.uuid4())[:8]
        st.rerun()

    # Chat history
    if st.session_state.chat_history:
        st.markdown('<div class="history-label">Recent</div>', unsafe_allow_html=True)
        for chat in st.session_state.chat_history[:15]:
            is_active = chat["id"] == st.session_state.active_chat_id
            label = chat["title"]
            if len(label) > 35:
                label = label[:35] + "..."
            if st.button(
                f"{'● ' if is_active else ''}{label}",
                key=f"hist_{chat['id']}",
                use_container_width=True,
            ):
                save_current_chat()
                st.session_state.messages = chat["messages"].copy()
                st.session_state.active_chat_id = chat["id"]
                st.rerun()

    # Bottom section
    st.markdown("---")

    # Dataset info
    if st.session_state.dataset_loaded:
        st.markdown(f"📄 **{st.session_state.dataset_name}**")
        st.markdown(f"`{st.session_state.dataset_shape}`")
    else:
        st.markdown("No dataset loaded")



# ── Main area header ──
if st.session_state.dataset_loaded:
    st.markdown(f"""
    <div class="main-header">
        <span class="dataset-pill loaded">📄 {st.session_state.dataset_name} · {st.session_state.dataset_shape}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Show error if agent failed ──
if st.session_state.agent_error:
    st.markdown(f"""
    <div style="max-width:500px; margin:80px auto; padding:32px; text-align:center;
         background:#2a1a1a; border:1px solid #5a2a2a; border-radius:12px;
         font-family:'DM Sans',sans-serif; color:#ff8888;">
        <h3 style="color:#ff8888;">⚠️ Setup Required</h3>
        <p style="color:#cc8888;">{st.session_state.agent_error}</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Chat History Display ──
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "⚡"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "charts" in message:
            for chart_path in message["charts"]:
                if os.path.exists(chart_path):
                    st.image(chart_path, use_container_width=True)


# ── Process message ──
def process_message(user_input: str):
    # FIX 5 — always ensure the dataset is available in the tool store
    if st.session_state.dataset_loaded and st.session_state.stored_df is not None:
        if get_dataset() is None:
            set_dataset(st.session_state.stored_df)

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Thinking..."):
            try:
                existing_charts = set(glob.glob("chart_*.png"))

                response = get_agent_response(
                    st.session_state.agent,
                    user_input,
                    # FIX 6 — use stable session thread ID instead of active_chat_id
                    thread_id=st.session_state.session_thread_id,
                )

                st.markdown(response)

                new_charts = sorted(set(glob.glob("chart_*.png")) - existing_charts)
                for chart_path in new_charts:
                    st.image(chart_path, use_container_width=True)

                msg = {"role": "assistant", "content": response}
                if new_charts:
                    msg["charts"] = new_charts
                st.session_state.messages.append(msg)

                # Auto-save to history
                save_current_chat()

            except Exception as e:
                error_msg = f"Something went wrong: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ── Welcome screen ──
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-logo">⚡</div>
        <div class="welcome-h">What can I help you analyze?</div>
        <div class="welcome-p">
            Upload a CSV dataset, then ask me anything or click a suggestion to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # File upload centered in chat area
    if not st.session_state.dataset_loaded:
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                label_visibility="collapsed",
                key="welcome_upload",
            )
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    set_dataset(df)
                    # FIX 7 — persist the DataFrame in session state
                    st.session_state.stored_df = df
                    st.session_state.dataset_loaded = True
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.dataset_shape = f"{df.shape[0]} rows × {df.shape[1]} cols"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reading file: {e}")

    # Suggestion buttons — only show when dataset is loaded
    if st.session_state.dataset_loaded:
        suggestions = [
            ("📊  Overview of the dataset", "Give me a complete overview of the dataset"),
            ("🧹  Clean and prepare data", "Check the data for issues and clean it — handle missing values, duplicates, and wrong types"),
            ("📈  Correlation heatmap", "Create a correlation heatmap for all numeric columns"),
            ("🤖  Build a prediction model", "Use dataset_overview to understand the dataset, then pick the most suitable target column yourself and call build_model with model_type='auto'. Show the results and explain what they mean."),
        ]

        col_l, col_c, col_r = st.columns([1, 3, 1])
        with col_c:
            st.markdown('<div class="sug-grid">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            for i, (label, prompt) in enumerate(suggestions):
                with (c1 if i % 2 == 0 else c2):
                    if st.button(label, key=f"sug_{i}", use_container_width=True):
                        st.session_state.pending_action = prompt
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ── Handle pending action ──
if st.session_state.pending_action:
    action = st.session_state.pending_action
    st.session_state.pending_action = None
    process_message(action)
    st.rerun()


# ── Chat input ──
user_input = st.chat_input("Message Pride...")

if user_input:
    if not st.session_state.dataset_loaded:
        st.warning("Upload a CSV dataset first.")
    else:
        process_message(user_input)
        st.rerun()