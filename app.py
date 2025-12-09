# app.py (updated styles: wider middle column + refined bubbles)
import streamlit as st
import importlib
import io
import json
from contextlib import redirect_stdout

# import your original module (back.py)
import back

st.set_page_config(layout="wide", page_title="ICD Chatbot — UI")

# -----------------------
# session init (safe)
# -----------------------
if "chat_list" not in st.session_state:
    st.session_state.chat_list = ["Chat 1"]

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"

if "conversations" not in st.session_state:
    st.session_state.conversations = {"Chat 1": []}

if "debug_text" not in st.session_state:
    st.session_state.debug_text = ""

# input buffer only (don't bind to widget key)
if "input_buffer" not in st.session_state:
    st.session_state.input_buffer = ""

# -----------------------
# layout columns (middle wider)
# -----------------------
col1, col2, col3 = st.columns([1, 3, 1])  # middle column more wide

# =========================================
# LEFT COLUMN — NAVIGATION
# =========================================
with col1:
    st.header("Chats")

    if st.button("New Chat", key="new_chat_btn"):
        name = f"Chat {len(st.session_state.chat_list) + 1}"
        st.session_state.chat_list.append(name)
        st.session_state.current_chat = name
        st.session_state.conversations[name] = []
        st.session_state.debug_text = ""
        st.session_state.input_buffer = ""

    st.markdown("---")
    st.subheader("Open chats")
    for idx, c in enumerate(st.session_state.chat_list):
        if st.button(c, key=f"chat_btn_{idx}"):
            st.session_state.current_chat = c
            st.session_state.debug_text = ""

    st.markdown("---")
    st.write(f"Active: **{st.session_state.current_chat}**")

# =========================================
# MIDDLE COLUMN — MAIN CHAT + INPUT
# =========================================
with col2:
    st.header("Chat Wall")

    conv = st.session_state.conversations.get(st.session_state.current_chat, [])
    # conversation bubbles (refined styling)
    for m in conv:
        if m["role"] == "user":
            st.markdown(
                f"""
                <div style="
                  background: #60c77f;
                  color: white;
                  padding: 12px 14px;
                  border-radius: 14px;
                  margin: 8px 0;
                  text-align: right;
                  max-width: 90%;
                  float: right;
                  clear: both;
                  font-size: 14px;
                ">
                  <b>You:</b> {m['content']}
                </div>
                <div style="clear: both;"></div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="
                  background: #F3F4F6;
                  color: #111;
                  padding: 12px 14px;
                  border-radius: 14px;
                  margin: 8px 0;
                  text-align: left;
                  max-width: 90%;
                  float: left;
                  clear: both;
                  font-size: 14px;
                ">
                  <b>Bot:</b> {m['content']}
                </div>
                <div style="clear: both;"></div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='clear:both'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # TEXT AREA (no key) — initial value from input_buffer
    input_text = st.text_area(
        "Paste discharge summary here:",
        value=st.session_state.input_buffer,
        height=220,
    )

    # QUICK SAMPLE BUBBLES
    st.markdown("### Quick Samples (click to load into input)")

    SAMPLE_1 = """DISCHARGE SUMMARY:
Chief complaint: Shortness of breath and fever.
History: 68-year-old male with COPD and type 2 diabetes.
X-ray: right lower lobe consolidation consistent with pneumonia.
Course: Treated with IV antibiotics.
Diagnosis: Community-acquired pneumonia, COPD, type 2 diabetes mellitus.
"""

    SAMPLE_2 = """DISCHARGE SUMMARY:
Chief complaint: Chest pain for 2 hours.
History: 55-year-old male, smoker, hypertension.
ECG: ST elevation in V2-V4.
Labs: Troponin markedly elevated.
Diagnosis: Acute myocardial infarction (STEMI).
"""

    SAMPLE_3 = """DISCHARGE SUMMARY:
Chief complaint: Reduced urine output.
History: 72-year-old female with dehydration.
Labs: Creatinine 3.2 mg/dL.
Diagnosis: Acute kidney injury, dehydration.
"""

    colA, colB, colC = st.columns(3)
    if colA.button("Pneumonia Sample", key="sample_pneu"):
        st.session_state.input_buffer = SAMPLE_1
        st.rerun()
    if colB.button("MI Sample", key="sample_mi"):
        st.session_state.input_buffer = SAMPLE_2
        st.rerun()
    if colC.button("AKI Sample", key="sample_aki"):
        st.session_state.input_buffer = SAMPLE_3
        st.rerun()

    # RUN pipeline
    if st.button("Run pipeline", key="run_pipeline_btn"):
        text_to_run = input_text.strip()
        if not text_to_run:
            st.warning("Please paste a discharge summary.")
        else:
            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    importlib.reload(back)  # reload for dev convenience
                    result = back.run_icd_chatbot(text_to_run)
                except Exception as e:
                    result = f"ERROR: {e}"

            captured = buf.getvalue()

            # append messages to conversation
            st.session_state.conversations.setdefault(st.session_state.current_chat, [])
            st.session_state.conversations[st.session_state.current_chat].append({"role": "user", "content": text_to_run})
            st.session_state.conversations[st.session_state.current_chat].append({"role": "assistant", "content": result})

            st.session_state.debug_text = captured
            st.session_state.input_buffer = ""
            st.rerun()

# =========================================
# RIGHT COLUMN — DEBUG OUTPUT
# =========================================
with col3:
    st.header("Debug Output")
    if st.session_state.debug_text:
        st.code(st.session_state.debug_text, language="bash")
    else:
        st.write("No debug output yet.")
