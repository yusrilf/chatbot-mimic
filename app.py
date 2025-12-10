# app.py - ICD Chatbot dengan UI Streamlit dan Backend Terintegrasi
import streamlit as st
import io
import json
import os
import time
import csv
import requests
from contextlib import redirect_stdout
from functools import lru_cache
from openai import OpenAI

# ==================================
# KONFIGURASI DAN DEBUG
# ==================================
DEBUG = True  # set False untuk tanpa log

def debug_print(title, data):
    if not DEBUG:
        return
    print("\n" + "=" * 80)
    print(f"🔍 DEBUG: {title}")
    print("=" * 80)
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(data)
    print("=" * 80 + "\n")

# ==================================
# ENV / API KEYS
# ==================================
HF_API_KEY = os.getenv("HF_API_KEY")
BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY")
BIOPORTAL_BASE = "https://data.bioontology.org"

# ==================================
# 0. SETUP HF LLM CLIENT
# ==================================
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_KEY,
)

def call_llm(messages, response_format="text", model="openai/gpt-oss-120b"):
    rf = {"type": response_format}
    debug_print("LLM Request Messages", messages)
    
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=rf,
    )
    raw = resp.choices[0].message.content
    debug_print("LLM Raw Response", raw)
    
    if raw and "<|start|>assistant" in raw:
        final = (
            raw.split("<|start|>assistant")[-1]
            .replace("<|channel|>final", "")
            .replace("<|message|>", "")
            .strip()
        )
        debug_print("LLM Cleaned Response", final)
        return final
    
    return raw

# ==================================
# 1. SYSTEM PROMPT GLOBAL
# ==================================
SYSTEM_PROMPT = """
You are an expert clinical coding assistant specialized in ICD-9-CM coding for hospital discharge summaries.
Follow ICD coding rules strictly.
Never invent diagnoses. Use unspecified codes only when documentation lacks specificity.
"""

# ==================================
# 2. ONTOLOGY / ANNOTATOR MODULE
# ==================================
def normalize_term(term: str) -> str:
    if not isinstance(term, str):
        return ""
    return term.lower().strip()

def parse_icd_code_from_uri(uri: str) -> str:
    """
    Given a BioPortal class URI from ICD ontologies, parse the code.
    Example ICD10CM URI: http://purl.bioontology.org/ontology/ICD10CM/J18.9 → 'J18.9'
    Example ICD9CM URI:  http://purl.bioontology.org/ontology/ICD9CM/486   → '486'
    """
    try:
        parts = uri.rstrip("/").split("/")
        return parts[-1]
    except Exception:
        return ""

def annotate_bioportal(text, ontologies=None, use_longest=False, require_exact=False):
    """
    Generic BioPortal annotator wrapper.
    ontologies: list like ["ICD9CM"], ["ICD10CM","ICD10"], etc.
    """
    url = f"{BIOPORTAL_BASE}/annotator"
    params = {
        "text": text,
        "apikey": BIOPORTAL_API_KEY,
        "longest_only": "true" if use_longest else "false",
        "require_exact_match": "true" if require_exact else "false"
    }
    if ontologies:
        params["ontologies"] = ",".join(ontologies)
    
    debug_print("Annotate BioPortal Params", params)
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        debug_print("BioPortal Annotator error", str(e))
        return []
    
    result = resp.json()
    annotations = []
    
    for ann in result:
        for c in ann.get("annotations", []):
            ont = c.get("ontologyConcept")
            if not ont:
                continue
            class_uri = ont.get("@id") or ont.get("id")
            label = ont.get("prefLabel") or ont.get("label")
            if class_uri and label:
                annotations.append({"class_uri": class_uri, "label": label})
    
    debug_print("Annotate BioPortal Raw Annotations", annotations)
    return annotations

def fallback_bioportal_search(text, ontologies=None, max_results=50):
    """
    Fallback to /search when annotator returns nothing.
    """
    url = f"{BIOPORTAL_BASE}/search"
    params = {
        "q": text,
        "apikey": BIOPORTAL_API_KEY,
        "pagesize": max_results
    }
    if ontologies:
        params["ontologies"] = ",".join(ontologies)
    
    debug_print("Search BioPortal Params", params)
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        debug_print("BioPortal search error", str(e))
        return []
    
    coll = resp.json().get("collection", [])
    results = []
    for item in coll:
        uri = item.get("@id") or item.get("id")
        label = item.get("prefLabel") or item.get("label")
        if uri and label:
            results.append({"class_uri": uri, "label": label})
    
    debug_print("Search BioPortal Raw Collection", results)
    return results

def pipeline_text_to_icd(text, bioportal_ontos=None):
    """
    Pipeline:
    1) Try annotate
    2) If empty → fallback search
    3) Deduplicate
    4) Parse ICD codes from URI
    Return list of dict:
      {"input": text, "icd_class_uri": uri, "icd_label": label, "icd_code": code}
    """
    if bioportal_ontos is None:
        bioportal_ontos = ["ICD10CM", "ICD10"]
    
    text_norm = text.strip()
    if not text_norm:
        return []
    
    anns = annotate_bioportal(
        text_norm,
        ontologies=bioportal_ontos,
        use_longest=False,
        require_exact=False
    )
    
    if not anns:
        anns = fallback_bioportal_search(
            text_norm,
            ontologies=bioportal_ontos,
            max_results=50
        )
    
    # de-duplicate
    seen = set()
    cleaned = []
    for a in anns:
        key = (a["class_uri"], a["label"])
        if key not in seen:
            seen.add(key)
            cleaned.append(a)
    
    results = []
    for a in cleaned:
        uri = a["class_uri"]
        label = a["label"]
        code = parse_icd_code_from_uri(uri)
        results.append({
            "input": text_norm,
            "icd_class_uri": uri,
            "icd_label": label,
            "icd_code": code
        })
    
    debug_print("pipeline_text_to_icd results", results)
    return results

# ==================================
# 3. ICD RETRIEVAL (DICTIONARY)
# ==================================
LOCAL_ICD_LIST = {
    "486": "Pneumonia, organism unspecified",
    "481": "Pneumococcal pneumonia",
    "491.21": "Chronic obstructive pulmonary disease with acute exacerbation",
    "496": "Chronic airway obstruction, not elsewhere classified",
    "250.00": "Type II diabetes mellitus without complications",
    "584.9": "Acute kidney failure, unspecified",
    "410.90": "Acute myocardial infarction, unspecified",
}

def local_fuzzy_icd(term: str, top_k=5):
    term_norm = normalize_term(term)
    hits = []
    for code, label in LOCAL_ICD_LIST.items():
        if term_norm in label.lower():
            hits.append({
                "code": code,
                "label": label,
                "uri": None,
                "score": 1.0
            })
    debug_print("Local fuzzy ICD hits", hits)
    return hits[:top_k]

def retrieve_icd_candidates(term: str, top_k: int = 10, ontologies=None):
    """
    Unified ICD candidate retrieval.
    """
    if ontologies is None:
        ontologies = ["ICD9CM"]
    
    term_norm = normalize_term(term)
    if not term_norm:
        return []
    
    debug_print("Retrieve ICD candidates for term", term_norm)
    
    # 1) Ontology-based dictionary via BioPortal (ICD9CM)
    mapped = pipeline_text_to_icd(term_norm, bioportal_ontos=ontologies)
    candidates = []
    for row in mapped:
        candidates.append({
            "code": row["icd_code"],
            "label": row["icd_label"],
            "uri": row["icd_class_uri"],
            "score": None
        })
    
    if candidates:
        debug_print("Ontology-based ICD candidates", candidates[:top_k])
        return candidates[:top_k]
    
    # 2) Fallback: local dictionary
    local = local_fuzzy_icd(term_norm, top_k=top_k)
    if local:
        debug_print("Using local fallback ICD candidates", local)
        return local
    
    debug_print("No ICD candidates found", term_norm)
    return []

# ==================================
# 4. STEP 1 – LLM: EKSTRAKSI KLINIS
# ==================================
EXTRACTION_PROMPT_TEMPLATE = """
You will receive a hospital discharge summary.

Extract all clinical information needed for ICD-9-CM coding.
Return JSON ONLY in the structure below.

{{
  "primary_diagnosis": {{
    "label": "",
    "supporting_evidence": []
  }},
  "secondary_diagnoses": [
    {{
      "label": "",
      "supporting_evidence": []
    }}
  ]
}}

<<<DISCHARGE_SUMMARY>>>
{discharge_summary}
<<<END>>>
"""

def extract_clinical_json(discharge_summary: str) -> dict:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(discharge_summary=discharge_summary)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    raw = call_llm(messages, response_format="text")
    
    try:
        parsed = json.loads(raw)
        debug_print("Clinical Extraction JSON", parsed)
        return parsed
    except Exception as e:
        debug_print("JSON Parse Error (Extraction)", raw)
        raise e

# ==================================
# 5. STEP 2 – LLM: ICD REASONING
# ==================================
ICD_REASONER_PROMPT = """
You will receive:
1) Extracted clinical JSON
2) ICD-9-CM candidate list (per term)

Rules:
- Select ICD ONLY from candidate list.
- Never invent codes outside list.
- Follow ICD-9-CM rules strictly.

Return JSON ONLY:

{{
  "primary_icd": {{
    "code": "",
    "description": "",
    "reasoning": ""
  }},
  "secondary_icd": []
}}

<<<CLINICAL>>>
{clinical_json}
<<<END_CLINICAL>>>

<<<CANDIDATES>>>
{icd_candidates}
<<<END_CANDIDATES>>>
"""

def predict_icd_from_clinical_json(clinical_json: dict) -> dict:
    primary = clinical_json.get("primary_diagnosis", {}).get("label", "")
    secondary = [s.get("label", "") for s in clinical_json.get("secondary_diagnoses", [])]
    
    terms = []
    if primary:
        terms.append(primary)
    terms.extend([t for t in secondary if t])
    
    # ICD retrieval per term
    candidate_map = {
        t: retrieve_icd_candidates(t, top_k=10, ontologies=["ICD9CM"])
        for t in terms
    }
    
    debug_print("ICD Candidate Map", candidate_map)
    
    prompt = ICD_REASONER_PROMPT.format(
        clinical_json=json.dumps(clinical_json, ensure_ascii=False),
        icd_candidates=json.dumps(candidate_map, ensure_ascii=False),
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    raw = call_llm(messages, response_format="text")
    
    try:
        parsed = json.loads(raw)
        debug_print("ICD Reasoner Output JSON", parsed)
        return parsed
    except Exception as e:
        debug_print("JSON Parse Error (Reasoner)", raw)
        raise e

# ==================================
# 6. STEP 3 – FORMAT OUTPUT CHATBOT
# ==================================
CHATBOT_OUTPUT_PROMPT = """
Generate Indonesian explanation from ICD JSON.

Format:
- Ringkasan singkat kasus
- Diagnosa utama (kode + deskripsi + alasan)
- Diagnosa lain (jika ada)
- Catatan / Limitasi (jika ada)

<<<ICD>>>
{icd_json}
<<<END>>>
"""

def format_chatbot_answer(icd_json: dict) -> str:
    prompt = CHATBOT_OUTPUT_PROMPT.format(icd_json=json.dumps(icd_json, ensure_ascii=False))
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant for Indonesian clinicians."},
        {"role": "user", "content": prompt},
    ]
    
    raw = call_llm(messages)
    debug_print("Final Chatbot Output", raw)
    return raw

# ==================================
# 7. PIPELINE END-TO-END
# ==================================
def run_icd_chatbot(discharge_summary: str) -> str:
    debug_print("INPUT DISCHARGE SUMMARY", discharge_summary)
    
    step1 = extract_clinical_json(discharge_summary)
    step2 = predict_icd_from_clinical_json(step1)
    step3 = format_chatbot_answer(step2)
    
    return step3

# ==================================
# 8. STREAMLIT UI
# ==================================
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
                    result = run_icd_chatbot(text_to_run)
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