# blablabla.py
import os
import json
import time
import csv
import requests
from functools import lru_cache
from openai import OpenAI

# ==================================
# CONFIG & DEBUG
# ==================================
DEBUG = True  # set False kalau mau tanpa log

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
# 2. ONTOLOGY / ANNOTATOR MODULE (FROM BIOPORTAL)
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
        # default: ICD10 style (original script)
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


def batch_process(input_texts, output_path="icd_mapped.csv", bioportal_ontos=None):
    """
    Batch pipeline untuk generate CSV mapping text → ICD.
    Tidak dipakai di chatbot, tapi berguna untuk bikin kamus offline.
    """
    if bioportal_ontos is None:
        bioportal_ontos = ["ICD10CM", "ICD10"]

    with open(output_path, "w", newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=["input", "icd_class_uri", "icd_label", "icd_code"]
        )
        writer.writeheader()
        for text in input_texts:
            mapped = pipeline_text_to_icd(text, bioportal_ontos=bioportal_ontos)
            for row in mapped:
                writer.writerow(row)
            time.sleep(0.3)
    print("Saved mapping to", output_path)


# ==================================
# 3. ICD RETRIEVAL (DICTIONARY FROM ONTOLOGY + FALLBACK LOCAL)
# ==================================

# Local fallback dictionary (boleh kamu perluas)
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
    Unified ICD candidate retrieval:
    - Dictionary diambil dari ontology (BioPortal: annotate + search + convert URI→code)
    - Jika kosong → fallback local fuzzy dictionary

    ontologies untuk pipeline chatbot → default ["ICD9CM"]
    """
    if ontologies is None:
        ontologies = ["ICD9CM"]  # konsisten dengan SYSTEM_PROMPT

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
# 5. STEP 2 – LLM: ICD REASONING (WITH CANDIDATES)
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
# 6. STEP 3 – FORMAT OUTPUT CHATBOT (BAHASA INDONESIA)
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
# 8. DEMO / MAIN
# ==================================

if __name__ == "__main__":
    # Contoh discharge summary dummy
    sample = """
    DISCHARGE SUMMARY:
    Chief complaint: Shortness of breath and fever.
    History: 68-year-old male with COPD and type 2 diabetes.
    X-ray: right lower lobe consolidation consistent with pneumonia.
    Course: Treated with IV antibiotics.
    Diagnosis: Community-acquired pneumonia, COPD, type 2 diabetes mellitus.
    """

    result = run_icd_chatbot(sample)
    print("\n\nFINAL OUTPUT:\n", result)

    # Kalau mau test batch ontology → CSV (ICD10 / ICD10CM), bisa panggil manual:
    # examples = ["pneumonia", "diabetes mellitus", "acute kidney injury"]
    # batch_process(examples, output_path="icd_mapped.csv", bioportal_ontos=["ICD10CM","ICD10"])
