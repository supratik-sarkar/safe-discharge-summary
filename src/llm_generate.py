import os, time, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from google import genai

load_dotenv(Path.home() / "Desktop/HealthcareAI_Stanford/.env")
OUT = Path.home() / "Desktop/HealthcareAI_Stanford/results"

source = pd.read_parquet(OUT/"cohort_notes_v2.parquet")
gt = pd.read_parquet(OUT/"ground_truth_ds.parquet")

PROMPT = """You are a clinician writing a hospital discharge summary.
Below are the clinical notes from a patient's hospital stay (excluding any prior discharge summary).
Write a complete discharge summary covering: Admission Diagnosis, Hospital Course, Procedures,
Medications on Discharge, and Discharge Condition. Be faithful to the notes — do not invent facts.

CLINICAL NOTES:
{notes}

DISCHARGE SUMMARY:"""

def build_input(hadm):
    texts = source[source.HADM_ID==hadm].TEXT.fillna("").tolist()
    joined = "\n---\n".join(texts)
    return joined[:25000]  # token budget

groq = Groq(api_key=os.environ["GROQ_API_KEY"])
gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

MODELS = [
    ("groq", "llama-3.3-70b-versatile"),
    ("groq", "llama-3.1-8b-instant"),
    ("gemini", "gemini-2.5-flash"),
]

rows = []
for hadm in gt.HADM_ID.unique():
    notes_blob = build_input(hadm)
    prompt = PROMPT.format(notes=notes_blob)
    for provider, model in MODELS:
        t0 = time.time()
        try:
            if provider == "groq":
                r = groq.chat.completions.create(
                    model=model, messages=[{"role":"user","content":prompt}],
                    max_tokens=1500, temperature=0.2)
                text = r.choices[0].message.content
            else:
                r = gemini.models.generate_content(model=model, contents=prompt)
                text = r.text
            rows.append({"HADM_ID": hadm, "MODEL": model, "GENERATED_DS": text,
                         "LATENCY_MS": int((time.time()-t0)*1000)})
            print(f"  {model[:30]:30s} hadm={hadm} ok ({len(text)} chars)")
        except Exception as e:
            print(f"  {model[:30]:30s} hadm={hadm} FAILED: {str(e)[:100]}")
        time.sleep(0.5)  # rate-limit politeness

pd.DataFrame(rows).to_parquet(OUT/"llm_summaries.parquet", index=False)
print(f"\nSaved {len(rows)} generations across {gt.HADM_ID.nunique()} admissions × {len(MODELS)} models")
