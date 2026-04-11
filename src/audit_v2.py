import pandas as pd, re, json, numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

OUT = Path.home()/"Desktop/HealthcareAI_Stanford/results"
LOGS = OUT/"logs"; LOGS.mkdir(exist_ok=True)

source = pd.read_parquet(OUT/"cohort_notes_v2.parquet")
gt = pd.read_parquet(OUT/"ground_truth_ds.parquet")
llm = pd.read_parquet(OUT/"llm_summaries.parquet")

print("Loading NLI model (first run downloads ~300MB)...")
nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")
LABELS = ["contradiction","entailment","neutral"]

def sents(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(t)) if 15 < len(s.strip()) < 400]
def tok(s): return re.findall(r"[a-z0-9]+", s.lower())

def audit(hadm, summary_text, label):
    ev_pool = source[source.HADM_ID==hadm]
    ev_sents = [(i,s) for _,r in ev_pool.iterrows() for i,s in enumerate(sents(r.TEXT))]
    if not ev_sents: return []
    bm25 = BM25Okapi([tok(s) for _,s in ev_sents])
    claims = sents(summary_text)[:40]
    out = []
    for c in claims:
        idx = bm25.get_scores(tok(c)).argsort()[-3:][::-1]
        evs = [ev_sents[i][1] for i in idx]
        pairs = [(c,e) for e in evs]
        scores = nli.predict(pairs)
        probs = np.exp(scores)/np.exp(scores).sum(axis=1,keepdims=True)
        best = probs.max(axis=0)
        ent, con = float(best[1]), float(best[0])
        if con > 0.5: d,r = "REFUSE","contradicted"
        elif ent > 0.7: d,r = "RELEASE","supported"
        elif ent > 0.4: d,r = "REFUSE","low_confidence"
        else: d,r = "REFUSE","no_evidence"
        out.append({"hadm":int(hadm),"model":label,"claim":c[:200],
                    "decision":d,"reason":r,"entail":round(ent,3),"contra":round(con,3)})
    return out

all_rows = []
# Ground truth baseline (ceiling)
for _,r in gt.iterrows():
    all_rows += audit(r.HADM_ID, r.text, "ground_truth")
    print(f"  GT hadm={r.HADM_ID} done")
# Each LLM
for _,r in llm.iterrows():
    all_rows += audit(r.HADM_ID, r.GENERATED_DS, r.MODEL)
    print(f"  {r.MODEL[:25]} hadm={r.HADM_ID} done")

df = pd.DataFrame(all_rows)
df.to_parquet(OUT/"audit_v2.parquet", index=False)
print(f"\nTotal claims: {len(df)}")
print(df.groupby(['model','decision']).size())
