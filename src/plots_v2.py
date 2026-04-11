import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
OUT = Path.home()/"Desktop/HealthcareAI_Stanford/results"
FIG = OUT/"figures"; FIG.mkdir(exist_ok=True)

df = pd.read_parquet(OUT/"audit_v2.parquet")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titleweight":"bold",
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
PALETTE = {"ground_truth":"#264653","llama-3.3-70b-versatile":"#2a9d8f",
           "llama-3.1-8b-instant":"#e9c46a","gemini-2.5-flash":"#e76f51"}

def bootstrap_ci(vals, n=1000):
    b = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return np.percentile(b,[2.5,97.5])

# Fig 1: Unsupported-claim rate per model with 95% CI
fig,ax = plt.subplots(figsize=(8,5))
models = df.model.unique()
rates, errs = [], []
for m in models:
    v = (df[df.model==m].decision=="REFUSE").astype(int).values
    rates.append(v.mean())
    lo,hi = bootstrap_ci(v)
    errs.append([v.mean()-lo, hi-v.mean()])
errs = np.array(errs).T
bars = ax.bar(range(len(models)), rates, yerr=errs, capsize=8,
    color=[PALETTE.get(m,"#999") for m in models], edgecolor="black",linewidth=1.2)
ax.set_xticks(range(len(models))); ax.set_xticklabels([m[:18] for m in models],rotation=15,ha="right")
ax.set_ylabel("Unsupported-claim rate (refused by gate)")
ax.set_title("Hallucination rate across LLMs vs ground truth\n(bootstrap 95% CI, n=1000)")
for i,r in enumerate(rates): ax.text(i,r+0.02,f"{r:.1%}",ha="center",fontweight="bold")
ax.set_ylim(0,max(rates)*1.25+0.05)
plt.tight_layout(); plt.savefig(FIG/"01_hallucination_by_model.png",dpi=300); plt.close()

# Fig 2: Decision breakdown stacked per model
piv = df.groupby(["model","decision"]).size().unstack(fill_value=0)
piv_pct = piv.div(piv.sum(axis=1),axis=0)
fig,ax = plt.subplots(figsize=(8,5))
piv_pct[["RELEASE","REFUSE"]].plot.barh(stacked=True,ax=ax,
    color=["#2a9d8f","#e76f51"],edgecolor="black")
ax.set_xlabel("Proportion of claims"); ax.set_title("Gate decisions per model")
ax.legend(loc="lower right"); plt.tight_layout()
plt.savefig(FIG/"02_decision_breakdown.png",dpi=300); plt.close()

# Fig 3: Refusal reason breakdown
ref = df[df.decision=="REFUSE"]
piv2 = ref.groupby(["model","reason"]).size().unstack(fill_value=0)
piv2_pct = piv2.div(piv2.sum(axis=1),axis=0).fillna(0)
fig,ax = plt.subplots(figsize=(8,5))
piv2_pct.plot.bar(stacked=True,ax=ax,
    color=["#c1121f","#fcbf49","#6a994e"],edgecolor="black")
ax.set_ylabel("Share of refusals"); ax.set_title("Why claims were refused (per model)")
ax.set_xticklabels([m[:18] for m in piv2_pct.index],rotation=15,ha="right")
ax.legend(title="reason"); plt.tight_layout()
plt.savefig(FIG/"03_refusal_reasons.png",dpi=300); plt.close()

# Fig 4: NLI entailment score distribution
fig,ax = plt.subplots(figsize=(9,5))
for m in models:
    sns.kdeplot(df[df.model==m].entail, ax=ax, label=m[:20],
        color=PALETTE.get(m,"#999"), linewidth=2.2, fill=True, alpha=0.2)
ax.axvline(0.7,color="red",ls="--",label="release threshold (0.7)")
ax.set_xlabel("NLI entailment probability"); ax.set_ylabel("Density")
ax.set_title("Claim-level entailment distributions")
ax.legend(); plt.tight_layout()
plt.savefig(FIG/"04_entailment_distribution.png",dpi=300); plt.close()

# Fig 5: Contradiction-catch rate
fig,ax = plt.subplots(figsize=(8,5))
con_rates = [(df[df.model==m].contra > 0.5).mean() for m in models]
bars = ax.bar(range(len(models)),con_rates,
    color=[PALETTE.get(m,"#999") for m in models],edgecolor="black")
ax.set_xticks(range(len(models))); ax.set_xticklabels([m[:18] for m in models],rotation=15,ha="right")
ax.set_ylabel("Contradiction-catch rate"); ax.set_title("Safety: fraction of claims flagged contradicted")
for i,r in enumerate(con_rates): ax.text(i,r+0.005,f"{r:.1%}",ha="center",fontweight="bold")
plt.tight_layout(); plt.savefig(FIG/"05_contradiction_catch.png",dpi=300); plt.close()

# Fig 6: Ground-truth vs LLM delta
gt_rate = (df[df.model=="ground_truth"].decision=="REFUSE").mean()
fig,ax = plt.subplots(figsize=(8,5))
llm_models = [m for m in models if m!="ground_truth"]
deltas = [(df[df.model==m].decision=="REFUSE").mean() - gt_rate for m in llm_models]
colors = ["#e76f51" if d>0 else "#2a9d8f" for d in deltas]
ax.barh(range(len(llm_models)),deltas,color=colors,edgecolor="black")
ax.axvline(0,color="black",linewidth=1)
ax.set_yticks(range(len(llm_models))); ax.set_yticklabels([m[:22] for m in llm_models])
ax.set_xlabel("Δ unsupported-claim rate vs ground truth")
ax.set_title("LLM fidelity gap: excess hallucination over human DS")
for i,d in enumerate(deltas): ax.text(d,i,f" {d:+.1%}",va="center",fontweight="bold")
plt.tight_layout(); plt.savefig(FIG/"06_llm_vs_groundtruth.png",dpi=300); plt.close()

print("6 figures saved to", FIG)
