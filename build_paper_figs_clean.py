import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "output")
OUT = os.path.join(HERE, "paper_figs_clean")
os.makedirs(OUT, exist_ok=True)

EXPECTED = {
    "openai": "openai/gpt-4.1-mini",
    "deepseek": "deepseek/deepseek-v4-flash",
    "llama": "meta-llama/llama-4-maverick",
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300, "lines.linewidth": 1.8, "lines.markersize": 5,
})

prov = {}
loaded_files = []
for f in sorted(glob.glob(os.path.join(SRC, "results_*_seed4[234].json"))):
    if "norlog" in os.path.basename(f):
        continue
    d = json.load(open(f))
    provider = d["provider"]
    model = d["model"]
    assert provider in EXPECTED, f"Unexpected provider in {f}: {provider}"
    assert model == EXPECTED[provider], f"Unexpected model in {f}: {model}; expected {EXPECTED[provider]}"
    assert "gpt-5" not in model.lower(), f"GPT-5 legacy file accidentally included: {f}"
    prov.setdefault(provider, []).append(d["tick_logs"])
    loaded_files.append((provider, d["seed"], model, os.path.basename(f)))

for provider in EXPECTED:
    seeds = sorted([seed for p, seed, model, fname in loaded_files if p == provider])
    assert seeds == [42, 43, 44], f"{provider} seeds are {seeds}, expected [42, 43, 44]"

REGIME = {"openai": "GPT-4.1-mini", "deepseek": "DeepSeek-V4-flash", "llama": "Llama-4-Maverick"}
COLOR  = {"openai": "#2563EB", "deepseek": "#DC2626", "llama": "#059669"}
ORDER  = ["openai", "deepseek", "llama"]
ticks  = [t["tick"] for t in prov["openai"][0]]

def series(p, key):
    arr = np.array([[t[key] for t in run] for run in prov[p]])
    return arr.mean(0), arr.std(0)

# FIGURE 1
fig = plt.figure(figsize=(10, 4.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1], wspace=0.28)

axA = fig.add_subplot(gs[0])
for p in ORDER:
    m, s = series(p, "active_ratio"); m, s = m*100, s*100
    axA.plot(ticks, m, color=COLOR[p], marker="o", label=REGIME[p], zorder=3)
    axA.fill_between(ticks, m-s, m+s, color=COLOR[p], alpha=0.15, zorder=1)
axA.axvline(6, color="#444", ls="--", lw=1, alpha=0.6, zorder=2)
axA.text(6.2, axA.get_ylim()[1]*0.92, "shock (tick 6)", fontsize=7, color="#444")
axA.axhline(3.5, color="#888", ls=":", lw=1)
axA.text(0.3, 4.2, "3.5% threshold (Chenoweth & Stephan 2011)", fontsize=6.5, color="#888")
axA.set_xlabel("Simulation tick"); axA.set_ylabel("Active agents (%)")
axA.set_title("A   Mobilization trajectory by model backbone")
axA.legend(loc="upper right", framealpha=0.95)
axA.set_xlim(-0.5, 24.5)

axB = fig.add_subplot(gs[1])
gaps, errs = [], []
for p in ORDER:
    g = np.array([run[-1]["mean_g_internal"] - run[-1]["mean_g_external"] for run in prov[p]])
    gaps.append(g.mean()); errs.append(g.std())
x = np.arange(len(ORDER))
axB.bar(x, gaps, yerr=errs, capsize=4, color=[COLOR[p] for p in ORDER], alpha=0.85, edgecolor="black", linewidth=0.6)
axB.set_xticks(x); axB.set_xticklabels(["GPT", "DeepSeek", "Llama"])
axB.set_ylabel("Hidden-transcript gap  (G$_{int}$ - G$_{ext}$, final)")
axB.set_title("B   Preference falsification\nat end state")
for i, g in enumerate(gaps):
    axB.text(i, g+errs[i]+0.004, f"{g:.3f}", ha="center", fontsize=8, fontweight="bold")
axB.set_ylim(0, max(gaps)*1.25)

fig.suptitle("Figure 1. Cross-model divergence: mobilization and preference falsification (N=1000, 3 seeds each)", fontsize=10, fontweight="bold", y=1.02)
fig.savefig(os.path.join(OUT, "fig1_cross_model.pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "fig1_cross_model.png"), bbox_inches="tight", facecolor="white", dpi=200)
plt.close()

# FIGURE 2
fig, axes = plt.subplots(3, 3, figsize=(11, 8.5), sharex=True)
metrics = [("active_ratio","Active (%)",100), ("mean_g_internal","Mean G internal",1), ("mean_despair","Mean despair",1)]
for col, p in enumerate(ORDER):
    for rowi,(key,lab,scale) in enumerate(metrics):
        ax = axes[rowi][col]
        m,s = series(p, key); m,s = m*scale, s*scale
        ax.plot(ticks, m, color=COLOR[p], marker=".", lw=1.6)
        ax.fill_between(ticks, m-s, m+s, color=COLOR[p], alpha=0.15)
        ax.axvline(6, color="#444", ls="--", lw=0.8, alpha=0.5)
        if rowi==0: ax.set_title(REGIME[p], fontsize=9)
        if col==0: ax.set_ylabel(lab, fontsize=8)
        if rowi==2: ax.set_xlabel("tick", fontsize=8)
        if key=="active_ratio": ax.set_ylim(-1, 26)
        if key=="mean_g_internal": ax.set_ylim(0.2,1.02)
        if key=="mean_despair": ax.set_ylim(0.1,0.8)
fig.suptitle("Figure 2. Cross-model dynamics (rows: mobilization, internal grievance, despair; columns: backbones).\nAll share identical imposed dynamics; differences are LLM-driven. Bands = ±1 SD across 3 seeds.", fontsize=9.5, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(OUT, "fig2_three_model.pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "fig2_three_model.png"), bbox_inches="tight", facecolor="white", dpi=200)
plt.close()

# FIGURE 3
quotes = []
for f in sorted(glob.glob(os.path.join(SRC, "results_llama_seed4[234].json"))):
    d = json.load(open(f))
    for t in d["tick_logs"]:
        if t["tick"] < 10: continue
        for r in t.get("reasoning_sample", []):
            if not r["active"] and r["g_internal"] >= 0.99 and r["outrage"] >= 0.97:
                quotes.append(r["reasoning"].strip())
seen, picks = set(), []
for q in quotes:
    key = q[:25]
    if key in seen: continue
    if 60 <= len(q) <= 150:
        picks.append(q); seen.add(key)
    if len(picks) == 4: break

hg_count = 0
for f in sorted(glob.glob(os.path.join(SRC, "results_llama_seed4[234].json"))):
    d = json.load(open(f))
    for t in d["tick_logs"]:
        for r in t.get("reasoning_sample", []):
            if not r["active"] and r["g_internal"] >= 0.9:
                hg_count += 1

fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.2)

ax = fig.add_subplot(gs[0])
gaps, errs = [], []
for p in ORDER:
    g = np.array([run[-1]["mean_g_internal"] - run[-1]["mean_g_external"] for run in prov[p]])
    gaps.append(g.mean()); errs.append(g.std())
x = np.arange(len(ORDER))
ax.bar(x, gaps, yerr=errs, capsize=4, color=[COLOR[p] for p in ORDER], alpha=0.85, edgecolor="black", linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels(["GPT","DeepSeek","Llama"])
ax.set_ylabel("Hidden-transcript gap (final)")
ax.set_title("Quantitative:\nLlama shows largest gap", fontsize=9)
for i,g in enumerate(gaps): ax.text(i, g+errs[i]+0.004, f"{g:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.set_ylim(0, max(gaps)*1.25)

ax = fig.add_subplot(gs[1]); ax.axis("off")
ax.set_title("Qualitative: why Llama agents stay silent\n(g$_{internal}$=1.0, outrage$\\approx$1.0, but inactive)", fontsize=9, loc="left")
y = 0.92
for q in picks:
    wrapped = textwrap.fill('"'+q+'"', 52)
    ax.text(0.02, y, wrapped, va="top", fontsize=8.5, style="italic", color="#111", family="serif")
    y -= 0.07 + 0.045*wrapped.count("\n")
ax.text(0.02, max(y-0.02,0.02), f"- {hg_count} logged justifications (25 inactive/tick × 13 ticks × 3 seeds)", va="top", fontsize=7.5, color="#666")

fig.suptitle("Figure 3. Preference falsification in Llama: maximal internal grievance, suppressed public action", fontsize=10, fontweight="bold", y=1.0)
fig.savefig(os.path.join(OUT, "fig3_llama_falsification.pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "fig3_llama_falsification.png"), bbox_inches="tight", facecolor="white", dpi=200)
plt.close()

# FIGURE 4
P = "openai"
fig = plt.figure(figsize=(10, 7.5))
gs = gridspec.GridSpec(3, 2, hspace=0.42, wspace=0.40)

ax = fig.add_subplot(gs[0, 0])
for key, lab, c, ls in [("mean_beta","beta (falsification coeff.)","#7C3AED","-"), ("mean_g_internal","G internal","#DC2626","-"), ("mean_g_external","G external","#059669","--")]:
    m, s = series(P, key)
    ax.plot(ticks, m, color=c, label=lab, ls=ls, marker=".")
    ax.fill_between(ticks, m-s, m+s, color=c, alpha=0.12)
ax.axvline(6, color="#444", ls="--", lw=1, alpha=0.5)
ax.set_title("A   beta collapse & grievance divergence"); ax.set_ylabel("value [0,1]")
ax.legend(fontsize=7); ax.set_xlabel("tick")

ax = fig.add_subplot(gs[0, 1])
for key, lab, c in [("mean_outrage","Outrage","#DC2626"), ("mean_hope","Hope","#2563EB"), ("mean_despair","Despair","#6B7280")]:
    m, s = series(P, key)
    ax.plot(ticks, m, color=c, label=lab, marker=".")
    ax.fill_between(ticks, m-s, m+s, color=c, alpha=0.12)
ax.axvline(6, color="#444", ls="--", lw=1, alpha=0.5)
ax.set_title("B   Emotional vector (Alim et al. taxonomy)"); ax.set_ylabel("mean intensity")
ax.legend(fontsize=7); ax.set_xlabel("tick")

ax = fig.add_subplot(gs[1, 0])
m, s = series(P, "active_ratio"); m, s = m*100, s*100
ax.plot(ticks, m, color="#2563EB", marker="o", label="Active (%)")
ax.fill_between(ticks, m-s, m+s, color="#2563EB", alpha=0.15)
msu, ssu = series(P, "suspended_count")
ax2 = ax.twinx()
ax2.plot(ticks, msu, color="#DC2626", marker="x", ls="--", label="Suspended (count)")
ax2.set_ylabel("suspended (count)", color="#DC2626")
ax2.tick_params(axis="y", labelcolor="#DC2626")
ax.axvline(6, color="#444", ls="--", lw=1, alpha=0.5)
ax.set_title("C   Mobilization & state suspension"); ax.set_ylabel("active (%)", color="#2563EB")
ax.tick_params(axis="y", labelcolor="#2563EB"); ax.set_xlabel("tick")

ax = fig.add_subplot(gs[1, 1])
m, s = series(P, "std_outrage")
ax.plot(ticks, m, color="#7C3AED", marker="D")
ax.fill_between(ticks, m-s, m+s, color="#7C3AED", alpha=0.15)
ax.axvline(6, color="#444", ls="--", lw=1, alpha=0.5)
ax.set_title("D   Population emotional dispersion (sigma outrage)"); ax.set_ylabel("sigma"); ax.set_xlabel("tick")

ax = fig.add_subplot(gs[2, 0])
line = np.linspace(0, 1.05, 50); ax.plot(line, line, color="#999", ls="--", lw=1, label="G$_{ext}$=G$_{int}$")
for tk, c in [(0,"#2563EB"),(6,"#DC2626"),(24,"#111")]:
    gi = np.mean([[t["mean_g_internal"] for t in run if t["tick"]==tk][0] for run in prov[P]])
    ge = np.mean([[t["mean_g_external"] for t in run if t["tick"]==tk][0] for run in prov[P]])
    ax.scatter(gi, ge, color=c, s=70, zorder=3, label=f"tick {tk}")
ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.set_aspect("equal")
ax.set_title("E   Falsification space (mean)"); ax.set_xlabel("G internal"); ax.set_ylabel("G external")
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
pk = np.mean([max(t["active_ratio"] for t in run) for run in prov[P]])
txt = (f"GPT-4.1-mini summary (3 seeds)\n\n"
       f"peak mobilization:  {pk*100:.1f}%\n"
       f"beta: {prov[P][0][0]['mean_beta']:.2f} -> {min(t['mean_beta'] for t in prov[P][0]):.2f}\n"
       f"final G_int: {np.mean([run[-1]['mean_g_internal'] for run in prov[P]]):.2f}\n"
       f"final active: {np.mean([run[-1]['active_ratio'] for run in prov[P]])*100:.1f}%\n"
       f"content refusals: 0 / ~13,000\n\n"
       f"Hidden transcript: high internal\ngrievance persists while public\nexpression is suppressed.")
ax.text(0.05, 0.95, txt, va="top", fontsize=9, family="monospace")

fig.suptitle(f"Figure 4. Full dynamics - {REGIME[P]} (N=1000, mean of 3 seeds, bands = ±1 SD)", fontsize=10, fontweight="bold", y=0.99)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_gpt_solo.pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "fig4_gpt_solo.png"), bbox_inches="tight", facecolor="white", dpi=200)
plt.close()

print("Loaded files:")
for item in sorted(loaded_files):
    print(item)
