# Protest and Preference Falsification

This repository contains a generative-agent agent-based model (ABM) for studying protest mobilization and preference falsification under authoritarian repression. Agents are embedded in a scale-free social network, interpret a fixed sequence of political events through a large language model (LLM), update internal grievance and affect, and decide whether to mobilize publicly. A deterministic falsification channel controls the gap between internal grievance and public expression, while a state algorithm performs targeted node suspension and, when triggered, network-level edge removal.

The repository accompanies the manuscript **“Generative-Agent Modeling of Protest Mobilization and Preference Falsification with Large Language Models.”** The included clean outputs reproduce the paper’s headline comparison across GPT-4.1-mini, DeepSeek-V4-Flash, and Llama-4-Maverick.

## Repository layout

```text
.
├── simulation.py                  # ABM core: agents, memory, beta dynamics, state algorithm, event schedule
├── providers.py                   # OpenRouter/OpenAI-compatible provider registry and JSON/refusal handling
├── run.py                         # Multi-provider, multi-seed experiment runner
├── build_paper_figs_clean.py      # Figure-generation script with model/assertion checks
├── requirements.txt               # Python dependencies
├── outputs_clean/                 # Curated paper-source outputs
│   ├── comparison_table.csv       # Flat 9-run summary table
│   ├── summary_all_runs.json      # JSON 9-run summary
│   ├── model_means.csv            # Per-model means and sample SDs
│   └── results_<provider>_seed<seed>.json
└── paper_figs_clean/              # Pre-generated paper figures
    ├── fig1_cross_model.png/.pdf
    ├── fig2_three_model.png/.pdf
    ├── fig3_llama_falsification.png/.pdf
    └── fig4_gpt_solo.png/.pdf
```

`outputs_clean/` is the curated paper output bundle. New runs from `run.py` are written to `output/` by default, so running new experiments will not overwrite the clean paper outputs unless files are manually moved.

## Main experiment

The paper experiment consists of 9 runs:

| Backbone | Provider key | Model ID | Seeds |
|---|---|---|---|
| GPT-4.1-mini | `openai` | `openai/gpt-4.1-mini` | 42, 43, 44 |
| DeepSeek-V4-Flash | `deepseek` | `deepseek/deepseek-v4-flash` | 42, 43, 44 |
| Llama-4-Maverick | `llama` | `meta-llama/llama-4-maverick` | 42, 43, 44 |

All three backbones use the same ABM parameters:

```text
N = 1000
lambda = 0.5
sigma = 0.6
w = 0.6
tau_node = 0.55
tau_global = 0.30
blackout_severity = 0.60
seeds = 42, 43, 44
temperature = 0.85
max_tokens = 400
```

Headline values from `outputs_clean/comparison_table.csv` and `outputs_clean/model_means.csv`:

| Model | Peak mobilization, mean ± sample SD | Final hidden-transcript gap, mean ± sample SD | Notes |
|---|---:|---:|---|
| GPT-4.1-mini | 23.3% ± 1.6% | 0.200 ± 0.002 | 0 content refusals |
| DeepSeek-V4-Flash | 14.8% ± 0.4% | 0.216 ± 0.004 | 61 content refusals; most missing responses are API/format failures |
| Llama-4-Maverick | 3.2% ± 1.0% | 0.268 ± 0.002 | reasoning rerun with 975 logged inactive-agent justifications |

## Result provenance

The clean source-of-truth results are the JSON files in `outputs_clean/`:

```text
results_openai_seed42.json
results_openai_seed43.json
results_openai_seed44.json
results_deepseek_seed42.json
results_deepseek_seed43.json
results_deepseek_seed44.json
results_llama_seed42.json
results_llama_seed43.json
results_llama_seed44.json
```

The Llama files in `outputs_clean/` are the rerun with `reasoning_sample` logging enabled. These logs support the qualitative preference-falsification analysis, where each tick retains the 25 highest-grievance inactive agents. Across 13 ticks and 3 seeds, this yields 975 logged justifications.

The repository is intentionally cleaned so that legacy GPT-5-mini aggregate rows are not part of `outputs_clean/`. The figure script also asserts the expected model IDs and stops if a GPT-5 legacy output is accidentally included.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The live LLM runs use OpenRouter through the OpenAI-compatible client. Set your API key before running non-stub experiments:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

On Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY="your-key-here"
```

## Running the simulation

Fast local sanity check without an API key:

```bash
python run.py --providers stub --seeds 1 --agents 50
```

Re-run the full three-model configuration reported in the paper:

```bash
python run.py --providers openai deepseek llama --seeds 3 --agents 1000 --concurrency 10
```

This writes new results to `output/`, not `outputs_clean/`.

## Regenerating figures

The clean figure script reads `outputs_clean/` by default and writes to `paper_figs_clean/`:

```bash
python build_paper_figs_clean.py
```

The script verifies that exactly seeds 42, 43, and 44 are present for the expected three model IDs:

```text
openai/gpt-4.1-mini
deepseek/deepseek-v4-flash
meta-llama/llama-4-maverick
```

If the wrong provider/model files are present, the script raises an assertion error rather than silently generating contaminated figures.

## Model architecture summary

Each simulation advances through 13 event ticks from austerity to a lethal crackdown, mass mobilization, social-media shutdown, and a final hidden-dissent state. The population is stratified into three tiers:

- 70% disinterested majority
- 20% vulnerable middle
- 10% urban vanguard

Agents occupy a Barabási–Albert scale-free network. Vanguard roles are always informed, most non-rural agents receive the broadcast directly, and rural agents become informed only through an informed neighbor.

For each informed non-suspended agent, the LLM returns JSON fields for grievance change, outrage, hope, despair, mobilization, and a one-sentence in-character justification. The LLM controls grievance, emotion, mobilization, and justification text; it does **not** control the falsification coefficient.

Preference falsification is imposed deterministically:

```text
G_external = G_internal * (1 - beta)
beta(t+1) = beta(t) * [1 - lambda * D(t) - sigma * S(t)]
```

where `D(t)` is the fraction of active neighbors and `S(t)` is the shock indicator. Because this channel is deterministic, the beta collapse is an implementation check, not an emergent finding.

The state algorithm scores active agents by expressed grievance and eigenvector centrality, suspends agents above `tau_node`, and can trigger an edge-removal blackout if active mobilization exceeds `tau_global`.

## Important limitations

- The falsification coefficient and its update rule are imposed by construction; the model does not validate this rule empirically.
- Cross-model differences are model-specific. The `regime` field is a provenance label only and should not be interpreted as a national or political category claim.
- DeepSeek-V4-Flash has substantial endpoint/API and formatting failures in the logged runs; these are tracked separately from content refusals.
- The blackout actuator does not fire under the included paper parameters because active mobilization remains below `tau_global`.
- Mobilization is a model-internal binary state and is not calibrated to a real-world participation denominator.

## Output file schema

Each `results_<provider>_seed<seed>.json` contains:

- `provider`, `regime`, `model`, `seed`
- `params`
- `signature`, including peak activity, beta-collapse tick/drop, final hidden gap, and total refusals
- `tick_logs`, one record per event tick

Each tick log includes active ratio/count, suspended count, mean internal/external grievance, mean beta, mean emotions, outrage dispersion, edge count, blackout status, node suspensions, refusal categories, and, for the Llama reasoning rerun, `reasoning_sample`.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
