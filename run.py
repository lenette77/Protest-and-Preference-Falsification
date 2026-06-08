"""
dissent_abm/run.py

Multi-provider, multi-seed orchestrator for the cross-model comparison.

Usage
-----
  # Stub mode (no API key, fast sanity check):
  python run.py --providers stub --seeds 1

  # All three backbones, three seeds each (the configuration reported in the paper).
  # Requires OPENROUTER_API_KEY in the environment or a .env file:
  python run.py --providers deepseek openai llama --seeds 3 --agents 1000 --concurrency 10

  # Single-parameter sweep (stub or single provider recommended for cost):
  python run.py --providers stub --sweep lam --seeds 3

Outputs (./output/)
  results_<provider>_seed<N>.json   per-run tick logs
  summary_all_runs.json             aggregated across providers/seeds
  comparison_table.csv              flat table for plotting / paper tables
"""

import os
import sys
import csv
import json
import asyncio
import logging
import argparse
from statistics import mean, stdev

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import run_simulation, run_simulation_async
from providers import LLMProvider, PROVIDERS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

OUT_DIR = "output"


def get_provider(name: str):
    if name == "stub":
        return None
    return LLMProvider(name)


def signature_metrics(tick_logs: list) -> dict:
    """Per-run summary of the dynamics compared across providers/seeds."""
    betas = [t["mean_beta"] for t in tick_logs]
    drops = [betas[i] - betas[i + 1] for i in range(len(betas) - 1)]
    return {
        "beta_collapse_drop": round(max(drops), 3) if drops else 0.0,
        "beta_collapse_tick": (tick_logs[drops.index(max(drops)) + 1]["tick"]
                               if drops else None),
        "peak_active":        round(max(t["active_ratio"] for t in tick_logs), 3),
        "sigma_outrage_peak": round(max(t["std_outrage"] for t in tick_logs), 3),
        "end_hidden_gap":     round(tick_logs[-1]["mean_g_internal"]
                                    - tick_logs[-1]["mean_g_external"], 3),
        "total_refusals":     sum(t.get("refusals", 0) for t in tick_logs),
    }


def main():
    ap = argparse.ArgumentParser(description="Dissent ABM multi-provider runner")
    ap.add_argument("--providers", nargs="+", default=["stub"],
                    help=f"any of: stub {' '.join(PROVIDERS)}")
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of seeds per provider (seed = 42 + i)")
    ap.add_argument("--agents", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=20,
                    help="max simultaneous API calls")
    ap.add_argument("--sync", action="store_true",
                    help="force sequential mode (debugging only)")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--w", type=float, default=0.6)
    ap.add_argument("--tau-node", type=float, default=0.55)
    ap.add_argument("--tau-global", type=float, default=0.30)
    ap.add_argument("--sweep", choices=["lam", "sigma", "w", "tau_node", "tau_global"],
                    default=None, help="sweep one parameter over [0.1..0.9]")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    base = {"lam": args.lam, "sigma": args.sigma, "w": args.w,
            "tau_node": args.tau_node, "tau_global": args.tau_global}
    if args.sweep:
        sweep_vals = [round(0.1 * k, 2) for k in range(1, 10)]
        configs = [{**base, args.sweep: v} for v in sweep_vals]
    else:
        configs = [base]

    all_runs = []
    total_runs = len(args.providers) * len(configs) * args.seeds

    with tqdm(total=total_runs, desc="Overall Progress", unit="run", position=0, leave=True) as pbar:
        for pname in args.providers:
            try:
                provider = get_provider(pname)
            except Exception as e:
                log.error(f"Skipping provider '{pname}': {e}")
                pbar.update(len(configs) * args.seeds)
                continue

            for cfg in configs:
                for s in range(args.seeds):
                    seed = 42 + s
                    verbose = (args.seeds <= 2 and not args.sweep)
                    try:
                        if args.sync:
                            res = run_simulation(
                                n_agents=args.agents, seed=seed, provider=provider,
                                lam=cfg["lam"], sigma=cfg["sigma"], w=cfg["w"],
                                tau_node=cfg["tau_node"], tau_global=cfg["tau_global"],
                                verbose=verbose,
                            )
                        else:
                            res = asyncio.run(run_simulation_async(
                                n_agents=args.agents, seed=seed, provider=provider,
                                lam=cfg["lam"], sigma=cfg["sigma"], w=cfg["w"],
                                tau_node=cfg["tau_node"], tau_global=cfg["tau_global"],
                                concurrency=args.concurrency,
                                verbose=verbose,
                            ))
                    except Exception as e:
                        log.error(f"Run FAILED [{pname} seed {seed}]: {e}")
                        pbar.update(1)
                        continue

                    sig = signature_metrics(res["tick_logs"])
                    record = {
                        "provider":  res["provider"],
                        "regime":    res["regime"],
                        "model":     res["model"],
                        "seed":      seed,
                        "params":    cfg,
                        "signature": sig,
                        "tick_logs": res["tick_logs"],
                    }
                    all_runs.append(record)

                    tag = f"{res['provider']}_seed{seed}"
                    if args.sweep:
                        tag += f"_{args.sweep}{cfg[args.sweep]}"
                    with open(f"{OUT_DIR}/results_{tag}.json", "w") as f:
                        json.dump(record, f, indent=2)

                    if verbose:
                        log.info(f"  [{tag}] beta-drop={sig['beta_collapse_drop']} "
                                 f"peak_active={sig['peak_active']} "
                                 f"sigma_peak={sig['sigma_outrage_peak']} "
                                 f"refusals={sig['total_refusals']}")
                    pbar.update(1)

    with open(f"{OUT_DIR}/summary_all_runs.json", "w") as f:
        json.dump({"n_runs": len(all_runs),
                   "runs": [{k: r[k] for k in
                             ("provider", "regime", "model", "seed", "params", "signature")}
                            for r in all_runs]}, f, indent=2)

    with open(f"{OUT_DIR}/comparison_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["provider", "regime", "model", "seed",
                    "lam", "sigma", "w", "tau_node", "tau_global",
                    "beta_collapse_drop", "beta_collapse_tick",
                    "peak_active", "sigma_outrage_peak",
                    "end_hidden_gap", "total_refusals"])
        for r in all_runs:
            p, s = r["params"], r["signature"]
            w.writerow([r["provider"], r["regime"], r["model"], r["seed"],
                        p["lam"], p["sigma"], p["w"], p["tau_node"], p["tau_global"],
                        s["beta_collapse_drop"], s["beta_collapse_tick"],
                        s["peak_active"], s["sigma_outrage_peak"],
                        s["end_hidden_gap"], s["total_refusals"]])

    log.info("\n" + "=" * 72)
    log.info("CROSS-PROVIDER SIGNATURE COMPARISON (mean across seeds)")
    log.info("=" * 72)
    by_provider = {}
    for r in all_runs:
        by_provider.setdefault(r["provider"], []).append(r["signature"])
    log.info(f"{'provider':<12}{'regime':<10}{'beta-drop':>11}{'peak_act':>10}"
             f"{'sigma_peak':>11}{'hid_gap':>9}{'refus':>8}")
    log.info("-" * 72)
    for prov, sigs in by_provider.items():
        def ms(key):
            vals = [s[key] for s in sigs]
            m = mean(vals)
            sd = stdev(vals) if len(vals) > 1 else 0.0
            return f"{m:.2f}+/-{sd:.2f}"
        regime = next(r["regime"] for r in all_runs if r["provider"] == prov)
        log.info(f"{prov:<12}{regime:<10}"
                 f"{ms('beta_collapse_drop'):>11}{ms('peak_active'):>10}"
                 f"{ms('sigma_outrage_peak'):>11}{ms('end_hidden_gap'):>9}"
                 f"{mean([s['total_refusals'] for s in sigs]):>8.0f}")
    log.info("=" * 72)
    log.info(f"\nWrote {len(all_runs)} runs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
