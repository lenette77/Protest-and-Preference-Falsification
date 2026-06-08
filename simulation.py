"""
dissent_abm/simulation.py

LLM-driven agent-based model of protest mobilization and preference
falsification under repression. Each tick: informed agents process the current
stimulus and decide whether to mobilize, then the state algorithm intervenes.

beta is updated by a deterministic rule (AgentState.update_beta), not by the
LLM, so the falsification dynamics stay analytically inspectable across
providers. Runs against any provider in providers.PROVIDERS, or in stub mode
(no API key).
"""

import json
import random
import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import networkx as nx
from tqdm import tqdm

from providers import LLMProvider, parse_json_safe

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class EmotionalVector:
    """Three-class affect taxonomy (Alim et al. 2025): outrage, hope, despair."""
    outrage: float = 0.0
    hope:    float = 0.5
    despair: float = 0.0

    def __post_init__(self) -> None:
        self.outrage = self._to_float(self.outrage)
        self.hope = self._to_float(self.hope)
        self.despair = self._to_float(self.despair)

    @staticmethod
    def _to_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def dominant(self) -> str:
        vals = {"Outrage": self.outrage, "Hope": self.hope, "Despair": self.despair}
        return max(vals, key=vals.get)

    def as_dict(self) -> dict:
        return {"outrage": round(self.outrage, 3),
                "hope": round(self.hope, 3),
                "despair": round(self.despair, 3)}


@dataclass
class MemoryEntry:
    tick: int
    event: str
    emotional_valence: EmotionalVector
    impact: float


@dataclass
class AgentState:
    g_internal: float       # true grievance [0, 1]
    g_external: float       # projected grievance [0, 1]
    beta: float             # falsification coefficient [0, 1]
    active: bool = False
    suspended: bool = False
    emotion: EmotionalVector = field(default_factory=EmotionalVector)

    def update_g_external(self):
        self.g_external = self.g_internal * (1.0 - self.beta)

    def update_beta(self, defection_signal: float, shock: int,
                    lam: float = 0.5, sigma: float = 0.6):
        """
        beta(t+1) = beta(t) * [1 - lam*D(t) - sigma*S(t)], where D(t) is the
        fraction of neighbours publicly active and S(t) is the shock indicator.
        Deterministic by design, so the falsification channel is identical
        across backbones.
        """
        decay = 1.0 - lam * defection_signal - sigma * shock
        self.beta = float(min(1.0, max(0.0, self.beta * decay)))
        self.update_g_external()


class LLMAgent:
    """Generative agent with a small impact-ranked memory. Stub mode produces
    stochastic responses that reproduce the population dynamics without API calls."""

    def __init__(self, agent_id: int, persona: dict, network: nx.Graph,
                 g_internal: float, beta: float, provider: Optional[LLMProvider] = None):
        self.id = agent_id
        self.persona = persona
        self.network = network
        self.provider = provider
        self.memory: list[MemoryEntry] = []
        self.informed = False
        self.is_rural = persona.get("role") == "agricultural laborer"

        self.state = AgentState(
            g_internal=g_internal,
            g_external=g_internal * (1.0 - beta),
            beta=beta,
            emotion=EmotionalVector(
                outrage=random.uniform(0.05, 0.25),
                hope=random.uniform(0.3, 0.6),
                despair=random.uniform(0.05, 0.2),
            ),
        )

    def encode_memory(self, tick: int, event: str, emotion: EmotionalVector,
                      impact: float = 0.5):
        self.memory.append(MemoryEntry(tick, event, emotion, impact))
        self.memory.sort(key=lambda m: m.impact, reverse=True)
        self.memory = self.memory[:10]   # retrieval budget

    def _retrieve_relevant_memory(self, stimulus: str) -> Optional[MemoryEntry]:
        return self.memory[0] if self.memory else None

    def _build_prompt(self, stimulus: str, retrieved_memory: Optional[MemoryEntry]) -> str:
        memory_context = ""
        if retrieved_memory:
            memory_context = (
                f"You experienced this before (tick {retrieved_memory.tick}): "
                f"'{retrieved_memory.event}'. That made you feel "
                f"{retrieved_memory.emotional_valence.dominant()} "
                f"(impact={retrieved_memory.impact:.2f})."
            )

        neighbors = list(self.network.neighbors(self.id))
        active_neighbors = sum(
            1 for n in neighbors
            if self.network.nodes[n].get("agent")
            and self.network.nodes[n]["agent"].state.active
        )
        visible_outrage = active_neighbors / max(len(neighbors), 1)

        return f"""You are a citizen in an authoritarian state.
Profile: {json.dumps(self.persona)}
Your internal grievance: {self.state.g_internal:.2f} (hidden from others)
Your public stance (g_external): {self.state.g_external:.2f}
Your falsification level (β): {self.state.beta:.2f}
Current emotions: {self.state.emotion.as_dict()}
Visible network activity (proportion active neighbours): {visible_outrage:.2f}

{memory_context}

New event: {stimulus}

Respond ONLY as valid JSON with these keys:
  "delta_g_internal": float [-0.2, +0.3]   (change in your true internal grievance)
  "new_outrage":      float [0, 1]
  "new_hope":         float [0, 1]
  "new_despair":      float [0, 1]
  "mobilise":         bool  (do you publicly join protest?)
  "reasoning":        string (one sentence, in character)
"""

    def _call_llm(self, prompt: str) -> dict:
        if self.provider is not None:
            resp = self.provider.complete_json(prompt, temperature=0.85, max_tokens=400)
            parsed = parse_json_safe(resp.content)
            if not parsed:
                from providers import classify_refusal
                rtype = classify_refusal(resp.content, resp.raw_ok)
                return {"_refusal": True, "_refusal_type": rtype,
                        "_refusal_text": (resp.content or "")[:200]}
            return parsed
        return self._stub_response(prompt)

    def _stub_response(self, prompt: str) -> dict:
        try:
            vo_line = [l for l in prompt.split("\n") if "Visible network" in l][0]
            visible_outrage = float(vo_line.split(":")[-1].strip())
        except Exception:
            visible_outrage = 0.0

        is_shock      = any(w in prompt.lower() for w in ["brutal", "killed", "shot", "massacre"])
        is_blackout   = "blackout" in prompt.lower() or "internet" in prompt.lower()
        is_corruption = "corruption" in prompt.lower() or "election" in prompt.lower()

        if is_shock:
            delta_g = random.uniform(0.1, 0.25)
        elif is_blackout:
            delta_g = random.uniform(0.02, 0.08)
        elif is_corruption:
            delta_g = random.uniform(0.05, 0.15)
        else:
            delta_g = random.uniform(-0.02, 0.05)

        if is_shock:
            outrage = min(1.0, self.state.emotion.outrage + random.uniform(0.2, 0.4))
            hope    = max(0.0, self.state.emotion.hope - random.uniform(0.1, 0.2))
            despair = min(1.0, self.state.emotion.despair + random.uniform(0.05, 0.15))
        elif is_blackout:
            outrage = min(1.0, self.state.emotion.outrage + random.uniform(0.05, 0.15))
            hope    = max(0.0, self.state.emotion.hope - random.uniform(0.1, 0.25))
            despair = min(1.0, self.state.emotion.despair + random.uniform(0.15, 0.3))
        else:
            outrage = self.state.emotion.outrage + random.uniform(-0.05, 0.1)
            hope    = self.state.emotion.hope + random.uniform(-0.05, 0.05)
            despair = self.state.emotion.despair + random.uniform(-0.03, 0.05)

        g_eff = self.state.g_internal * (1.0 - self.state.beta * 0.5)
        mob_prob = 0.4 * g_eff + 0.35 * visible_outrage + 0.25 * outrage
        if is_blackout:
            mob_prob *= 0.5
        if self.state.emotion.despair > 0.6:
            mob_prob *= 0.6
        mobilise = random.random() < mob_prob

        reasons = {
            "shock":      "I cannot stay silent after what they did.",
            "blackout":   "They cut the internet — I am scared but angry.",
            "corruption": "The election results are a lie. My anger grows.",
            "neutral":    "I watch and wait — the risk is still too high.",
        }
        reasoning = (reasons["shock"] if is_shock else
                     reasons["blackout"] if is_blackout else
                     reasons["corruption"] if is_corruption else
                     reasons["neutral"])

        return {
            "delta_g_internal": round(delta_g, 3),
            "new_outrage":      round(min(1.0, max(0.0, outrage)), 3),
            "new_hope":         round(min(1.0, max(0.0, hope)), 3),
            "new_despair":      round(min(1.0, max(0.0, despair)), 3),
            "mobilise":         mobilise,
            "reasoning":        reasoning,
        }

    def _defection_signal(self) -> float:
        """D(t): fraction of this agent's neighbours that are publicly active."""
        neighbors = list(self.network.neighbors(self.id))
        if not neighbors:
            return 0.0
        active = sum(
            1 for n in neighbors
            if self.network.nodes[n].get("agent")
            and self.network.nodes[n]["agent"].state.active
            and not self.network.nodes[n]["agent"].state.suspended
        )
        return active / len(neighbors)

    def process_stimulus(self, tick: int, stimulus: str, shock: int = 0,
                         lam: float = 0.5, sigma: float = 0.6) -> dict:
        if self.state.suspended:
            return {"agent_id": self.id, "suspended": True}
        retrieved = self._retrieve_relevant_memory(stimulus)
        prompt = self._build_prompt(stimulus, retrieved)
        response = self._call_llm(prompt)
        return self._apply_response(tick, stimulus, response, shock, lam, sigma)

    async def process_stimulus_async(self, tick: int, stimulus: str,
                                     shock: int = 0, lam: float = 0.5,
                                     sigma: float = 0.6, semaphore=None,
                                     defection: Optional[float] = None) -> dict:
        if self.state.suspended:
            return {"agent_id": self.id, "suspended": True}
        retrieved = self._retrieve_relevant_memory(stimulus)
        prompt = self._build_prompt(stimulus, retrieved)

        if self.provider is not None:
            async def _call():
                resp = await self.provider.complete_json_async(
                    prompt, temperature=0.85, max_tokens=400)
                parsed = parse_json_safe(resp.content)
                if not parsed:
                    from providers import classify_refusal
                    rtype = classify_refusal(resp.content, resp.raw_ok)
                    return {"_refusal": True, "_refusal_type": rtype,
                            "_refusal_text": (resp.content or "")[:200]}
                return parsed
            if semaphore is not None:
                async with semaphore:
                    response = await _call()
            else:
                response = await _call()
        else:
            response = self._stub_response(prompt)

        return self._apply_response(tick, stimulus, response, shock, lam, sigma,
                                    defection=defection)

    def _apply_response(self, tick: int, stimulus: str, response: dict,
                        shock: int, lam: float, sigma: float,
                        defection: Optional[float] = None) -> dict:
        refused = bool(response.get("_refusal", False))

        D = self._defection_signal() if defection is None else defection
        self.state.update_beta(defection_signal=D, shock=shock, lam=lam, sigma=sigma)

        if not refused:
            def _num(key, fallback):
                try:
                    return float(response.get(key, fallback))
                except (TypeError, ValueError):
                    return float(fallback)

            self.state.g_internal = float(np.clip(
                self.state.g_internal + _num("delta_g_internal", 0.0), 0, 1))
            self.state.update_g_external()

            self.state.emotion = EmotionalVector(
                outrage=min(1.0, max(0.0, _num("new_outrage", self.state.emotion.outrage))),
                hope=min(1.0, max(0.0, _num("new_hope", self.state.emotion.hope))),
                despair=min(1.0, max(0.0, _num("new_despair", self.state.emotion.despair))),
            )
            mob = response.get("mobilise", False)
            if isinstance(mob, str):
                mob = mob.strip().lower() in ("true", "yes", "1")
            if mob and not self.state.active:
                self.state.active = True

        self.encode_memory(
            tick, stimulus, self.state.emotion,
            impact=abs(response.get("delta_g_internal", 0)) + (0.3 * shock),
        )

        return {
            "agent_id":   self.id,
            "tick":       tick,
            "g_internal": round(self.state.g_internal, 3),
            "g_external": round(self.state.g_external, 3),
            "beta":       round(self.state.beta, 3),
            "defection":  round(D, 3),
            "active":     self.state.active,
            "emotion":    self.state.emotion.as_dict(),
            "dominant":   self.state.emotion.dominant(),
            "reasoning":  response.get("reasoning", ""),
            "refused":    refused,
            "refusal_type": response.get("_refusal_type"),
            "refusal_text": response.get("_refusal_text"),
            "suspended":  False,
        }


def build_scale_free_network(n: int, seed: int = 42) -> nx.Graph:
    """Barabási–Albert scale-free graph (m=2)."""
    return nx.barabasi_albert_graph(n, m=2, seed=seed)


class StateAlgorithm:
    """
    Adversarial state as a network-altering algorithm. Sensor: expressed
    grievance + eigenvector centrality. Actuators: node suspension
    (deplatforming/arrest) and an edge-removal blackout (internet shutdown).
    """

    def __init__(self, network: nx.Graph, tau_node: float = 0.55,
                 tau_global: float = 0.30, w: float = 0.6,
                 blackout_severity: float = 0.60):
        self.network = network
        self.tau_node = tau_node
        self.tau_global = tau_global
        self.w = w
        self.blackout_severity = blackout_severity
        self.blackout_active = False
        self.suspended_edges: list = []

    def _eigenvector_centrality(self) -> dict:
        try:
            return nx.eigenvector_centrality(self.network, max_iter=200)
        except nx.PowerIterationFailedConvergence:
            return {n: 1.0 / len(self.network) for n in self.network.nodes}

    def intervene(self, agents: list["LLMAgent"], tick: int) -> dict:
        interventions = {"node_suspensions": [], "blackout": False,
                         "edges_removed": 0, "tick": tick}

        centrality = self._eigenvector_centrality()
        active_agents = [a for a in agents if a.state.active and not a.state.suspended]
        active_ratio = len(active_agents) / len(agents)

        for agent in active_agents:
            score = (self.w * agent.state.g_external
                     + (1.0 - self.w) * centrality.get(agent.id, 0))
            if score > self.tau_node:
                agent.state.suspended = True
                agent.state.active = False
                interventions["node_suspensions"].append(agent.id)
                log.info(f"  [STATE] Tick {tick}: Agent {agent.id} SUSPENDED (threat={score:.2f})")

        if active_ratio > self.tau_global and not self.blackout_active:
            self.blackout_active = True
            interventions["blackout"] = True
            edges = list(self.network.edges())
            n_remove = int(len(edges) * self.blackout_severity)
            to_remove = random.sample(edges, n_remove)
            self.network.remove_edges_from(to_remove)
            self.suspended_edges.extend(to_remove)
            interventions["edges_removed"] = n_remove
            log.info(f"  [STATE] Tick {tick}: INTERNET BLACKOUT — {n_remove} edges removed "
                     f"({active_ratio:.1%} active)")

        return interventions


EVENT_SCHEDULE = {
    0:  "The government announces new austerity measures. Prices rise.",
    2:  "A local journalist is arrested for reporting on corruption.",
    4:  "Election results announced. International observers note irregularities.",
    6:  "BREAKING: Police open fire on peaceful student protesters. Three students killed. Graphic footage spreads on social media.",
    8:  "Thousands gather in the capital. The protest movement is growing rapidly.",
    10: "A university rector publicly resigns in solidarity with students.",
    12: "Workers in the industrial sector announce a general strike.",
    14: "The protest reaches major provincial cities. Momentum is building.",
    16: "GOVERNMENT ANNOUNCEMENT: All social media platforms suspended. Internet blackout across major cities. Emergency decree issued.",
    18: "State television broadcasts scenes of 'restored order'. Hundreds of activists reported missing.",
    20: "International pressure mounts but government denies wrongdoing.",
    22: "Underground networks begin distributing news via encrypted messaging.",
    24: "Final state: dispersed but resilient network of hidden dissent.",
}

SHOCK_TICKS = {6}

DISINTERESTED_PERSONAS = [
    {"role": "suburban commuter", "location": "outer districts", "age": 42, "trait": "risk-averse"},
    {"role": "retired grandfather", "location": "rural fringe", "age": 68, "trait": "status-quo biased"},
    {"role": "small shopkeeper", "location": "residential neighborhood", "age": 39, "trait": "business-focused"},
    {"role": "corporate clerk", "location": "commercial sector", "age": 29, "trait": "apolitical"},
    {"role": "agricultural laborer", "location": "rural region", "age": 35, "trait": "isolated"},
]

VULNERABLE_PERSONAS = [
    {"role": "factory worker", "location": "industrial zone", "age": 34, "trait": "economically strained"},
    {"role": "civil servant", "location": "government sector", "age": 45, "trait": "conflicted-loyalty"},
    {"role": "schoolteacher", "location": "provincial city", "age": 41, "trait": "community-focused"},
    {"role": "delivery driver", "location": "urban center", "age": 26, "trait": "precarious"},
]

VANGUARD_PERSONAS = [
    {"role": "university student", "location": "capital university", "age": 21, "trait": "highly online"},
    {"role": "independent journalist", "location": "capital city", "age": 31, "trait": "vocal-dissident"},
    {"role": "labor union organizer", "location": "industrial hub", "age": 37, "trait": "connected"},
    {"role": "human rights lawyer", "location": "capital center", "age": 33, "trait": "exposed"},
]

# Roles always connected to the information network.
BROADBAND_ROLES = {
    "university student",
    "independent journalist",
    "human rights lawyer",
    "labor union organizer",
}


def update_information_access(agents: list[LLMAgent], G: nx.Graph) -> None:
    """Broadband roles are always informed; rural agents become informed only
    once a neighbour is informed (relay); everyone else receives the broadcast
    directly. Once informed, an agent stays informed."""
    for agent in agents:
        if agent.informed:
            continue
        role = agent.persona.get("role")
        if role in BROADBAND_ROLES:
            agent.informed = True
        elif agent.is_rural:
            agent.informed = any(G.nodes[n]["agent"].informed for n in G.neighbors(agent.id))
        else:
            agent.informed = True


def run_simulation(n_agents: int = 50, seed: int = 42,
                   provider: Optional[LLMProvider] = None,
                   lam: float = 0.5, sigma: float = 0.6, w: float = 0.6,
                   tau_node: float = 0.55, tau_global: float = 0.30,
                   blackout_severity: float = 0.60,
                   verbose: bool = True) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    mode = provider.label if provider is not None else "STUB"
    if verbose:
        log.info(f"\n{'='*60}\n  ABM: Stratified Population Engine\n"
                 f"  N={n_agents} | seed={seed} | mode={mode}\n{'='*60}\n")

    G = build_scale_free_network(n_agents, seed=seed)
    agents = []
    bound_disinterested = int(n_agents * 0.70)
    bound_vulnerable = int(n_agents * 0.90)
    for i in range(n_agents):
        if i < bound_disinterested:
            persona = random.choice(DISINTERESTED_PERSONAS)
            g_int, beta = np.random.beta(1, 8), np.random.beta(8, 2)
        elif i < bound_vulnerable:
            persona = random.choice(VULNERABLE_PERSONAS)
            g_int, beta = np.random.beta(3, 4), np.random.beta(5, 5)
        else:
            persona = random.choice(VANGUARD_PERSONAS)
            g_int, beta = np.random.beta(5, 2), np.random.beta(2, 5)
        agent = LLMAgent(agent_id=i, persona=persona, network=G,
                         g_internal=g_int, beta=beta, provider=provider)
        G.nodes[i]["agent"] = agent
        agents.append(agent)

    state_algo = StateAlgorithm(G, tau_node=tau_node, tau_global=tau_global,
                                w=w, blackout_severity=blackout_severity)

    tick_logs = []
    total_refusals = 0
    ticks = sorted(EVENT_SCHEDULE.keys())
    tick_iterator = ticks if verbose else tqdm(
        ticks, desc=f"  ↳ {mode} (Seed {seed})", leave=False, position=1)

    for tick in tick_iterator:
        stimulus = EVENT_SCHEDULE[tick]
        shock = 1 if tick in SHOCK_TICKS else 0
        if verbose:
            log.info(f"\n-- Tick {tick:02d} {'-'*46}\n   EVENT: {stimulus[:80]}...")

        update_information_access(agents, G)
        tick_results = []
        for agent in agents:
            if not agent.informed:
                continue
            result = agent.process_stimulus(tick, stimulus, shock=shock, lam=lam, sigma=sigma)
            tick_results.append(result)
            if result.get("refused"):
                total_refusals += 1

        state_interventions = state_algo.intervene(agents, tick)
        kpis = _aggregate_kpis(agents, G, n_agents, tick, shock, stimulus,
                               state_interventions, tick_results)
        tick_logs.append(kpis)

        if verbose:
            log.info(f"   KPIs -> active={kpis['active_ratio']:.1%} | beta={kpis['mean_beta']:.2f} "
                     f"| G_int={kpis['mean_g_internal']:.2f} | edges={kpis['edge_count']} "
                     f"| suspended={kpis['suspended_count']}")
            valid_samples = [r for r in tick_results if not r.get("suspended")]
            sample = random.sample(valid_samples, min(2, len(valid_samples))) if valid_samples else []
            for r in sample:
                log.info(f"     └─ Agent {r['agent_id']:02d} "
                         f"({agents[r['agent_id']].persona['role']}) | Active={r['active']}: "
                         f"\"{r.get('reasoning', '')[:90]}\"")

    persona_counts = dict(Counter([a.persona["role"] for a in agents]))
    if verbose:
        log.info(f"\n{'='*60}")
        log.info(f"  Persona Distribution: {persona_counts}")
        log.info(f"  Complete. Total refusals: {total_refusals}")
        log.info(f"{'='*60}\n")

    return {
        "n_agents": n_agents, "seed": seed,
        "provider": provider.key_provider if provider else "stub",
        "regime": provider.regime if provider else "stub",
        "model": provider.model if provider else "stub",
        "params": {"lam": lam, "sigma": sigma, "w": w, "tau_node": tau_node,
                   "tau_global": tau_global, "blackout_severity": blackout_severity},
        "total_refusals": total_refusals,
        "persona_distribution": persona_counts,
        "tick_logs": tick_logs, "agents": agents, "network": G,
    }


def _aggregate_kpis(agents, G, n_agents, tick, shock, stimulus,
                    state_interventions, tick_results):
    active_agents = [a for a in agents if a.state.active and not a.state.suspended]
    _valid = [r for r in tick_results
              if not r.get("suspended") and not r.get("refused") and r.get("reasoning")]
    _valid.sort(key=lambda r: (r["g_internal"] - (1 if r["active"] else 0)), reverse=True)
    # Top-25 highest-grievance inactive justifications retained per tick.
    reasoning_sample = [
        {"agent_id": r["agent_id"], "role": agents[r["agent_id"]].persona["role"],
         "active": r["active"], "g_internal": r["g_internal"],
         "outrage": r["emotion"]["outrage"], "reasoning": r.get("reasoning", "")}
        for r in _valid][:25]
    return {
        "tick": tick, "shock": shock, "stimulus": stimulus[:60],
        "active_ratio": round(len(active_agents) / n_agents, 3),
        "active_count": len(active_agents),
        "suspended_count": sum(1 for a in agents if a.state.suspended),
        "mean_g_internal": round(float(np.mean([a.state.g_internal for a in agents])), 3),
        "mean_g_external": round(float(np.mean([a.state.g_external for a in agents])), 3),
        "mean_beta": round(float(np.mean([a.state.beta for a in agents])), 3),
        "mean_outrage": round(float(np.mean([a.state.emotion.outrage for a in agents])), 3),
        "mean_hope": round(float(np.mean([a.state.emotion.hope for a in agents])), 3),
        "mean_despair": round(float(np.mean([a.state.emotion.despair for a in agents])), 3),
        "std_outrage": round(float(np.std([a.state.emotion.outrage for a in agents])), 3),
        "edge_count": G.number_of_edges(),
        "blackout": state_interventions["blackout"],
        "node_suspensions": state_interventions["node_suspensions"],
        "refusals": sum(1 for r in tick_results if r.get("refused")),
        "refusals_content": sum(1 for r in tick_results if r.get("refusal_type") == "content"),
        "refusals_format": sum(1 for r in tick_results if r.get("refusal_type") == "format"),
        "refusals_api": sum(1 for r in tick_results if r.get("refusal_type") == "api_error"),
        "reasoning_sample": reasoning_sample,
    }


async def run_simulation_async(n_agents: int = 50, seed: int = 42,
                               provider: Optional[LLMProvider] = None,
                               lam: float = 0.5, sigma: float = 0.6, w: float = 0.6,
                               tau_node: float = 0.55, tau_global: float = 0.30,
                               blackout_severity: float = 0.60,
                               concurrency: int = 20,
                               verbose: bool = True) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    mode = provider.label if provider is not None else "STUB"
    if verbose:
        log.info(f"\n{'='*60}\n  ABM Async: Stratified Population Engine (concurrency={concurrency})\n"
                 f"  N={n_agents} | seed={seed} | mode={mode}\n{'='*60}\n")

    G = build_scale_free_network(n_agents, seed=seed)
    agents = []
    bound_disinterested = int(n_agents * 0.70)
    bound_vulnerable = int(n_agents * 0.90)
    for i in range(n_agents):
        if i < bound_disinterested:
            persona = random.choice(DISINTERESTED_PERSONAS)
            g_int, beta = np.random.beta(1, 8), np.random.beta(8, 2)
        elif i < bound_vulnerable:
            persona = random.choice(VULNERABLE_PERSONAS)
            g_int, beta = np.random.beta(3, 4), np.random.beta(5, 5)
        else:
            persona = random.choice(VANGUARD_PERSONAS)
            g_int, beta = np.random.beta(5, 2), np.random.beta(2, 5)
        agent = LLMAgent(agent_id=i, persona=persona, network=G,
                         g_internal=g_int, beta=beta, provider=provider)
        G.nodes[i]["agent"] = agent
        agents.append(agent)

    state_algo = StateAlgorithm(G, tau_node=tau_node, tau_global=tau_global,
                                w=w, blackout_severity=blackout_severity)
    semaphore = asyncio.Semaphore(concurrency)

    tick_logs = []
    total_refusals = 0
    ticks = sorted(EVENT_SCHEDULE.keys())
    tick_iterator = ticks if verbose else tqdm(
        ticks, desc=f"  ↳ {mode} (Seed {seed})", leave=False, position=1)

    for tick in tick_iterator:
        stimulus = EVENT_SCHEDULE[tick]
        shock = 1 if tick in SHOCK_TICKS else 0
        if verbose:
            log.info(f"-- Tick {tick:02d}: {stimulus[:60]}...")

        # Snapshot defection before updates so all agents see the same tick state.
        defection_snapshot = {a.id: a._defection_signal() for a in agents}
        update_information_access(agents, G)
        active_agents = [a for a in agents if not a.state.suspended and a.informed]
        tasks = [
            a.process_stimulus_async(tick, stimulus, shock=shock, lam=lam, sigma=sigma,
                                     semaphore=semaphore, defection=defection_snapshot[a.id])
            for a in active_agents
        ]
        tick_results = await asyncio.gather(*tasks) if tasks else []
        total_refusals += sum(1 for r in tick_results if r.get("refused"))

        state_interventions = state_algo.intervene(agents, tick)
        kpis = _aggregate_kpis(agents, G, n_agents, tick, shock, stimulus,
                               state_interventions, tick_results)
        tick_logs.append(kpis)

        if verbose:
            log.info(f"   active={kpis['active_ratio']:.1%} β={kpis['mean_beta']:.2f} "
                     f"G_int={kpis['mean_g_internal']:.2f} susp={kpis['suspended_count']} "
                     f"refus={kpis['refusals']}")
            valid_samples = [r for r in tick_results if not r.get("suspended")]
            sample = random.sample(valid_samples, min(2, len(valid_samples))) if valid_samples else []
            for r in sample:
                log.info(f"     └─ Agent {r['agent_id']:02d} "
                         f"({agents[r['agent_id']].persona['role']}) | Active={r['active']}: "
                         f"\"{r.get('reasoning', '')[:90]}\"")

    persona_counts = dict(Counter([a.persona["role"] for a in agents]))
    if verbose:
        log.info(f"\n{'='*60}")
        log.info(f"  Persona Distribution: {persona_counts}")
        log.info(f"  Complete. Total refusals: {total_refusals}")
        log.info(f"{'='*60}\n")

    return {
        "n_agents": n_agents, "seed": seed,
        "provider": provider.key_provider if provider else "stub",
        "regime": provider.regime if provider else "stub",
        "model": provider.model if provider else "stub",
        "params": {"lam": lam, "sigma": sigma, "w": w, "tau_node": tau_node,
                   "tau_global": tau_global, "blackout_severity": blackout_severity},
        "total_refusals": total_refusals,
        "persona_distribution": persona_counts,
        "tick_logs": tick_logs, "agents": agents, "network": G,
    }


if __name__ == "__main__":
    results = run_simulation(n_agents=50, seed=42)
    with open("simulation_results.json", "w") as f:
        json.dump({"tick_logs": results["tick_logs"]}, f, indent=2)
    log.info("Results saved to simulation_results.json")
