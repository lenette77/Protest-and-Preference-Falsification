"""Offline, isolated verification of paper model mechanics and archived outputs.

Usage: python verify_paper_model.py PATH_TO_PUBLIC_REPOSITORY
No API calls. Extracts named definitions with AST, without importing provider code.
Uses a minimal graph fixture and fixed centrality scores to test state logic;
does not test NetworkX's eigenvector algorithm or live provider behavior.
"""
import ast
import json
import math
from pathlib import Path
import random
import statistics
import sys
import types
import unittest
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

ROOT = Path(sys.argv.pop(1))

class GraphFixture:
    def __init__(self, n, edges=()):
        self.nodes = {i: {} for i in range(n)}
        self._edges = {tuple(sorted(e)) for e in edges}
    def neighbors(self, i):
        return [b if a == i else a for a, b in sorted(self._edges) if i in (a, b)]
    def edges(self):
        return sorted(self._edges)
    def remove_edges_from(self, edges):
        self._edges.difference_update(tuple(sorted(e)) for e in edges)
    def __len__(self):
        return len(self.nodes)

names = {'EmotionalVector', 'MemoryEntry', 'AgentState', 'LLMAgent',
         'StateAlgorithm', 'update_information_access'}
tree = ast.parse((ROOT / 'simulation.py').read_text())
nodes = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in names]
ns = dict(dataclass=dataclass, field=field, Optional=Optional, np=np,
          random=random, json=json, nx=types.SimpleNamespace(Graph=GraphFixture),
          LLMProvider=object, log=types.SimpleNamespace(info=lambda *a: None),
          BROADBAND_ROLES={'university student', 'independent journalist',
                           'human rights lawyer', 'labor union organizer'})
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'isolated_public_definitions', 'exec'), ns)
A, State, Gov = ns['LLMAgent'], ns['AgentState'], ns['StateAlgorithm']

def agents(n=3, edges=((0, 1), (1, 2))):
    g = GraphFixture(n, edges)
    aa = [A(i, {'role': 'clerk'}, g, .8, .5) for i in range(n)]
    for a in aa:
        g.nodes[a.id]['agent'] = a
    return g, aa

class Verification(unittest.TestCase):
    def test_beta_boundaries_and_external_identity(self):
        for d, shock, expected in [(0,0,.5),(0,1,.2),(1,1,0),(1,0,.25)]:
            s = State(.8,.4,.5)
            s.update_beta(d,shock)
            self.assertAlmostEqual(s.beta, expected)
            self.assertAlmostEqual(s.g_external, s.g_internal*(1-s.beta))
    def test_absorbing_activity(self):
        _, aa=agents(); a=aa[0]
        a._apply_response(0,'test',{'mobilise':True},0,.5,.6,defection=0)
        a._apply_response(2,'test',{'mobilise':False},0,.5,.6,defection=0)
        self.assertTrue(a.state.active)
    def test_failure_updates_beta_and_memory_only(self):
        _, aa=agents(); a=aa[0]; emotion=a.state.emotion.as_dict()
        a._apply_response(6,'shock',{'_refusal':True},1,.5,.6,defection=0)
        self.assertEqual(a.state.g_internal,.8)
        self.assertFalse(a.state.active)
        self.assertEqual(a.state.emotion.as_dict(),emotion)
        self.assertAlmostEqual(a.state.beta,.2)
        self.assertAlmostEqual(a.state.g_external,.64)
        self.assertEqual(len(a.memory),1)
    def test_information_relay_and_order(self):
        g,aa=agents(); aa[0].persona={'role':'agricultural laborer'};aa[0].is_rural=True
        ns['update_information_access'](aa,g)
        self.assertFalse(aa[0].informed)
        ns['update_information_access'](aa,g)
        self.assertTrue(aa[0].informed)
    def test_suspension_strict_threshold_and_retained_edges(self):
        g,aa=agents(); aa[0].state.active=True;aa[0].state.g_external=.55
        gov=Gov(g,w=1,tau_node=.55,tau_global=1)
        gov._eigenvector_centrality=lambda:{i:0 for i in g.nodes}
        self.assertEqual(gov.intervene(aa,0)['node_suspensions'],[])
        aa[0].state.g_external=.551
        self.assertEqual(gov.intervene(aa,2)['node_suspensions'],[0])
        self.assertFalse(aa[0].state.active)
        self.assertEqual(len(g.edges()),2)
    def test_suspended_agent_skips_processing(self):
        _,aa=agents();a=aa[0];a.state.suspended=True
        a._call_llm=lambda p: self.fail('Suspended agent called provider')
        self.assertTrue(a.process_stimulus(6,'shock',1)['suspended'])
        self.assertEqual(a.state.beta,.5)
    def test_blackout_uses_pre_suspension_ratio_and_fires_once(self):
        g,aa=agents(4,((0,1),(0,2),(0,3),(1,2),(1,3)))
        for a in aa:a.state.active=True;a.state.g_external=1
        gov=Gov(g,w=1,tau_node=.5,tau_global=.3,blackout_severity=.6)
        gov._eigenvector_centrality=lambda:{i:0 for i in g.nodes}
        r=gov.intervene(aa,0)
        self.assertEqual(len(r['node_suspensions']),4)
        self.assertTrue(r['blackout']);self.assertEqual(r['edges_removed'],3)
        self.assertFalse(gov.intervene(aa,2)['blackout'])
    def test_blackout_threshold_equality(self):
        g,aa=agents(4,((0,1),(1,2),(2,3)))
        aa[0].state.active=True
        gov=Gov(g,tau_node=2,tau_global=.25)
        gov._eigenvector_centrality=lambda:{i:0 for i in g.nodes}
        self.assertFalse(gov.intervene(aa,0)['blackout'])
    def test_snapshot_overrides_live_defection(self):
        _,aa=agents();aa[1].state.active=True
        aa[0]._apply_response(0,'test',{},0,.5,.6,defection=0)
        self.assertEqual(aa[0].state.beta,.5)

if __name__ == '__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Verification))
    if not result.wasSuccessful():sys.exit(1)
    print('\nArchived-output recalculation (sample SD):')
    for p in ['openai','deepseek','llama']:
        runs=[json.loads((ROOT/'outputs_clean'/f'results_{p}_seed{s}.json').read_text()) for s in [42,43,44]]
        peaks=[max(t['active_ratio'] for t in r['tick_logs']) for r in runs]
        gaps=[r['tick_logs'][-1]['mean_g_internal']-r['tick_logs'][-1]['mean_g_external'] for r in runs]
        for r,v in zip(runs,peaks):assert math.isclose(r['signature']['peak_active'],v,abs_tol=.001)
        samples=[q for r in runs for t in r['tick_logs'] for q in t.get('reasoning_sample',[])]
        print(json.dumps({'provider':p,'peak_mean':statistics.mean(peaks),'peak_sample_sd':statistics.stdev(peaks),'gap_mean':statistics.mean(gaps),'gap_sample_sd':statistics.stdev(gaps),'failures':{k:sum(t.get('refusals_'+k,0) for r in runs for t in r['tick_logs']) for k in ['content','format','api']},'sample_records':len(samples),'sample_high_grievance':sum(q['g_internal']>=.9 for q in samples),'sample_active':sum(bool(q['active']) for q in samples)}))
