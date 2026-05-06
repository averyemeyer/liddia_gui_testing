import pickle
import sys
import types

# Replicate the _load_memory logic
run_dir = 'log/26-04-06-09-08-45_KEAP1'
mem_path = f'{run_dir}/KEAP1_memory.pkl'

class DummyMemory:
    def __init__(self):
        self.stream = {}
        self.history = []

liddia_mod = types.ModuleType('liddia')
liddia_mod.__path__ = []
submods = ['memory', 'action', 'environment', 'evaluate', 'utils', 'prompt_template', 'agent']
for name in submods:
    mod = types.ModuleType(f'liddia.{name}')
    sys.modules[f'liddia.{name}'] = mod
sys.modules['liddia'] = liddia_mod
sys.modules['liddia.memory'].Memory = DummyMemory
for fn in ['sample_zinc', 'graph_ga_optimizer', 'run_code', 'sample_pocket2mol']:
    setattr(sys.modules['liddia.action'], fn, lambda *a, **k: None)

with open(mem_path, 'rb') as f:
    mem = pickle.load(f)

print('History length:', len(mem.history))
for i, h in enumerate(mem.history):
    print(f'History {i}: action_output = {h.get("action_output")}')

# Test _iteration_pool_ids logic
pool_ids = []
for h in mem.history:
    pool_id = h.get('action_output')
    if pool_id and pool_id != 'EMPTY SET':
        pool_ids.append(pool_id)
print('Pool IDs:', pool_ids)