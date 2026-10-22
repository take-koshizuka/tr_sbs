import importlib.util,pathlib,json,copy,unittest
from types import SimpleNamespace
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ext',ROOT/'scripts/run_aldp_outer_extension.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Guard(unittest.TestCase):
 def setUp(self):self.r=json.loads((ROOT/'tests/fixtures/outer_extension_registration.json').read_text())
 def test_registered_pass(self):m.validate_registration(self.r)
 def test_cached_and_terminal_update_counts(self):
  for key in ['cached_global_step','final_global_step','final_Adam_u_steps','final_Adam_h_steps']:
   r=copy.deepcopy(self.r);r[key]+=1
   with self.assertRaises(ValueError):m.validate_registration(r)
 def test_scope_cannot_expand(self):
  for key,value in [('total_outer_loops',4),('seed',1),('u_epochs',1500),('OU_rate',.5),('test','open')]:
   r=copy.deepcopy(self.r);r[key]=value
   with self.assertRaises(ValueError):m.validate_registration(r)
 def test_marginal_cost_excludes_cached_work(self):
  cfg=SimpleNamespace(buffer_samples_per_epoch=128,num_samples_for_dual_loss=[32,8],eval_samples=64,eval_samples_final=1024)
  self.assertEqual(m.marginal_cost(cfg),260224)
if __name__=='__main__':unittest.main()
