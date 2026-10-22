import importlib.util,unittest,copy,pathlib
spec=importlib.util.spec_from_file_location('bc',pathlib.Path(__file__).resolve().parents[1]/'scripts/run_aldp_BC_weight_preflight.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Derivation(unittest.TestCase):
 def setUp(self):self.p={'sampling_metadata':{'bc_loss_weight':'legacy','source':'fixed'},'counters':dict(global_step=200000,current_epoch=2000,outer_step=0,phase='u',phase_epoch=2000),'models':{'a':[1,2]},'rng':{'torch':[3,4]}}
 def test_only_one_metadata_item_changes(self):
  original=copy.deepcopy(self.p);d=m.derive_metadata(self.p);self.assertEqual(self.p,original);d['sampling_metadata']['bc_loss_weight']='legacy';self.assertEqual(d,original)
 def test_failed_comparison_saved_before_assertion(self):
  source=pathlib.Path(m.__file__).read_text();self.assertLess(source.index("write_json(out/'paired_diagnostic.json'"),source.index("assert all(v['status']=='passed'"))
 def test_original_precision_before_model_construction(self):
  s=pathlib.Path(m.__file__).read_text();self.assertLess(s.index('jax.config.update("jax_default_matmul_precision", "highest")'),s.index('module=GammaASBSModule(cfg)'));self.assertIn('torch.set_float32_matmul_precision("highest")',s)
 def test_wrong_prefix_rejected(self):
  self.p['counters']['phase_epoch']=1999
  with self.assertRaises(ValueError):m.derive_metadata(self.p)
 def test_wrong_parent_weight_rejected(self):
  self.p['sampling_metadata']['bc_loss_weight']='none'
  with self.assertRaises(ValueError):m.derive_metadata(self.p)
if __name__=='__main__':unittest.main()
