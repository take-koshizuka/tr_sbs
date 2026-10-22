import unittest,importlib.util,pathlib,copy
spec=importlib.util.spec_from_file_location('fullbc',pathlib.Path(__file__).resolve().parents[1]/'scripts/run_aldp_BC_weight_full.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Registration(unittest.TestCase):
 def plan(self):return dict(registration=dict(only_scientific_change='bc_loss_weight legacy -> reference_variance',new_total_updates=400000,final_total_updates=600000,expected_new_OpenMM=263040,control_retrained=False),case=dict(saved_phase='u',saved_phase_epoch=2000,cached_global_step=200000))
 def test_registered(self):m.validate(self.plan())
 def test_cached_prefix_rejected(self):
  p=self.plan();p['case']['cached_global_step']=300000
  with self.assertRaises(ValueError):m.validate(p)
 def test_control_retraining_rejected(self):
  p=self.plan();p['registration']['control_retrained']=True
  with self.assertRaises(ValueError):m.validate(p)
 def test_budget_change_rejected(self):
  p=self.plan();p['registration']['expected_new_OpenMM']=320001
  with self.assertRaises(ValueError):m.validate(p)
if __name__=='__main__':unittest.main()
