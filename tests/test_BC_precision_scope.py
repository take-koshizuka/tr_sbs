import unittest,importlib.util,pathlib
spec=importlib.util.spec_from_file_location('precision',pathlib.Path(__file__).resolve().parents[1]/'scripts/run_aldp_BC_precision_diagnostic.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Scope(unittest.TestCase):
 def plan(self):return dict(precision_diagnostic_only=True,expected_OpenMM=0,tolerances=dict(parameter_max_absolute=1e-5),gradient_evidence_path='saved',derived_snapshot_path='saved')
 def test_valid(self):m.validate_plan(self.plan())
 def test_training_scope_rejected(self):
  p=self.plan();p['precision_diagnostic_only']=False
  with self.assertRaises(ValueError):m.validate_plan(p)
 def test_tolerance_change_rejected(self):
  p=self.plan();p['tolerances']['parameter_max_absolute']=1e-4
  with self.assertRaises(ValueError):m.validate_plan(p)
 def test_new_physical_budget_rejected(self):
  p=self.plan();p['expected_OpenMM']=1
  with self.assertRaises(ValueError):m.validate_plan(p)
if __name__=='__main__':unittest.main()
