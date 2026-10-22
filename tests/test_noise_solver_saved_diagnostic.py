import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_aldp_noise_solver_numerical_diagnostic import variance_audit,gradient_audit

def test_variance_distinguishes_finite_input_from_reduction_overflow():
    audit=variance_audit(torch.tensor([1e25,-1e25]))
    assert audit['input']['finite'] and not audit['fp32']['finite'] and audit['fp64']['finite']

def test_gradient_audit_distinguishes_norm_overflow_and_nonfinite_elements():
    m=torch.nn.Linear(2,1,bias=False);m.weight.grad=torch.full_like(m.weight,1e25)
    a=gradient_audit(m)
    assert a['nonfinite_elements']==0 and a['norm32']==float('inf') and a['norm64']<float('inf')
    m.weight.grad[0,0]=float('nan');assert gradient_audit(m)['nonfinite_elements']==1


def test_nonfinite_diagnostic_evidence_is_preserved_in_strict_json():
    import json
    from run_aldp_noise_solver_numerical_diagnostic import json_safe
    values={'rows':[{'norm':float('inf'),'loss':float('nan'),'min':-float('inf'),'finite':2.0}]}
    out=json.loads(json.dumps(json_safe(values),allow_nan=False))
    assert out['rows']==[{'norm':'+inf','loss':'nan','min':'-inf','finite':2.0}]
