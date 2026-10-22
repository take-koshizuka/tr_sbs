"""A-only physical/reference gates; differently weighted BC is disclosure."""
import math

def analyze(candidate,control,limits):
 d,n=candidate['diagnostics'],control['diagnostics'];p,r=d['trained_policy'],n['trained_policy'];a,ar=p['metrics']['a'],r['metrics']['a'];checks={}
 def check(name,value,bound,lower=False):
  checks[name]=dict(value=float(value),bound=float(bound),passed=math.isfinite(value) and math.isfinite(bound) and (value>=bound if lower else value<=bound))
 for name,key in [('torsion_js_mean','torsion_js_absolute_increase'),('coarse_basin_tvd','coarse_basin_tvd_absolute_increase'),('missing_reference_mass','missing_mass_absolute_increase')]:check(name,a[name],ar[name]+limits[key])
 check('physical_energy_w2',a['physical_energy_w2'],ar['physical_energy_w2']*limits['physical_energy_w2_ratio'])
 check('close_contact_fraction',p['geometry']['fraction_below_reference_q01'],r['geometry']['fraction_below_reference_q01']+limits['close_contact_fraction_absolute_increase'])
 for i in (0,1):check('near_cut_'+str(i),p['near_cut_fraction_phi_psi'][i],r['near_cut_fraction_phi_psi'][i]+limits['near_cut_fraction_absolute_increase'])
 check('raw_log_path_ratio_variance',p['raw_log_path_ratio_variance'],r['raw_log_path_ratio_variance']*limits['raw_log_path_ratio_variance_ratio'])
 check('raw_path_ess_fraction',p['raw_path_ess_fraction'],r['raw_path_ess_fraction']*limits['raw_path_ess_fraction_ratio_min'],True)
 improve=ar['torsion_js_mean']-a['torsion_js_mean']>=.01 or ar['coarse_basin_tvd']-a['coarse_basin_tvd']>=.02
 return dict(schema='aldp_BC_weight_comparison_v1',physical_checks=checks,meaningful_A_improvement=improve,conditional_candidate=all(x['passed'] for x in checks.values()) and improve,decision_bank='a',disclosure_b=p['metrics']['b'],test='closed',geometry_admission='not_established',common_legacy_BC=d['next_outer_validation'],control_common_legacy_BC=n['next_outer_validation'],own_weighted_BC=d['inner_boundaries'][-1]['validation'],candidate_cost=candidate['cost'],cached_control_retrained=False)
