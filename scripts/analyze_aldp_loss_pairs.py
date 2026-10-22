"""Physical comparisons only; objective values and own-path RN are disclosures."""


def compare_original_surface(reports):
    control=reports.get('brownian_geometric__legacy')
    rows=[]
    for name,r in reports.items():
        row=dict(candidate=name,A=r['metrics']['a'],B=r['metrics']['b'],cost=r['cost'],
            phase_curve=r['diagnostics']['phase_end'],own_reference_path_disclosure=r['diagnostics']['dual'],
            geometry_admission='not_established',physical_screen='control_not_complete')
        if control:
            a,b=r['metrics']['a'],control['metrics']['a']
            contact=lambda x:x['diagnostics']['trained_policy']['geometry']['fraction_below_reference_q01']
            gates=dict(js=a['torsion_js_mean']<=b['torsion_js_mean']+.01,
                basin=a['coarse_basin_tvd']<=b['coarse_basin_tvd']+.02,
                missing_mass=a['missing_reference_mass']<=b['missing_reference_mass']+.02,
                energy=a['physical_energy_w2']<=b['physical_energy_w2']*1.1,
                contact=contact(r)<=contact(control)+.02,
                near_cut_phi=r['near_cut_fraction_phi_psi'][0]<=control['near_cut_fraction_phi_psi'][0]+.02,
                near_cut_psi=r['near_cut_fraction_phi_psi'][1]<=control['near_cut_fraction_phi_psi'][1]+.02)
            improvement=a['torsion_js_mean']<=b['torsion_js_mean']-.01 or a['coarse_basin_tvd']<=b['coarse_basin_tvd']-.02
            row.update(A_relative_gates=gates,A_improvement=improvement,
                physical_screen='conditional_candidate' if all(gates.values()) and improvement else 'not_admitted')
        rows.append(row)
    return dict(rows=rows,control='brownian_geometric__legacy',selection_bank='a',disclosure_bank='b',test_bank='closed',
        geometry_admission='not_established',objective_values_used_for_selection=False,
        own_path_RN_ESS_used_for_selection=False,limits='Original-loss surface; finite seed0/outer5 diagnosis. Joint LV comparison requires verified same-cell full pairs.')


def compare_loss_pairs(original, lv):
    rows = []
    for candidate in sorted(set(original) | set(lv)):
        a, b = original.get(candidate), lv.get(candidate)
        row = dict(candidate=candidate, status='pending_verified_full_pair',
                   geometry_admission='not_established')
        if a is not None and b is not None:
            for report in (a, b):
                if report['rejected_training_steps'] or report['status'] != 'artifact_complete':
                    raise ValueError('Incomplete or failed result cannot become a loss pair')
            if a['sampling_metadata']['fc_loss_mode'] != 'cross_entropy' or b['sampling_metadata']['fc_loss_mode'] != 'log_variance':
                raise ValueError('Wrong loss identity')
            fixed = lambda report:{key:value for key,value in report['sampling_metadata'].items() if key != 'fc_loss_mode'}
            if fixed(a) != fixed(b):
                raise ValueError('Different fixed conditions cannot become a same-cell loss comparison')
            if a['global_step'] != b['global_step'] or a['calibration_sha256'] != b['calibration_sha256'] or a['seed'] != b['seed']:
                raise ValueError('Different updates/banks/seed cannot become a same-cell loss comparison')
            metric_keys = ('phi_psi_tvd','torsion_js_mean','physical_energy_w2','coarse_basin_tvd','missing_reference_mass')
            row.update(status='verified_finite_budget_pair',
                backward_numerics=dict(original=a['diagnostics'].get('controller_backward_numerics',dict(mode='native')),
                                       LV=b['diagnostics'].get('controller_backward_numerics',dict(mode='native'))),
                original_A=a['metrics']['a'], LV_A=b['metrics']['a'],
                physical_A_original_minus_LV={key:a['metrics']['a'][key]-b['metrics']['a'][key] for key in metric_keys},
                original_B=a['metrics']['b'], LV_B=b['metrics']['b'],
                original_cost=a['cost'], LV_cost=b['cost'],
                original_phase_curve=a['diagnostics']['phase_end'], LV_phase_curve=b['diagnostics']['phase_end'],
                own_path_disclosure=dict(original=a['diagnostics']['dual'], LV=b['diagnostics']['dual']))
        rows.append(row)
    return dict(rows=rows,selection_bank='a',disclosure_bank='b',test_bank='closed',
        objective_values_used_for_selection=False,own_path_RN_ESS_used_for_selection=False,
        limits='Same kernel/settings/seed/update count with method-specific draws and cost; no paired-trajectory, global minimum or scientific-admission claim')
