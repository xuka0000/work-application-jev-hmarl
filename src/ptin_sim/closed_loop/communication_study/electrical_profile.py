"""Frozen study assumptions sized only from the intact fixed-demand feeder.

These are synthetic engineering ratings, not recovered equipment nameplates.
No restoration policy, fault realization or evaluation score enters sizing.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from ..publication_island_power import ElectricalAssumptions, IslandACDispatch


def electrical_signature(scenario):
    physical = dict(nodes=sorted((n.node_id, n.active_power_kw, n.reactive_power_kvar,
                                  n.base_voltage_kv) for n in scenario.pdn_nodes),
                    edges=sorted((e.edge_id, e.from_node, e.to_node, e.r_pu, e.x_pu,
                                  e.normal_switch_state) for e in scenario.pdn_edges))
    return hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()


def profile_digest(profile):
    return hashlib.sha256(json.dumps({k: v for k, v in profile.items() if k != 'sha256'},
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


def intact_network(scenario, profile=None):
    """Fixed P AND Q at every bus; PF cannot conceal an infeasible case by shedding."""
    import pandapower as pp
    if any(abs(n.base_voltage_kv-4.8) > 1e-9 for n in scenario.pdn_nodes):
        raise ValueError('This assumed balanced single-voltage profile requires 4.8 kV')
    if any(e.normal_switch_state != 'closed' for e in scenario.pdn_edges):
        raise ValueError('Unrated normally-open circuits require a separate design basis')
    net = pp.create_empty_network(sn_mva=10.)
    buses = {n.node_id: pp.create_bus(net, vn_kv=4.8, name=n.node_id)
             for n in scenario.pdn_nodes}
    pp.create_ext_grid(net, buses['799'], vm_pu=1.)
    for n in scenario.pdn_nodes:
        pp.create_load(net, buses[n.node_id], p_mw=n.active_power_kw/1000,
                       q_mvar=n.reactive_power_kvar/1000, name=n.node_id)
    for e in scenario.pdn_edges:
        row = profile['branches'][e.edge_id] if profile else dict(
            r_ohm=e.r_pu*2.304, x_ohm=e.x_pu*2.304, rating_ka=.2)
        pp.create_line_from_parameters(net, buses[e.from_node], buses[e.to_node],
            length_km=1., r_ohm_per_km=row['r_ohm'], x_ohm_per_km=row['x_ohm'],
            c_nf_per_km=0., max_i_ka=row['rating_ka'], name=e.edge_id)
    pp.runpp(net, calculate_voltage_angles=True, tolerance_mva=1e-10, numba=False)
    if not net.converged or not net.res_bus.vm_pu.notna().all():
        raise RuntimeError('Intact fixed-demand power flow must serve every bus')
    return net


def derive_profile(scenario):
    net = intact_network(scenario)
    vmin, vmax = float(net.res_bus.vm_pu.min()), float(net.res_bus.vm_pu.max())
    if vmin < .95 or vmax > 1.05:
        raise RuntimeError('Assumed total impedance fails voltage limits; no automatic impedance tuning')
    by_id = {e.edge_id: e for e in scenario.pdn_edges}
    branches = {}
    for i, row in net.line.iterrows():
        e = by_id[row['name']]
        current = float(net.res_line.at[i, 'i_ka'])
        rating = max(.2, math.ceil(1.25*current/.05)*.05)
        branches[e.edge_id] = dict(from_bus=e.from_node, to_bus=e.to_node,
            r_ohm=e.r_pu*2.304, x_ohm=e.x_pu*2.304, rating_ka=round(rating, 8),
            normal_current_ka=current, historical_rating_ka=.2)
    p = float(net.res_ext_grid.p_mw.sum())*1000
    q = float(net.res_ext_grid.q_mvar.sum())*1000
    kva = math.ceil(1.25*math.hypot(p, q)/500)*500.
    profile = dict(schema=1, kind='assumed_normal_service_engineering_profile',
        physical_signature=electrical_signature(scenario), base_mva=10., voltage_kv=4.8,
        impedance_convention='inherited total pu on an explicitly assumed 10 MVA base; no geographic length multiplier',
        rating_rule='max(0.2 kA, ceil(1.25 * intact AC current / 0.05 kA) * 0.05 kA)',
        source_rule='ceil(1.25 * intact AC apparent power / 500 kVA) * 500 kVA',
        voltage_min_pu=.95, voltage_max_pu=1.05, source_apparent_power_kva=kva,
        branches=branches, added_cn_load_kw=0.,
        normal=dict(demand_kw=sum(n.active_power_kw for n in scenario.pdn_nodes),
                    demand_kvar=sum(n.reactive_power_kvar for n in scenario.pdn_nodes),
                    min_voltage_pu=vmin, max_voltage_pu=vmax,
                    upstream_p_kw=p, upstream_q_kvar=q, loss_kw=float(net.res_line.pl_mw.sum())*1000),
        limitations=['Balanced single-voltage equivalent, not a calibrated three-phase IEEE37 reproduction',
                    '709-775 remains the study same-voltage equivalent branch, not the stock transformer',
                    'Ratings and 25 percent headroom are study design assumptions, not equipment standards',
                    'No policy, seed, fault set or ranking was used to set these parameters'])
    profile['sha256'] = profile_digest(profile)
    return profile


def load_profile(path, scenario):
    profile = json.loads(Path(path).read_text(encoding='utf-8'))
    if profile.get('sha256') != profile_digest(profile):
        raise ValueError('Electrical profile digest mismatch')
    if profile['physical_signature'] != electrical_signature(scenario):
        raise ValueError('Frozen profile does not match scenario topology, demand or impedance')
    if set(profile['branches']) != {e.edge_id for e in scenario.pdn_edges}:
        raise ValueError('Incomplete electrical profile')
    return profile


def dispatch_from_profile(profile):
    a = ElectricalAssumptions(base_mva=profile['base_mva'], voltage_kv=profile['voltage_kv'],
        voltage_min_pu=profile['voltage_min_pu'], voltage_max_pu=profile['voltage_max_pu'],
        source_power_kw=profile['source_apparent_power_kva'],
        source_apparent_power_kva=profile['source_apparent_power_kva'])
    return IslandACDispatch(a,
        branch_ohms={k: (v['r_ohm'], v['x_ohm']) for k, v in profile['branches'].items()},
        branch_ratings_ka={k: v['rating_ka'] for k, v in profile['branches'].items()})
