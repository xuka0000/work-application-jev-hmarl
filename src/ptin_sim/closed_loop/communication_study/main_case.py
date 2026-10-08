"""Bind immutable fault cases to the user's approved EV/UAV main-study choices."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from ..scenario import load_closed_loop_scenario
from .electrical_profile import load_profile
from .uav_execution import UAVExecution


def load_main_case(workspace,case_id,*,input_manifest=None):
    root=Path(workspace).resolve()
    study=root/'outputs/complete_communication_study_20260918'
    registration=json.loads((study/'approved_resource_profile_20260919.json').read_text(encoding='utf-8'))
    manifest=study/'input_manifest.json' if input_manifest is None else Path(input_manifest)
    if not manifest.is_absolute():manifest=root/manifest
    if not manifest.resolve().is_relative_to(root.resolve()):raise ValueError('Case manifest escapes workspace')
    inputs=json.loads(manifest.read_text(encoding='utf-8'))
    # The manifest is authored on Windows, while frozen campaign bundles can
    # execute on Linux.  Normalize the registered relative path before both
    # matching and resolving it so pathlib does not treat backslashes as
    # literal filename characters on POSIX.
    candidates=[]
    for row in inputs['artifacts']:
        normalized=row['path'].replace('\\','/')
        if Path(normalized).name==case_id+'.json' and '/cases/' in normalized:
            candidates.append((row,normalized))
    if len(candidates)!=1:raise ValueError('Unknown or ambiguous registered case')
    candidate,relative=candidates[0];source=root/relative
    if not source.resolve().is_relative_to(root.resolve()):raise ValueError('Case artifact escapes workspace')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=candidate['sha256']:
        raise ValueError('Registered fault case was modified')
    case=json.loads(source.read_text(encoding='utf-8'))
    # Scenario override is explicit. Preserve the case's historical source label
    # for provenance; never resolve it back to the old 1.5 percent runtime.
    main_path=Path(registration['main_scenario'].replace('\\','/'))
    for filename in ('resource_fleet.yaml','disaster_scenario.yaml'):
        rel=str(main_path/filename).replace('\\','/')
        expected=next(v for k,v in registration['source_sha256'].items() if k.replace('\\','/')==rel)
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=expected:
            raise ValueError('Approved main scenario was modified')
    scenario=load_closed_loop_scenario(root/main_path)
    if scenario.fleet.v2g_willingness!=.3:raise ValueError('Main study requires approved 30 percent EV participation')
    scenario=replace(scenario,scenario_id=f'communication_main_ev30_{case_id}',
                     failed_pdn_edges=tuple(case['failed_edges']),
                     disaster=replace(scenario.disaster,failure_release_step_by_edge={
                         k:int(t/300) for k,t in case['private_release_time_s'].items()}))
    electrical=load_profile(study/'electrical_assumption_frozen/frozen_profile.json',scenario)
    values=registration['user_confirmed']
    depot=next(n for n in scenario.pdn_nodes if n.node_id=='744')
    uavs=UAVExecution(count=scenario.fleet.uav_count,depot_xy=(depot.coord_x_km,depot.coord_y_km),
        speed_km_h=scenario.fleet.uav_flight_speed_km_h,endurance_km=scenario.fleet.uav_flight_endurance_km,
        operation_s=values['uav_operation_s'],operation_equivalent_km=values['uav_operation_equivalent_km'],
        replacement_s=values['uav_replacement_s'])
    return scenario,case,electrical,uavs


def configure_main_capabilities(environment, *, max_age_s, require_freshness=True,
                                require_reconciliation=True):
    """User-approved five supply/repair ground units and four switch-only UAVs.

    Capabilities do not create units or teleport them. Ground presence requires
    an available, stationary ledger entry; UAV presence requires actual arrival
    at a bus and the operating phase of its finite mission.
    """
    env=environment
    ground={k:u for k,u in env.ledger.units.items() if u.kind in {'mess','ugv'}}
    if len(ground)!=5 or env.uav_execution is None or len(env.uav_execution.units)!=4:
        raise ValueError('Approved main capability profile requires exactly five ground units and four UAVs')
    capabilities={k:{'repair'} for k in ground}
    capabilities.update({k:{'switch'} for k in env.uav_execution.units})
    def at_bus(key):
        if key in ground:
            u=env.ledger.units[key]
            return u.location if u.available and u.destination is None else None
        u=env.uav_execution.units.get(key)
        if u is None or u['phase']!='operating':return None
        xy=tuple(u['position'])
        candidates=[n.node_id for n in env.scenario.pdn_nodes if (n.coord_x_km,n.coord_y_km)==xy]
        if len(candidates)!=1:return None
        return candidates[0]
    env.configure_control(capabilities=capabilities,unit_at_bus=at_bus,
        max_age_s=max_age_s,require_freshness=require_freshness,
        require_reconciliation=require_reconciliation)
    return capabilities
