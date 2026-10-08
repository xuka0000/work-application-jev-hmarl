"""Build the individual physical plant from an explicit resource configuration.

No numerical resource defaults are supplied here. Configuration provenance is
recorded by the caller; an integration configuration is not a formal freeze.
"""
import math

from .main_case import load_main_case,configure_main_capabilities
from .resource_ledger import BatteryUnit,ResourceLedger
from .sited_resource_dispatch import Inverter
from .ev_calendar import build_ev_calendar
from .road_execution import RoadFleetExecution
from .physical_event_environment import PhysicalEventEnvironment
from .physical_actions import PhysicalActionSession


def build_physical_case(workspace,case_id,*,resources,channel,trace=None,input_manifest=None):
    s,case,profile,uavs=load_main_case(workspace,case_id,input_manifest=input_manifest)
    horizon=channel.horizon_s
    sites=resources['connection_sites'];depot=resources['depot_bus']
    v2g_enabled=bool(resources.get('v2g_supply_enabled',True))
    mess_supply_enabled=bool(resources.get('mess_supply_enabled',True))
    identities=[f'EV_{i+1:04d}' for i in range(math.floor(s.fleet.ev_count*s.fleet.v2g_willingness))]
    if resources['ev_site_assignment']!='cyclic_identity_order':raise ValueError('Unregistered site assignment')
    assigned={key:sites[i%len(sites)] for i,key in enumerate(identities)}
    ev,availability,calendar=build_ev_calendar(s.fleet,site_by_unit=assigned,
        initial_soc_by_unit={key:resources['ev_initial_soc'] for key in identities},horizon_s=horizon)
    if not v2g_enabled:
        for unit in ev:unit.max_power_kw=0.
    ground=[BatteryUnit(f'MESS_{i+1:02d}','mess',s.fleet.mess_unit_energy_kwh,
        s.fleet.mess_unit_energy_kwh*resources['mess_initial_soc'],
        s.fleet.mess_unit_power_kw if mess_supply_enabled else 0.,depot)
        for i in range(s.fleet.mess_count)]
    ledger=ResourceLedger(ground+ev,discharge_efficiency=resources['discharge_efficiency'])
    connections=[dict(time_s=0.,unit_id=u.unit_id,bus_id=u.location) for u in ev if u.available]
    # Availability is explicitly defined as a vehicle ready at its station.
    # Every return is a separate connection event; residual energy is retained.
    if resources['ev_availability_meaning']!='station_ready_to_connect':
        raise ValueError('Explicit EV connection semantics required')
    connections += [dict(time_s=e['time_s'],unit_id=e['unit_id'],bus_id=assigned[e['unit_id']])
                    for e in availability if e['available']]
    inverters={key:Inverter(u.max_power_kw*resources['q_to_p_rating'],
        u.max_power_kw*resources['s_to_p_rating'],resources['mess_grid_forming'] if u.kind=='mess'
        else resources['ev_grid_forming']) for key,u in ledger.units.items()}
    roads=RoadFleetExecution(s,ledger,connection_sites=sites,connection_s=resources['connection_s'],
        traction_kwh_per_km=resources['traction_kwh_per_km'],traffic_factor=resources['road_time_factor'],
        service_sites=[n.node_id for n in s.pdn_nodes],service_setup_s=resources['repair_site_setup_s'])
    point=next(n for n in s.pdn_nodes if n.node_id==depot)
    depot_nodes=[n.node_id for n in s.cn_nodes if math.dist((point.coord_x_km,point.coord_y_km),
        (n.coord_x_km,n.coord_y_km))<=n.coverage_radius_km]
    if not depot_nodes:raise ValueError('Depot has no source-radius CN coverage')
    env=PhysicalEventEnvironment(s,profile,ledger,inverters,channel=channel,
        release_time_s=case['private_release_time_s'],availability_events=availability,
        connection_events=connections,horizon_s=horizon,trace=trace,road_execution=roads,
        uav_execution=uavs,onsite_operation_check=lambda *a:(False,'not_configured'),
        authorization_ttl_s=resources['authorization_ttl_s'],depot_cn_nodes=depot_nodes,
        mobile_radio=resources['stationary_mobile_radio'],
        relay_radius_km=s.fleet.uav_communication_radius_km if resources.get('finite_relay_enabled',False) else None,
        cached_local_repair=resources.get('cached_local_repair',False),
        fixed_relay_buses=resources.get('fixed_relay_buses',()))
    caps=configure_main_capabilities(env,max_age_s=resources['authorization_max_age_s'],
        require_freshness=resources.get('require_freshness',True),
        require_reconciliation=resources.get('require_reconciliation',True))
    env.configure_uav_control()
    env.local_repair_durations={e.edge_id:resources['repair_work_s'] for e in s.pdn_edges}
    session=PhysicalActionSession(env,capabilities=caps,
        repair_duration_s={e.edge_id:resources['repair_work_s'] for e in s.pdn_edges},
        switch_duration_s=uavs.operation_s,max_age_s=resources['observation_max_age_s'],
        lifetime_s=resources['command_lifetime_s'],decision_period_s=resources['decision_period_s'],
        reward_metric=resources.get('reward_metric','system_ens_kwh'))
    env.resource_configuration=dict(resources,uav_operation_s=uavs.operation_s,
        uav_replacement_s=uavs.replacement_s,relay_radius_km=env.relay_radius_km)
    if trace:trace.emit('resource_configuration_bound',0.,dict(resources=resources,ev_calendar=calendar,
        assigned_sites=assigned,capabilities=caps,depot_cn_nodes=depot_nodes,initial_devices=ledger.snapshot()))
    return env,session
