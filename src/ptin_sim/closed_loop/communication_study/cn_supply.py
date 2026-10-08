"""User-confirmed common-load-switch interpretation of CN power availability.

No 5 kW/site load is added to the fixed PDN demand. A CN node can operate only
when its assigned bus's full recorded demand is served and voltage is within
bounds. This is a modelling assumption, not independently metered CN allocation.
Source discrepancy remains visible in the evidence returned to every event.
"""
from dataclasses import dataclass
import math

from .routing import NetworkState


@dataclass(frozen=True)
class CommonLoadSwitchCNSupply:
    voltage_min_pu: float = .95
    voltage_max_pu: float = 1.05
    full_service_tolerance_kw: float = .001
    voltage_tolerance_pu: float = 1e-6

    def __post_init__(self):
        values=(self.voltage_min_pu,self.voltage_max_pu,self.full_service_tolerance_kw,self.voltage_tolerance_pu)
        if not all(math.isfinite(x) for x in values) or not 0 < self.voltage_min_pu <= self.voltage_max_pu:
            raise ValueError("Invalid CN supply voltage limits")
        if min(self.full_service_tolerance_kw,self.voltage_tolerance_pu) < 0:
            raise ValueError("CN supply tolerances cannot be negative")

    def __call__(self, scenario, audit):
        buses={node.node_id:node for node in scenario.pdn_nodes}
        served=audit.get("served_load_by_node_kw",{})
        voltages=audit.get("bus_voltage_pu",{})
        converged=audit.get("status") in {"opf_ok","opf_fallback_power_flow_ok"}
        nodes=[]
        powered=set()
        for cn in scenario.cn_nodes:
            bus=buses.get(cn.power_supply_bus)
            power=served.get(cn.power_supply_bus)
            voltage=voltages.get(cn.power_supply_bus)
            reasons=[]
            if not converged:
                reasons.append("electrical_solution_not_converged")
            if bus is None or bus.active_power_kw <= 0:
                reasons.append("no_positive_recorded_bus_load")
            if power is None or not math.isfinite(power):
                reasons.append("bus_service_unknown")
            elif bus is not None and power+self.full_service_tolerance_kw < bus.active_power_kw:
                reasons.append("bus_load_not_fully_served")
            if voltage is None or not math.isfinite(voltage):
                reasons.append("bus_voltage_unknown")
            elif not self.voltage_min_pu-self.voltage_tolerance_pu <= voltage <= self.voltage_max_pu+self.voltage_tolerance_pu:
                reasons.append("bus_voltage_out_of_bounds")
            if not reasons:
                powered.add(cn.node_id)
            nodes.append(dict(cn_id=cn.node_id,power_supply_bus=cn.power_supply_bus,
                bus_requested_kw=bus.active_power_kw if bus else None,bus_served_kw=power,bus_voltage_pu=voltage,
                available=not reasons,blockers=reasons,
                cn_nameplate_kw_not_added=cn.demand_kw))
        evidence=dict(interpretation="user_confirmed_common_bus_load_switch_no_additional_CN_demand",
            added_demand_kw=0.,pdn_fixed_demand_kw=sum(max(0.,n.active_power_kw) for n in scenario.pdn_nodes),
            voltage_limits_pu=[self.voltage_min_pu,self.voltage_max_pu],
            full_service_tolerance_kw=self.full_service_tolerance_kw,voltage_tolerance_pu=self.voltage_tolerance_pu,
            separately_metered_CN_power=False,nodes=nodes)
        return NetworkState(float(audit["time_s"]),frozenset(powered)),evidence
