from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PDNNode:
    node_id: str
    coord_x_km: float
    coord_y_km: float
    active_power_kw: float
    reactive_power_kvar: float
    base_voltage_kv: float
    criticality: str
    raw: dict[str, str] = field(repr=False)


@dataclass(frozen=True)
class PDNEdge:
    edge_id: str
    from_node: str
    to_node: str
    status: str
    length_km: float
    r_pu: float
    x_pu: float
    normal_switch_state: str
    raw: dict[str, str] = field(repr=False)


@dataclass(frozen=True)
class CNNode:
    node_id: str
    role: str
    coord_x_km: float
    coord_y_km: float
    coverage_radius_km: float
    demand_kw: float
    power_supply_bus: str
    raw: dict[str, str] = field(repr=False)


@dataclass(frozen=True)
class CNEdge:
    edge_id: str
    from_node: str
    to_node: str
    raw: dict[str, str] = field(repr=False)


@dataclass(frozen=True)
class UTNNode:
    node_id: str
    coord_x_km: float
    coord_y_km: float
    raw: dict[str, str] = field(repr=False)


@dataclass(frozen=True)
class UTNEdge:
    edge_id: str
    from_node: str
    to_node: str
    length_km: float
    raw: dict[str, str] = field(repr=False)


@dataclass(frozen=True)
class PTINDependency:
    dependency_id: str
    source_network: str
    source_id: str
    target_network: str
    target_id: str
    dependency_type: str
    status: str
    raw: dict[str, str] = field(repr=False)


@dataclass(frozen=True)
class EVAvailabilityPoint:
    time_min: float
    availability_factor: float


@dataclass(frozen=True)
class FleetSpec:
    mess_count: int
    mess_unit_power_kw: float
    mess_unit_energy_kwh: float
    uav_count: int
    uav_communication_radius_km: float
    uav_flight_speed_km_h: float
    uav_flight_endurance_km: float
    ev_unit_power_kw: float
    ev_unit_energy_kwh: float
    ev_count: int = 0
    ev_ratio: float = 0.0
    v2g_willingness: float = 0.0
    ev_availability_profile: tuple[EVAvailabilityPoint, ...] = ()


def ev_availability_factor_at(fleet: FleetSpec, time_min: float) -> float:
    profile = sorted(fleet.ev_availability_profile, key=lambda point: point.time_min)
    if not profile:
        return 1.0
    selected = profile[0].availability_factor
    for point in profile:
        if float(point.time_min) <= float(time_min) + 1.0e-9:
            selected = point.availability_factor
        else:
            break
    return max(0.0, min(1.0, float(selected)))


def dispatchable_v2g_vehicle_count_at(fleet: FleetSpec, time_min: float) -> int:
    return int(
        round(
            max(0, fleet.ev_count)
            * max(0.0, min(1.0, fleet.v2g_willingness))
            * ev_availability_factor_at(fleet, time_min)
        )
    )


def v2g_energy_capacity_kwh_at(fleet: FleetSpec, time_min: float) -> float:
    return round(
        dispatchable_v2g_vehicle_count_at(fleet, time_min)
        * max(0.0, fleet.ev_unit_energy_kwh),
        6,
    )


@dataclass(frozen=True)
class DisasterProfile:
    disaster_category: str
    control_paradigm: str
    traffic_uncertainty_mode: str
    robust_travel_time_multiplier: float
    predeployment_quality_index: float
    topology_vulnerability_weight: float
    failure_release_step_by_edge: dict[str, int]


@dataclass(frozen=True)
class EmbodiedExecutionConstraints:
    task_authorized: bool = True
    task_packet_ttl_min: float = 120.0
    task_packet_age_min: float = 0.0
    local_memory_age_min: float = 0.0
    local_memory_confidence: float = 1.0
    reconnect_available: bool = True
    control_margin: float = 1.0
    risk_budget: float = 10.0
    risk_cost_base: float = 0.05
    risk_cost_vulnerability_weight: float = 0.25
    risk_cost_relay_penalty: float = 0.02
    communication_perturbation_identity: str = ""
    requested_packet_loss_rate: float = 0.0
    requested_packet_delay_ms: float = 0.0


@dataclass(frozen=True)
class ClosedLoopAction:
    action_id: str
    action_type: str
    target_id: str
    resource_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClosedLoopStepMetrics:
    step_index: int
    restored_load_kw: float
    min_voltage_pu: float | None
    max_line_loading_pct: float | None
    packet_delivery_rate: float
    mean_travel_time_s: float
    reward: float
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class PTINScenario:
    scenario_id: str
    scenario_dir: str
    pdn_nodes: tuple[PDNNode, ...]
    pdn_edges: tuple[PDNEdge, ...]
    cn_nodes: tuple[CNNode, ...]
    cn_edges: tuple[CNEdge, ...]
    utn_nodes: tuple[UTNNode, ...]
    utn_edges: tuple[UTNEdge, ...]
    dependencies: tuple[PTINDependency, ...]
    failed_pdn_edges: tuple[str, ...]
    fleet: FleetSpec
    disaster: DisasterProfile
    embodied_constraints: EmbodiedExecutionConstraints = field(
        default_factory=EmbodiedExecutionConstraints
    )

    def dependencies_by_target(self, target_network: str) -> dict[str, tuple[PTINDependency, ...]]:
        grouped: dict[str, list[PTINDependency]] = {}
        for dependency in self.dependencies:
            if dependency.target_network == target_network:
                grouped.setdefault(dependency.target_id, []).append(dependency)
        return {key: tuple(value) for key, value in grouped.items()}
