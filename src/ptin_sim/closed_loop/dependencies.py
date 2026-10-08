from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import PTINDependency, PTINScenario


@dataclass(frozen=True)
class CrossLayerFeasibility:
    feasible: bool
    blockers: tuple[str, ...]
    switch_controller_count: int
    powered_switch_controller_count: int
    uav_relay_used: bool
    traffic_feasible: bool
    resource_energy_feasible: bool
    resource_energy_required_kwh: float
    resource_energy_remaining_kwh: float
    v2g_energy_feasible: bool = True
    v2g_energy_required_kwh: float = 0.0
    v2g_energy_remaining_kwh: float = 0.0
    uav_relay_coverage_feasible: bool = True
    uav_communication_radius_km: float = 0.0
    uav_relay_required_radius_km: float = 0.0


def dependency_sources(
    scenario: PTINScenario,
    *,
    target_network: str,
    dependency_type: str | None = None,
) -> dict[str, tuple[PTINDependency, ...]]:
    grouped: dict[str, list[PTINDependency]] = {}
    for dependency in scenario.dependencies:
        if dependency.target_network != target_network:
            continue
        if dependency_type is not None and dependency.dependency_type != dependency_type:
            continue
        grouped.setdefault(dependency.target_id, []).append(dependency)
    return {key: tuple(value) for key, value in grouped.items()}


def failed_edge_restoration_load_kw(scenario: PTINScenario, edge_id: str) -> float:
    edge = next((item for item in scenario.pdn_edges if item.edge_id == edge_id), None)
    if edge is None:
        return 0.0
    node = next((item for item in scenario.pdn_nodes if item.node_id == edge.to_node), None)
    return node.active_power_kw if node is not None else 0.0


def critical_load_weight(scenario: PTINScenario, edge_id: str) -> float:
    edge = next((item for item in scenario.pdn_edges if item.edge_id == edge_id), None)
    if edge is None:
        return 1.0
    node = next((item for item in scenario.pdn_nodes if item.node_id == edge.to_node), None)
    if node is None:
        return 1.0
    return 1.35 if node.criticality.lower() == "critical" else 1.0


def topology_vulnerability_score(scenario: PTINScenario, edge_id: str) -> float:
    edge = next((item for item in scenario.pdn_edges if item.edge_id == edge_id), None)
    if edge is None:
        return 0.0
    degree: dict[str, int] = {}
    max_length = max((item.length_km for item in scenario.pdn_edges), default=1.0)
    for item in scenario.pdn_edges:
        degree[item.from_node] = degree.get(item.from_node, 0) + 1
        degree[item.to_node] = degree.get(item.to_node, 0) + 1
    endpoint_degree = max(1, min(degree.get(edge.from_node, 1), degree.get(edge.to_node, 1)))
    radial_exposure = 1.0 / endpoint_degree
    length_exposure = edge.length_km / max(max_length, 1.0e-6)
    return round(radial_exposure + 0.5 * length_exposure, 6)


def weighted_restoration_value_kw(scenario: PTINScenario, edge_id: str) -> float:
    raw_load = failed_edge_restoration_load_kw(scenario, edge_id)
    critical_weight = critical_load_weight(scenario, edge_id)
    vulnerability = topology_vulnerability_score(scenario, edge_id)
    vulnerability_weight = max(0.0, scenario.disaster.topology_vulnerability_weight)
    return round(raw_load * (critical_weight + vulnerability_weight * vulnerability), 6)


def robust_mean_travel_time_s(scenario: PTINScenario, mean_travel_time_s: float) -> float:
    multiplier = max(1.0, scenario.disaster.robust_travel_time_multiplier)
    return round(max(0.0, mean_travel_time_s) * multiplier, 6)


def target_utn_edge_ids_for_pdn_edge(
    scenario: PTINScenario,
    target_pdn_edge_id: str,
) -> tuple[str, ...]:
    edge = next((item for item in scenario.pdn_edges if item.edge_id == target_pdn_edge_id), None)
    if edge is None:
        return ()
    endpoint_buses = {edge.from_node, edge.to_node}
    edge_ids = sorted(
        {
            dependency.target_id
            for dependency in scenario.dependencies
            if dependency.source_network == "PDN"
            and dependency.source_id in endpoint_buses
            and dependency.target_network == "UTN_EDGE"
            and dependency.dependency_type == "road_lighting_power"
        }
    )
    return tuple(edge_ids)


def target_travel_time_s(
    scenario: PTINScenario,
    target_pdn_edge_id: str,
    traffic_result: Any,
) -> tuple[float, tuple[str, ...]]:
    travel_times = getattr(traffic_result, "edge_travel_time_s", {}) or {}
    target_edges = target_utn_edge_ids_for_pdn_edge(scenario, target_pdn_edge_id)
    matched = [
        float(travel_times[edge_id])
        for edge_id in target_edges
        if edge_id in travel_times
    ]
    if matched:
        return round(sum(matched) / len(matched), 6), tuple(
            edge_id for edge_id in target_edges if edge_id in travel_times
        )
    return round(float(getattr(traffic_result, "mean_travel_time_s", 0.0)), 6), ()


def uav_relay_required_radius_km(
    scenario: PTINScenario,
    target_pdn_edge_id: str,
) -> float:
    edge = next(
        (item for item in scenario.pdn_edges if item.edge_id == target_pdn_edge_id),
        None,
    )
    if edge is None:
        return 0.0
    from_node = next(
        (item for item in scenario.pdn_nodes if item.node_id == edge.from_node),
        None,
    )
    to_node = next(
        (item for item in scenario.pdn_nodes if item.node_id == edge.to_node),
        None,
    )
    if from_node is None or to_node is None:
        return 0.0
    controller_ids = switch_controller_cn_ids(
        scenario,
        target_pdn_edge_id=target_pdn_edge_id,
    )
    controller_distances: list[float] = []
    for controller_id in controller_ids:
        controller = next(
            (item for item in scenario.cn_nodes if item.node_id == controller_id),
            None,
        )
        if controller is None:
            continue
        controller_distances.append(
            _point_to_segment_distance_km(
                point_x=controller.coord_x_km,
                point_y=controller.coord_y_km,
                start_x=from_node.coord_x_km,
                start_y=from_node.coord_y_km,
                end_x=to_node.coord_x_km,
                end_y=to_node.coord_y_km,
            )
        )
    if not controller_distances:
        return 0.0
    return round(max(controller_distances), 6)


def _point_to_segment_distance_km(
    *,
    point_x: float,
    point_y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> float:
    dx = end_x - start_x
    dy = end_y - start_y
    length_sq = dx * dx + dy * dy
    if length_sq <= 1.0e-12:
        nearest_x = start_x
        nearest_y = start_y
    else:
        raw_t = ((point_x - start_x) * dx + (point_y - start_y) * dy) / length_sq
        t = max(0.0, min(1.0, raw_t))
        nearest_x = start_x + t * dx
        nearest_y = start_y + t * dy
    return ((point_x - nearest_x) ** 2 + (point_y - nearest_y) ** 2) ** 0.5


def evaluate_cross_layer_feasibility(
    scenario: PTINScenario,
    *,
    target_pdn_edge_id: str,
    closed_pdn_edges: set[str],
    communication_control_available: bool,
    traffic_result: Any,
    resource_energy_remaining_kwh: float,
    projected_mess_support_kw: float,
    step_minutes: float,
    v2g_energy_remaining_kwh: float = 0.0,
    projected_v2g_support_kw: float = 0.0,
    source_node_id: str = "799",
    max_mean_travel_time_s: float = 3600.0,
    requested_uav_relay: bool = False,
) -> CrossLayerFeasibility:
    energized_nodes = energized_pdn_node_ids(
        scenario,
        source_node_id=source_node_id,
        closed_pdn_edges=closed_pdn_edges,
    )
    powered_cn = powered_cn_node_ids(scenario, energized_pdn_nodes=energized_nodes)
    switch_controllers = switch_controller_cn_ids(
        scenario,
        target_pdn_edge_id=target_pdn_edge_id,
    )
    powered_controller_count = sum(1 for cn_id in switch_controllers if cn_id in powered_cn)
    relay_required_radius_km = uav_relay_required_radius_km(
        scenario,
        target_pdn_edge_id,
    )
    uav_radius_km = round(max(0.0, scenario.fleet.uav_communication_radius_km), 6)
    relay_needed_for_switch_control = (
        bool(switch_controllers) and powered_controller_count == 0
    )
    relay_coverage_feasible = (
        not (requested_uav_relay and relay_needed_for_switch_control)
        or relay_required_radius_km <= uav_radius_km + 1.0e-9
    )
    uav_relay_used = (
        relay_needed_for_switch_control
        and scenario.fleet.uav_count > 0
        and communication_control_available
        and requested_uav_relay
        and relay_coverage_feasible
    )
    switch_control_feasible = (
        not switch_controllers
        or powered_controller_count > 0
        or uav_relay_used
    )
    target_time_s, _target_edges = target_travel_time_s(
        scenario,
        target_pdn_edge_id,
        traffic_result,
    )
    robust_travel_time = robust_mean_travel_time_s(scenario, target_time_s)
    traffic_feasible = (
        str(getattr(traffic_result, "status", "")) == "ok"
        and robust_travel_time <= max_mean_travel_time_s
    )
    required_kwh = round(max(0.0, projected_mess_support_kw) * max(0.0, step_minutes) / 60.0, 6)
    remaining_kwh = round(max(0.0, resource_energy_remaining_kwh), 6)
    mess_energy_feasible = remaining_kwh + 1.0e-9 >= required_kwh
    v2g_required_kwh = round(max(0.0, projected_v2g_support_kw) * max(0.0, step_minutes) / 60.0, 6)
    v2g_remaining_kwh = round(max(0.0, v2g_energy_remaining_kwh), 6)
    v2g_energy_feasible = v2g_remaining_kwh + 1.0e-9 >= v2g_required_kwh
    resource_energy_feasible = mess_energy_feasible and v2g_energy_feasible

    blockers: list[str] = []
    if not communication_control_available:
        blockers.append("communication_control_unavailable")
    if requested_uav_relay and scenario.fleet.uav_count <= 0:
        blockers.append("uav_relay_unavailable")
    if (
        requested_uav_relay
        and scenario.fleet.uav_count > 0
        and relay_needed_for_switch_control
        and not relay_coverage_feasible
    ):
        blockers.append("uav_relay_coverage_unavailable")
    if not switch_control_feasible:
        blockers.append("switch_controller_unpowered")
    if not traffic_feasible:
        blockers.append("traffic_support_unavailable")
    if not resource_energy_feasible:
        blockers.append("resource_energy_unavailable")
    if mess_energy_feasible and not v2g_energy_feasible:
        blockers[-1] = "v2g_energy_unavailable"

    return CrossLayerFeasibility(
        feasible=not blockers,
        blockers=tuple(blockers),
        switch_controller_count=len(switch_controllers),
        powered_switch_controller_count=powered_controller_count,
        uav_relay_used=uav_relay_used,
        traffic_feasible=traffic_feasible,
        resource_energy_feasible=resource_energy_feasible,
        resource_energy_required_kwh=required_kwh,
        resource_energy_remaining_kwh=remaining_kwh,
        v2g_energy_feasible=v2g_energy_feasible,
        v2g_energy_required_kwh=v2g_required_kwh,
        v2g_energy_remaining_kwh=v2g_remaining_kwh,
        uav_relay_coverage_feasible=relay_coverage_feasible,
        uav_communication_radius_km=uav_radius_km,
        uav_relay_required_radius_km=relay_required_radius_km,
    )


def energized_pdn_node_ids(
    scenario: PTINScenario,
    *,
    source_node_id: str,
    closed_pdn_edges: set[str],
) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for edge in scenario.pdn_edges:
        if edge.edge_id not in closed_pdn_edges:
            continue
        adjacency.setdefault(edge.from_node, set()).add(edge.to_node)
        adjacency.setdefault(edge.to_node, set()).add(edge.from_node)

    visited: set[str] = set()
    stack = [source_node_id]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.extend(sorted(adjacency.get(node_id, set()) - visited))
    return visited


def powered_cn_node_ids(
    scenario: PTINScenario,
    *,
    energized_pdn_nodes: set[str],
) -> set[str]:
    powered: set[str] = set()
    for dependency in scenario.dependencies:
        if (
            dependency.source_network == "PDN"
            and dependency.target_network == "CN"
            and dependency.dependency_type == "power_supply"
            and dependency.source_id in energized_pdn_nodes
        ):
            powered.add(dependency.target_id)
    return powered


def switch_controller_cn_ids(
    scenario: PTINScenario,
    *,
    target_pdn_edge_id: str,
) -> tuple[str, ...]:
    return tuple(
        dependency.source_id
        for dependency in scenario.dependencies
        if dependency.source_network == "CN"
        and dependency.target_network == "PDN_EDGE"
        and dependency.target_id == target_pdn_edge_id
        and dependency.dependency_type == "switch_controllability"
    )
