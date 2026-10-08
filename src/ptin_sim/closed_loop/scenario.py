from __future__ import annotations

from pathlib import Path

from ptin_sim.scenario_loader import read_csv_rows

from .types import (
    CNEdge,
    CNNode,
    DisasterProfile,
    EVAvailabilityPoint,
    EmbodiedExecutionConstraints,
    FleetSpec,
    PDNEdge,
    PDNNode,
    PTINDependency,
    PTINScenario,
    UTNEdge,
    UTNNode,
)


def load_closed_loop_scenario(scenario_dir: str | Path) -> PTINScenario:
    root = Path(scenario_dir)
    disaster_path = root / "disaster_scenario.yaml"
    data_root = _scenario_data_root(root, disaster_path)
    pdn_nodes = tuple(_pdn_node(row) for row in read_csv_rows(data_root / "pdn_nodes.csv"))
    pdn_edges = tuple(_pdn_edge(row) for row in read_csv_rows(data_root / "pdn_edges.csv"))
    cn_nodes = tuple(_cn_node(row) for row in read_csv_rows(data_root / "cn_nodes.csv"))
    cn_edges = tuple(_cn_edge(row) for row in read_csv_rows(data_root / "cn_edges.csv"))
    utn_nodes = tuple(_utn_node(row) for row in read_csv_rows(data_root / "utn_nodes.csv"))
    utn_edges = tuple(_utn_edge(row) for row in read_csv_rows(data_root / "utn_edges.csv"))
    dependencies = tuple(
        _dependency(row) for row in read_csv_rows(data_root / "dependencies.csv")
    )
    failed_pdn_edges = tuple(_failed_pdn_edges(disaster_path))
    return PTINScenario(
        scenario_id=_scenario_id(disaster_path),
        scenario_dir=str(root),
        pdn_nodes=pdn_nodes,
        pdn_edges=pdn_edges,
        cn_nodes=cn_nodes,
        cn_edges=cn_edges,
        utn_nodes=utn_nodes,
        utn_edges=utn_edges,
        dependencies=dependencies,
        failed_pdn_edges=failed_pdn_edges,
        fleet=_fleet_spec(_scenario_local_or_base_file(root, data_root, "resource_fleet.yaml")),
        disaster=_disaster_profile(disaster_path, failed_pdn_edges),
        embodied_constraints=_embodied_execution_constraints(disaster_path),
    )


def _scenario_data_root(root: Path, disaster_path: Path) -> Path:
    text = disaster_path.read_text(encoding="utf-8")
    base_dir = (
        _section_value(text, "source_data", "base_scenario_dir")
        or _section_value(text, "scenario", "base_scenario_dir")
        or _top_level_value(text, "base_scenario_dir")
    )
    if not base_dir:
        return root
    path = Path(base_dir)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _scenario_local_or_base_file(root: Path, data_root: Path, name: str) -> Path:
    local = root / name
    return local if local.exists() else data_root / name


def _pdn_node(row: dict[str, str]) -> PDNNode:
    return PDNNode(
        node_id=row["node_id"],
        coord_x_km=_float(row.get("coord_x_km")),
        coord_y_km=_float(row.get("coord_y_km")),
        active_power_kw=_float(row.get("active_power_kw") or row.get("base_load_kw")),
        reactive_power_kvar=_float(row.get("reactive_power_kvar")),
        base_voltage_kv=_float(row.get("base_voltage_kv"), 4.8),
        criticality=row.get("criticality", "normal"),
        raw=dict(row),
    )


def _pdn_edge(row: dict[str, str]) -> PDNEdge:
    return PDNEdge(
        edge_id=row["edge_id"],
        from_node=row["from_node"],
        to_node=row["to_node"],
        status=row.get("status", "normal"),
        length_km=_float(row.get("length_km"), 1.0),
        r_pu=_float(row.get("r_pu")),
        x_pu=_float(row.get("x_pu")),
        normal_switch_state=row.get("normal_switch_state", "closed"),
        raw=dict(row),
    )


def _cn_node(row: dict[str, str]) -> CNNode:
    return CNNode(
        node_id=row["node_id"],
        role=row.get("role", "base_station"),
        coord_x_km=_float(row.get("coord_x_km")),
        coord_y_km=_float(row.get("coord_y_km")),
        coverage_radius_km=_float(row.get("coverage_radius_km"), 3.0),
        demand_kw=_float(row.get("demand_kw"), 0.0),
        power_supply_bus=row.get("power_supply_bus", ""),
        raw=dict(row),
    )


def _cn_edge(row: dict[str, str]) -> CNEdge:
    return CNEdge(
        edge_id=row.get("edge_id") or f"{row.get('from_node')}_{row.get('to_node')}",
        from_node=row["from_node"],
        to_node=row["to_node"],
        raw=dict(row),
    )


def _utn_node(row: dict[str, str]) -> UTNNode:
    return UTNNode(
        node_id=row["node_id"],
        coord_x_km=_float(row.get("coord_x_km")),
        coord_y_km=_float(row.get("coord_y_km")),
        raw=dict(row),
    )


def _utn_edge(row: dict[str, str]) -> UTNEdge:
    return UTNEdge(
        edge_id=row["edge_id"],
        from_node=row["from_node"],
        to_node=row["to_node"],
        length_km=_float(row.get("length_km"), 1.0),
        raw=dict(row),
    )


def _dependency(row: dict[str, str]) -> PTINDependency:
    return PTINDependency(
        dependency_id=row["dependency_id"],
        source_network=row["source_network"],
        source_id=row["source_id"],
        target_network=row["target_network"],
        target_id=row["target_id"],
        dependency_type=row["dependency_type"],
        status=row.get("status", ""),
        raw=dict(row),
    )


def _scenario_id(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("id:"):
            return stripped.split(":", 1)[1].strip()
    return path.parent.name


def _failed_pdn_edges(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    inside = False
    failed: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "failed_pdn_line_switch_ids:":
            inside = True
            continue
        if inside and stripped.endswith(":") and not stripped.startswith("-"):
            break
        if inside and stripped.startswith("-"):
            edge_id = stripped[1:].strip()
            if edge_id:
                failed.append(edge_id)
    return failed


def _disaster_profile(path: Path, failed_pdn_edges: tuple[str, ...]) -> DisasterProfile:
    text = path.read_text(encoding="utf-8")
    scenario_type = _section_value(text, "scenario", "type") or ""
    default_category = "progressive" if "progressive" in scenario_type else "instantaneous"
    default_paradigm = (
        "mpc_rolling_horizon"
        if default_category == "progressive"
        else "two_stage_deterministic"
    )
    release_steps = _failure_release_steps(text, failed_pdn_edges)
    if not release_steps:
        release_steps = {edge_id: 0 for edge_id in failed_pdn_edges}
    return DisasterProfile(
        disaster_category=_section_value(
            text, "framework_alignment", "disaster_category"
        )
        or default_category,
        control_paradigm=_section_value(
            text, "framework_alignment", "control_paradigm"
        )
        or default_paradigm,
        traffic_uncertainty_mode=_section_value(
            text, "framework_alignment", "traffic_uncertainty_mode"
        )
        or "interval_robust",
        robust_travel_time_multiplier=_section_float(
            text, "framework_alignment", "robust_travel_time_multiplier", 1.0
        ),
        predeployment_quality_index=_section_float(
            text, "framework_alignment", "predeployment_quality_index", 1.0
        ),
        topology_vulnerability_weight=_section_float(
            text, "framework_alignment", "topology_vulnerability_weight", 0.0
        ),
        failure_release_step_by_edge=release_steps,
    )


def _embodied_execution_constraints(path: Path) -> EmbodiedExecutionConstraints:
    text = path.read_text(encoding="utf-8")
    section = "embodied_execution_constraints"
    return EmbodiedExecutionConstraints(
        task_authorized=_section_bool(text, section, "task_authorized", True),
        task_packet_ttl_min=_section_float(text, section, "task_packet_ttl_min", 120.0),
        task_packet_age_min=_section_float(text, section, "task_packet_age_min", 0.0),
        local_memory_age_min=_section_float(text, section, "local_memory_age_min", 0.0),
        local_memory_confidence=_section_float(
            text,
            section,
            "local_memory_confidence",
            1.0,
        ),
        reconnect_available=_section_bool(text, section, "reconnect_available", True),
        control_margin=_section_float(text, section, "control_margin", 1.0),
        risk_budget=_section_float(text, section, "risk_budget", 10.0),
        risk_cost_base=_section_float(text, section, "risk_cost_base", 0.05),
        risk_cost_vulnerability_weight=_section_float(
            text,
            section,
            "risk_cost_vulnerability_weight",
            0.25,
        ),
        risk_cost_relay_penalty=_section_float(
            text,
            section,
            "risk_cost_relay_penalty",
            0.02,
        ),
        communication_perturbation_identity=(
            _section_value(text, section, "communication_perturbation_identity") or ""
        ),
        requested_packet_loss_rate=_section_float(
            text, section, "requested_packet_loss_rate", 0.0
        ),
        requested_packet_delay_ms=_section_float(
            text, section, "requested_packet_delay_ms", 0.0
        ),
    )


def _failure_release_steps(text: str, failed_pdn_edges: tuple[str, ...]) -> dict[str, int]:
    raw_steps = _section_mapping(text, "failure_release_steps")
    out: dict[str, int] = {}
    for edge_id in failed_pdn_edges:
        try:
            out[edge_id] = int(float(raw_steps.get(edge_id, 0)))
        except (TypeError, ValueError):
            out[edge_id] = 0
    return out


def _fleet_spec(path: Path) -> FleetSpec:
    text = path.read_text(encoding="utf-8")
    vehicle_count = _section_int(
        text, "traffic_vehicle_population", "stable_operating_vehicle_count", 0
    )
    ev_ratio = _section_float(text, "ev_unit", "ev_ratio", 0.0)
    ev_count = int(round(max(0, vehicle_count) * max(0.0, ev_ratio)))
    return FleetSpec(
        mess_count=_section_int(text, "mess_ugv", "count", 0),
        mess_unit_power_kw=_section_float(text, "mess_ugv", "unit_power_kw", 0.0),
        mess_unit_energy_kwh=_section_float(text, "mess_ugv", "unit_energy_kwh", 0.0),
        uav_count=_section_int(text, "uav", "count", 0),
        uav_communication_radius_km=_section_float(text, "uav", "communication_radius_km", 0.0),
        uav_flight_speed_km_h=_section_float(text, "uav", "flight_speed_km_per_h", 0.0),
        uav_flight_endurance_km=_section_float(text, "uav", "flight_endurance_km", 0.0),
        ev_unit_power_kw=_section_float(text, "ev_unit", "unit_power_kw", 0.0),
        ev_unit_energy_kwh=_section_float(text, "ev_unit", "unit_energy_kwh", 0.0),
        ev_count=ev_count,
        ev_ratio=ev_ratio,
        v2g_willingness=_section_float(text, "ev_unit", "v2g_willingness", 0.0),
        ev_availability_profile=_ev_availability_profile(text),
    )


def _ev_availability_profile(text: str) -> tuple[EVAvailabilityPoint, ...]:
    rows = _section_list_mappings(text, "ev_availability_profile")
    profile: list[EVAvailabilityPoint] = []
    for row in rows:
        profile.append(
            EVAvailabilityPoint(
                time_min=_float(row.get("time_min")),
                availability_factor=_float(row.get("availability_factor"), 1.0),
            )
        )
    return tuple(sorted(profile, key=lambda point: point.time_min))


def _section_float(text: str, section: str, key: str, default: float) -> float:
    return _float(_section_value(text, section, key), default)


def _section_int(text: str, section: str, key: str, default: int) -> int:
    try:
        return int(float(_section_value(text, section, key)))
    except (TypeError, ValueError):
        return default


def _section_bool(text: str, section: str, key: str, default: bool) -> bool:
    value = _section_value(text, section, key)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _section_value(text: str, section: str, key: str) -> str | None:
    parent_indent: int | None = None
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped == f"{section}:":
            parent_indent = indent
            inside = True
            continue
        if inside and parent_indent is not None and indent <= parent_indent and stripped.endswith(":"):
            return None
        if inside and stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip().strip("'\"")
    return None


def _top_level_value(text: str, key: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0 and stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip().strip("'\"")
    return None


def _section_mapping(text: str, section: str) -> dict[str, str]:
    parent_indent: int | None = None
    inside = False
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped == f"{section}:":
            parent_indent = indent
            inside = True
            continue
        if inside and parent_indent is not None and indent <= parent_indent:
            break
        if inside and ":" in stripped and not stripped.startswith("-"):
            key, value = stripped.split(":", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def _section_list_mappings(text: str, section: str) -> list[dict[str, str]]:
    parent_indent: int | None = None
    inside = False
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped == f"{section}:":
            parent_indent = indent
            inside = True
            continue
        if inside and parent_indent is not None and indent <= parent_indent:
            break
        if not inside:
            continue
        if stripped.startswith("-"):
            if current is not None:
                rows.append(current)
            current = {}
            item = stripped[1:].strip()
            if ":" in item:
                key, value = item.split(":", 1)
                current[key.strip()] = value.strip().strip("'\"")
            continue
        if current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = value.strip().strip("'\"")
    if current is not None:
        rows.append(current)
    return rows


def _float(value: str | float | int | None, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
