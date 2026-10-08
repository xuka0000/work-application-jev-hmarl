"""Load and validate the reconstructed PTIN scenario skeleton.

This script intentionally performs only lightweight CSV/YAML-adjacent checks.
The first project gate is to verify that the paper-reconstructed scenario has
the expected network and fleet counts before simulator implementation begins.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_DIR = ROOT / "data" / "scenario_reconstruction"
DEFAULT_EXPECTED_COUNTS = {
    "pdn_nodes": 37,
    "utn_nodes": 24,
    "cn_nodes": 42,
    "mess_ugv_count": 5,
    "uav_count": 5,
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def count_resource_ids(resource_text: str, prefix: str) -> int:
    """Count resource groups from simple YAML content when no YAML parser is required."""
    if prefix == "mess_ugv":
        marker = "mess_ugv:"
    elif prefix == "uav":
        marker = "uav:"
    else:
        marker = prefix
    return 1 if marker in resource_text else 0


def extract_simple_count(resource_text: str, section: str) -> int | None:
    lines = resource_text.splitlines()
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped == f"{section}:":
            inside = True
            continue
        if inside and stripped.endswith(":") and not line.startswith(" "):
            return None
        if inside and stripped.startswith("count:"):
            return int(stripped.split(":", 1)[1].strip())
    return None


def extract_target_counts(config_text: str) -> dict[str, int]:
    """Extract the simple target_counts block without requiring a YAML dependency."""
    counts: dict[str, int] = {}
    inside = False
    aliases = {
        "pdn_nodes": "pdn_nodes",
        "utn_nodes": "utn_nodes",
        "cn_nodes": "cn_nodes",
        "mess_ugv_units": "mess_ugv_count",
        "uav_units": "uav_count",
    }
    for line in config_text.splitlines():
        stripped = line.strip()
        if stripped == "target_counts:":
            inside = True
            continue
        if inside and stripped.endswith(":") and not line.startswith(" "):
            break
        if not inside or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = aliases.get(key.strip())
        if key is None:
            continue
        try:
            counts[key] = int(value.strip())
        except ValueError:
            continue
    return counts


def tbd_count(rows: Iterable[dict[str, str]]) -> int:
    total = 0
    for row in rows:
        for value in row.values():
            if value is not None and "TBD" in value:
                total += 1
    return total


def connected_node_count(nodes: list[dict[str, str]], edges: list[dict[str, str]]) -> int:
    ids = {row["node_id"] for row in nodes}
    if not ids:
        return 0
    adjacency = {node_id: set() for node_id in ids}
    for edge in edges:
        source = edge.get("from_node")
        target = edge.get("to_node")
        if source in ids and target in ids:
            adjacency[source].add(target)
            adjacency[target].add(source)
    start = next(iter(ids))
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen)


def validate(scenario_dir: Path) -> tuple[dict[str, int | str], list[str]]:
    pdn_nodes = read_csv_rows(scenario_dir / "pdn_nodes.csv")
    pdn_edges = read_csv_rows(scenario_dir / "pdn_edges.csv")
    utn_nodes = read_csv_rows(scenario_dir / "utn_nodes.csv")
    utn_edges = read_csv_rows(scenario_dir / "utn_edges.csv")
    cn_nodes = read_csv_rows(scenario_dir / "cn_nodes.csv")
    cn_edges_path = scenario_dir / "cn_edges.csv"
    cn_edges = read_csv_rows(cn_edges_path) if cn_edges_path.exists() else []
    dependencies = read_csv_rows(scenario_dir / "dependencies.csv")
    resource_text = (scenario_dir / "resource_fleet.yaml").read_text(encoding="utf-8")
    config_path = scenario_dir / "ptin_case_config.yaml"
    expected = dict(DEFAULT_EXPECTED_COUNTS)
    if config_path.exists():
        expected.update(extract_target_counts(config_path.read_text(encoding="utf-8")))

    summary: dict[str, int | str] = {
        "pdn_nodes": len(pdn_nodes),
        "pdn_edges": len(pdn_edges),
        "pdn_connected_nodes": connected_node_count(pdn_nodes, pdn_edges),
        "utn_nodes": len(utn_nodes),
        "utn_edges": len(utn_edges),
        "utn_connected_nodes": connected_node_count(utn_nodes, utn_edges),
        "cn_nodes": len(cn_nodes),
        "cn_edges": len(cn_edges),
        "cn_connected_nodes": connected_node_count(cn_nodes, cn_edges) if cn_edges else 0,
        "dependency_rules": len(dependencies),
        "mess_ugv_count": extract_simple_count(resource_text, "mess_ugv") or "missing",
        "uav_count": extract_simple_count(resource_text, "uav") or "missing",
        "pdn_tbd_fields": tbd_count(pdn_nodes) + tbd_count(pdn_edges),
        "utn_tbd_fields": tbd_count(utn_nodes) + tbd_count(utn_edges),
        "cn_tbd_fields": tbd_count(cn_nodes) + tbd_count(cn_edges),
        "dependency_tbd_fields": tbd_count(dependencies),
    }

    warnings: list[str] = []
    for key, target in expected.items():
        if summary[key] != target:
            warnings.append(f"{key}: expected {target}, got {summary[key]}")

    if summary["pdn_edges"] <= 1:
        warnings.append("pdn_edges are still placeholders; feeder connectivity must be reconstructed.")
    if summary["pdn_connected_nodes"] != summary["pdn_nodes"]:
        warnings.append("PDN graph is not connected under current edge table.")
    if summary["utn_connected_nodes"] != summary["utn_nodes"]:
        warnings.append("UTN graph is not connected under current edge table.")
    if summary["cn_edges"] == 0:
        warnings.append("CN edge table is missing.")
    elif summary["cn_connected_nodes"] != summary["cn_nodes"]:
        warnings.append("CN graph is not connected under current edge table.")
    if summary["dependency_tbd_fields"]:
        warnings.append("PTIN dependency mappings are rule placeholders, not node-level mappings.")
    if summary["cn_tbd_fields"]:
        warnings.append("CN coordinates/links are synthetic placeholders and require reconstruction.")
    return summary, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    args = parser.parse_args()

    summary, warnings = validate(args.scenario_dir)
    print("Scenario skeleton summary")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    if warnings:
        print("Warnings")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("Warnings: none")
    return 0 if not any("expected" in w for w in warnings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
