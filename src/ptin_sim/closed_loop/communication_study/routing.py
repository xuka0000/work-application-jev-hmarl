"""Target-specific routing over the source CN graph and supplied power events.

The source graph is treated as bidirectional. Power availability is an explicit
input, not inferred from PDN connectivity or fabricated from a radio radius.
This supplies physical admission to an end-to-end synthetic channel; it is not
a per-hop radio simulation. UAV links require a separate deployment model.
"""
from collections import deque
from dataclasses import dataclass
import math

from ..dependencies import switch_controller_cn_ids


_RELAY_LINK_NODE_PREFIX = {
    'relay:': 'uav:',
    'fixed_relay:': 'fixed:',
}


def _relay_link_node_prefix(link_id):
    return next((node_prefix for link_prefix,node_prefix in _RELAY_LINK_NODE_PREFIX.items()
                 if link_id.startswith(link_prefix)),None)


@dataclass(frozen=True)
class NetworkState:
    time_s: float
    powered_nodes: frozenset[str]
    disabled_edges: frozenset[str] = frozenset()
    mobile_attachments: tuple = ()
    relay_nodes: frozenset[str] = frozenset()
    relay_links: tuple = ()


class TargetRouter:
    def __init__(self, scenario, *, states, dispatcher_node=None, depot_nodes=()):
        self.scenario = scenario
        self.nodes = {n.node_id for n in scenario.cn_nodes}
        self.depot_nodes=tuple(sorted(set(depot_nodes)))
        if not set(self.depot_nodes)<=self.nodes:raise ValueError('Unknown depot CN endpoint')
        centers = [n.node_id for n in scenario.cn_nodes if n.role == "center"]
        if dispatcher_node is None:
            if len(centers) != 1:
                raise ValueError("An unambiguous dispatcher CN node is required")
            dispatcher_node = centers[0]
        if dispatcher_node not in self.nodes:
            raise ValueError("Unknown dispatcher CN node")
        self.dispatcher_node = dispatcher_node
        self.edges = {e.edge_id: e for e in scenario.cn_edges}
        if len(self.edges) != len(scenario.cn_edges):
            raise ValueError("Duplicate CN edge IDs")
        if any(e.from_node not in self.nodes or e.to_node not in self.nodes for e in self.edges.values()):
            raise ValueError("Unknown CN edge endpoint")
        self.adjacency = {node: [] for node in self.nodes}
        for edge in self.edges.values():
            self.adjacency[edge.from_node].append((edge.to_node, edge.edge_id))
            self.adjacency[edge.to_node].append((edge.from_node, edge.edge_id))
        for neighbors in self.adjacency.values():
            neighbors.sort()
        self.states = []
        for state in states:
            self.append_state(state)
        if not self.states or self.states[0].time_s != 0:
            raise ValueError("Network state at physical time zero is required")

    def append_state(self, state):
        if not math.isfinite(state.time_s) or state.time_s < 0:
            raise ValueError("Invalid network event time")
        if self.states and state.time_s <= self.states[-1].time_s:
            raise ValueError("Network events must be strictly ordered")
        if not state.powered_nodes <= self.nodes or not state.disabled_edges <= self.edges.keys():
            raise ValueError("Unknown powered node or disabled edge")
        available=self.nodes|state.relay_nodes
        if (any(not k.startswith(('uav:','fixed:')) for k in state.relay_nodes)
                or len({link[2] for link in state.relay_links})!=len(state.relay_links)
                or any(a not in available or b not in available
                       or _relay_link_node_prefix(k) is None
                       or not any(node.startswith(_relay_link_node_prefix(k))
                                  for node in ({a,b}&state.relay_nodes))
                       for a,b,k in state.relay_links)):
            raise ValueError('Invalid finite relay topology')
        if (len(dict(state.mobile_attachments))!=len(state.mobile_attachments)
                or any(not key.startswith('unit:') or not set(nodes)<=available for key,nodes in state.mobile_attachments)):
            raise ValueError('Invalid mobile radio attachment')
        self.states.append(state)

    def state_at(self, time_s):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Invalid route query time")
        return next(state for state in reversed(self.states) if state.time_s <= time_s)

    def _endpoints(self, message, state):
        controllers = tuple(sorted(set(switch_controller_cn_ids(self.scenario, target_pdn_edge_id=message.target_id))))
        if not controllers or any(node not in self.nodes for node in controllers):
            return None
        def resolve(endpoint):
            if endpoint == "dispatcher":
                return (self.dispatcher_node,)
            if endpoint == 'depot':
                return self.depot_nodes
            if endpoint.startswith('unit:'):
                return dict(state.mobile_attachments).get(endpoint,())
            if endpoint in {"terminal", "plant"}:
                return controllers
            # Explicit CN endpoint must belong to the target's controller set.
            if endpoint in controllers:
                return (endpoint,)
            return ()
        return resolve(message.source), resolve(message.destination)

    def admit(self, message, time_s):
        state = self.state_at(time_s)
        endpoints = self._endpoints(message,state)
        base = dict(target_id=message.target_id, model="source_CN_bidirectional_power_gated_end_to_end_channel")
        if endpoints is None or not all(endpoints):
            return dict(base, admitted=False, reason="unmapped_target_or_endpoint", nodes=[], edges=[])
        sources, destinations = endpoints
        # An explicitly configured target endpoint is required even for telemetry;
        # a global PTIN payload cannot silently use a single abstract route.
        powered=state.powered_nodes|state.relay_nodes
        adjacency={k:list(v) for k,v in self.adjacency.items()}
        for a,b,key in state.relay_links:
            adjacency.setdefault(a,[]).append((b,key));adjacency.setdefault(b,[]).append((a,key))
        queue = deque((source, [source], []) for source in sources if source in powered)
        seen = {item[0] for item in queue}
        while queue:
            node, nodes, edges = queue.popleft()
            if node in destinations:
                return dict(base, admitted=True, reason="powered_path", nodes=nodes, edges=edges,
                            sent_s=time_s, network_state_time_s=state.time_s)
            for neighbor, edge_id in sorted(adjacency.get(node,())):
                if neighbor not in seen and neighbor in powered and edge_id not in state.disabled_edges:
                    seen.add(neighbor)
                    queue.append((neighbor, nodes + [neighbor], edges + [edge_id]))
        return dict(base, admitted=False, reason="no_powered_target_path", nodes=[], edges=[])

    def validate(self, message, route, arrival_s):
        """Check the original route through every intervening power/link event.

        No implicit reroute or successful delivery after a mid-flight outage.
        Event sources must have supplied all changes up to the arrival clock.
        """
        if route["target_id"] != message.target_id or not route["admitted"]:
            return False, "route_identity_mismatch"
        if arrival_s < route["sent_s"]:
            raise ValueError("Arrival predates sending")
        states = [self.state_at(route["sent_s"])] + [s for s in self.states if route["sent_s"] < s.time_s <= arrival_s]
        for state in states:
            endpoints=self._endpoints(message,state)
            if not endpoints or route['nodes'][0] not in endpoints[0] or route['nodes'][-1] not in endpoints[1]:
                return False,'endpoint_attachment_lost_in_transit'
            if not set(route["nodes"]) <= state.powered_nodes|state.relay_nodes:
                return False, "route_power_lost_in_transit"
            if set(route["edges"]) & state.disabled_edges:
                return False, "route_link_lost_in_transit"
            if not {k for k in route['edges'] if _relay_link_node_prefix(k) is not None} <= {
                    r[2] for r in state.relay_links}:
                return False,'relay_link_lost_in_transit'
        return True, "powered_route_valid"
