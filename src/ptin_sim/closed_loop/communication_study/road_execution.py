"""Road-constrained MESS motion with observed route selection and local execution.

Each road segment holds its entry speed; subsequent segments read their local
facility supply on entry. This is the source normal/failed speed model, not
microscopic traffic flow. Traction and connection time are explicit inputs.
"""
import math
import networkx as nx


class RoadFleetExecution:
    def __init__(self, scenario, ledger, *, connection_sites, connection_s,
                 traction_kwh_per_km, traffic_factor=1., service_sites=(), service_setup_s=None):
        self.scenario, self.ledger = scenario, ledger
        self.sites = set(connection_sites)
        self.service_sites=set(service_sites)
        if self.service_sites and (service_setup_s is None or not math.isfinite(service_setup_s) or service_setup_s<=0):
            raise ValueError('Service destinations require an explicit positive site setup time')
        self.service_setup_s=service_setup_s
        self.connection_s, self.traction_rate, self.traffic_factor = connection_s, traction_kwh_per_km, traffic_factor
        if not all(math.isfinite(v) for v in (connection_s, traction_kwh_per_km, traffic_factor)) or (
                connection_s < 0 or traction_kwh_per_km < 0 or traffic_factor <= 0):
            raise ValueError('Explicit finite motion and connection parameters required')
        self.buses = {n.node_id:n for n in scenario.pdn_nodes}
        self.nodes = {n.node_id:n for n in scenario.utn_nodes}
        self.edges = {e.edge_id:e for e in scenario.utn_edges}
        if not (self.sites|self.service_sites) <= self.buses.keys() or not self.nodes:
            raise ValueError('Unknown connection site or empty road graph')
        self.bus_to_road = {k:min(self.nodes,key=lambda v: (
            (self.nodes[v].coord_x_km-n.coord_x_km)**2+(self.nodes[v].coord_y_km-n.coord_y_km)**2,v))
            for k,n in self.buses.items()}
        self.position = {k:self.bus_to_road[u.location] for k,u in ledger.units.items()}
        self.jobs, self.events = {}, []

    def _speed(self, edge, powered_buses):
        name = 'normal_speed_km_h' if edge.raw['power_supply_bus'] in powered_buses else 'failed_speed_km_h'
        value = float(edge.raw[name])/self.traffic_factor
        if not math.isfinite(value) or value <= 0 or not math.isfinite(edge.length_km) or edge.length_km <= 0:
            raise ValueError('Positive finite sourced road length and speed required')
        return value

    def route(self, unit_id, bus_id, *, observed_powered_buses, observed_s, observed_road_position=None,
              purpose='connect'):
        if not math.isfinite(observed_s) or not 0 <= observed_s <= self.ledger.time_s:
            raise ValueError('Route planning cannot use a future road observation')
        if purpose not in {'connect','service'}:raise ValueError('Unknown road trip purpose')
        sites=self.sites if purpose=='connect' else self.service_sites
        setup=self.connection_s if purpose=='connect' else self.service_setup_s
        if bus_id not in sites or (observed_road_position is None and unit_id in self.jobs):
            raise ValueError('Unknown connection site or vehicle already travelling')
        graph=nx.Graph();graph.add_nodes_from(self.nodes)
        for edge in sorted(self.edges.values(),key=lambda e:e.edge_id):
            weight=edge.length_km/self._speed(edge,observed_powered_buses)*3600
            if graph.has_edge(edge.from_node,edge.to_node):
                raise ValueError('Ambiguous parallel road edge requires explicit routing')
            graph.add_edge(edge.from_node,edge.to_node,seconds=weight,edge_id=edge.edge_id)
        # Policy proposals use the received position, never the live simulator
        # position. Execution calls the default path and rejects stale proposals.
        position=self.position[unit_id] if observed_road_position is None else observed_road_position
        if position not in self.nodes:raise ValueError('Unknown observed road position')
        path=nx.shortest_path(graph,position,self.bus_to_road[bus_id],weight='seconds')
        edges=[graph[a][b]['edge_id'] for a,b in zip(path,path[1:])]
        return dict(unit_id=unit_id,destination_bus=bus_id,purpose=purpose,site_setup_s=setup,
                    road_nodes=path,road_edges=edges,
                    observed_s=observed_s,observed_powered_buses=sorted(observed_powered_buses),
                    distance_km=sum(self.edges[e].length_km for e in edges),
                    estimated_duration_s=sum(graph[a][b]['seconds'] for a,b in zip(path,path[1:]))+setup)

    def start(self, route, *, powered_buses):
        key, target=route['unit_id'],route['destination_bus']
        # Reconstruct the proposal from its observable inputs. Neither a stale
        # position nor a caller-edited shortcut may move a vehicle off-network.
        expected=self.route(key,target,observed_powered_buses=set(route['observed_powered_buses']),
                            observed_s=route['observed_s'],purpose=route['purpose'])
        if expected != route:
            raise ValueError('Route proposal does not match current position and observation')
        unit=self.ledger.units[key]
        if unit.kind not in {'mess','ugv'}:
            raise ValueError('Road dispatch is limited to explicitly controlled MESS/UGV units')
        if not unit.available or target==unit.connected_bus:
            raise ValueError('Unavailable vehicle or already-connected destination')
        traction=route['distance_km']*self.traction_rate
        before=self.ledger.snapshot()
        setup=route['site_setup_s']
        if not route['road_edges']:
            # Co-located road attachment is a disclosed nearest-node mapping.
            # It does not waive a connection interval.
            if setup <= 0:
                raise ValueError('Zero-duration relocation requires an explicit local connection')
            arrival=self.ledger.time_s+setup
            self.ledger.depart(key,target,arrival,traction_kwh=traction)
            self.jobs[key]=dict(route=route,index=0,next_s=arrival,phase='connection')
        else:
            duration=self.edges[route['road_edges'][0]].length_km/self._speed(self.edges[route['road_edges'][0]],powered_buses)*3600
            predicted=sum(self.edges[e].length_km/self._speed(self.edges[e],powered_buses)*3600 for e in route['road_edges'])+setup
            self.ledger.depart(key,target,self.ledger.time_s+predicted,traction_kwh=traction)
            self.jobs[key]=dict(route=route,index=0,next_s=self.ledger.time_s+duration,phase='road')
        self.events.append(dict(event='road_departure',time_s=self.ledger.time_s,unit_id=key,
            route=route,traction_kwh=traction,before=before,after=self.ledger.snapshot()))

    def next_event_s(self):
        return min((j['next_s'] for j in self.jobs.values()),default=math.inf)

    def process(self, *, powered_buses):
        now=self.ledger.time_s
        for key in sorted(tuple(self.jobs)):
            job=self.jobs[key]
            if job['next_s'] != now:continue
            route=job['route'];before=self.ledger.snapshot()
            setup=route['site_setup_s']
            if job['phase']=='road':
                self.position[key]=route['road_nodes'][job['index']+1]
                job['index']+=1
                if job['index'] < len(route['road_edges']):
                    remaining=route['road_edges'][job['index']:]
                    edge=self.edges[remaining[0]]
                    speed=self._speed(edge,powered_buses)
                    job['next_s']=now+edge.length_km/speed*3600
                    self.ledger.units[key].arrival_s=now+sum(self.edges[e].length_km/self._speed(self.edges[e],powered_buses)*3600 for e in remaining)+setup
                elif setup:
                    job['phase']='connection';job['next_s']=now+setup
                    self.ledger.units[key].arrival_s=job['next_s']
                else:
                    job['phase']='connection'
            if job['phase']=='connection' and job['next_s']==now:
                unit=self.ledger.units[key]
                if unit.destination is not None:
                    raise RuntimeError('Physical arrival must be settled before connection')
                if route['purpose']=='connect':self.ledger.connect(key,route['destination_bus'])
                del self.jobs[key]
            self.events.append(dict(event='road_segment_or_connection_completed',time_s=now,unit_id=key,
                road_position=self.position[key],job=dict(job),before=before,after=self.ledger.snapshot()))
