"""Two explicit internal heuristics over the common causal joint catalogue.

Scores are public-load proxies, not simulator lookahead or predicted AC gains.
Critical-load greedy maximizes distinct task load (critical weight two).
Gain/time divides the same per-task proxy by declared travel/work duration.
Inspection and relay scores use missing received reports, never hidden faults.
"""
import math
from .physical_actions import received_state


class PhysicalRuleBaseline:
    def __init__(self,scenario,resources,*,method):
        if method not in {'critical_load_greedy','gain_per_execution_time'}:
            raise ValueError('Unknown internal rule baseline')
        self.method=method;self.resources=dict(resources)
        self.nodes={n.node_id:n for n in scenario.pdn_nodes}
        self.edges={e.edge_id:e for e in scenario.pdn_edges}
        self.speed=scenario.fleet.uav_flight_speed_km_h
        self.radius=scenario.fleet.uav_communication_radius_km

    def weighted_load(self,bus):
        n=self.nodes[bus];return n.active_power_kw*(2. if n.criticality=='critical' else 1.)

    def action_benefits(self,a,known):
        kind=a['kind'];benefits={};duration=1.
        if kind=='wait':return benefits,duration
        if kind=='deploy':
            r=a['route'];bus=r['destination_bus'];duration=r['estimated_duration_s']
            if r['purpose']=='connect':
                if bus not in known['powered_buses']:benefits['supply:'+bus]=self.weighted_load(bus)
            else:
                for target,e in self.edges.items():
                    if bus not in {e.from_node,e.to_node}:continue
                    row=known['targets'].get(target)
                    if row is None or row['failed']:
                        benefits['inspect:'+target]=self.weighted_load(e.to_node)
        elif kind in {'control','uav_control'}:
            e=self.edges[a['target']];row=known['targets'][a['target']]
            if (a['operation']=='repair' and row['failed']) or (a['operation']=='close' and not row['failed']):
                benefits['restore:'+a['target']]=self.weighted_load(e.to_node)
            elif a['operation']=='open' and not row['failed']:
                benefits['disconnect:'+a['target']]=-self.weighted_load(e.to_node)
            duration=a.get('duration_s',self.resources['uav_operation_s'])
        elif kind=='uav_relay':
            n=self.nodes[a['bus_id']]
            for target,e in self.edges.items():
                if target in known['targets']:continue
                if any(math.dist((n.coord_x_km,n.coord_y_km),
                    (self.nodes[b].coord_x_km,self.nodes[b].coord_y_km))<=self.radius for b in (e.from_node,e.to_node)):
                    benefits['observe:'+target]=self.weighted_load(e.to_node)
            duration=self.resources['uav_operation_s']
        elif kind=='uav_replace':
            # Maintenance has no credited load restoration. A depleted fleet
            # may still need replacement: this is an explicit deterministic tie
            # preference, not an invented supply improvement.
            duration=self.resources['uav_replacement_s']
        else:raise ValueError('Unknown physical action')
        if kind in {'uav_control','uav_relay'}:
            d=self.nodes[self.resources['depot_bus']];n=self.nodes[a['bus_id']]
            duration+=2*math.dist((d.coord_x_km,d.coord_y_km),(n.coord_x_km,n.coord_y_km))/self.speed*3600
        return benefits,max(1.,duration)

    def select(self,observation,joints):
        known=received_state(observation,max_age_s=self.resources['observation_max_age_s'])
        rows=[]
        for joint in joints:
            gains={};maintenance=0
            for a in joint:
                benefit,duration=self.action_benefits(a,known)
                for key,value in benefit.items():
                    score=value/duration if self.method=='gain_per_execution_time' else value
                    gains[key]=max(gains[key],score) if key in gains else score
                maintenance+=a['kind']=='uav_replace'
            rows.append(dict(score=sum(gains.values()),distinct_benefits=gains,maintenance_actions=maintenance))
        index=max(range(len(joints)),key=lambda i:(rows[i]['score'],rows[i]['maintenance_actions'],-len(joints[i]),-i))
        self.last_decision_trace=dict(method=self.method,information_scope='received_reports_and_public_topology',
            score_meaning='weighted_load_proxy_not_AC_counterfactual',candidates=rows,selected_index=index)
        return joints[index]
