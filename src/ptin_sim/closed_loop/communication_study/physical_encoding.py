"""Received-information encoding of resource-disjoint physical joint actions.

Every device occupies a fixed feature slot. No full simulator state, future
fault calendar, final reward or counterfactual power solution is consulted.
"""
import hashlib
import json
import math
import torch
from copy import deepcopy

from .physical_actions import action_id,received_state


STATE_FEATURES=('time_fraction','known_target_fraction','known_fault_fraction',
    'known_repaired_fraction','report_age_fraction','known_ground_fraction',
    'received_ground_soc_mean','received_warehouse_uav_fraction')
DEVICE_ACTION_FEATURES=('wait','deploy_supply','deploy_service','repair','open','close','uav',
    'target_load_fraction','target_critical','flight_or_road_distance_fraction','duration_fraction',
    'received_energy_fraction','report_age_fraction','known_target_failed','known_target_repaired')


class PhysicalJointEncoder:
    def __init__(self,scenario,resources,*,horizon_s=14400.):
        if not math.isfinite(horizon_s) or horizon_s<=0:raise ValueError('Invalid horizon')
        # Learning objective is recorded by the replay, not an observation feature.
        resources={k:v for k,v in resources.items() if k!='reward_metric'}
        self.scenario,self.resources,self.horizon=scenario,dict(resources),horizon_s
        self.state_features=STATE_FEATURES
        self.edges={e.edge_id:e for e in scenario.pdn_edges};self.nodes={n.node_id:n for n in scenario.pdn_nodes}
        self.ground_ids=tuple(f'MESS_{i+1:02d}' for i in range(scenario.fleet.mess_count))
        self.uav_ids=tuple(f'UAV_{i+1:02d}' for i in range(scenario.fleet.uav_count))
        self.units=self.ground_ids+self.uav_ids
        self.total_load=max(1.,sum(n.active_power_kw for n in scenario.pdn_nodes))
        self.action_features=tuple(f'{key}:{feature}' for key in self.units for feature in DEVICE_ACTION_FEATURES)+('joint_size',)
        self.identity=dict(schema='physical_joint_v1',state_features=STATE_FEATURES,
            action_semantics_version=3,relay_encoding='uav=1 with open=close=wait=0; finite site operation',
            replacement_encoding='uav=wait=1; warehouse replacement occupies the device',
            action_features=self.action_features,normalization='static_source_and_common_resource_parameters',
            nodes=[dict(id=n.node_id,load_kw=n.active_power_kw,critical=n.criticality,
                        x=n.coord_x_km,y=n.coord_y_km) for n in scenario.pdn_nodes],
            edges=[dict(id=e.edge_id,a=e.from_node,b=e.to_node) for e in scenario.pdn_edges],
            resources=resources,horizon_s=horizon_s)
        self.sha256=hashlib.sha256(json.dumps(self.identity,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def encode(self,observation,joint_candidates):
        now=observation['time_s']
        if not math.isfinite(now) or not 0<=now<=self.horizon:raise ValueError('Invalid observation time')
        s=received_state(observation,max_age_s=self.resources['observation_max_age_s'])
        if not set(s['targets'])<=self.edges.keys():raise ValueError('Unknown received target')
        ground={k:v for k,v in s['devices'].items() if k in self.ground_ids}
        report=observation.get('uav_observation');uavs={}
        if report:
            if not 0<=report['observed_s']<=report['received_s']<=now:raise ValueError('Noncausal UAV information')
            if now-report['observed_s']<=self.resources['observation_max_age_s']:uavs=report['units']
        n=max(1,len(self.edges));age=(now-s['oldest_observed_s'])/self.horizon
        state=[now/self.horizon,len(s['targets'])/n,sum(r['failed'] for r in s['targets'].values())/n,
            sum(r['restored'] for r in s['targets'].values())/n,age,len(ground)/max(1,len(self.ground_ids)),
            sum(v['energy_kwh']/v['capacity_kwh'] for v in ground.values())/max(1,len(ground)),
            sum(v['phase']=='warehouse' for v in uavs.values())/max(1,len(self.uav_ids))]
        rows=[];seen=set()
        for joint in joint_candidates:
            if not joint:raise ValueError('Joint action must include wait or at least one device')
            slots={k:[1.]+[0.]*(len(DEVICE_ACTION_FEATURES)-1) for k in self.units};used=set()
            signature=tuple(sorted(action_id(a) for a in joint))
            if signature in seen:raise ValueError('Duplicate joint candidate')
            seen.add(signature)
            for action in joint:
                kind=action['kind']
                if kind=='wait':
                    if len(joint)!=1:raise ValueError('Wait cannot be combined with physical work')
                    continue
                if kind not in {'deploy','control','uav_control','uav_relay','uav_replace'}:raise ValueError('Unknown physical action')
                key=action['route']['unit_id'] if kind=='deploy' else action['unit_id']
                if key not in slots or key in used:raise ValueError('Invalid or repeated device assignment')
                used.add(key);operation=action.get('operation');target=action.get('target')
                target_row=s['targets'].get(target,{})
                if target is not None and not target_row:raise ValueError('Control target not received')
                distance=0.;duration=action.get('duration_s',0.)
                if kind=='deploy':
                    route=action['route'];bus=route['destination_bus'];distance=route['distance_km']
                    duration=route['estimated_duration_s'];unit=ground.get(key)
                    if route['observed_s']>now:raise ValueError('Future route information')
                elif kind in {'uav_control','uav_relay','uav_replace'}:
                    bus=self.resources['depot_bus'] if kind=='uav_replace' else action['bus_id'];unit=uavs.get(key)
                    depot=self.nodes[self.resources['depot_bus']];dest=self.nodes[bus]
                    distance=math.dist((depot.coord_x_km,depot.coord_y_km),(dest.coord_x_km,dest.coord_y_km))
                    duration=2*distance/self.scenario.fleet.uav_flight_speed_km_h*3600+self.resources['uav_operation_s']
                    if kind=='uav_replace':duration=self.resources['uav_replacement_s']
                else:
                    unit=ground.get(key);bus=unit['location'] if unit else None
                if unit is None:raise ValueError('Device state has not been received')
                node=self.nodes[bus]
                energy=unit['energy_km']/self.scenario.fleet.uav_flight_endurance_km if kind in {'uav_control','uav_relay','uav_replace'} else unit['energy_kwh']/unit['capacity_kwh']
                slots[key]=[float(kind=='uav_replace'),float(kind=='deploy' and action['route']['purpose']=='connect'),
                    float(kind=='deploy' and action['route']['purpose']=='service'),float(operation=='repair'),
                    float(operation=='open'),float(operation=='close'),float(kind in {'uav_control','uav_relay','uav_replace'}),
                    node.active_power_kw/self.total_load,float(node.criticality=='critical'),
                    distance/self.scenario.fleet.uav_flight_endurance_km,duration/self.horizon,energy,age,
                    float(target_row.get('failed',False)),float(target_row.get('restored',False))]
            rows.append([x for key in self.units for x in slots[key]]+[len(used)/len(self.units)])
        if not rows:raise ValueError('Nonempty joint catalogue required')
        states,actions=torch.tensor(state,dtype=torch.float32),torch.tensor(rows,dtype=torch.float32)
        if not torch.isfinite(states).all() or not torch.isfinite(actions).all():raise ValueError('Nonfinite features')
        return states,actions


def collate_physical_transitions(rows):
    if not rows:raise ValueError('Empty training batch')
    width=len(rows[0]['candidates'][0]);state_width=len(rows[0]['state'])
    maximum=max(len(r['candidates']) for r in rows)
    candidates=torch.zeros(len(rows),maximum,width);valid=torch.zeros(len(rows),maximum,dtype=torch.bool)
    for i,r in enumerate(rows):
        count=len(r['candidates'])
        if (not count or not 0<=r['chosen']<count or len(r['state'])!=state_width
                or len(r['next_state'])!=state_width or any(len(a)!=width for a in r['candidates'])):
            raise ValueError('Inconsistent physical training transition')
        candidates[i,:count]=torch.tensor(r['candidates']);valid[i,:count]=True
    return dict(states=torch.tensor([r['state'] for r in rows],dtype=torch.float32),candidates=candidates,valid=valid,
        chosen=torch.tensor([r['chosen'] for r in rows],dtype=torch.long),
        rewards=torch.tensor([r['reward'] for r in rows],dtype=torch.float32),
        next_states=torch.tensor([r['next_state'] for r in rows],dtype=torch.float32),
        terminated=torch.tensor([float(r['task_terminal']) for r in rows]))


class PhysicalIQLPolicy:
    def __init__(self,encoder,model,*,checkpoint_evidence):
        if checkpoint_evidence.get('encoder_sha256')!=encoder.sha256:
            raise ValueError('Physical encoder identity differs from checkpoint')
        if model.config.state_dim!=len(encoder.state_features) or model.config.action_dim!=len(encoder.action_features):
            raise ValueError('Physical joint checkpoint dimension mismatch')
        self.encoder,self.model=encoder,model.eval()

    def select(self,observation,joint_candidates):
        state,actions=self.encoder.encode(observation,joint_candidates)
        with torch.no_grad():
            probabilities=self.model.probabilities(state[None],actions[None],
                torch.ones((1,len(joint_candidates)),dtype=torch.bool))[0]
        index=int(probabilities.argmax())
        self.last_decision_trace=dict(encoder_sha256=self.encoder.sha256,encoded_state=state.tolist(),
            joint_candidates=[dict(actions=deepcopy(a),features=actions[i].tolist(),probability=float(probabilities[i]))
                              for i,a in enumerate(joint_candidates)],selected_index=index)
        return deepcopy(joint_candidates[index])
