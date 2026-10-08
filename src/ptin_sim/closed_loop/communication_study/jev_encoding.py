"""Causal target identities shared by all students in the Jev study.

The legacy vector aliases deployment commands sent through different field
controllers. Explicit one-hot targets preserve those distinct decisions on
the registered fixed network. They do not imply transfer to a new topology.
"""
import hashlib
import json
import torch
from .physical_encoding import PhysicalJointEncoder
from .physical_actions import received_state


class TargetAwarePhysicalEncoder(PhysicalJointEncoder):
    def __init__(self,scenario,resources,*,horizon_s=14400.):
        super().__init__(scenario,resources,horizon_s=horizon_s)
        self.edge_ids=tuple(sorted(self.edges));self.bus_ids=tuple(sorted(self.nodes))
        self.edge_index={k:i for i,k in enumerate(self.edge_ids)}
        self.bus_index={k:i for i,k in enumerate(self.bus_ids)}
        self.base_state_dim=len(self.state_features);self.base_action_dim=len(self.action_features)
        self.state_features+=tuple(f'{edge}:{field}' for edge in self.edge_ids
            for field in ('received','failed','restored','closed','report_age_fraction','delivery_age_fraction'))
        self.action_features+=tuple(f'{unit}:{field}:{target}' for unit in self.units
            for field,targets in [('control_target',self.edge_ids),('destination',self.bus_ids)] for target in targets)
        self.identity=dict(schema='physical_target_identity_v2',base=self.identity,
            state_features=self.state_features,action_features=self.action_features,
            scope='fixed public topology; received reports only; no unobserved faults or channel truth')
        self.sha256=hashlib.sha256(json.dumps(self.identity,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def encode(self,observation,joint_candidates):
        state,actions=super().encode(observation,joint_candidates)
        reports=received_state(observation,max_age_s=self.resources['observation_max_age_s'])['targets']
        extra=[];now=observation['time_s']
        for edge in self.edge_ids:
            row=reports.get(edge)
            extra.extend([1.,float(row['failed']),float(row['restored']),float(row['switch_closed']),
                (now-row['observed_s'])/self.horizon,(now-row['received_s'])/self.horizon] if row else [0.]*6)
        state=torch.cat((state,torch.tensor(extra,dtype=state.dtype)))
        width=len(self.edge_ids)+len(self.bus_ids)
        targets=torch.zeros(len(joint_candidates),len(self.units)*width,dtype=actions.dtype)
        for i,joint in enumerate(joint_candidates):
            for action in joint:
                if action['kind']=='wait':continue
                if action['kind']=='deploy':
                    unit=action['route']['unit_id'];edge=action['controller_target'];bus=action['route']['destination_bus']
                else:
                    unit=action['unit_id'];edge=action.get('target')
                    bus=self.resources['depot_bus'] if action['kind']=='uav_replace' else action.get('bus_id')
                offset=self.units.index(unit)*width
                if edge is not None:targets[i,offset+self.edge_index[edge]]=1.
                if bus is not None:targets[i,offset+len(self.edge_ids)+self.bus_index[bus]]=1.
        return state,torch.cat((actions,targets),dim=1)
