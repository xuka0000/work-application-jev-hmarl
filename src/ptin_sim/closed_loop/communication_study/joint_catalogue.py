"""Policy-independent finite joint proposals from received atomic candidates.

Keep every singleton. Additional joint assignments use a fixed public seed and
the same resource/target conflict rules for every method. No selected action,
reward, hidden plant state or case failure calendar enters proposal generation.
"""
from copy import deepcopy
import hashlib
import json
import random
from .physical_actions import action_id


def unit_id(action):
    return action['route']['unit_id'] if action['kind']=='deploy' else action.get('unit_id')


def compatible(action,selected):
    if action['kind']=='wait':return not selected
    if any(unit_id(a)==unit_id(action) for a in selected):return False
    target=action.get('target')
    if target is not None and any(a.get('target')==target for a in selected):return False
    if action['kind']=='uav_relay' and any(a['kind']=='uav_relay' and a['bus_id']==action['bus_id'] for a in selected):return False
    return True


def build_joint_catalogue(atomics,*,seed,joint_draws):
    if type(seed)is not int or type(joint_draws)is not int or joint_draws<0:
        raise ValueError('Explicit integer seed and nonnegative proposal count required')
    ids=[action_id(a) for a in atomics]
    if len(ids)!=len(set(ids)) or not atomics:raise ValueError('Unique atomic catalogue required')
    active=sorted((deepcopy(a) for a in atomics if a['kind']!='wait'),key=action_id)
    wait=[a for a in atomics if a['kind']=='wait']
    if wait!=[{'kind':'wait'}]:raise ValueError('Exactly one explicit wait action required')
    groups={}
    for a in active:groups.setdefault(unit_id(a),[]).append(a)
    result=[[{'kind':'wait'}]]+[[a] for a in active]
    seen={tuple(action_id(a) for a in row) for row in result}
    # The hash makes proposal draws invariant to incoming catalogue ordering.
    signature=hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
    rng=random.Random(f'{seed}:{signature}')
    for _ in range(joint_draws):
        keys=sorted(groups);rng.shuffle(keys);chosen=[]
        for key in keys:
            if rng.random()<.25:continue
            options=[a for a in groups[key] if compatible(a,chosen)]
            if options:chosen.append(rng.choice(options))
        chosen.sort(key=action_id);identity=tuple(action_id(a) for a in chosen)
        if chosen and identity not in seen:result.append(chosen);seen.add(identity)
    return deepcopy(result)
