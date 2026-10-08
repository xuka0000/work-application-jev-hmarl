"""Finite physical action proposals derived only from received local reports.

Reports may be stale or incomplete. Proposals are not execution certificates:
the actual message, vehicle and control executors still check them on receipt.
The caller supplies capabilities and work durations from the common profile.
"""
from copy import deepcopy
import hashlib
import json
import math


def action_id(action):
    return hashlib.sha256(json.dumps(action,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def received_state(observation, *, max_age_s):
    now=observation['time_s']
    if not math.isfinite(max_age_s) or max_age_s<=0:raise ValueError('Finite positive report age required')
    targets={};power={};devices={};provenance={}
    reports=list(sorted(observation['target_observations'].items()))
    reports += [(r['target_id'],r) for _,r in sorted(observation.get('source_observations',{}).items())]
    for target,row in reports:
        observed,received=row['observed_s'],row['received_s']
        if not 0<=observed<=received<=now:raise ValueError('Observation has noncausal timestamps')
        if now-observed>max_age_s:continue
        old=targets.get(target)
        if old is None or (row['state_version'],observed,received)>=(old['state_version'],old['observed_s'],old['received_s']):
            targets[target]=deepcopy(row)
        # Version is target-local; timestamps, not versions of different edges,
        # determine which of two reports of the same endpoint is newer.
        stamp=(observed,received,row['message_id'])
        for bus,value in row.get('endpoint_power',{}).items():
            if bus not in power or stamp>power[bus][0]:power[bus]=(stamp,bool(value))
        for key,value in row.get('stationary_devices',{}).items():
            if key not in devices or stamp>devices[key][0]:
                devices[key]=(stamp,deepcopy(value));provenance[key]=row['message_id']
    return dict(targets=targets,powered_buses={k for k,(_,v) in power.items() if v},
                devices={k:v for k,(_,v) in devices.items()},device_provenance=provenance,
                oldest_observed_s=min((r['observed_s'] for r in targets.values()),default=now))


def build_actions(scenario, observation, *, road_execution, capabilities,
                  repair_duration_s, switch_duration_s, max_age_s, lifetime_s,uav_execution=None,
                  relay_radius_km=None):
    if not all(math.isfinite(x) and x>0 for x in (switch_duration_s,lifetime_s)):
        raise ValueError('Explicit positive command and switch durations required')
    state=received_state(observation,max_age_s=max_age_s)
    edges={e.edge_id:e for e in scenario.pdn_edges}
    actions=[{'kind':'wait'}]
    for key,unit in sorted(state['devices'].items()):
        if unit['kind'] not in {'mess','ugv'} or not unit['available'] or unit.get('work_busy',False):continue
        origins=[target for target in state['targets'] if unit['location'] in
                 {edges[target].from_node,edges[target].to_node}]
        for target,row in sorted(state['targets'].items()):
            edge=edges[target]
            endpoints={edge.from_node,edge.to_node}
            kind='repair' if row['failed'] else ('open' if row['switch_closed'] else 'close')
            required='repair' if kind=='repair' else 'switch'
            duration=repair_duration_s.get(target) if kind=='repair' else switch_duration_s
            if (required in capabilities.get(key,()) and unit['location'] in endpoints
                    and duration is not None):
                if not math.isfinite(duration) or duration<=0:raise ValueError('Invalid sourced repair duration')
                actions.append(dict(kind='control',unit_id=key,target=target,operation=kind,
                    duration_s=duration,lifetime_s=lifetime_s,
                    evidence=[row['message_id'],state['device_provenance'][key]]))
        if road_execution is None or unit.get('road_position') is None:continue
        # A destination need not already have power or have reported a fault.
        # The command must reach the vehicle at its observed departure site.
        destinations=[(bus,'connect') for bus in road_execution.sites]
        if 'repair' in capabilities.get(key,()):
            # A field inspection cannot require an already received fault:
            # loss of power may prevent the first fault report. These public
            # sites permit travel only, never assert a fault or authorize work.
            destinations += [(bus,'service') for bus in road_execution.service_sites]
        for bus,purpose in sorted(destinations):
            if bus==unit.get('connected_bus') or (purpose=='service' and bus==unit['location']):continue
            route=road_execution.route(key,bus,observed_powered_buses=state['powered_buses'],
                observed_s=state['oldest_observed_s'],observed_road_position=unit['road_position'],purpose=purpose)
            if route['distance_km']*road_execution.traction_rate>unit['energy_kwh']:continue
            for origin in sorted(origins):
                actions.append(dict(kind='deploy',controller_target=origin,route=route,
                    lifetime_s=lifetime_s,evidence=sorted({r['message_id'] for r in state['targets'].values()})))
    report=observation.get('uav_observation')
    if uav_execution is not None and report is not None:
        now=observation['time_s']
        if not 0<=report['observed_s']<=report['received_s']<=now:raise ValueError('Noncausal UAV report')
        if now-report['observed_s']<=max_age_s:
            nodes={n.node_id:n for n in scenario.pdn_nodes}
            for key,unit in sorted(report['units'].items()):
                if unit['phase']!='warehouse' or 'switch' not in capabilities.get(key,()):continue
                if unit['energy_km']<uav_execution.endurance-1e-9:
                    actions.append(dict(kind='uav_replace',unit_id=key,evidence=[report['message_id']]))
                if relay_radius_km is not None:
                    for bus,n in sorted(nodes.items()):
                        distance=math.dist(uav_execution.depot,(n.coord_x_km,n.coord_y_km))
                        if 2*distance+uav_execution.operation_cost<=unit['energy_km']:
                            actions.append(dict(kind='uav_relay',unit_id=key,bus_id=bus,
                                evidence=[report['message_id']]))
                for target,row in sorted(state['targets'].items()):
                    # Faulted lines may be opened, never closed by a UAV repair fiction.
                    operation='open' if row['switch_closed'] else 'close'
                    if operation=='close' and row['failed']:continue
                    for bus in sorted({edges[target].from_node,edges[target].to_node}):
                        n=nodes[bus];distance=math.dist(uav_execution.depot,(n.coord_x_km,n.coord_y_km))
                        if 2*distance+uav_execution.operation_cost>unit['energy_km']:continue
                        actions.append(dict(kind='uav_control',unit_id=key,target=target,bus_id=bus,
                            operation=operation,evidence=[row['message_id'],report['message_id']]))
    return actions


class PhysicalActionSession:
    """Policy boundary with exact candidate/observation evidence for each choice."""
    def __init__(self, environment, *, capabilities, repair_duration_s,
                 switch_duration_s, max_age_s, lifetime_s, decision_period_s,
                 reward_metric='system_ens_kwh'):
        if not math.isfinite(decision_period_s) or decision_period_s<=0:
            raise ValueError('Finite positive decision period required')
        self.environment=environment
        self.parameters=dict(capabilities=deepcopy(capabilities),repair_duration_s=dict(repair_duration_s),
            switch_duration_s=switch_duration_s,max_age_s=max_age_s,lifetime_s=lifetime_s)
        self.period=decision_period_s
        if reward_metric not in {'system_ens_kwh','critical_ens_kwh'}:
            raise ValueError('Unknown physical restoration reward metric')
        self.reward_metric=reward_metric
        self.decisions=[]

    def candidates(self):
        env=self.environment
        return build_actions(env.scenario,env.observe(),road_execution=env.road_execution,
                             uav_execution=env.uav_execution,relay_radius_km=env.relay_radius_km,**self.parameters)

    def step(self, action):
        return self.step_joint([action])

    def step_joint(self, actions):
        """Issue a resource-disjoint joint decision on one shared physical clock."""
        self.submit_joint(actions)
        return self.advance_transition(min(self.environment.horizon_s,
                                           self.environment.time_s+self.period))

    def submit_joint(self, actions):
        """Issue now without advancing time; online callers own the fixed grid."""
        env=self.environment
        if env.failure is not None or env.time_s>=env.horizon_s:
            raise RuntimeError('Cannot decide in a failed or finished attempt')
        candidates=self.candidates()
        actions=deepcopy(list(actions))
        if not actions or any(a not in candidates for a in actions):
            raise ValueError('Action is not supported by the received observation')
        if len(actions)>1 and any(a['kind']=='wait' for a in actions):raise ValueError('Wait cannot accompany active work')
        keys=[a['route']['unit_id'] if a['kind']=='deploy' else a.get('unit_id') for a in actions]
        if len(set(keys))!=len(keys):raise ValueError('A joint action assigns a unit more than once')
        targets=[a['target'] for a in actions if a['kind'] in {'control','uav_control'}]
        if len(set(targets))!=len(targets):raise ValueError('A joint action assigns conflicting target work')
        record=dict(time_s=env.time_s,observation=env.observe(),candidates=candidates,
                    selected_joint=actions,selected_joint_sha256=[action_id(a) for a in actions])
        if len(actions)==1:record.update(selected=actions[0],selected_sha256=action_id(actions[0]))
        self.decisions.append(record)
        if env.trace:env.trace.emit('policy_decision',env.time_s,record)
        for action in actions:
            if action['kind']=='deploy':
                env.request_deployment(action['route'],controller_target=action['controller_target'],
                                       lifetime_s=action['lifetime_s'])
            elif action['kind']=='control':
                env.request_control(unit_id=action['unit_id'],target=action['target'],kind=action['operation'],
                                    duration_s=action['duration_s'],lifetime_s=action['lifetime_s'])
            elif action['kind']=='uav_control':
                env.request_uav_control(**{k:action[k] for k in ('unit_id','target','bus_id','operation')})
            elif action['kind']=='uav_relay':
                env.request_uav_relay(unit_id=action['unit_id'],bus_id=action['bus_id'])
            elif action['kind']=='uav_replace':
                env.request_uav_replacement(unit_id=action['unit_id'])
        return record

    def advance_transition(self, until_s, *, before=None):
        """Account for a whole decision interval, including inference waiting."""
        env=self.environment
        if env.failure is not None or not env.time_s<=until_s<=env.horizon_s:
            raise ValueError('Invalid physical transition boundary')
        before=env.service_metrics() if before is None else before
        observation=env.advance(until_s) if until_s>env.time_s else env.observe()
        after=env.service_metrics()
        rewards={key:before[key]-after[key] for key in ('system_ens_kwh','critical_ens_kwh')}
        transition=dict(observation=observation,reward_negative_ens_kwh=rewards['system_ens_kwh'],
                        reward_negative_critical_ens_kwh=rewards['critical_ens_kwh'],
                        reward=rewards[self.reward_metric],reward_metric=self.reward_metric,
                        terminal=env.time_s==env.horizon_s)
        if env.trace:env.trace.emit('policy_transition',env.time_s,transition)
        return transition
