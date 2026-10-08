"""Physical clock coupling sited AC, batteries, CN power and routed telemetry.

This replaces aggregate power accounting for the event core. Scheduling of
vehicle routes, UAVs and repair commands is deliberately not inferred here.
Only received target telemetry is exposed by observe(); full truth is audited.
"""
import math
from copy import deepcopy
from dataclasses import replace

from .cn_supply import CommonLoadSwitchCNSupply
from .packets import Message, MessageBus, SyntheticChannel
from .routing import NetworkState, TargetRouter
from .sited_resource_dispatch import SitedResourceDispatch
from .control_execution import PDNControlExecution
from .finite_relay import with_finite_relays,with_fixed_relays
from ..dependencies import switch_controller_cn_ids


class PhysicalEventEnvironment:
    def __init__(self, scenario, profile, ledger, inverters, *, channel,
                 release_time_s, availability_events=(), connection_events=(),control_period_s=30.,
                 report_period_s=30., horizon_s=14400., trace=None, road_execution=None,
                 uav_execution=None, onsite_operation_check=None, authorization_ttl_s=900.,depot_cn_nodes=(),
                 mobile_radio=False,relay_radius_km=None,cached_local_repair=False,
                 fixed_relay_buses=()):
        if ledger.time_s != 0:
            raise ValueError('A fresh physical ledger is required')
        if not all(math.isfinite(x) and x > 0 for x in (control_period_s, report_period_s, horizon_s)):
            raise ValueError('Positive finite periods and horizon required')
        edges = {e.edge_id for e in scenario.pdn_edges}
        if not set(release_time_s) <= edges or any(not math.isfinite(t) or t < 0 or t >= horizon_s
                                                 for t in release_time_s.values()):
            raise ValueError('Invalid private fault calendar')
        self.scenario, self.ledger = scenario, ledger
        self.mobile_radio=mobile_radio
        if relay_radius_km is not None and (not math.isfinite(relay_radius_km) or relay_radius_km<=0):
            raise ValueError('Positive finite relay radius required')
        self.relay_radius_km=relay_radius_km
        self.fixed_relay_buses=tuple(sorted(set(fixed_relay_buses)))
        if self.fixed_relay_buses and self.relay_radius_km is None:
            raise ValueError('Fixed relays require a registered relay radius')
        self.cached_local_repair=cached_local_repair
        self.local_repair_durations={}
        self.local_decisions=[]
        self._local_offset=0
        if not math.isfinite(authorization_ttl_s) or authorization_ttl_s<=0:
            raise ValueError('Positive authorization lifetime required')
        self.authorization_ttl_s=authorization_ttl_s
        if road_execution is not None and road_execution.ledger is not ledger:
            raise ValueError('Road execution must share the physical ledger')
        self.road_execution = road_execution
        if uav_execution is not None and (uav_execution.time_s != 0 or not callable(onsite_operation_check)):
            raise ValueError('UAV integration requires a fresh fleet and explicit onsite execution check')
        self.uav_execution, self.onsite_operation_check = uav_execution, onsite_operation_check
        self.onsite_operation_start_check=None
        self.command_events = []
        self.plant = SitedResourceDispatch(scenario, profile, ledger, inverters)
        self.releases = dict(release_time_s)
        self.availability_events = sorted((dict(e) for e in availability_events), key=lambda e:e['time_s'])
        self.connection_events=sorted((dict(e) for e in connection_events),key=lambda e:e['time_s'])
        for event in self.connection_events:
            if (event['unit_id'] not in ledger.units or event['bus_id'] not in {n.node_id for n in scenario.pdn_nodes}
                    or not math.isfinite(event['time_s']) or not 0<=event['time_s']<horizon_s):
                raise ValueError('Invalid explicit station connection calendar')
        for event in self.availability_events:
            if (event['unit_id'] not in ledger.units or type(event['available']) is not bool
                    or not math.isfinite(event['time_s']) or not 0 <= event['time_s'] < horizon_s):
                raise ValueError('Invalid per-device availability calendar')
        identities = [(e['time_s'], e['unit_id']) for e in self.availability_events]
        if len(identities) != len(set(identities)):
            raise ValueError('Conflicting simultaneous availability events')
        self.control_period_s, self.report_period_s, self.horizon_s = control_period_s, report_period_s, horizon_s
        self.trace = trace
        self.router = TargetRouter(scenario, states=[NetworkState(0., frozenset())],depot_nodes=depot_cn_nodes)
        self.bus = MessageBus(SyntheticChannel(channel), router=self.router)
        self.resolver = CommonLoadSwitchCNSupply()
        self.snapshots, self.physical_audits = {}, []
        self.source_snapshots={}
        self.uav_snapshot=None
        self.pending_uav_authorizations={}
        self.repaired = set()
        self.switch_closed={e.edge_id for e in scenario.pdn_edges if e.normal_switch_state=='closed'}
        self.controls=None
        self.control_versions={e.edge_id:0 for e in scenario.pdn_edges}
        self._control_offset=0
        self._sequence = 0
        self._message_offset = 0
        self._resource_offset = 0
        self._road_offset = 0
        self._command_offset = 0
        self._uav_offset = 0
        self._command_changed_power = False
        self.failure = None
        self._apply_availability(0.)
        self._solve_and_publish_power()
        self._report()
        self.bus.advance(0., on_receive=self._receive)
        self._flush_evidence()

    @property
    def time_s(self):
        return self.ledger.time_s

    def _failed(self):
        return {k for k,t in self.releases.items() if t <= self.time_s and k not in self.repaired}

    def _closed(self):
        return self.switch_closed-self._failed()

    def _version(self,target):
        return int(self.releases.get(target,math.inf)<=self.time_s)+self.control_versions[target]

    def configure_control(self,*,capabilities,unit_at_bus,max_age_s=900.,
                          require_freshness=True,require_reconciliation=True):
        if self.controls is not None:raise RuntimeError('Control executor already configured')
        self.controls=PDNControlExecution(self.scenario,self.bus,repaired=self.repaired,
            switch_closed=self.switch_closed,active_faults=self._failed,capabilities=capabilities,
            unit_at_bus=unit_at_bus,controller_available=lambda target:bool(
                set(switch_controller_cn_ids(self.scenario,target_pdn_edge_id=target)) &
                set(self.physical_audits[-1]['powered_cn_nodes'])),
            observed_version=self._version,max_age_s=max_age_s,
            require_freshness=require_freshness,
            require_reconciliation=require_reconciliation)
        # Local switch actuation needs power at the actual operating bus.
        # CN station power determines packet routing, not actuator power.
        self.controls.local_controller_available=lambda target,key:unit_at_bus(key) in self._powered_buses()

    def _next_tick(self, period):
        return (math.floor(self.time_s/period)+1)*period

    def _apply_availability(self, at_s):
        for event in self.availability_events:
            if event['time_s'] == at_s:
                self.ledger.set_available(event['unit_id'], event['available'])
        for event in self.connection_events:
            if event['time_s']==at_s:
                self.ledger.connect(event['unit_id'],event['bus_id'])

    def _solve_and_publish_power(self):
        result = self.plant.plan(min(self.horizon_s, self._next_tick(self.control_period_s)),
                                 closed_pdn_edges=self._closed())
        voltage = {n.node_id: None for n in self.scenario.pdn_nodes}
        for component in result['components']:
            voltage.update(component.get('voltages', {}))
        served = {n.node_id: 0. for n in self.scenario.pdn_nodes}
        for row in result['loads']:
            served[row['bus_id']] += row['served_kw']
        audit = dict(time_s=self.time_s, status='opf_ok',
                     solver_meaning='verified_local_AC_feasible_solution_not_global_optimum',
                     served_load_by_node_kw=served, bus_voltage_pu=voltage,
                     line_loading_pct={r['edge_id']:r['loading_percent'] for r in result['lines']},
                     profile_sha256=self.plant.profile['sha256'], electrical=result,
                     resource_state=self.ledger.snapshot())
        state, evidence = self.resolver(self.scenario, audit)
        relay_points={}
        if self.relay_radius_km is not None and self.uav_execution is not None:
            state,mobile_points=with_finite_relays(state,self.scenario,self.uav_execution.snapshot(),
                radius_km=self.relay_radius_km,now_s=self.time_s)
            relay_points.update(mobile_points)
        if self.fixed_relay_buses:
            state,fixed_points=with_fixed_relays(state,self.scenario,self.fixed_relay_buses,
                radius_km=self.relay_radius_km)
            relay_points.update(fixed_points)
        if self.mobile_radio:
            nodes={n.node_id:n for n in self.scenario.pdn_nodes}
            attachments=[]
            for key,u in self.ledger.units.items():
                if u.kind not in {'mess','ugv'} or not u.available or u.destination is not None or u.energy_kwh<=0:continue
                n=nodes[u.location]
                xy=(n.coord_x_km,n.coord_y_km)
                reachable=tuple(sorted([c.node_id for c in self.scenario.cn_nodes
                    if math.dist(xy,(c.coord_x_km,c.coord_y_km))<=c.coverage_radius_km]+
                    [k for k,p in relay_points.items() if math.dist(xy,p)<=self.relay_radius_km]))
                attachments.append((f'unit:{key}',reachable))
            state=replace(state,mobile_attachments=tuple(sorted(attachments)))
        if self.router.states[-1].time_s == self.time_s:
            self.router.states[-1] = state
        else:
            self.router.append_state(state)
        audit['cn_supply_evidence'] = evidence
        audit['powered_cn_nodes'] = sorted(state.powered_nodes)
        audit['finite_relay_nodes']=sorted(state.relay_nodes)
        audit['finite_relay_links']=state.relay_links
        audit['fixed_relay_buses']=list(self.fixed_relay_buses)
        self.physical_audits.append(audit)
        if self.trace:
            self.trace.emit('physical_state', self.time_s, audit)

    def _report(self):
        # Static cadence and target set prevent hidden failures changing traffic.
        failed = self._failed()
        powered=self._powered_buses()
        devices=self.ledger.snapshot()
        for edge in self.scenario.pdn_edges:
            self._sequence += 1
            version = self._version(edge.edge_id)
            self.bus.send(Message(message_id=f'physical_telemetry_{self._sequence}', source='plant',
                destination='dispatcher', kind='telemetry', target_id=edge.edge_id,
                generated_s=self.time_s, sent_s=self.time_s,
                expires_s=self.time_s+self.report_period_s, state_version=version,
                payload=dict(failed=edge.edge_id in failed, restored=edge.edge_id in self.repaired,
                    switch_closed=edge.edge_id in self.switch_closed,
                    endpoint_power={k:k in powered for k in (edge.from_node,edge.to_node)},
                    stationary_devices={k:dict(v,work_busy=bool(self.controls and k in self.controls.jobs),
                        road_position=self.road_execution.position[k]
                        if self.road_execution else None) for k,v in devices.items()
                        if v['location'] in {edge.from_node,edge.to_node} and v['destination'] is None})))
            if self.mobile_radio:
                for key,u in self.ledger.units.items():
                    if u.kind not in {'mess','ugv'} or not u.available or u.destination is not None:continue
                    if u.location not in {edge.from_node,edge.to_node}:continue
                    self._sequence+=1
                    self.bus.send(Message(message_id=f'field_telemetry_{self._sequence}',source=f'unit:{key}',
                        destination='dispatcher',kind='telemetry',target_id=edge.edge_id,generated_s=self.time_s,
                        sent_s=self.time_s,expires_s=self.time_s+self.report_period_s,state_version=version,
                        payload=dict(failed=edge.edge_id in failed,restored=edge.edge_id in self.repaired,
                            switch_closed=edge.edge_id in self.switch_closed,
                            endpoint_power={u.location:u.location in powered},
                            stationary_devices={key:dict(devices[key],work_busy=bool(self.controls and key in self.controls.jobs),
                                road_position=self.road_execution.position[key] if self.road_execution else None)})))
        if self.cached_local_repair and self.controls is not None:
            self._execute_cached_local_repairs()
        if self.uav_execution and self.router.depot_nodes:
            self._sequence+=1
            self.bus.send(Message(message_id=f'depot_telemetry_{self._sequence}',source='depot',destination='dispatcher',
                kind='depot_telemetry',target_id=self.scenario.pdn_edges[0].edge_id,generated_s=self.time_s,
                sent_s=self.time_s,expires_s=self.time_s+self.report_period_s,state_version=0,
                payload={'units':{k:v for k,v in self.uav_execution.snapshot().items()
                                  if v['phase'] in {'warehouse','replacing'}}}))

    def _execute_cached_local_repairs(self):
        # Preinstalled local policy: observe adjacent hardware only and use an
        # already received, reconciled, still-valid authorization. This never
        # synthesizes a command from hidden global failure information.
        for key,u in sorted(self.ledger.units.items()):
            if (not u.available or u.destination is not None or key in self.controls.jobs
                    or 'repair' not in self.controls.capabilities.get(key,())):continue
            for edge in sorted(self.scenario.pdn_edges,key=lambda e:e.edge_id):
                if u.location not in {edge.from_node,edge.to_node}:continue
                target=edge.edge_id
                if target not in self._failed():continue
                duration=self.local_repair_durations.get(target)
                if duration is None:continue
                received=self.bus.latest(f'unit:{key}','authorization',target)
                if received is None:continue
                if any(j['target']==target for j in self.controls.jobs.values()):continue
                accepted,reason=self.controls.start(key,target=target,kind='repair',received=received,
                    now_s=self.time_s,duration_s=duration)
                self.local_decisions.append(dict(time_s=self.time_s,unit_id=key,target=target,
                    policy='cached_authorized_local_repair',observed_at_bus=u.location,
                    locally_observed_failed=True,locally_observed_version=self._version(target),
                    authorization_id=received.message.message_id,accepted=accepted,reason=reason))
                if accepted:break

    def _receive(self, received):
        m = received.message
        if m.kind=='depot_telemetry':
            self.uav_snapshot=dict(deepcopy(m.payload),observed_s=m.generated_s,received_s=received.received_s,
                                   message_id=m.message_id)
        elif m.kind == 'telemetry':
            row = dict(m.payload, observed_s=m.generated_s,
                received_s=received.received_s, state_version=m.state_version, message_id=m.message_id,
                reported_by=m.source,target_id=m.target_id)
            old=self.snapshots.get(m.target_id)
            if old is None or (row['state_version'],row['observed_s'],row['received_s']) >= (
                    old['state_version'],old['observed_s'],old['received_s']):
                self.snapshots[m.target_id] = deepcopy(row)
            self.source_snapshots[f'{m.source}|{m.target_id}']=deepcopy(row)
            if self.controls is not None:
                self._sequence+=1
                self.bus.send(Message(message_id=f'authorization_{self._sequence}',source='dispatcher',
                    destination=m.source if m.source.startswith('unit:') else 'terminal',kind='authorization',target_id=m.target_id,
                    generated_s=m.generated_s,sent_s=self.bus.now_s,expires_s=m.generated_s+self.authorization_ttl_s,
                    state_version=m.state_version,payload={'authorized':True,
                    'allowed_actions':['repair'] if m.payload['failed'] else ['open','close']}))
        elif m.kind in {'authorization','relay_authorization'}:
            self.bus.acknowledge(received,deadline_s=30.)
            if m.destination=='depot' and 'uav_mission' in m.payload:
                # Reserve the request until its real acknowledgement arrives.
                # A short flight must not race the reconciliation handshake.
                self.pending_uav_authorizations[m.message_id]=received
        elif m.kind=='ack' and self.bus.acknowledged(m.reply_to):
            pending=self.pending_uav_authorizations.get(m.reply_to)
            if pending is not None and self.time_s<=pending.message.expires_s:
                # The depot cannot observe the dispatcher's ack receipt for
                # free. Send a real final launch commitment back to the depot.
                self._sequence+=1
                self.bus.send(Message(message_id=f'uav_launch_{self._sequence}',source='dispatcher',
                    destination='depot',kind='uav_launch',target_id=pending.message.target_id,
                    generated_s=self.time_s,sent_s=self.time_s,expires_s=pending.message.expires_s,
                    state_version=self._sequence,payload={'authorization_id':m.reply_to}))
        elif m.kind=='uav_launch':
            identity=m.payload.get('authorization_id')
            pending=self.pending_uav_authorizations.get(identity)
            if (m.source=='dispatcher' and m.destination=='depot' and pending is not None
                    and m.target_id==pending.message.target_id and self.bus.acknowledged(identity)):
                del self.pending_uav_authorizations[identity]
                self._receive_uav_mission(pending)
        elif m.kind=='depot_service':
            outcome=dict(time_s=self.time_s,message_id=m.message_id,applied=False,
                         meaning='battery_replacement_started_not_completed')
            try:
                if m.source!='dispatcher' or m.destination!='depot' or m.payload.get('operation')!='replace_battery':
                    raise ValueError('Invalid depot service command')
                self.uav_execution.replace_battery(m.payload['unit_id'])
                outcome['applied']=True
            except (ValueError,KeyError) as exc:outcome['reason']=str(exc)
            self.command_events.append(outcome)
        elif m.kind=='control_command':
            if self.controls is None:
                self.command_events.append(dict(time_s=self.time_s,message_id=m.message_id,
                    applied=False,reason='control_executor_not_configured'))
            else:
                p=m.payload
                authorization=self.bus.latest(m.destination,'authorization',m.target_id)
                try:
                    accepted,reason=self.controls.start(p['unit_id'],target=m.target_id,kind=p['kind'],
                        received=authorization,now_s=self.time_s,duration_s=p['duration_s'])
                except ValueError as exc:
                    accepted,reason=False,str(exc)
                self.command_events.append(dict(time_s=self.time_s,message_id=m.message_id,
                    applied=accepted,reason=reason,meaning='work_started_not_completed'))
        elif m.kind == 'deployment_command':
            outcome = dict(time_s=received.received_s,message_id=m.message_id,
                           target_id=m.target_id,applied=False)
            try:
                if self.road_execution is None or m.source != 'dispatcher' or m.destination not in {'terminal',f"unit:{m.payload['route']['unit_id']}"}:
                    raise ValueError('Deployment command has no executable recipient')
                if received.received_s != self.time_s:
                    raise RuntimeError('Command delivery must coincide with physical clock')
                route = m.payload['route']
                edge = next(e for e in self.scenario.pdn_edges if e.edge_id == m.target_id)
                unit=self.ledger.units[route['unit_id']]
                if self.controls and route['unit_id'] in self.controls.jobs:
                    raise ValueError('Vehicle is occupied by unfinished control or repair work')
                if unit.destination is not None or unit.location not in {edge.from_node,edge.to_node}:
                    raise ValueError('Deployment command must reach the vehicle at its departure site')
                self.road_execution.start(route,powered_buses=self._powered_buses())
                outcome['applied'] = True
            except (ValueError, KeyError, StopIteration) as exc:
                outcome['reason'] = str(exc)
            self.command_events.append(outcome)
            if outcome['applied']:
                # Keep numerical failures outside the command-rejection handler:
                # a departed vehicle must not be reported as an unstarted job.
                self._solve_and_publish_power()

    def _powered_buses(self):
        audit = self.physical_audits[-1]
        return {n.node_id for n in self.scenario.pdn_nodes
                if audit['served_load_by_node_kw'].get(n.node_id,0.) >= n.active_power_kw-.001
                and audit['bus_voltage_pu'].get(n.node_id) is not None
                and .95-1e-6 <= audit['bus_voltage_pu'][n.node_id] <= 1.05+1e-6}

    def request_deployment(self, route, *, controller_target, lifetime_s):
        """Send an actual command; loss/delay never starts a vehicle implicitly."""
        if not math.isfinite(lifetime_s) or lifetime_s <= 0:
            raise ValueError('Positive command lifetime required')
        if self.failure is not None or self.time_s >= self.horizon_s:
            raise RuntimeError('Cannot command a failed or finished attempt')
        self._sequence += 1
        self.bus.send(Message(message_id=f'deployment_{self._sequence}',source='dispatcher',
            destination=f"unit:{route['unit_id']}" if self.mobile_radio else 'terminal',kind='deployment_command',target_id=controller_target,
            generated_s=self.time_s,sent_s=self.time_s,expires_s=self.time_s+lifetime_s,
            state_version=0,payload={'route':route}))
        self._flush_evidence()

    def request_control(self,*,unit_id,target,kind,duration_s,lifetime_s):
        if self.failure is not None or self.time_s>=self.horizon_s:
            raise RuntimeError('Cannot command a failed or finished attempt')
        if self.controls is None:raise RuntimeError('Control executor is not configured')
        if (not all(math.isfinite(x) and x>0 for x in (duration_s,lifetime_s))
                or target not in self.control_versions or kind not in {'repair','open','close'}):
            raise ValueError('Invalid finite control work request')
        self._sequence+=1
        self.bus.send(Message(message_id=f'control_command_{self._sequence}',source='dispatcher',
            destination=f'unit:{unit_id}' if self.mobile_radio and unit_id in self.ledger.units else 'terminal',
            kind='control_command',target_id=target,generated_s=self.time_s,
            sent_s=self.time_s,expires_s=self.time_s+lifetime_s,state_version=0,
            payload=dict(unit_id=unit_id,kind=kind,duration_s=duration_s)))
        self._flush_evidence()

    def observe(self):
        return dict(time_s=self.time_s, target_observations=deepcopy(self.snapshots),
                    observation_available=bool(self.snapshots),uav_observation=deepcopy(self.uav_snapshot),
                    source_observations=deepcopy(self.source_snapshots))

    def configure_uav_control(self):
        if self.uav_execution is None or self.controls is None:
            raise RuntimeError('Finite UAV fleet and control executor required')
        def begin(key,mission,at):
            packet=self.bus._all_received[mission['authorization']['message_id']]
            if mission.get('mode')=='relay':
                valid=self.bus.acknowledged(packet.message.message_id) and at<=packet.message.expires_s
                return valid,'finite_relay_authorized' if valid else 'relay_authorization_not_reconciled'
            try:
                return self.controls.start(key,target=mission['target_id'],kind=packet.message.payload['uav_mission']['operation'],
                    received=packet,now_s=at,duration_s=self.uav_execution.operation_s)
            except ValueError as exc:return False,str(exc)
        def finish(key,mission,at):
            if mission.get('mode')=='relay':return True,'finite_relay_interval_completed'
            offset=len(self.controls.events)
            self.controls.complete_at(at,unit_id=key)
            events=self.controls.events[offset:]
            for e in events:
                if e['applied']:self.control_versions[e['target']]+=1
            own=next((e for e in events if e['unit_id']==key),None)
            return (own['applied'],own['reason']) if own else (False,'no_started_onsite_work')
        self.onsite_operation_start_check=begin
        self.onsite_operation_check=finish

    def request_uav_control(self,*,unit_id,target,bus_id,operation):
        if self.failure is not None or self.time_s>=self.horizon_s:
            raise RuntimeError('Cannot command a failed or finished attempt')
        if self.onsite_operation_start_check is None or operation not in {'open','close'}:
            raise ValueError('Configured switch-only UAV execution required')
        edge=next(e for e in self.scenario.pdn_edges if e.edge_id==target)
        if bus_id not in {edge.from_node,edge.to_node}:raise ValueError('UAV must operate at a target endpoint')
        row=self.snapshots.get(target)
        if row is None or (operation=='close' and row['failed']):
            raise ValueError('Received target state does not support this switch operation')
        expires=row['observed_s']+self.authorization_ttl_s
        if expires<=self.time_s:
            # Proposals are not execution certificates. Expiry is an expected
            # rejection, not a failure of the complete physical trajectory.
            rejection=dict(time_s=self.time_s,unit_id=unit_id,target=target,
                operation=operation,applied=False,reason='received_target_report_expired',
                observed_s=row['observed_s'],expires_s=expires,
                meaning='uav_authorization_not_sent')
            self.command_events.append(rejection)
            if self.trace:self.trace.emit('uav_request_rejected',self.time_s,rejection)
            self._flush_evidence()
            return False
        self._sequence+=1
        self.bus.send(Message(message_id=f'uav_authorization_{self._sequence}',source='dispatcher',destination='depot',
            kind='authorization',target_id=target,generated_s=row['observed_s'],sent_s=self.time_s,
            expires_s=expires,state_version=row['state_version'],payload=dict(authorized=True,
                allowed_actions=[operation],uav_mission=dict(unit_id=unit_id,bus_id=bus_id,operation=operation))))
        self._flush_evidence()

    def request_uav_relay(self,*,unit_id,bus_id):
        if self.failure is not None or self.time_s>=self.horizon_s:
            raise RuntimeError('Cannot command a failed or finished attempt')
        if self.relay_radius_km is None or self.onsite_operation_start_check is None:
            raise ValueError('Finite relay operation is not configured')
        edge=next(e for e in self.scenario.pdn_edges if bus_id in {e.from_node,e.to_node})
        self._sequence+=1
        self.bus.send(Message(message_id=f'relay_authorization_{self._sequence}',source='dispatcher',
            destination='depot',kind='relay_authorization',target_id=edge.edge_id,generated_s=self.time_s,
            sent_s=self.time_s,expires_s=self.time_s+self.authorization_ttl_s,state_version=0,
            payload=dict(authorized=True,allowed_actions=['relay'],
                uav_mission=dict(unit_id=unit_id,bus_id=bus_id,operation='relay'))))
        self._flush_evidence()

    def request_uav_replacement(self,*,unit_id):
        if self.failure is not None or self.time_s>=self.horizon_s:
            raise RuntimeError('Cannot command a failed or finished attempt')
        if self.uav_execution is None:raise ValueError('No finite UAV fleet')
        self._sequence+=1
        self.bus.send(Message(message_id=f'uav_replacement_{self._sequence}',source='dispatcher',
            destination='depot',kind='depot_service',target_id=self.scenario.pdn_edges[0].edge_id,
            generated_s=self.time_s,sent_s=self.time_s,expires_s=self.time_s+self.authorization_ttl_s,
            state_version=0,payload=dict(unit_id=unit_id,operation='replace_battery')))
        self._flush_evidence()

    def _receive_uav_mission(self,received):
        m=received.message;p=m.payload['uav_mission']
        result=dict(time_s=self.time_s,message_id=m.message_id,applied=False,meaning='mission_started_not_switch_completed')
        try:
            if self.uav_execution is None or self.onsite_operation_start_check is None:
                raise ValueError('UAV control integration is unavailable')
            node=next(n for n in self.scenario.pdn_nodes if n.node_id==p['bus_id'])
            mission_target=f"relay:{p['bus_id']}" if p['operation']=='relay' else m.target_id
            self.uav_execution.dispatch(p['unit_id'],target_id=mission_target,target_xy=(node.coord_x_km,node.coord_y_km),
                mode='relay' if p['operation']=='relay' else 'control',
                authorization=dict(message_id=m.message_id,target_id=mission_target,authorized=True,
                    received_s=received.received_s,expires_s=m.expires_s))
            result['applied']=True
        except (ValueError,KeyError,StopIteration) as exc:result['reason']=str(exc)
        self.command_events.append(result)

    def _flush_evidence(self):
        if self.trace:
            for event in self.local_decisions[self._local_offset:]:
                self.trace.emit('local_policy_decision',event['time_s'],event)
            for event in self.bus.events[self._message_offset:]:
                self.trace.emit('message', event['event_s'], event)
            for event in self.ledger.events[self._resource_offset:]:
                self.trace.emit('resource_event', event['time_s'], event)
            if self.road_execution:
                for event in self.road_execution.events[self._road_offset:]:
                    self.trace.emit('road_event',event['time_s'],event)
            for event in self.command_events[self._command_offset:]:
                self.trace.emit('command_execution',event['time_s'],event)
            if self.uav_execution:
                for event in self.uav_execution.events[self._uav_offset:]:
                    self.trace.emit('uav_event',event['time_s'],event)
            if self.controls:
                for event in self.controls.events[self._control_offset:]:
                    self.trace.emit('control_work',event['time_s'],event)
        self._message_offset = len(self.bus.events)
        self._resource_offset = len(self.ledger.events)
        self._road_offset = len(self.road_execution.events) if self.road_execution else 0
        self._command_offset = len(self.command_events)
        self._uav_offset = len(self.uav_execution.events) if self.uav_execution else 0
        self._control_offset=len(self.controls.events) if self.controls else 0
        self._local_offset=len(self.local_decisions)

    def advance(self, until_s):
        if self.failure is not None:
            raise RuntimeError('Failed physical attempt requires diagnosis; automatic continuation is disabled')
        try:
            return self._advance(until_s)
        except Exception as exc:
            self.failure = dict(time_s=self.time_s, requested_until_s=until_s,
                                exception=type(exc).__name__, message=str(exc),
                                resource_state=self.ledger.snapshot())
            self._flush_evidence()
            if self.trace:
                self.trace.emit('physical_failure', self.time_s, self.failure)
            raise

    def _advance(self, until_s):
        if not math.isfinite(until_s) or not self.time_s < until_s <= self.horizon_s:
            raise ValueError('Advance must remain within the common physical horizon')
        while self.time_s < until_s:
            physical_times = [self._next_tick(self.control_period_s)]
            physical_times += [t for t in self.releases.values() if t > self.time_s]
            physical_times += [e['time_s'] for e in self.availability_events if e['time_s'] > self.time_s]
            physical_times += [e['time_s'] for e in self.connection_events if e['time_s'] > self.time_s]
            physical_times += [u.arrival_s for u in self.ledger.units.values()
                               if u.arrival_s is not None and u.arrival_s > self.time_s]
            if self.road_execution:
                physical_times.append(self.road_execution.next_event_s())
            if self.uav_execution:
                physical_times.append(self.uav_execution.next_event_s())
            if self.controls:
                physical_times.append(self.controls.next_event_s())
            physical_at = min(physical_times)
            report_at = self._next_tick(self.report_period_s)
            stop = min(until_s, physical_at, report_at, self.bus.next_arrival_s())
            # Deliver earlier packets against the still-active physical state.
            # Endpoint delivery waits until fault/availability/power updates.
            self.bus.advance(stop, include_endpoint=False, on_receive=self._receive)
            if stop > self.time_s:
                interval = self.plant.commit_until(stop)
                if self.trace:
                    self.trace.emit('physical_interval', stop, interval)
            if stop == physical_at and stop < self.horizon_s:
                self._apply_availability(stop)
                self._solve_and_publish_power()
                if self.road_execution:
                    self.road_execution.process(powered_buses=self._powered_buses())
                    self._solve_and_publish_power()
            if self.uav_execution:
                self.uav_execution.advance(stop,operation_check=self.onsite_operation_check,
                    operation_start_check=self.onsite_operation_start_check)
                # A local execution callback may change switch state. Rebuild
                # supply before endpoint messages, never invent repair success.
                if stop == physical_at and stop < self.horizon_s:
                    self._solve_and_publish_power()
            if self.controls:
                offset=len(self.controls.events)
                if self.controls.complete_at(stop):
                    for event in self.controls.events[offset:]:
                        if event['applied']:self.control_versions[event['target']]+=1
                    if stop<self.horizon_s:self._solve_and_publish_power()
            self.bus.advance(stop, on_receive=self._receive)
            for identity,pending in list(self.pending_uav_authorizations.items()):
                if stop>pending.message.expires_s:
                    self.command_events.append(dict(time_s=stop,message_id=identity,applied=False,
                        reason='uav_authorization_expired_before_reconciliation'))
                    del self.pending_uav_authorizations[identity]
            if self._command_changed_power:
                self._solve_and_publish_power()
                self._command_changed_power = False
            if stop == report_at and stop < self.horizon_s:
                self._report()
                self.bus.advance(stop, on_receive=self._receive)
            self._flush_evidence()
        return self.observe()

    def service_metrics(self):
        critical = {n.node_id for n in self.scenario.pdn_nodes if n.criticality == 'critical'}
        total = critical_total = 0.
        for row in self.plant.intervals:
            dt = (row['end_s']-row['start_s'])/3600
            total += (row['electrical']['demand_kw']-row['electrical']['served_kw'])*dt
            critical_total += sum(b['p_kw']-b['served_kw'] for b in row['electrical']['loads']
                                  if b['bus_id'] in critical)*dt
        return dict(system_ens_kwh=total, critical_ens_kwh=critical_total,
                    observed_until_s=self.time_s, complete=self.time_s == self.horizon_s)
