"""Separate finite repair work from switching; a UAV cannot repair a conductor.

Capabilities, actual device presence and controller power are explicit physical
inputs. Authorization is checked at both start and completion, never invented
from a policy proposal or a send event.
"""
from copy import deepcopy
import math

from .packets import execution_evidence


class PDNControlExecution:
    def __init__(self, scenario, bus, *, repaired, switch_closed, active_faults,
                 capabilities, unit_at_bus, controller_available, observed_version,
                 max_age_s=900., require_freshness=True,
                 require_reconciliation=True,local_controller_available=None):
        self.edges={e.edge_id:e for e in scenario.pdn_edges}
        self.bus=bus
        self.repaired,self.switch_closed=repaired,switch_closed
        self.active_faults=active_faults
        self.capabilities={k:frozenset(v) for k,v in capabilities.items()}
        self.unit_at_bus,self.controller_available=unit_at_bus,controller_available
        self.local_controller_available=local_controller_available
        self.observed_version=observed_version
        if not math.isfinite(max_age_s) or max_age_s<=0:raise ValueError('Invalid authorization age')
        self.max_age_s=max_age_s
        self.require_freshness=bool(require_freshness)
        self.require_reconciliation=bool(require_reconciliation)
        self.jobs,self.events={},[]

    def _check(self,key,target,kind,received,at):
        if target not in self.edges or kind not in {'repair','open','close'}:
            return False,'unknown_target_or_action'
        required='repair' if kind=='repair' else 'switch'
        if required not in self.capabilities.get(key,()):return False,'device_not_capable'
        if received is None:return False,'no_received_authorization'
        if received.message.kind!='authorization' or received.message.source!='dispatcher':
            return False,'wrong_authorization_source_or_kind'
        destination=received.message.destination
        if destination not in {'terminal','depot',f'unit:{key}'}:
            return False,'authorization_addressed_to_another_unit'
        if destination=='depot' and received.message.payload.get('uav_mission',{}).get('unit_id')!=key:
            return False,'authorization_addressed_to_another_uav'
        # Receiving is an actual event in this bus, not a caller-supplied timestamp.
        if self.bus._all_received.get(received.message.message_id)!=received:
            return False,'authorization_not_received_on_this_bus'
        valid,reasons=execution_evidence(cached=received,now_s=at,target_id=target,
            observed_state_version=self.observed_version(target),max_age_s=self.max_age_s,
            require_freshness=self.require_freshness,
            reconciled=self.bus.acknowledged(received.message.message_id),
            require_reconciliation=self.require_reconciliation)
        if not valid:return False,';'.join(reasons)
        allowed=received.message.payload.get('allowed_actions',())
        if kind not in allowed:return False,'authorization_does_not_cover_action'
        e=self.edges[target]
        if self.unit_at_bus(key) not in {e.from_node,e.to_node}:return False,'device_not_at_target'
        faults=self.active_faults()
        if kind=='repair' and target not in faults:return False,'no_active_fault_to_repair'
        if kind=='close' and target in faults:return False,'unrepaired_fault_cannot_be_closed'
        powered=(self.local_controller_available(target,key) if self.local_controller_available
                 else self.controller_available(target))
        if kind!='repair' and not powered:return False,'controller_not_powered'
        return True,'physical_and_packet_checks_passed'

    def start(self,key,*,target,kind,received,now_s,duration_s):
        if not math.isfinite(duration_s) or duration_s<=0 or not math.isfinite(now_s) or now_s<0:
            raise ValueError('Finite positive work duration required')
        if key in self.jobs or any(j['target']==target for j in self.jobs.values()):
            raise ValueError('Device or target already has an active task')
        accepted,reason=self._check(key,target,kind,received,now_s)
        row=dict(time_s=now_s,event='control_work_started' if accepted else 'control_work_rejected',
                 unit_id=key,target=target,kind=kind,packet_id=received.message.message_id if received else None,
                 accepted=accepted,reason=reason,duration_s=duration_s)
        self.events.append(row)
        if accepted:self.jobs[key]=dict(target=target,kind=kind,received=received,done_s=now_s+duration_s)
        return accepted,reason

    def next_event_s(self):
        return min((j['done_s'] for j in self.jobs.values()),default=math.inf)

    def complete_at(self,at,*,unit_id=None):
        changed=False
        for key in sorted(tuple(self.jobs)):
            if unit_id is not None and key!=unit_id:continue
            job=self.jobs[key]
            if job['done_s']!=at:continue
            accepted,reason=self._check(key,job['target'],job['kind'],job['received'],at)
            before=dict(repaired=sorted(self.repaired),switch_closed=sorted(self.switch_closed))
            if accepted:
                target=job['target']
                if job['kind']=='repair':
                    self.repaired.add(target)
                    # Repair does not imply permission to energize. A distinct
                    # closing command is needed after repaired-state telemetry.
                    self.switch_closed.discard(target)
                elif job['kind']=='open':self.switch_closed.discard(target)
                else:self.switch_closed.add(target)
                changed=True
            self.events.append(dict(time_s=at,event='control_work_finished',unit_id=key,
                target=job['target'],kind=job['kind'],packet_id=job['received'].message.message_id,
                applied=accepted,reason=reason,before=before,
                after=dict(repaired=sorted(self.repaired),switch_closed=sorted(self.switch_closed))))
            del self.jobs[key]
        return changed
