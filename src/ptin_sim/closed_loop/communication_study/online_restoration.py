"""Timed policy execution on the existing event-driven physical evaluator."""
from copy import deepcopy
import math
import time

from .physical_actions import action_id
from .joint_catalogue import build_joint_catalogue


class DecisionUnavailable(RuntimeError):
    """Known transport/admission failure; a local policy may take over."""


def semantic_action(action):
    """Keep the requested operation and route; only report identifiers may refresh."""
    value=deepcopy(action);value.pop('evidence',None)
    if 'route' in value:value['route'].pop('observed_s',None)
    return action_id(value)


def report_expiries(observation,max_age_s):
    rows=list(observation.get('target_observations',{}).values())
    rows+=list(observation.get('source_observations',{}).values())
    if observation.get('uav_observation'):rows.append(observation['uav_observation'])
    result={}
    for row in rows:
        if not 0<=row['observed_s']<=row['received_s']<=observation['time_s']:
            raise ValueError('Noncausal response lease')
        expiry=row['observed_s']+max_age_s
        key=row['message_id']
        if key in result and result[key]!=expiry:raise ValueError('Conflicting report identity')
        result[key]=expiry
    return result


def response_lease(joints,expires,start,end):
    """Keep a query possible while at least one active joint retains evidence.

    Each joint is bounded by its own earliest report expiry. A short-lived
    alternative must not prevent selection of a longer-lived alternative.
    The selected joint still undergoes the original per-report arrival gate.
    Wait does not extend a query whose active alternatives have all expired.
    """
    active=[joint for joint in joints if any(a['kind']!='wait' for a in joint)]
    if not active:return end
    return max(min((expires.get(key,start) for action in joint
                    for key in action.get('evidence',[])),default=start)
               for joint in active)


def revalidate_joint(joint,original_observation,session):
    """Require original evidence to be unexpired and present-day received support.

    New reports may refresh identical actions, but never change the selected
    operation, destination, route, duration, or observed route power assumptions.
    The physical executor subsequently checks delivery and actual feasibility.
    """
    expires=report_expiries(original_observation,session.parameters['max_age_s'])
    now=session.environment.time_s
    for action in joint:
        for key in action.get('evidence',[]):
            if key not in expires or expires[key]<now:
                return None,'original_evidence_expired_or_missing'
    current={semantic_action(a):a for a in session.candidates()}
    if any(semantic_action(a) not in current for a in joint):
        return None,'action_no_longer_supported'
    return [deepcopy(current[semantic_action(a)]) for a in joint],None


class TimedPhysicalController:
    """One joint command per fixed interval, with measured waiting and takeover.

    Selectors only receive received observations and a common candidate list.
    Their result is {'joint': candidate-or-None, ...audit fields}. Simulator
    integration time is not policy latency. A clock can be injected for tests.
    """
    def __init__(self,session,selector,fallback,*,proposal_seed,proposal_draws,
                 timeout_s=45.,clock=time.perf_counter):
        if not math.isfinite(timeout_s) or not 0<timeout_s<=45:
            raise ValueError('Online timeout must be in (0,45] seconds')
        self.session=session;self.selector=selector;self.fallback=fallback
        self.seed=proposal_seed;self.draws=proposal_draws
        self.timeout=timeout_s;self.clock=clock;self.records=[]

    def _catalogue(self):
        return build_joint_catalogue(self.session.candidates(),seed=self.seed,joint_draws=self.draws)

    def step(self):
        api=self.session;env=api.environment;start=env.time_s
        if env.failure is not None or start>=env.horizon_s:
            raise RuntimeError('Cannot continue a failed or finished trial')
        grid=start/api.period
        if not math.isclose(grid,round(grid),abs_tol=1e-9):
            raise ValueError('Online decisions must start on the original fixed grid')
        end=min(env.horizon_s,(round(grid)+1)*api.period)
        before=env.service_metrics();wall=self.clock()
        observation=env.observe();joints=self._catalogue()
        expires=report_expiries(observation,api.parameters['max_age_s'])
        lease=response_lease(joints,expires,start,end)
        timeout=max(0.,min(self.timeout,end-start,lease-start)-(self.clock()-wall))
        if len(joints)==1:
            result=dict(joint=joints[0],source='singleton',reason=None)
        elif timeout<=0:
            result=dict(joint=None,source='local',reason='no_response_time_remaining')
        else:
            try:result=self.selector(deepcopy(observation),deepcopy(joints),timeout_s=timeout)
            except DecisionUnavailable as exc:
                result=dict(joint=None,source='unavailable',reason=str(exc))
        elapsed=self.clock()-wall
        if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid policy wall clock')
        arrival=min(end,start+elapsed)
        if arrival>env.time_s:env.advance(arrival)
        chosen=result.get('joint');reason=result.get('reason')
        if elapsed>=end-start or (timeout>0 and elapsed>min(self.timeout,lease-start) and len(joints)>1):
            chosen=None;reason='late_response'
        elif chosen is not None and chosen not in joints:
            chosen=None;reason='not_in_snapshot_catalogue'
        validated=None;fallback_used=False;fallback_elapsed=0.
        if chosen is not None and env.time_s<end:
            validated,reason=revalidate_joint(chosen,observation,api)
        if validated is None and env.time_s<end:
            fallback_used=True
            # A bounded local decision gets its own current observation and clock.
            fw=self.clock();fallback_observation=env.observe();fresh=self._catalogue()
            fallback_joint=self.fallback(fallback_observation,fresh)
            fallback_elapsed=self.clock()-fw
            if not math.isfinite(fallback_elapsed) or fallback_elapsed<0:
                raise ValueError('Invalid fallback wall clock')
            until=min(end,env.time_s+fallback_elapsed)
            if until>env.time_s:env.advance(until)
            if fallback_joint not in fresh:raise ValueError('Fallback selected an unsupported joint')
            if env.time_s<end:
                validated,fallback_rejection=revalidate_joint(fallback_joint,fallback_observation,api)
                if validated is None:
                    reason=(reason or 'abstain')+';fallback_'+fallback_rejection
                    validated=[{'kind':'wait'}]
        issued_at=None
        if validated is not None and env.time_s<end:
            api.submit_joint(validated);issued_at=env.time_s
        record=dict(grid_start_s=start,grid_end_s=end,observation=observation,
            response=result,policy_elapsed_s=elapsed,fallback_elapsed_s=fallback_elapsed,
            response_arrival_s=start+elapsed,time_accounted_until_s=arrival,response_lease_s=lease,rejection_reason=reason,
            response_lease_rule='latest_remaining_active_joint; original selected evidence revalidated at arrival',
            fallback_used=fallback_used,issued_at_s=issued_at,issued_joint=validated,
            response_accepted=chosen is not None and not fallback_used and issued_at is not None,
            latency_scope='catalogue plus selector plus fallback; final gate and simulator integration excluded')
        self.records.append(record)
        if env.trace:env.trace.emit('online_decision',env.time_s,record)
        transition=api.advance_transition(end,before=before)
        record['interval_critical_ens_kwh']=-transition['reward_negative_critical_ens_kwh']
        record['interval_system_ens_kwh']=-transition['reward_negative_ens_kwh']
        return transition
