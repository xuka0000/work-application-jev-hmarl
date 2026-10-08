"""Finite warehouse--site--warehouse UAV missions from source Eqs.90--98.

Battery is expressed in equivalent flight kilometres (source endurance 50 km).
Operation cost, operation time and battery-replacement time must be supplied
explicitly. This onsite-control executor is not an unlimited radio relay.
"""
from copy import deepcopy
import math


class UAVExecution:
    def __init__(self, *, count, depot_xy, speed_km_h, endurance_km,
                 operation_s, operation_equivalent_km, replacement_s):
        if not isinstance(count,int) or count < 1:
            raise ValueError('Positive integral UAV count required')
        values=(*depot_xy,speed_km_h,endurance_km,operation_s,operation_equivalent_km,replacement_s)
        if not all(math.isfinite(v) for v in values) or min(speed_km_h,endurance_km,operation_s,replacement_s)<=0 or operation_equivalent_km<0:
            raise ValueError('Explicit finite UAV flight, operation and replacement parameters required')
        self.depot=tuple(depot_xy)
        self.speed,self.endurance=speed_km_h,endurance_km
        self.operation_s,self.operation_cost,self.replacement_s=operation_s,operation_equivalent_km,replacement_s
        self.time_s=0.
        self.units={f'UAV_{i+1:02d}':dict(phase='warehouse',position=self.depot,
            energy_km=endurance_km,flight_used_km=0.,operation_used_km=0.,replacement_added_km=0.,mission=None)
            for i in range(count)}
        self.events=[]
        self.failure=None

    def _record(self,kind,key,before,**extra):
        self.events.append(dict(time_s=self.time_s,event=kind,unit_id=key,
            before=before,after=deepcopy(self.units[key]),**extra))

    def dispatch(self,key,*,target_id,target_xy,authorization,mode='control'):
        if mode not in {'control','relay'}:raise ValueError('Unknown UAV mission mode')
        u=self.units[key]
        if u['phase']!='warehouse' or not target_id or len(target_xy)!=2 or not all(math.isfinite(v) for v in target_xy):
            raise ValueError('Mission requires a warehouse UAV and a finite target')
        if (not authorization or authorization.get('authorized') is not True
                or not authorization.get('message_id') or authorization.get('received_s',math.inf)>self.time_s):
            raise ValueError('Mission requires a received authorization record')
        if authorization.get('target_id')!=target_id or self.time_s>authorization.get('expires_s',-1):
            raise ValueError('Wrong or expired mission authorization')
        distance=math.dist(self.depot,target_xy)
        required=2*distance+self.operation_cost
        if required>u['energy_km']+1e-9:
            raise ValueError('Insufficient energy for outbound, operation AND return')
        if any(v['mission'] and v['mission']['target_id']==target_id for v in self.units.values()):
            raise ValueError('Target already assigned to a UAV')
        before=deepcopy(u)
        travel=distance/self.speed*3600
        if mode=='relay' and self.time_s+travel+self.operation_s>authorization['expires_s']:
            raise ValueError('Relay authorization must cover the complete finite operation')
        u.update(phase='outbound',mission=dict(target_id=target_id,target_xy=tuple(target_xy),
            mode=mode,
            distance_km=distance,depart_s=self.time_s,arrival_s=self.time_s+travel,
            operation_done_s=self.time_s+travel+self.operation_s,
            return_s=self.time_s+2*travel+self.operation_s,authorization=deepcopy(authorization)))
        self._record('uav_departed',key,before)

    def replace_battery(self,key):
        u=self.units[key]
        if u['phase']!='warehouse':raise ValueError('Battery replacement is warehouse-only')
        before=deepcopy(u);u['phase']='replacing';u['replacement_done_s']=self.time_s+self.replacement_s
        self._record('battery_replacement_started',key,before)

    def next_event_s(self):
        times=[]
        for u in self.units.values():
            if u['phase']=='replacing':times.append(u['replacement_done_s'])
            elif u['phase'] in {'outbound','operating','returning'}:
                field={'outbound':'arrival_s','operating':'operation_done_s','returning':'return_s'}[u['phase']]
                times.append(u['mission'][field])
        return min(times,default=math.inf)

    def advance(self,until_s,*,operation_check,operation_start_check=None):
        if self.failure is not None:
            raise RuntimeError('Failed UAV execution requires diagnosis')
        try:
            return self._advance(until_s,operation_check=operation_check,operation_start_check=operation_start_check)
        except Exception as exc:
            self.failure=dict(time_s=self.time_s,exception=type(exc).__name__,message=str(exc),state=self.snapshot())
            raise

    def _advance(self,until_s,*,operation_check,operation_start_check):
        if not math.isfinite(until_s) or until_s<self.time_s:
            raise ValueError('UAV physical time cannot go backwards')
        while self.next_event_s()<=until_s:
            self.time_s=self.next_event_s()
            for key,u in self.units.items():
                before=deepcopy(u);m=u['mission']
                if u['phase']=='outbound' and m['arrival_s']==self.time_s:
                    u['energy_km']-=m['distance_km'];u['flight_used_km']+=m['distance_km']
                    u['position']=m['target_xy'];u['phase']='operating'
                    if operation_start_check is not None:
                        ok,reason=operation_start_check(key,deepcopy(m),self.time_s)
                        m['start_permitted'],m['start_reason']=bool(ok),reason
                    self._record('uav_arrived',key,before)
                elif u['phase']=='operating' and m['operation_done_s']==self.time_s:
                    # Recheck actual expiry and local execution conditions on site.
                    live=self.time_s<=m['authorization']['expires_s']
                    if not m.get('start_permitted',True):accepted,reason=False,m['start_reason']
                    else:accepted,reason=operation_check(key,deepcopy(m),self.time_s) if live else (False,'authorization_expired')
                    u['energy_km']-=self.operation_cost;u['operation_used_km']+=self.operation_cost
                    u['phase']='returning'
                    self._record('onsite_operation_finished',key,before,applied=bool(accepted),reason=reason)
                elif u['phase']=='returning' and m['return_s']==self.time_s:
                    u['energy_km']-=m['distance_km'];u['flight_used_km']+=m['distance_km']
                    u.update(phase='warehouse',position=self.depot,mission=None)
                    self._record('uav_returned_without_refill',key,before)
                elif u['phase']=='replacing' and u['replacement_done_s']==self.time_s:
                    u['replacement_added_km']+=self.endurance-u['energy_km']
                    u['energy_km']=self.endurance;u['phase']='warehouse';del u['replacement_done_s']
                    self._record('battery_replacement_completed',key,before)
        self.time_s=until_s

    def snapshot(self):
        states=deepcopy(self.units)
        for u in states.values():
            m=u['mission']
            if u['phase'] in {'outbound','returning'}:
                if u['phase']=='outbound':a,b,start,end=self.depot,m['target_xy'],m['depart_s'],m['arrival_s']
                else:a,b,start,end=m['target_xy'],self.depot,m['operation_done_s'],m['return_s']
                f=min(1.,max(0.,(self.time_s-start)/(end-start))) if end>start else 1.
                u['position']=tuple(x+f*(y-x) for x,y in zip(a,b))
                # Energy remains committed until leg completion; expose remaining
                # free energy separately to avoid claiming spent flight is free.
                u['inflight_used_km']=f*m['distance_km']
        return states

    def conservation_residuals_km(self):
        return {k:u['energy_km']-(self.endurance+u['replacement_added_km']-
                    u['flight_used_km']-u['operation_used_km']) for k,u in self.units.items()}
