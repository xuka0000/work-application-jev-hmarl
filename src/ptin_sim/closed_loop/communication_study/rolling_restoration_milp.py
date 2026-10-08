"""Multiperiod linearized electrical dispatch over causal physical missions.

This is a local rolling-horizon adaptation, not the complete 2025 benchmark.
It jointly optimizes a shared first-action catalogue, nodal P/Q service and
finite storage use over future periods. Lossless LinDistFlow and conservative
diamond apparent-power limits approximate AC operation. The real executor
retains packet, travel, energy and nonlinear electrical checks.
"""
from copy import deepcopy
import math
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from .physical_actions import received_state


class LinearModel:
    def __init__(self):
        self.keys=[];self.index={};self.low=[];self.high=[];self.integral=[];self.cost=[]
        self.rows=[];self.row_low=[];self.row_high=[]

    def var(self,key,lo=0.,hi=math.inf,cost=0.,binary=False):
        if key in self.index:raise ValueError('Duplicate optimization variable')
        index=len(self.keys);self.index[key]=index;self.keys.append(key)
        self.low.append(lo);self.high.append(hi);self.cost.append(cost);self.integral.append(int(binary))
        return index

    def constraint(self,values,lo=-math.inf,hi=math.inf):
        self.rows.append(values);self.row_low.append(lo);self.row_high.append(hi)

    def solve(self,wall_limit_s):
        r=[];c=[];data=[]
        for i,row in enumerate(self.rows):
            for key,value in row.items():
                if value:r.append(i);c.append(key);data.append(value)
        matrix=coo_matrix((data,(r,c)),shape=(len(self.rows),len(self.keys))).tocsc()
        return milp(np.array(self.cost),integrality=np.array(self.integral),
            bounds=Bounds(self.low,self.high),constraints=LinearConstraint(matrix,self.row_low,self.row_high),
            options=dict(time_limit=wall_limit_s,mip_rel_gap=.001,disp=False))


def solve_dispatch_disaggregated(problem,*,wall_limit_s=10.,fixed_candidate=None):
    """The input contains forecasts only; no simulator or release calendar."""
    if not math.isfinite(wall_limit_s) or wall_limit_s<=0:raise ValueError('Finite solver budget required')
    nodes=problem['nodes'];edges=problem['edges'];forecasts=problem['forecasts']
    periods=problem['periods'];hours=problem['period_s']/3600.;count=len(forecasts)
    if count<1 or periods<1 or hours<=0:raise ValueError('Nonempty finite optimization horizon required')
    model=LinearModel();z=[model.var(('choice',i),hi=1.,binary=True) for i in range(count)]
    model.constraint({i:1. for i in z},lo=1.,hi=1.)
    if fixed_candidate is not None:
        if not 0<=fixed_candidate<count:raise ValueError('Fixed candidate outside catalogue')
        model.constraint({z[fixed_candidate]:1.},lo=1.,hi=1.)
    # Every candidate supplies its own physical forecast. Exactly one is active.
    for choice,forecast in enumerate(forecasts):
        supplies=forecast['supplies'];pcap=problem['source_kva'];ncount=len(nodes)
        p_by_unit={u['id']:[] for u in supplies}
        for t in range(periods):
            served={};voltage={};energized={};pflow={};qflow={};reach={};pg={};qg={};root={}
            for node,n in nodes.items():
                served[node]=model.var(('served',choice,t,node),hi=n['p_kw'],
                    cost=-hours if n['critical'] else 0.)
                energized[node]=model.var(('energized',choice,t,node),hi=1.,binary=True)
                voltage[node]=model.var(('voltage2',choice,t,node),lo=.95**2,hi=1.05**2)
                model.constraint({served[node]:1.,energized[node]:-n['p_kw']},hi=0.)
                model.constraint({energized[node]:1.,z[choice]:-1.},hi=0.)
                root[node]=model.var(('rootflow',choice,t,node),hi=ncount)
            for edge,e in edges.items():
                cap=e['kva'] if forecast['closed'][edge][t] else 0.
                pflow[edge]=model.var(('P',choice,t,edge),lo=-cap,hi=cap)
                qflow[edge]=model.var(('Q',choice,t,edge),lo=-cap,hi=cap)
                reach[edge]=model.var(('reach',choice,t,edge),lo=-ncount if cap else 0.,hi=ncount if cap else 0.)
                for sp in (-1.,1.):
                    for sq in (-1.,1.):
                        model.constraint({pflow[edge]:sp,qflow[edge]:sq,z[choice]:-cap},hi=0.)
                if cap:
                    # r*P/(1000*kV^2) uses kW and ohm, yielding squared pu.
                    equation={voltage[e['a']]:1.,voltage[e['b']]:-1.,
                        pflow[edge]:-2*e['r_ohm']/(1000*problem['voltage_kv']**2),
                        qflow[edge]:-2*e['x_ohm']/(1000*problem['voltage_kv']**2)}
                    model.constraint(equation,lo=0.,hi=0.)
            sources=[dict(id='upstream',node=problem['source'],pmax=pcap,qmax=pcap,smax=pcap,forming=True)]
            sources += [dict(id=u['id'],node=u['nodes'][t],pmax=u['pmax'],qmax=u['qmax'],smax=u['smax'],forming=u['forming'])
                        for u in supplies if u['nodes'][t] is not None]
            for supply in sources:
                key=supply['id'];node=supply['node']
                pg[key]=model.var(('Pg',choice,t,key),hi=supply['pmax'])
                qg[key]=model.var(('Qg',choice,t,key),lo=-supply['qmax'],hi=supply['qmax'])
                model.constraint({pg[key]:1.,energized[node]:-supply['pmax']},hi=0.)
                for sign in (-1.,1.):model.constraint({pg[key]:1.,qg[key]:sign,z[choice]:-supply['smax']},hi=0.)
                if key!='upstream':p_by_unit[key].append(pg[key])
            for node,n in nodes.items():
                ps={served[node]:-1.};qs={served[node]:-n['q_kvar']/n['p_kw'] if n['p_kw'] else 0.}
                connectivity={root[node]:1.,energized[node]:-1.}
                for edge,e in edges.items():
                    sign=1. if e['b']==node else -1. if e['a']==node else 0.
                    if sign:ps[pflow[edge]]=sign;qs[qflow[edge]]=sign;connectivity[reach[edge]]=sign
                at_node=[s for s in sources if s['node']==node]
                for s in at_node:ps[pg[s['id']]]=1.;qs[qg[s['id']]]=1.
                model.constraint(ps,lo=0.,hi=0.);model.constraint(qs,lo=0.,hi=0.)
                model.constraint(connectivity,lo=0.,hi=0.)
                forming=any(s['forming'] for s in at_node)
                model.constraint({root[node]:1.,z[choice]:-ncount if forming else 0.},hi=0.)
            model.constraint({voltage[problem['source']]:1.},lo=1.,hi=1.)
        for u in supplies:model.constraint({p:hours for p in p_by_unit[u['id']]},hi=u['energy_kwh'])
    started=time.monotonic();result=model.solve(wall_limit_s)
    if result.x is None:
        raise RuntimeError(f'Multiperiod MILP returned no incumbent, status={result.status}')
    vector=np.asarray(result.x)
    violation=max([0.]+[max(lo-sum(vector[k]*v for k,v in row.items()),
                                  sum(vector[k]*v for k,v in row.items())-hi,0.)
                            for row,lo,hi in zip(model.rows,model.row_low,model.row_high)])
    if violation>1e-5 or max(abs(vector[z]-np.round(vector[z])))>1e-5:
        raise RuntimeError('MILP incumbent failed independent constraint residual check')
    index=int(np.argmax(vector[z]));total=sum(n['p_kw'] for n in nodes.values() if n['critical'])*hours*periods
    return dict(selected_index=index,predicted_critical_ens_kwh=max(0.,total+float(result.fun)),
        status=int(result.status),optimal=result.status==0,mip_gap=float(result.mip_gap),
        bound=float(result.mip_dual_bound),max_constraint_violation=violation,
        elapsed_s=time.monotonic()-started,variables=len(model.keys),constraints=len(model.rows),
        served_kwh={node:sum(vector[model.index[('served',index,t,node)]]*hours for t in range(periods)) for node in nodes},
        model='lossless_LinDistFlow_diamond_thermal_limits_finite_storage')


def solve_dispatch(problem,*,wall_limit_s=10.,fixed_candidate=None):
    """Share physical variables across mutually exclusive mission forecasts.

This avoids duplicating a feeder for each candidate. All physical availability
and finite energy bounds are linked to the selected forecast with linear terms.
"""
    if not math.isfinite(wall_limit_s) or wall_limit_s<=0:raise ValueError('Finite solver budget required')
    nodes=problem['nodes'];edges=problem['edges'];forecasts=problem['forecasts']
    periods=problem['periods'];hours=problem['period_s']/3600.;count=len(forecasts);ncount=len(nodes)
    if count<1 or periods<1 or hours<=0:raise ValueError('Nonempty finite optimization horizon required')
    model=LinearModel();z=[model.var(('choice',i),hi=1.,binary=True) for i in range(count)]
    model.constraint({i:1. for i in z},lo=1.,hi=1.)
    if fixed_candidate is not None:
        if not 0<=fixed_candidate<count:raise ValueError('Fixed candidate outside catalogue')
        model.constraint({z[fixed_candidate]:1.},lo=1.,hi=1.)
    units=[{u['id']:u for u in f['supplies']} for f in forecasts]
    unit_ids=sorted(set().union(*(u.keys() for u in units)))
    all_power={key:[] for key in unit_ids};served_all={};pcap=problem['source_kva']
    for t in range(periods):
        served={};voltage={};energized={};root={};pflow={};qflow={};reach={};supply_p={};supply_q={}
        for node,n in nodes.items():
            served[node]=model.var(('served',t,node),hi=n['p_kw'],cost=-hours if n['critical'] else 0.)
            served_all[t,node]=served[node]
            energized[node]=model.var(('energized',t,node),hi=1.,binary=True)
            voltage[node]=model.var(('voltage2',t,node),lo=.95**2,hi=1.05**2)
            root[node]=model.var(('rootflow',t,node),hi=ncount)
            model.constraint({served[node]:1.,energized[node]:-n['p_kw']},hi=0.)
        for edge,e in edges.items():
            cap=e['kva'];availability={z[i]:1. for i,f in enumerate(forecasts) if f['closed'][edge][t]}
            pflow[edge]=model.var(('P',t,edge),lo=-cap,hi=cap)
            qflow[edge]=model.var(('Q',t,edge),lo=-cap,hi=cap)
            reach[edge]=model.var(('reach',t,edge),lo=-ncount,hi=ncount)
            for sp in (-1.,1.):
                for sq in (-1.,1.):model.constraint({pflow[edge]:sp,qflow[edge]:sq,
                    **{key:-cap*value for key,value in availability.items()}},hi=0.)
            for sign in (-1.,1.):model.constraint({reach[edge]:sign,
                **{key:-ncount*value for key,value in availability.items()}},hi=0.)
            equation={voltage[e['a']]:1.,voltage[e['b']]:-1.,
                pflow[edge]:-2*e['r_ohm']/(1000*problem['voltage_kv']**2),
                qflow[edge]:-2*e['x_ohm']/(1000*problem['voltage_kv']**2)}
            big_m=1.05**2-.95**2
            model.constraint({**equation,**{key:big_m for key in availability}},hi=big_m)
            model.constraint({**equation,**{key:-big_m for key in availability}},lo=-big_m)
        source=problem['source']
        supply_p['upstream',source]=model.var(('Pg',t,'upstream'),hi=pcap)
        supply_q['upstream',source]=model.var(('Qg',t,'upstream'),lo=-pcap,hi=pcap)
        for sign in (-1.,1.):model.constraint({supply_p['upstream',source]:1.,supply_q['upstream',source]:sign},hi=pcap)
        for key in unit_ids:
            locations=sorted({u[key]['nodes'][t] for u in units if key in u and u[key]['nodes'][t] is not None})
            for node in locations:
                available={i:u[key] for i,u in enumerate(units) if key in u and u[key]['nodes'][t]==node}
                pmax=max(u['pmax'] for u in available.values());qmax=max(u['qmax'] for u in available.values())
                pv=model.var(('Pg',t,key,node),hi=pmax);qv=model.var(('Qg',t,key,node),lo=-qmax,hi=qmax)
                supply_p[key,node]=pv;supply_q[key,node]=qv;all_power[key].append(pv)
                model.constraint({pv:1.,energized[node]:-pmax},hi=0.)
                model.constraint({pv:1.,**{z[i]:-u['pmax'] for i,u in available.items()}},hi=0.)
                for sign in (-1.,1.):
                    model.constraint({pv:1.,qv:sign,**{z[i]:-u['smax'] for i,u in available.items()}},hi=0.)
                    model.constraint({qv:sign,**{z[i]:-u['qmax'] for i,u in available.items()}},hi=0.)
        for node,n in nodes.items():
            ps={served[node]:-1.};qs={served[node]:-n['q_kvar']/n['p_kw'] if n['p_kw'] else 0.}
            connectivity={root[node]:1.,energized[node]:-1.}
            for edge,e in edges.items():
                sign=1. if e['b']==node else -1. if e['a']==node else 0.
                if sign:ps[pflow[edge]]=sign;qs[qflow[edge]]=sign;connectivity[reach[edge]]=sign
            for (key,bus),pv in supply_p.items():
                if bus==node:ps[pv]=1.;qs[supply_q[key,bus]]=1.
            model.constraint(ps,lo=0.,hi=0.);model.constraint(qs,lo=0.,hi=0.)
            model.constraint(connectivity,lo=0.,hi=0.)
            forming={z[i]:1. for i,u in enumerate(units) if any(
                value['forming'] and value['energy_kwh']>0 and value['nodes'][t]==node for value in u.values())}
            if node!=source:model.constraint({root[node]:1.,**{key:-ncount for key in forming}},hi=0.)
        model.constraint({voltage[source]:1.},lo=1.,hi=1.)
    for key in unit_ids:model.constraint({**{p:hours for p in all_power[key]},
        **{z[i]:-u[key]['energy_kwh'] for i,u in enumerate(units) if key in u}},hi=0.)
    started=time.monotonic();result=model.solve(wall_limit_s)
    if result.x is None:raise RuntimeError(f'Multiperiod MILP returned no incumbent, status={result.status}')
    vector=np.asarray(result.x)
    violation=max([0.]+[max(lo-sum(vector[k]*v for k,v in row.items()),
        sum(vector[k]*v for k,v in row.items())-hi,0.) for row,lo,hi in zip(model.rows,model.row_low,model.row_high)])
    violation=max(violation,float(np.maximum(np.array(model.low)-vector,vector-np.array(model.high)).max()))
    discrete=vector[np.array(model.integral,dtype=bool)]
    if violation>1e-5 or max(abs(discrete-np.round(discrete)))>1e-5:
        raise RuntimeError('MILP incumbent failed independent feasibility verification')
    total=sum(n['p_kw'] for n in nodes.values() if n['critical'])*hours*periods
    return dict(selected_index=int(np.argmax(vector[z])),predicted_critical_ens_kwh=max(0.,total+float(result.fun)),
        status=int(result.status),optimal=result.status==0,mip_gap=float(result.mip_gap),bound=float(result.mip_dual_bound),
        max_constraint_violation=violation,elapsed_s=time.monotonic()-started,
        variables=len(model.keys),constraints=len(model.rows),
        served_kwh={node:sum(vector[served_all[t,node]]*hours for t in range(periods)) for node in nodes},
        model='lossless_LinDistFlow_diamond_thermal_limits_finite_storage_shared_physical_variables')


class CausalRollingMILP:
    """Forecast one admitted mission per device and replan next decision.

Unknown branches are unavailable in the conservative forecast. Communication
failures after dispatch and future reports are not predicted. This limitation
is retained in every trace and prevents a full-benchmark-equivalence claim.
"""
    def __init__(self,scenario,resources,profile,*,periods=4,wall_limit_s=10.,horizon_s=14400.):
        self.scenario=scenario;self.resources=dict(resources);self.profile=profile
        self.periods=periods;self.wall_limit_s=wall_limit_s;self.horizon_s=horizon_s

    def problem(self,observation,joints):
        s=self.scenario;r=self.resources;dt=r['decision_period_s'];now=observation['time_s']
        known=received_state(observation,max_age_s=r['observation_max_age_s'])
        periods=min(self.periods,max(1,math.ceil((self.horizon_s-now)/dt)))
        nodes={n.node_id:dict(p_kw=n.active_power_kw,q_kvar=n.reactive_power_kvar,critical=n.criticality=='critical') for n in s.pdn_nodes}
        positions={n.node_id:(n.coord_x_km,n.coord_y_km) for n in s.pdn_nodes}
        edges={e.edge_id:dict(a=e.from_node,b=e.to_node,r_ohm=self.profile['branches'][e.edge_id]['r_ohm'],
            x_ohm=self.profile['branches'][e.edge_id]['x_ohm'],
            kva=math.sqrt(3)*self.profile['voltage_kv']*self.profile['branches'][e.edge_id]['rating_ka']*1000) for e in s.pdn_edges}
        forecasts=[]
        for joint in joints:
            closed={key:[bool(known['targets'].get(key,{}).get('switch_closed',False)
                             and not known['targets'].get(key,{}).get('failed',True))]*periods for key in edges}
            units={key:dict(id=key,nodes=[u.get('connected_bus')]*periods,pmax=u['max_power_kw'],
                qmax=u['max_power_kw']*r['q_to_p_rating'],smax=u['max_power_kw']*r['s_to_p_rating'],
                energy_kwh=u['energy_kwh']*r['discharge_efficiency'],forming=r['mess_grid_forming'] if u['kind']=='mess' else r['ev_grid_forming'])
                for key,u in known['devices'].items() if u['available'] and u.get('destination') is None}
            for action in joint:
                kind=action['kind']
                if kind=='deploy':
                    route=action['route'];unit=units.get(route['unit_id'])
                    if unit:
                        unit['nodes']=[route['destination_bus'] if route['purpose']=='connect' and t*dt>=route['estimated_duration_s'] else None for t in range(periods)]
                        unit['energy_kwh']=max(0.,unit['energy_kwh']-route['distance_km']*r['traction_kwh_per_km'])
                elif kind in {'control','uav_control'}:
                    duration=action.get('duration_s',r['uav_operation_s'])
                    if kind=='uav_control':duration+=math.dist(positions[r['depot_bus']],positions[action['bus_id']])/s.fleet.uav_flight_speed_km_h*3600
                    for t in range(periods):
                        if t*dt>=duration:closed[action['target']][t]=action['operation']!='open'
            forecasts.append(dict(closed=closed,supplies=list(units.values())))
        return dict(nodes=nodes,edges=edges,forecasts=forecasts,periods=periods,period_s=dt,
                    source='799',source_kva=self.profile['source_apparent_power_kva'],voltage_kv=self.profile['voltage_kv'])

    def select(self,observation,joints):
        if len(joints)==1:
            self.last_decision_trace=dict(selected_index=0,singleton=True,solver_run=False)
            return deepcopy(joints[0])
        result=solve_dispatch(self.problem(observation,joints),wall_limit_s=self.wall_limit_s)
        result.update(information_scope='received reports and public physical ratings',
            benchmark_equivalence=False,limitations=['no future report or channel-outcome prediction',
                'one current mission per device; no later contingent repair planning',
                'nonlinear AC and packet execution are checked by the real plant'])
        self.last_decision_trace=result
        return deepcopy(joints[result['selected_index']])
