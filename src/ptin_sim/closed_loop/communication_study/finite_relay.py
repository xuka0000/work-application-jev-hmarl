"""Geometry of explicitly deployed, finite UAV relay operations.

Relays bridge live radios. They never energize a CN station or the dispatcher.
Flight, on-site duration, operation energy and return are handled by UAVExecution.
"""
from dataclasses import replace
import math


def with_finite_relays(state, scenario, uav_states, *, radius_km, now_s):
    if not math.isfinite(radius_km) or radius_km<=0:
        raise ValueError('Explicit positive relay radius required')
    active={}
    for key,u in uav_states.items():
        m=u.get('mission')
        if (u['phase']=='operating' and m and m.get('mode')=='relay'
                and m.get('start_permitted') and now_s<m['operation_done_s']
                and now_s<=m['authorization']['expires_s']):
            active['uav:'+key]=tuple(u['position'])
    points={n.node_id:(n.coord_x_km,n.coord_y_km) for n in scenario.cn_nodes
            if n.node_id in state.powered_nodes}
    links=set()
    for key,xy in sorted(active.items()):
        for other,target in sorted(dict(points,**active).items()):
            if key==other or math.dist(xy,target)>radius_km:continue
            a,b=sorted((key,other));links.add((a,b,'relay:'+a+'|'+b))
    return replace(state,relay_nodes=frozenset(active),relay_links=tuple(sorted(links))),active


def with_fixed_relays(state, scenario, bus_ids, *, radius_km):
    """Add registered always-on relay sites without consuming mobile UAV state."""
    if not math.isfinite(radius_km) or radius_km <= 0:
        raise ValueError('Explicit positive fixed-relay radius required')
    nodes={n.node_id:n for n in scenario.pdn_nodes}
    unknown=set(bus_ids)-nodes.keys()
    if unknown:
        raise ValueError(f'Unknown fixed-relay buses: {sorted(unknown)}')
    points={f'fixed:{key}':(nodes[key].coord_x_km,nodes[key].coord_y_km)
            for key in sorted(set(bus_ids))}
    cn={n.node_id:(n.coord_x_km,n.coord_y_km) for n in scenario.cn_nodes}
    links=set(state.relay_links);active=set(state.relay_nodes)
    for key,point in points.items():
        active.add(key)
        for other,xy in cn.items():
            if math.dist(point,xy)<=radius_km:
                a,b=sorted((key,other));links.add((a,b,'fixed_relay:'+a+'|'+b))
    return replace(state,relay_nodes=frozenset(active),relay_links=tuple(sorted(links))),points
