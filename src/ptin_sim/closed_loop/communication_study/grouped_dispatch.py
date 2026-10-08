"""Exact aggregation of co-located homothetic inverter feasible sets.

Group only equal Pmax/Smax and Qmax/Smax ratios and equal grid-forming roles.
The sum of scaled identical convex feasible sets equals their scaled sum.
Proportional-S disaggregation preserves every original P/Q/S bound. No voltage,
branch, source, load or battery constraint is loosened by this transformation.
"""
from collections import defaultdict, OrderedDict
from copy import deepcopy
import hashlib
import math
from ..publication_island_power import SitedSupply


class GroupedSitedDispatch:
    def __init__(self,backend,*,cache_size=512):
        if not isinstance(cache_size,int) or cache_size<0:raise ValueError('Nonnegative integer cache size required')
        self.backend,self.cache_size=backend,cache_size
        self.cache=OrderedDict()
        self.actual_solves,self.cache_hits=0,0

    def run(self,scenario,*,closed_pdn_edges,supplies=()):
        groups=defaultdict(list);direct=[]
        if len({s.unit_id for s in supplies})!=len(supplies):raise ValueError('Duplicate source identity')
        for s in supplies:
            if not all(math.isfinite(x) and x>=0 for x in (s.max_p_kw,s.max_q_kvar,s.max_s_kva)):
                raise ValueError('Invalid source bounds')
            # Keep reference-forming identities unchanged: the physical backend
            # selects an island voltage reference by stable unit identity.
            if s.max_s_kva==0 or s.grid_forming:direct.append(s);continue
            # Exact ratios: no tolerance-based merge of unlike feasible sets.
            key=(s.bus_id,s.grid_forming,(s.max_p_kw/s.max_s_kva).hex(),(s.max_q_kvar/s.max_s_kva).hex())
            groups[key].append(s)
        mapping={};reduced=list(direct)
        original_ids={s.unit_id for s in supplies}
        for key,members in sorted(groups.items()):
            if len(members)==1:reduced.extend(members);continue
            name='group_'+hashlib.sha256(repr(key).encode()).hexdigest()[:20]
            if name in original_ids:raise ValueError('Reserved group identity collision')
            mapping[name]=members
            reduced.append(SitedSupply(name,key[0],sum(s.max_p_kw for s in members),
                sum(s.max_q_kvar for s in members),sum(s.max_s_kva for s in members),key[1]))
        # Battery energy matters through the exact per-interval supply limits.
        # Cache only identical physical problems, never rounded energies/limits.
        signature=(tuple((n.node_id,n.active_power_kw,n.reactive_power_kvar,n.base_voltage_kv,
                          n.criticality,tuple(sorted(n.raw.items()))) for n in scenario.pdn_nodes),
                   tuple((e.edge_id,e.from_node,e.to_node,e.r_pu,e.x_pu,e.length_km) for e in scenario.pdn_edges),
                   tuple(sorted(closed_pdn_edges)),tuple(reduced),
                   repr(getattr(self.backend,'assumptions',None)),
                   repr(getattr(self.backend,'branch_ohms',None)),
                   repr(getattr(self.backend,'branch_ratings_ka',None)))
        if signature in self.cache:
            self.cache_hits+=1;self.cache.move_to_end(signature)
            result=deepcopy(self.cache[signature])
            cache_hit=True
        else:
            result=self.backend.run(scenario,closed_pdn_edges=closed_pdn_edges,supplies=tuple(reduced))
            self.actual_solves+=1;cache_hit=False
            if self.cache_size:
                self.cache[signature]=deepcopy(result)
                while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        expanded=[];audit=[]
        for row in result['supplies']:
            if row['unit_id'] not in mapping:expanded.append(row);continue
            members=mapping[row['unit_id']];total=sum(s.max_s_kva for s in members)
            for s in members:
                fraction=s.max_s_kva/total
                expanded.append(dict(unit_id=s.unit_id,bus_id=s.bus_id,
                                     p_kw=row['p_kw']*fraction,q_kvar=row['q_kvar']*fraction))
            audit.append(dict(group_id=row['unit_id'],bus_id=row['bus_id'],
                members=[dict(unit_id=s.unit_id,p_max_kw=s.max_p_kw,q_max_kvar=s.max_q_kvar,
                              s_max_kva=s.max_s_kva) for s in members],
                p_kw=row['p_kw'],q_kvar=row['q_kvar'],rule='proportional_apparent_rating'))
        return dict(result,supplies=expanded,exact_supply_groups=audit,
                    physical_supply_count=len(supplies),solver_supply_count=len(reduced),
                    identical_physics_cache_hit=cache_hit)
