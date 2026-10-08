"""Location-resolved, balanced radial AC dispatch for model development.

The branch-flow equalities follow Farivar and Low, TPWRS 2013, Part I.
A conic relaxation initializes SLSQP, which enforces the original equalities.
Relaxation exactness is not assumed for this bounded-load restoration model.
This is a physical backend, not a published restoration-policy comparator.
Electrical ratings and the inherited impedance base remain explicit assumptions.
"""
from dataclasses import dataclass
import math

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from scipy import sparse


@dataclass(frozen=True)
class ElectricalAssumptions:
    base_mva: float = 10.
    voltage_kv: float = 4.8
    line_rating_ka: float = .2
    voltage_min_pu: float = .95
    voltage_max_pu: float = 1.05
    # This models the study's upstream source, not the stock 2.5 MVA transformer.
    source_power_kw: float = 100000.
    source_apparent_power_kva: float | None = None
    include_facility_load: bool = True
    facility_accounting: str = 'included_in_bus_total'
    independent_facility_switch: bool = False
    facility_q_per_p: float = 0.  # Only used in the separate additional-load diagnostic.
    facility_weight: float = 1.5
    ordinary_weight: float = 1.
    verify_with_pandapower: bool = True


@dataclass(frozen=True)
class SitedSupply:
    unit_id: str
    bus_id: str
    max_p_kw: float
    max_q_kvar: float
    max_s_kva: float
    grid_forming: bool = True


def demand_blocks(scenario, assumptions):
    blocks = []
    for n in scenario.pdn_nodes:
        p = float(n.raw.get('cn_utn_load_kw', 0)) if assumptions.include_facility_load else 0.
        included = assumptions.facility_accounting == 'included_in_bus_total'
        if included and p > n.active_power_kw+1e-7:
            raise ValueError('Facility subset exceeds the recorded bus active load')
        # Reference Eqs.43-46 use one bus load and one load-switch state. The
        # default partitions that load for reporting instead of adding it twice.
        fraction = p/n.active_power_kw if included and n.active_power_kw > 0 else 0.
        blocks.append(dict(bus_id=n.node_id, kind='ordinary',
            p_kw=n.active_power_kw-p if included else n.active_power_kw,
            q_kvar=n.reactive_power_kvar*(1-fraction), weight=assumptions.ordinary_weight))
        if assumptions.include_facility_load:
            if p > 0:
                blocks.append(dict(bus_id=n.node_id, kind='facility', p_kw=p,
                    q_kvar=n.reactive_power_kvar*fraction if included else p*assumptions.facility_q_per_p,
                    weight=assumptions.facility_weight))
    return blocks


class IslandACDispatch:
    def __init__(self, assumptions=None, *, branch_ohms=None, branch_ratings_ka=None):
        self.assumptions = assumptions or ElectricalAssumptions()
        a = self.assumptions
        if a.facility_accounting not in ('included_in_bus_total', 'additional_diagnostic'):
            raise ValueError('Unknown facility load accounting convention')
        if (a.base_mva <= 0 or a.voltage_kv <= 0 or a.line_rating_ka <= 0
                or not 0 < a.voltage_min_pu <= 1 <= a.voltage_max_pu
                or a.source_power_kw <= 0 or a.facility_q_per_p < 0
                or min(a.facility_weight, a.ordinary_weight) <= 0):
            raise ValueError('Invalid electrical assumptions')
        # A supplied physical-ohm profile must cover all lines. Never silently
        # mix a partial source mapping with a guessed transformer impedance.
        self.branch_ohms = branch_ohms
        self.branch_ratings_ka = dict(branch_ratings_ka) if branch_ratings_ka is not None else None
        if a.source_apparent_power_kva is not None and (
                not math.isfinite(a.source_apparent_power_kva) or a.source_apparent_power_kva <= 0):
            raise ValueError('Invalid upstream apparent power rating')
        if self.branch_ratings_ka is not None and any(
                not math.isfinite(v) or v <= 0 for v in self.branch_ratings_ka.values()):
            raise ValueError('Invalid branch current rating')

    def run(self, scenario, *, closed_pdn_edges, supplies=()):
        a = self.assumptions
        nodes = {n.node_id: n for n in scenario.pdn_nodes}
        edges = {e.edge_id: e for e in scenario.pdn_edges}
        if not set(closed_pdn_edges) <= edges.keys():
            raise ValueError('Unknown closed branch')
        if self.branch_ohms is not None and set(self.branch_ohms) != set(edges):
            raise ValueError('Physical impedance profile must cover every branch')
        if self.branch_ratings_ka is not None and set(self.branch_ratings_ka) != set(edges):
            raise ValueError('Current rating profile must cover every branch')
        if any(abs(n.base_voltage_kv-a.voltage_kv) > 1e-9 for n in nodes.values()):
            raise ValueError('Mixed voltage levels require an explicit transformer model')
        if len({s.unit_id for s in supplies}) != len(supplies) or any(s.unit_id == 'upstream' for s in supplies):
            raise ValueError('Supply IDs must be unique and cannot be upstream')
        for s in supplies:
            if s.bus_id not in nodes or not all(math.isfinite(v) and v >= 0
                    for v in (s.max_p_kw, s.max_q_kvar, s.max_s_kva)):
                raise ValueError('Invalid sited supply')
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for key in sorted(closed_pdn_edges):
            e = edges[key]
            if graph.has_edge(e.from_node, e.to_node):
                raise ValueError('Parallel branches require an explicit circuit model')
            graph.add_edge(e.from_node, e.to_node, edge=e)
        if not nx.is_forest(graph):
            raise ValueError('Radial branch-flow backend rejects a meshed topology')
        blocks = demand_blocks(scenario, a)
        all_loads, all_units, all_lines, components = [], [], [], []
        for component in sorted(nx.connected_components(graph), key=lambda c: min(c)):
            units = sorted((s for s in supplies if s.bus_id in component and s.max_p_kw > 1e-8
                            and s.max_s_kva > 1e-8), key=lambda s: s.unit_id)
            loads = [b for b in blocks if b['bus_id'] in component]
            forming = [s for s in units if s.grid_forming]
            root = '799' if '799' in component else forming[0].bus_id if forming else None
            if root is None:
                all_loads.extend(dict(b, fraction=0., served_kw=0., served_kvar=0.) for b in loads)
                all_units.extend(dict(unit_id=s.unit_id, bus_id=s.bus_id, p_kw=0., q_kvar=0.) for s in units)
                components.append(dict(root=None, buses=sorted(component), status='unpowered',
                    served_kw=0., loss_kw=0., independent_ac_verified=False))
                continue
            if root == '799':
                units.insert(0, SitedSupply('upstream', root, a.source_power_kw,
                                           a.source_power_kw, a.source_apparent_power_kva
                                           if a.source_apparent_power_kva is not None else 2*a.source_power_kw))
            else:
                # The actual grid-forming unit supplies the island reference.
                # Its capacity and energy are never replaced by an unlimited grid.
                units.sort(key=lambda s: (s.unit_id != forming[0].unit_id, s.unit_id))
            part = self._solve(graph.subgraph(component), root, loads, units)
            all_loads.extend(part.pop('loads'))
            all_units.extend(part.pop('supplies'))
            all_lines.extend(part.pop('lines'))
            components.append(part)
        # Preserve zero rows for depleted/disconnected device accounting.
        recorded = {r['unit_id'] for r in all_units}
        all_units.extend(dict(unit_id=s.unit_id, bus_id=s.bus_id, p_kw=0., q_kvar=0.)
                         for s in supplies if s.unit_id not in recorded)
        return dict(demand_kw=sum(b['p_kw'] for b in blocks),
            served_kw=sum(b['served_kw'] for b in all_loads),
            critical_demand_kw=sum(b['p_kw'] for b in blocks if b['kind'] == 'facility'),
            critical_served_kw=sum(b['served_kw'] for b in all_loads if b['kind'] == 'facility'),
            mobile_dispatch_kw=sum(s['p_kw'] for s in all_units if s['unit_id'] != 'upstream'),
            loss_kw=sum(c['loss_kw'] for c in components), loads=all_loads,
            supplies=all_units, lines=all_lines, components=components)

    def _solve(self, graph, root, loads, supplies):
        a = self.assumptions
        buses = list(nx.bfs_tree(graph, root))
        bi = {b: i for i, b in enumerate(buses)}
        branches = [(u, v, graph[u][v]['edge']) for u, v in nx.bfs_edges(graph, root)]
        n, m, d, g = len(buses), len(branches), len(loads), len(supplies)
        base_kw, zbase = a.base_mva*1000, a.voltage_kv**2/a.base_mva
        ibase = a.base_mva/(math.sqrt(3)*a.voltage_kv)
        ratings = np.array([self.branch_ratings_ka[e.edge_id] if self.branch_ratings_ka is not None
                            else a.line_rating_ka for _, _, e in branches])
        parent = np.array([bi[u] for u, _, _ in branches], dtype=int)
        child = np.array([bi[v] for _, v, _ in branches], dtype=int)
        r = np.array([self.branch_ohms[e.edge_id][0]/zbase if self.branch_ohms is not None
                      else e.r_pu for _, _, e in branches])
        x = np.array([self.branch_ohms[e.edge_id][1]/zbase if self.branch_ohms is not None
                      else e.x_pu for _, _, e in branches])
        if np.any(r <= 0) or np.any(x < 0) or not np.isfinite(r+x).all():
            raise ValueError('Invalid total series impedance')
        # Variables are served fractions, squared voltages, sending-end P/Q,
        # squared branch currents and individual source P/Q, all on one base.
        sizes = (d, n, m, m, m, g, g)
        off = np.r_[0, np.cumsum(sizes)]
        sl = [slice(off[i], off[i+1]) for i in range(len(sizes))]
        ia, iv, ip, iq, il, igp, igq = sl
        nv = off[-1]
        demandp = np.array([b['p_kw']/base_kw for b in loads])
        demandq = np.array([b['q_kvar']/base_kw for b in loads])
        gpmax = np.array([s.max_p_kw/base_kw for s in supplies])
        gqmax = np.array([s.max_q_kvar/base_kw for s in supplies])
        gsmax = np.array([s.max_s_kva/base_kw for s in supplies])
        lower = np.r_[np.zeros(d), np.full(n, a.voltage_min_pu**2),
                      np.full(2*m, -100.), np.zeros(m+g), -gqmax]
        upper = np.r_[np.ones(d), np.full(n, a.voltage_max_pu**2),
                      np.full(2*m, 100.), (ratings/ibase)**2, gpmax, gqmax]
        # Linear balance at every bus, including the bounded island reference.
        linear = np.zeros((2*n+m+1, nv))
        for j, b in enumerate(loads):
            k = bi[b['bus_id']]
            linear[k, ia.start+j] = -demandp[j]
            linear[n+k, ia.start+j] = -demandq[j]
        for j, s in enumerate(supplies):
            k = bi[s.bus_id]
            linear[k, igp.start+j] = 1
            linear[n+k, igq.start+j] = 1
        for j, (u, v, _) in enumerate(branches):
            ui, vi = bi[u], bi[v]
            linear[ui, ip.start+j], linear[vi, ip.start+j] = -1, 1
            linear[n+ui, iq.start+j], linear[n+vi, iq.start+j] = -1, 1
            linear[vi, il.start+j], linear[n+vi, il.start+j] = -r[j], -x[j]
            row = 2*n+j
            linear[row, iv.start+vi], linear[row, iv.start+ui] = 1, -1
            linear[row, ip.start+j], linear[row, iq.start+j] = 2*r[j], 2*x[j]
            linear[row, il.start+j] = -(r[j]**2+x[j]**2)
        linear[-1, iv.start] = 1
        rhs = np.zeros(linear.shape[0])
        rhs[-1] = 1.
        if not a.independent_facility_switch:
            common_switch = []
            for j, block in enumerate(loads):
                if block['kind'] == 'facility':
                    other = next(k for k, b in enumerate(loads)
                                 if b['bus_id'] == block['bus_id'] and b['kind'] == 'ordinary')
                    row = np.zeros(nv)
                    row[ia.start+j], row[ia.start+other] = 1., -1.
                    common_switch.append(row)
            if common_switch:
                linear = np.vstack([linear, common_switch])
                rhs = np.r_[rhs, np.zeros(len(common_switch))]
        def eq(y):
            return np.r_[linear@y-rhs, y[iv][parent]*y[il]-y[ip]**2-y[iq]**2]
        def eq_jac(y):
            jac = np.zeros((m, nv))
            jj = np.arange(m)
            jac[jj, iv.start+parent] = y[il]
            jac[jj, il.start+jj] = y[iv][parent]
            jac[jj, ip.start+jj] = -2*y[ip]
            jac[jj, iq.start+jj] = -2*y[iq]
            return np.vstack([linear, jac])
        def ineq(y):
            return gsmax**2-y[igp]**2-y[igq]**2
        def ineq_jac(y):
            jac = np.zeros((g, nv))
            jj = np.arange(g)
            jac[jj, igp.start+jj] = -2*y[igp]
            jac[jj, igq.start+jj] = -2*y[igq]
            return jac
        objective = np.zeros(nv)
        objective[ia] = -demandp*np.array([b['weight'] for b in loads])
        objective[il] = .001*r
        # A small explicit operating penalty prevents cost-free circulation.
        objective[igp] = [.0001 if s.unit_id == 'upstream' else .001 for s in supplies]
        # Zero-load AC is feasible but singular for a direct SQP start. A convex
        # branch-flow relaxation supplies a useful initial dispatch. Its lower
        # objective bound is diagnostic; it is not accepted as AC without checks.
        import clarabel
        # Clarabel uses A*y+s=b with s in the product cone. Direct sparse
        # assembly avoids importing unrelated optional solver binaries.
        matrices = [linear, -np.eye(nv), np.eye(nv)]
        vectors = [rhs, -lower, upper]
        cones = [clarabel.ZeroConeT(len(rhs)), clarabel.NonnegativeConeT(nv), clarabel.NonnegativeConeT(nv)]
        for j in range(m):
            cone = np.zeros((4, nv))
            cone[0, iv.start+parent[j]] = cone[0, il.start+j] = 1.
            cone[1, ip.start+j], cone[2, iq.start+j] = 2., 2.
            cone[3, iv.start+parent[j]], cone[3, il.start+j] = 1., -1.
            matrices.append(-cone); vectors.append(np.zeros(4)); cones.append(clarabel.SecondOrderConeT(4))
        for j in range(g):
            cone = np.zeros((3, nv))
            cone[1, igp.start+j], cone[2, igq.start+j] = -1., -1.
            matrices.append(cone); vectors.append(np.array([gsmax[j], 0., 0.])); cones.append(clarabel.SecondOrderConeT(3))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-11
        settings.max_iter = 300
        problem = clarabel.DefaultSolver(sparse.csc_matrix((nv, nv)), objective,
            sparse.csc_matrix(np.vstack(matrices)), np.concatenate(vectors), cones, settings).solve()
        if str(problem.status) not in ('Solved', 'AlmostSolved'):
            raise RuntimeError(f'Conic initialization failed at {root}: {problem.status}')
        initial = np.clip(problem.x, lower, upper)
        result = minimize(lambda y: objective@y, initial, jac=lambda y: objective,
            method='SLSQP', bounds=list(zip(lower, upper)),
            constraints=[dict(type='eq', fun=eq, jac=eq_jac),
                         dict(type='ineq', fun=ineq, jac=ineq_jac)],
            options=dict(ftol=1e-11, maxiter=250, disp=False))
        solver_attempts = [dict(status=int(result.status), iterations=int(result.nit))]
        attempt_start = initial
        if result.status == 9:
            # Continue an iteration-limited solve at the same precision. This
            # branch cannot change a previously accepted first-attempt result.
            # The same recovery applies to every planning algorithm.
            attempt_start = result.x.copy()
            result = minimize(lambda y: objective@y, attempt_start, jac=lambda y: objective,
                method='SLSQP', bounds=list(zip(lower, upper)),
                constraints=[dict(type='eq', fun=eq, jac=eq_jac),
                             dict(type='ineq', fun=ineq, jac=ineq_jac)],
                options=dict(ftol=1e-11, maxiter=1000, disp=False))
            solver_attempts.append(dict(status=int(result.status), iterations=int(result.nit)))
        if result.status == 4:
            # A failed SQP subproblem does not prove that the original AC
            # constraints are infeasible. Retry from a contracted starting
            # dispatch, with voltage contracted towards the zero-load state.
            # This branch never modifies a previously accepted solution.
            for fraction in (.5, .1, .01):
                restart = attempt_start*fraction
                restart[iv] = 1.+fraction*(attempt_start[iv]-1.)
                candidate = minimize(lambda y: objective@y, restart, jac=lambda y: objective,
                    method='SLSQP', bounds=list(zip(lower, upper)),
                    constraints=[dict(type='eq', fun=eq, jac=eq_jac),
                                 dict(type='ineq', fun=ineq, jac=ineq_jac)],
                    options=dict(ftol=1e-11, maxiter=1000, disp=False))
                solver_attempts.append(dict(status=int(candidate.status), iterations=int(candidate.nit),
                    restart_fraction=fraction))
                trial = candidate.x
                if (candidate.success and np.max(np.abs(eq(trial))) <= 2e-8
                        and max(0., np.max(lower-trial), np.max(trial-upper), -np.min(ineq(trial))) <= 2e-8):
                    result = candidate
                    break
        y = result.x
        # Near-zero branch flow can have a small squared-current residual but
        # an appreciable current error. Refine the same constraints, not their
        # acceptance limits, before the independent bus-injection AC check.
        implied_current = np.sqrt((y[ip]**2+y[iq]**2)/y[iv][parent])*ibase
        current_residual = np.max(np.abs(np.sqrt(np.maximum(0., y[il]))*ibase-implied_current), initial=0.)
        if result.success and current_residual > 1e-6:
            result = minimize(lambda z: objective@z, y, jac=lambda z: objective,
                method='SLSQP', bounds=list(zip(lower, upper)),
                constraints=[dict(type='eq', fun=eq, jac=eq_jac),
                             dict(type='ineq', fun=ineq, jac=ineq_jac)],
                options=dict(ftol=1e-14, maxiter=250, disp=False))
            solver_attempts.append(dict(status=int(result.status), iterations=int(result.nit),
                reason='near_zero_current_refinement', prior_current_residual_ka=float(current_residual)))
            y = result.x
        residual = float(np.max(np.abs(eq(y))))
        violation = float(max(0., np.max(lower-y), np.max(y-upper), -np.min(ineq(y))))
        if not result.success or residual > 2e-8 or violation > 2e-8:
            raise RuntimeError(f'Island AC dispatch failed at {root}: {result.message}; '
                               f'equality={residual}, violation={violation}')
        load_rows = [dict(b, fraction=float(y[ia][j]), served_kw=float(y[ia][j]*b['p_kw']),
                         served_kvar=float(y[ia][j]*b['q_kvar'])) for j, b in enumerate(loads)]
        supply_rows = [dict(unit_id=s.unit_id, bus_id=s.bus_id, p_kw=float(y[igp][j]*base_kw),
                           q_kvar=float(y[igq][j]*base_kw)) for j, s in enumerate(supplies)]
        line_rows = [dict(edge_id=e.edge_id, from_bus=u, to_bus=v,
            p_from_kw=float(y[ip][j]*base_kw), q_from_kvar=float(y[iq][j]*base_kw),
            current_ka=float(math.sqrt(max(0., y[il][j]))*ibase),
            rating_ka=float(ratings[j]),
            loading_percent=float(math.sqrt(max(0., y[il][j]))*ibase/ratings[j]*100),
            loss_kw=float(r[j]*y[il][j]*base_kw)) for j, (u, v, e) in enumerate(branches)]
        voltages = {b: float(math.sqrt(y[iv][bi[b]])) for b in buses}
        check = self._independent_pf(buses, branches, r*zbase, x*zbase,
                                    load_rows, supply_rows, voltages, line_rows) if a.verify_with_pandapower else {}
        return dict(root=root, buses=buses, status='ac_feasible_local_solution',
            served_kw=sum(b['served_kw'] for b in load_rows), loss_kw=sum(b['loss_kw'] for b in line_rows),
            equality_residual_pu=residual, constraint_violation_pu=violation,
            iterations=int(result.nit), solver_attempts=solver_attempts, voltages=voltages,
            conic_status=str(problem.status), conic_objective_bound=float(problem.obj_val),
            local_objective=float(objective@y),
            independent_ac_verified=bool(check), independent_ac=check,
            loads=load_rows, supplies=supply_rows, lines=line_rows)

    def _independent_pf(self, buses, branches, r, x, loads, supplies, voltages, lines):
        """Rebuild in pandapower and solve bus-injection AC equations at fixed dispatch.

        Its slack is used only for verification. Its computed P/Q must match the
        bounded generator solved above, so it cannot add unaccounted island energy.
        """
        import pandapower as pp
        a = self.assumptions
        net = pp.create_empty_network(sn_mva=a.base_mva)
        indices = {b: pp.create_bus(net, vn_kv=a.voltage_kv) for b in buses}
        pp.create_ext_grid(net, indices[supplies[0]['bus_id']], vm_pu=1.)
        for j, (u, v, e) in enumerate(branches):
            pp.create_line_from_parameters(net, indices[u], indices[v], length_km=1.,
                r_ohm_per_km=r[j], x_ohm_per_km=x[j], c_nf_per_km=0.,
                max_i_ka=self.branch_ratings_ka[e.edge_id] if self.branch_ratings_ka is not None
                else a.line_rating_ka, name=e.edge_id)
        for b in loads:
            pp.create_load(net, indices[b['bus_id']], p_mw=b['served_kw']/1000,
                           q_mvar=b['served_kvar']/1000)
        for s in supplies[1:]:
            pp.create_sgen(net, indices[s['bus_id']], p_mw=s['p_kw']/1000, q_mvar=s['q_kvar']/1000)
        pp.runpp(net, calculate_voltage_angles=True, tolerance_mva=1e-10, numba=False)
        voltage_error = max(abs(net.res_bus.vm_pu.at[indices[b]]-voltages[b]) for b in buses)
        source_p_error = abs(net.res_ext_grid.p_mw.sum()*1000-supplies[0]['p_kw'])
        source_q_error = abs(net.res_ext_grid.q_mvar.sum()*1000-supplies[0]['q_kvar'])
        current_error = max((abs(net.res_line.i_ka.iloc[j]-row['current_ka']) for j, row in enumerate(lines)), default=0.)
        if max(source_p_error, source_q_error) > .002 or voltage_error > 2e-7 or current_error > 2e-6:
            raise RuntimeError(f'Independent AC reconstruction disagrees with bounded dispatch: '
                               f'P={source_p_error} kW, Q={source_q_error} kvar, '
                               f'V={voltage_error} pu, I={current_error} kA')
        return dict(voltage_error_pu=float(voltage_error), slack_p_error_kw=float(source_p_error),
                    slack_q_error_kvar=float(source_q_error), current_error_ka=float(current_error))
