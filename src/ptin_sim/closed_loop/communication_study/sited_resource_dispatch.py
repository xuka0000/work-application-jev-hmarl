"""Couple frozen AC limits to individual batteries over one physical interval.

The event scheduler must split at arrivals, departures, faults and controls.
This kernel does not infer device locations, inverter ratings or free energy.
"""
from dataclasses import dataclass
import math

from ..publication_island_power import SitedSupply
from .electrical_profile import dispatch_from_profile, electrical_signature, profile_digest
from .grouped_dispatch import GroupedSitedDispatch


@dataclass(frozen=True)
class Inverter:
    max_q_kvar: float
    max_s_kva: float
    grid_forming: bool


class SitedResourceDispatch:
    def __init__(self, scenario, profile, ledger, inverters):
        if profile['sha256'] != profile_digest(profile) or profile['physical_signature'] != electrical_signature(scenario):
            raise ValueError('Frozen electrical profile mismatch')
        if set(inverters) != set(ledger.units):
            raise ValueError('Every battery needs an explicit inverter specification')
        for spec in inverters.values():
            if not all(math.isfinite(x) and x >= 0 for x in (spec.max_q_kvar, spec.max_s_kva)):
                raise ValueError('Invalid inverter rating')
        self.scenario, self.profile, self.ledger = scenario, profile, ledger
        self.inverters = dict(inverters)
        self.backend = GroupedSitedDispatch(dispatch_from_profile(profile))
        self.intervals = []
        self.pending = None

    def plan(self, hold_until_s, *, closed_pdn_edges):
        start = self.ledger.time_s
        if not math.isfinite(hold_until_s) or hold_until_s <= start:
            raise ValueError('Positive finite physical interval required')
        # Dispatch depends on a declared hold horizon, never on the next hidden
        # fault time. Events can interrupt this setpoint without anticipating them.
        hours = (hold_until_s-start)/3600
        supplies = []
        for key, unit in self.ledger.units.items():
            if unit.available and unit.destination is None and unit.connected_bus is not None:
                spec = self.inverters[key]
                cap = min(unit.max_power_kw, unit.energy_kwh*self.ledger.discharge_efficiency/hours)
                supplies.append(SitedSupply(key, unit.connected_bus, cap, spec.max_q_kvar,
                                             spec.max_s_kva, spec.grid_forming))
        result = self.backend.run(self.scenario, closed_pdn_edges=closed_pdn_edges, supplies=supplies)
        dispatch = {r['unit_id']: max(0., r['p_kw']) for r in result['supplies'] if r['unit_id'] != 'upstream'}
        self.pending = dict(hold_until_s=hold_until_s, result=result, dispatch=dispatch,
                            closed_pdn_edges=sorted(closed_pdn_edges))
        return result

    def commit_until(self, until_s):
        if self.pending is None or until_s > self.pending['hold_until_s']:
            raise ValueError('A valid held dispatch is required')
        start = self.ledger.time_s
        before = self.ledger.snapshot()
        result, dispatch = self.pending['result'], self.pending['dispatch']
        self.ledger.advance(until_s, dispatch_kw=dispatch)
        audit = dict(start_s=start, end_s=until_s, profile_sha256=self.profile['sha256'],
                     closed_pdn_edges=self.pending['closed_pdn_edges'], before=before,
                     after=self.ledger.snapshot(), dispatch_kw=dispatch, electrical=result)
        self.intervals.append(audit)
        return audit

    def advance(self, until_s, *, closed_pdn_edges):
        if any(u.arrival_s is not None and self.ledger.time_s < u.arrival_s < until_s
               for u in self.ledger.units.values()):
            raise ValueError('Split the physical interval at every arrival')
        self.plan(until_s, closed_pdn_edges=closed_pdn_edges)
        return self.commit_until(until_s)
