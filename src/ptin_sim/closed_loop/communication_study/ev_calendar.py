"""Explicit EV identities for a frozen aggregate availability series.

This deterministic nested-identity construction is a study assumption, not
measured driver trips. Vehicle SOC and site assignment must be explicit.
Returning identities keep their own residual energy in ResourceLedger.
"""
import math
from .resource_ledger import BatteryUnit


def build_ev_calendar(fleet, *, site_by_unit, initial_soc_by_unit, horizon_s):
    # FleetSpec.ev_count is already vehicle_population * ev_ratio.
    # Applying ev_ratio again would halve the source 390 eligible vehicles.
    nominal=fleet.ev_count*fleet.v2g_willingness
    if not math.isfinite(nominal) or nominal<0 or not math.isfinite(horizon_s) or horizon_s<=0:
        raise ValueError('Invalid EV population or horizon')
    points=sorted((float(p.time_min)*60,float(p.availability_factor)) for p in fleet.ev_availability_profile)
    if not points:points=[(0.,1.)]
    if points[0][0]!=0 or len({t for t,f in points})!=len(points):
        raise ValueError('Unique EV availability times beginning at zero required')
    if any(not math.isfinite(t) or t<0 or not math.isfinite(f) or not 0<=f<=1 for t,f in points):
        raise ValueError('Invalid EV availability factor')
    identities=[f'EV_{i+1:04d}' for i in range(math.floor(nominal))]
    if set(site_by_unit)!=set(identities) or set(initial_soc_by_unit)!=set(identities):
        raise ValueError('Explicit site and initial SOC required for each eligible EV')
    initial_count=math.floor(nominal*points[0][1])
    units=[]
    for i,key in enumerate(identities):
        soc=initial_soc_by_unit[key]
        if not math.isfinite(soc) or not 0<=soc<=1 or not site_by_unit[key]:
            raise ValueError('Invalid initial EV SOC or site')
        units.append(BatteryUnit(key,'ev',fleet.ev_unit_energy_kwh,
            fleet.ev_unit_energy_kwh*soc,fleet.ev_unit_power_kw,site_by_unit[key],available=i<initial_count))
    prior=set(identities[:initial_count]);events=[];counts=[]
    for at,factor in points:
        if at>=horizon_s:continue
        active=set(identities[:math.floor(nominal*factor)])
        counts.append(dict(time_s=at,factor=factor,available_count=len(active)))
        for key in sorted(prior^active):
            events.append(dict(time_s=at,unit_id=key,available=key in active))
        prior=active
    return units,events,dict(assumption='nested deterministic identities; explicit static sites and initial SOC',
        nominal_eligible=nominal,identity_count=len(identities),counts=counts,
        no_automatic_connection=True,no_return_refill=True)
