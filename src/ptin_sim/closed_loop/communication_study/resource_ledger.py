"""Individual stationary/vehicle batteries with explicit energy conservation.

Availability does not create or recharge a battery. Transit excludes injection.
Travel/charging powers must be explicit caller inputs; no implicit free-energy
fallback or clamp hides an infeasible dispatch. Each atomic change is recorded.
"""
from dataclasses import asdict, dataclass
import math


@dataclass
class BatteryUnit:
    unit_id: str
    kind: str
    capacity_kwh: float
    energy_kwh: float
    max_power_kw: float
    location: str
    available: bool = True
    connected_bus: str | None = None
    destination: str | None = None
    arrival_s: float | None = None
    discharged_kwh: float = 0.
    charged_kwh: float = 0.
    traction_kwh: float = 0.


class ResourceLedger:
    def __init__(self, units, *, discharge_efficiency=1., charge_efficiency=1.):
        values=(discharge_efficiency,charge_efficiency)
        if not all(math.isfinite(v) and 0 < v <= 1 for v in values):
            raise ValueError("Efficiencies must be in (0,1]")
        units=list(units)
        if not units or len({u.unit_id for u in units}) != len(units):
            raise ValueError("Nonempty unique battery IDs required")
        # Own independent state, not aliases to caller-owned dataclasses.
        self.units={u.unit_id:BatteryUnit(**asdict(u)) for u in units}
        for unit in self.units.values():
            if (not unit.unit_id or not unit.location
                    or not all(math.isfinite(v) for v in (unit.capacity_kwh,unit.energy_kwh,unit.max_power_kw))
                    or not 0 <= unit.energy_kwh <= unit.capacity_kwh or unit.capacity_kwh <= 0 or unit.max_power_kw < 0):
                raise ValueError("Invalid battery initial state")
            if unit.destination is not None or unit.arrival_s is not None:
                raise ValueError("Initial transit needs an explicit departure event")
            if not unit.available and unit.connected_bus is not None:
                raise ValueError("Unavailable battery cannot be initially connected")
        self.initial_energy={key:u.energy_kwh for key,u in self.units.items()}
        self.discharge_efficiency,self.charge_efficiency=values
        self.time_s=0.
        self.events=[]

    def _record(self,kind,before,**extra):
        self.events.append(dict(time_s=self.time_s,event=kind,before=before,
                                after=self.snapshot(),**extra))

    def snapshot(self):
        return {key:asdict(unit) for key,unit in self.units.items()}

    def set_available(self,unit_id,available):
        unit=self.units[unit_id]
        if unit.destination is not None:
            raise ValueError("Availability change during transit requires explicit trip handling")
        before=self.snapshot()
        unit.available=bool(available)
        if not available:
            unit.connected_bus=None
        self._record("availability_changed",before,unit_id=unit_id)

    def connect(self,unit_id,bus_id):
        unit=self.units[unit_id]
        if not unit.available or unit.destination is not None or unit.location != bus_id:
            raise ValueError("Battery must be present and available at the connection bus")
        before=self.snapshot();unit.connected_bus=bus_id
        self._record("connected",before,unit_id=unit_id)

    def depart(self,unit_id,destination,arrival_s,*,traction_kwh):
        unit=self.units[unit_id]
        if (not unit.available or unit.destination is not None or not destination
                or not math.isfinite(arrival_s) or arrival_s <= self.time_s
                or not math.isfinite(traction_kwh) or not 0 <= traction_kwh <= unit.energy_kwh):
            raise ValueError("Invalid trip or insufficient traction energy")
        before=self.snapshot()
        unit.connected_bus=None
        unit.destination,unit.arrival_s=destination,arrival_s
        unit.energy_kwh-=traction_kwh;unit.traction_kwh+=traction_kwh
        self._record("departed",before,unit_id=unit_id,traction_kwh=traction_kwh)

    def advance(self,until_s,*,dispatch_kw,charging_kw=None):
        if not math.isfinite(until_s) or until_s <= self.time_s:
            raise ValueError("Strictly increasing finite resource time required")
        charging_kw=charging_kw or {}
        if not (set(dispatch_kw)|set(charging_kw)) <= self.units.keys():
            raise ValueError("Unknown battery dispatch")
        arrivals=[u.arrival_s for u in self.units.values() if u.arrival_s is not None and self.time_s < u.arrival_s < until_s]
        if arrivals:
            raise ValueError("Split the power interval at the next arrival event")
        duration_h=(until_s-self.time_s)/3600.
        changes={}
        for key,unit in self.units.items():
            output=float(dispatch_kw.get(key,0.));charge=float(charging_kw.get(key,0.))
            if (not all(math.isfinite(v) and v >= 0 for v in (output,charge))
                    or output > unit.max_power_kw+1e-8 or charge > unit.max_power_kw+1e-8
                    or (output > 0 and charge > 0)):
                raise ValueError("Invalid battery power or simultaneous charge/discharge")
            if (output or charge) and (not unit.available or unit.connected_bus is None or unit.destination is not None):
                raise ValueError("Unavailable, disconnected or travelling battery cannot inject/charge")
            used=output*duration_h/self.discharge_efficiency
            added=charge*duration_h*self.charge_efficiency
            after=unit.energy_kwh-used+added
            if after < -1e-8 or after > unit.capacity_kwh+1e-8:
                raise ValueError("Battery energy bound violated")
            changes[key]=(max(0.,min(unit.capacity_kwh,after)),used,added)
        # Validate every unit before applying any update.
        before=self.snapshot();start=self.time_s
        for key,(energy,used,added) in changes.items():
            unit=self.units[key];unit.energy_kwh=energy
            unit.discharged_kwh+=used;unit.charged_kwh+=added
        self.time_s=until_s
        self._record("power_interval",before,start_s=start,dispatch_kw=dict(dispatch_kw),charging_kw=dict(charging_kw))
        for unit in self.units.values():
            if unit.arrival_s is not None and unit.arrival_s <= until_s:
                before=self.snapshot();unit.location=unit.destination
                unit.destination=unit.arrival_s=None
                self._record("arrived_not_yet_connected",before,unit_id=unit.unit_id)

    def conservation_residuals_kwh(self):
        return {key:unit.energy_kwh-(self.initial_energy[key]+unit.charged_kwh-unit.discharged_kwh-unit.traction_kwh)
                for key,unit in self.units.items()}
