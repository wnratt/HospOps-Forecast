from __future__ import annotations
from typing import Dict, Optional
from pydantic import BaseModel, Field, PositiveFloat

class HKMultipliers(BaseModel):
    SoloBusiness: float = 1.0
    LeisureCouple: float = 1.15
    FamilyWithKids: float = 1.80
    TourGroup: float = 1.30
    Other: float = 1.10

class HousekeepingConfig(BaseModel):
    minutes_per_checkout: PositiveFloat = 45
    minutes_per_stayover: PositiveFloat = 20
    target_utilization: float = Field(0.85, ge=0.5, le=0.99)
    default_shift_hours: PositiveFloat = 8
    archetype_multipliers: HKMultipliers = HKMultipliers()

class MealItem(BaseModel):
    adult: PositiveFloat
    child: PositiveFloat

class MealConfig(BaseModel):
    base: Dict[str, MealItem]
    multipliers: Dict[str, Dict[str, float]]

class FNBMeals(BaseModel):
    breakfast: MealConfig
    lunch: MealConfig
    dinner: MealConfig

class DistributionConfig(BaseModel):
    distributions: Dict[str, Dict[str, float]]

class ReceptionConfig(BaseModel):
    transactions_per_agent_per_hour: PositiveFloat = 12
    utilization: float = Field(0.85, ge=0.5, le=0.99)
    distributions: Dict[str, Dict[str, float]]

class BreakfastConfig(BaseModel):
    covers_per_staff_per_hour: PositiveFloat = 25
    utilization: float = Field(0.85, ge=0.5, le=0.99)
    distributions: Dict[str, Dict[str, float]]

class ServiceLoadConfig(BaseModel):
    reception: ReceptionConfig
    breakfast: BreakfastConfig

class ServiceSLA(BaseModel):
    default_target_wait_min: PositiveFloat = 5.0

class DeptSpa(BaseModel):
    minutes_per_treatment: PositiveFloat = 50
    treatments_per_guest_day: Dict[str, float]
    utilization: float = 0.8

class DeptConcierge(BaseModel):
    requests_per_guest_day: Dict[str, float]
    minutes_per_request: PositiveFloat = 8
    utilization: float = 0.85

class DeptValet(BaseModel):
    cars_per_room_day: Dict[str, float]
    minutes_per_transaction: PositiveFloat = 6
    utilization: float = 0.85

class DeptEngineering(BaseModel):
    tickets_per_room_day: float = 0.05
    minutes_per_ticket: PositiveFloat = 20
    utilization: float = 0.8

class Departments(BaseModel):
    spa: DeptSpa
    concierge: DeptConcierge
    valet: DeptValet
    engineering: DeptEngineering

class AirlineArea(BaseModel):
    pax_per_agent_per_hour: PositiveFloat
    sla_target_wait_min: PositiveFloat
    distributions: Dict[str, float]

class AirlineConfig(BaseModel):
    boarding: AirlineArea
    gate: AirlineArea
    lounge: AirlineArea

class AppConfig(BaseModel):
    housekeeping: HousekeepingConfig
    fnb_meals: FNBMeals
    service_load: ServiceLoadConfig
    service_sla: ServiceSLA
    departments: Departments
    airline: AirlineConfig
