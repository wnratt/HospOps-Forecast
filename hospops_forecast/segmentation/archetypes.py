from __future__ import annotations
from enum import Enum

class Archetype(str, Enum):
    SoloBusiness = "SoloBusiness"
    LeisureCouple = "LeisureCouple"
    FamilyWithKids = "FamilyWithKids"
    TourGroup = "TourGroup"
    Other = "Other"
