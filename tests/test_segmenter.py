import pandas as pd
from hospops_forecast.segmentation import Segmenter, Archetype

def test_segmenter_basic_rules():
    df = pd.DataFrame([
        {"reservation_id":"R1","arrival_date":"2025-01-06","departure_date":"2025-01-08","adults":1,"children":0,"room_type":"Standard","channel":"Corp","company":"ACME"},
        {"reservation_id":"R2","arrival_date":"2025-01-11","departure_date":"2025-01-13","adults":2,"children":0,"room_type":"Deluxe","channel":"OTA","company":""},
        {"reservation_id":"R3","arrival_date":"2025-01-11","departure_date":"2025-01-14","adults":2,"children":2,"room_type":"Suite","channel":"OTA","company":""},
    ])
    seg = Segmenter()
    out = seg.enrich(df)
    a = out.set_index("reservation_id")["archetype"].to_dict()
    assert a["R1"] == Archetype.SoloBusiness.value
    assert a["R2"] == Archetype.LeisureCouple.value
    assert a["R3"] == Archetype.FamilyWithKids.value
