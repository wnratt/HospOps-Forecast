# Data Schemas

## Reservations (input) — required cols
reservation_id, arrival_date, departure_date (exclusive), adults, children,
room_type, channel, company, nationality

## Flights (input)
flight_id, date, gate_time (HH:MM), pax_count, mix_business_share (0..1)

## Housekeeping actuals
date, hk_man_hours
