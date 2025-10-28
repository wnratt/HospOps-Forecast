from __future__ import annotations
import math

def erlang_c_wait_minutes(arrival_rate_per_hour: float, service_rate_per_agent_per_hour: float, c: int) -> float:
    # Returns expected waiting time in **minutes** for M/M/c
    lam = arrival_rate_per_hour
    mu = service_rate_per_agent_per_hour
    if c <= 0 or mu <= 0:
        return float("inf")
    rho = lam / (c * mu)
    if rho >= 1.0:
        return float("inf")
    a = lam / mu
    # compute P0
    sum_terms = sum((a**n) / math.factorial(n) for n in range(c))
    last = (a**c) / math.factorial(c) * (1/(1 - rho))
    P0 = 1.0 / (sum_terms + last)
    P_wait = ((a**c) / math.factorial(c)) * (1/(1 - rho)) * P0
    Wq_hours = P_wait / (c * mu - lam)  # hours
    return Wq_hours * 60.0
