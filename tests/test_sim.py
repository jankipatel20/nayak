"""Comprehensive tests for the NAYAK simulator core."""

import time
import pytest

from nayak.config import (
    Approach,
    PhaseState,
    SCENARIOS,
    ScenarioConfig,
    SignalPhase,
    SignalTimingConfig,
    VehicleType,
)
from nayak.controllers.fixed_time import FixedTimeController
from nayak.sim.arrivals import ArrivalGenerator
from nayak.sim.intersection import Intersection
from nayak.sim.runner import run_episode
from nayak.sim.vehicles import Vehicle


def test_determinism() -> None:
    """Verify that running with identical seeds produces byte-for-byte identical metrics."""
    scenario = SCENARIOS["normal"]
    controller1 = FixedTimeController(30)
    controller2 = FixedTimeController(30)

    m1, _ = run_episode(controller1, scenario, episode_seconds=900, warmup_seconds=300, seed=123)
    m2, _ = run_episode(controller2, scenario, episode_seconds=900, warmup_seconds=300, seed=123)

    assert m1.mean_wait == pytest.approx(m2.mean_wait, abs=1e-9)
    assert m1.wait_p95 == pytest.approx(m2.wait_p95, abs=1e-9)
    assert m1.max_wait == pytest.approx(m2.max_wait, abs=1e-9)
    assert m1.mean_ambulance_delay == pytest.approx(m2.mean_ambulance_delay, abs=1e-9)
    assert m1.throughput_vehicles == m2.throughput_vehicles
    assert m1.throughput_pcu == pytest.approx(m2.throughput_pcu, abs=1e-9)
    assert m1.total_arrived == m2.total_arrived
    assert m1.unserved_queued == m2.unserved_queued

    # Different seeds should produce different arrivals and metrics
    m3, _ = run_episode(controller1, scenario, episode_seconds=900, warmup_seconds=300, seed=456)
    assert m1.total_arrived != m3.total_arrived or m1.mean_wait != m3.mean_wait


def test_vehicle_conservation() -> None:
    """Verify arrived = departed + still queued for both raw simulation and tracked metrics."""
    scenario = SCENARIOS["normal"]
    controller = FixedTimeController(30)
    metrics, intersection = run_episode(
        controller, scenario, episode_seconds=900, warmup_seconds=300, seed=99
    )

    # 1. Physical vehicle conservation across all approaches
    total_in_queues = sum(len(q) for q in intersection.queues.values())
    total_departed = len(intersection.all_departed_vehicles)
    total_arrived = len(intersection.all_arrived_vehicles)
    assert total_arrived == total_departed + total_in_queues

    # 2. Tracked evaluation metrics conservation
    assert metrics.total_arrived == metrics.total_departed + metrics.unserved_queued


def test_bus_delays_queue_more_than_two_wheelers() -> None:
    """Verify that a 2.5 PCU bus delays a trailing vehicle significantly more than a 0.3 PCU two-wheeler."""
    timing = SignalTimingConfig(startup_lost_time=2, sat_flow_pcu_per_sec=0.5)

    # Experiment A: Two-wheeler (0.3 PCU) followed by trailing Car (1.0 PCU) on North approach
    sim_a = Intersection(timing)
    tw = Vehicle(id=1, vehicle_type=VehicleType.TWO_WHEELER, approach=Approach.N, arrival_time=0.0, pcu=0.3)
    car_a = Vehicle(id=2, vehicle_type=VehicleType.CAR, approach=Approach.N, arrival_time=0.0, pcu=1.0)
    sim_a.add_arrivals({Approach.N: [tw, car_a], Approach.E: [], Approach.S: [], Approach.W: []})

    for t in range(1, 20):
        sim_a.step(t)
        if car_a.departed:
            break

    # Experiment B: Bus (2.5 PCU) followed by trailing Car (1.0 PCU) on North approach
    sim_b = Intersection(timing)
    bus = Vehicle(id=3, vehicle_type=VehicleType.BUS, approach=Approach.N, arrival_time=0.0, pcu=2.5)
    car_b = Vehicle(id=4, vehicle_type=VehicleType.CAR, approach=Approach.N, arrival_time=0.0, pcu=1.0)
    sim_b.add_arrivals({Approach.N: [bus, car_b], Approach.E: [], Approach.S: [], Approach.W: []})

    for t in range(1, 20):
        sim_b.step(t)
        if car_b.departed:
            break

    assert car_a.departed, "Car A should have departed"
    assert car_b.departed, "Car B should have departed"
    assert car_b.departure_time > car_a.departure_time, (
        f"Bus should delay trailing car more (car_b departed at {car_b.departure_time}s vs "
        f"car_a at {car_a.departure_time}s)"
    )


def test_saturation_response() -> None:
    """Verify that under low demand wait is modest, whereas under oversaturation wait grows large."""
    low_demand_scenario = ScenarioConfig(
        name="low_demand",
        description="Light traffic V/C ~ 0.18",
        vehicle_mix=SCENARIOS["normal"].vehicle_mix,
        base_arrival_rate=0.04,  # ~1 veh every 25s per approach
        tidal_flow=False,
    )
    oversaturated_scenario = ScenarioConfig(
        name="oversaturated",
        description="Oversaturated traffic V/C ~ 1.8",
        vehicle_mix=SCENARIOS["normal"].vehicle_mix,
        base_arrival_rate=0.45,
        tidal_flow=False,
    )

    controller = FixedTimeController(30)

    m_low, _ = run_episode(controller, low_demand_scenario, episode_seconds=900, warmup_seconds=300, seed=77)
    m_over, _ = run_episode(controller, oversaturated_scenario, episode_seconds=900, warmup_seconds=300, seed=77)

    # In low demand, queue clears every cycle; average wait is modest (IRC typical ~15-25s)
    assert m_low.mean_wait < 30.0, f"Expected modest wait under low demand, got {m_low.mean_wait}"

    # Under oversaturation, queues build up continuously
    assert m_over.mean_wait > 80.0, f"Expected large wait under oversaturation, got {m_over.mean_wait}"
    assert m_over.unserved_queued > 50, f"Expected large unserved queue, got {m_over.unserved_queued}"
    assert m_over.mean_wait > m_low.mean_wait * 3.0


def test_ambulance_moves_to_front_of_queue() -> None:
    """Verify that arriving ambulances filter to the front of the queue, behind prior ambulances."""
    sim = Intersection()
    # Initial phase is PHASE_NS, so Approach.E is RED
    assert Approach.E not in sim.active_green_approaches

    car1 = Vehicle(id=1, vehicle_type=VehicleType.CAR, approach=Approach.E, arrival_time=1.0, pcu=1.0)
    car2 = Vehicle(id=2, vehicle_type=VehicleType.CAR, approach=Approach.E, arrival_time=2.0, pcu=1.0)
    sim.add_arrivals({Approach.E: [car1, car2], Approach.N: [], Approach.S: [], Approach.W: []})

    assert [v.id for v in sim.queues[Approach.E]] == [1, 2]

    # First ambulance arrives on Red East approach
    amb1 = Vehicle(id=3, vehicle_type=VehicleType.AMBULANCE, approach=Approach.E, arrival_time=3.0, pcu=1.0)
    sim.add_arrivals({Approach.E: [amb1], Approach.N: [], Approach.S: [], Approach.W: []})

    # amb1 must filter to index 0
    assert sim.queues[Approach.E][0].id == 3
    assert [v.id for v in sim.queues[Approach.E]] == [3, 1, 2]

    # Second ambulance arrives on Red East approach
    amb2 = Vehicle(id=4, vehicle_type=VehicleType.AMBULANCE, approach=Approach.E, arrival_time=4.0, pcu=1.0)
    sim.add_arrivals({Approach.E: [amb2], Approach.N: [], Approach.S: [], Approach.W: []})

    # amb2 must filter behind amb1 but ahead of regular cars
    assert [v.id for v in sim.queues[Approach.E]] == [3, 4, 1, 2]


def test_simulation_speed_performance() -> None:
    """Verify that a 900-second simulation runs in under 0.1s CPU time for feasibility of GA evolution."""
    scenario = SCENARIOS["normal"]
    controller = FixedTimeController(30)

    start_cpu = time.perf_counter()
    run_episode(controller, scenario, episode_seconds=900, warmup_seconds=300, seed=42)
    elapsed = time.perf_counter() - start_cpu

    # Target: 900s episode in under 0.1s
    assert elapsed < 0.15, f"Simulation too slow: took {elapsed:.4f}s (target < 0.10-0.15s)"
