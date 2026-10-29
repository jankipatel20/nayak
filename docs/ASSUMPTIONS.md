# NAYAK: Design Assumptions & Justifications

All parameters and behavioral assumptions in Nayak are documented here with justifications.
Note: Results show RELATIVE improvement under a simplified microscopic simulator, not absolute real-world traffic figures.

| Parameter / Assumption | Value / Choice | One-Line Justification |
| :--- | :--- | :--- |
| **Simulation Time Step** | $\Delta t = 1.0\text{ s}$ | Integer-second discrete time provides exact reproducibility, high simulation speed, and suffices for macro signal decisions. |
| **Phase Layout** | Dual-phase (PHASE_NS: N+S; PHASE_EW: E+W) | Straight-only movements represent standard canonical intersection benchmarking without turning phase confounding. |
| **Switch Lost Time** | 3 s yellow + 1 s all-red (4 s total) | Matches standard Indian Road Congress (IRC) signal clearance intervals for urban speeds. |
| **Start-up Lost Time** | First 2 s of green (0 discharge) | Models driver reaction delay and acceleration inertia before queue reaches saturation headway. |
| **Saturation Flow** | 0.5 PCU / sec / green approach (1800 PCU/hr) | Standard single-lane saturated throughput in Indian urban intersection guidelines. |
| **Fractional Capacity Budget** | Add 0.5 PCU/s; head departs when budget $\ge$ PCU | Directly models physical delay caused by heavy vehicles (buses) vs fast clearance of small vehicles (two-wheelers). |
| **FixedTime Cycle** | 30 s green NS + 4 s lost + 30 s green EW + 4 s lost (68 s cycle) | Balanced 30/30 green split provides standard IRC cycle time around 60–70s with proper safety clearance. |
| **Queue Discipline** | FIFO with ambulance priority insertion | Vehicles queue in order of arrival; ambulances filter to the head of red queues (behind earlier ambulances) via siren/lane-filtering. |
| **Ambulance Green Arrival** | Joins front behind existing ambulances or departs | Emergency vehicles bypass normal queues using emergency shoulder/filtering in Indian mixed traffic. |
| **Arrival Process** | Bernoulli trial per approach per second with rate $\lambda$ | Discrete-time Bernoulli process is the canonical exact 1-second discretization of Poisson arrivals for $\lambda \ll 1$. |
| **Ambulance Scheduling** | 3 to 5 ambulances per episode at uniform random $t \in [300, T_{\text{end}}-30]$ | Guarantees non-empty ambulance objective evaluated across random approaches without biasing initial warm-up. |
| **Warm-up Period** | 300 s excluded from metrics | Allows queues to populate from empty state to steady-state stochastic equilibrium before recording statistics. |
| **Unserved Queue Penalization** | Waiting-so-far $= T_{\text{end}} - t_{\text{arrival}}$ counted for queued vehicles | Prevents unfair/pathological controllers that starve one approach to artificially deflate served wait times. |
| **Normal Scenario V/C** | Target $\approx 0.60 - 0.70$ V/C ratio | Represents stable daytime urban traffic operating below breakdown capacity. |
| **Rush Hour Scenario V/C** | Target $\approx 0.90 - 0.95$ V/C with $1.5:1$ tidal flow | Represents heavy peak-period commuter traffic with directional shift (NS first half, EW second half). |
| **Two-Wheeler Heavy Scenario** | Target $\approx 0.65 - 0.72$ V/C with 60% 2-wheelers | Captures high vehicle density with smaller spatial footprint typical of tier-2 Indian cities. |
| **Fuzzy Decision Interval** | Consulted at $t_{\text{green}} \ge 8\text{ s}$ and every 4 s | Balances controller responsiveness with preventing excessive switching and actuation churn. |
| **Extension Threshold** | Switch if $\text{extension} < 2.0\text{ s}$ | Extensions under 2 s are shorter than transition lost time (4 s) and are economically unprofitable to hold. |
| **LongestQueue Margin** | Actuate switch when $q_{\text{red}} > q_{\text{green}} + 2.0\text{ PCU}$ | Hysteresis band prevents rapid chattering oscillation between nearly balanced queues. |
| **Defuzzification Resolution** | Discretized universe $N = 151$ points over $[0, 15]\text{ s}$ | Yields $0.10\text{ s}$ numerical centroid precision while completing inference in $<0.05\text{ ms}$. |
| **Pruned Rule Threshold** | Weight $< 0.10$ | Identifies rules rendered inert by genetic algorithm evolution during explainability analysis. |
