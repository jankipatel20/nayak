# NAYAK Hand-Designed Fuzzy Rule Base & Initial Membership Report

## 1. Membership Function Breakpoints (Natural Units)

| Variable | Term | Type | Breakpoints (Natural Units) |
| :--- | :--- | :--- | :--- |
| `q_green` | **LOW** | TrapezoidalMF | `(0.00, 0.00, 5.00, 15.00)` |
| `q_green` | **MEDIUM** | TriangularMF | `(10.00, 20.00, 30.00)` |
| `q_green` | **HIGH** | TrapezoidalMF | `(25.00, 35.00, 50.00, 50.00)` |
| `q_red` | **LOW** | TrapezoidalMF | `(0.00, 0.00, 5.00, 15.00)` |
| `q_red` | **MEDIUM** | TriangularMF | `(10.00, 20.00, 30.00)` |
| `q_red` | **HIGH** | TrapezoidalMF | `(25.00, 35.00, 50.00, 50.00)` |
| `bus_share_red` | **LOW** | TrapezoidalMF | `(0.00, 0.00, 0.10, 0.25)` |
| `bus_share_red` | **MEDIUM** | TriangularMF | `(0.15, 0.35, 0.55)` |
| `bus_share_red` | **HIGH** | TrapezoidalMF | `(0.45, 0.65, 1.00, 1.00)` |
| `two_wheeler_share_green` | **LOW** | TrapezoidalMF | `(0.00, 0.00, 0.20, 0.35)` |
| `two_wheeler_share_green` | **MEDIUM** | TriangularMF | `(0.25, 0.50, 0.70)` |
| `two_wheeler_share_green` | **HIGH** | TrapezoidalMF | `(0.60, 0.75, 1.00, 1.00)` |
| `green_elapsed` | **LOW** | TrapezoidalMF | `(0.00, 0.00, 12.00, 22.00)` |
| `green_elapsed` | **MEDIUM** | TriangularMF | `(18.00, 30.00, 42.00)` |
| `green_elapsed` | **HIGH** | TrapezoidalMF | `(38.00, 48.00, 60.00, 60.00)` |
| `ambulance_red` | `0` / `1` | Binary Crisp | `ambulance_red in [0, 1]` |
| `ambulance_green` | `0` / `1` | Binary Crisp | `ambulance_green in [0, 1]` |
| `extension` | **ZERO** | TrapezoidalMF | `(0.00, 0.00, 0.50, 2.00)` |
| `extension` | **SHORT** | TriangularMF | `(1.50, 4.00, 7.00)` |
| `extension` | **MEDIUM** | TriangularMF | `(5.00, 8.50, 12.00)` |
| `extension` | **LONG** | TrapezoidalMF | `(10.00, 13.00, 15.00, 15.00)` |

## 2. Active & Pruned Fuzzy Rule Base

> Rules with weight $< 0.10$ are flagged as effectively pruned/inactive.

| # | Antecedents (IF) | Consequent (THEN) | Weight | Status |
| :--- | :--- | :--- | :---: | :---: |
| 1 | `ambulance_red` is **1** | `extension` is **ZERO** | 1.000 | Active |
| 2 | `ambulance_green` is **1** | `extension` is **LONG** | 1.000 | Active |
| 3 | `ambulance_green` is **1** AND `ambulance_red` is **1** | `extension` is **SHORT** | 1.000 | Active |
| 4 | `q_green` is **LOW** AND `q_red` is **HIGH** | `extension` is **ZERO** | 1.000 | Active |
| 5 | `q_green` is **HIGH** AND `q_red` is **LOW** | `extension` is **LONG** | 1.000 | Active |
| 6 | `q_green` is **HIGH** AND `q_red` is **HIGH** | `extension` is **SHORT** | 1.000 | Active |
| 7 | `q_green` is **LOW** AND `q_red` is **LOW** | `extension` is **SHORT** | 1.000 | Active |
| 8 | `q_green` is **MEDIUM** AND `q_red` is **MEDIUM** | `extension` is **MEDIUM** | 1.000 | Active |
| 9 | `q_green` is **MEDIUM** AND `q_red` is **HIGH** | `extension` is **SHORT** | 1.000 | Active |
| 10 | `q_green` is **HIGH** AND `q_red` is **MEDIUM** | `extension` is **MEDIUM** | 1.000 | Active |
| 11 | `q_green` is **LOW** AND `q_red` is **MEDIUM** | `extension` is **ZERO** | 1.000 | Active |
| 12 | `bus_share_red` is **HIGH** AND `q_red` is **HIGH** | `extension` is **ZERO** | 1.000 | Active |
| 13 | `bus_share_red` is **HIGH** AND `q_red` is **MEDIUM** | `extension` is **SHORT** | 1.000 | Active |
| 14 | `two_wheeler_share_green` is **HIGH** AND `q_green` is **MEDIUM** | `extension` is **MEDIUM** | 1.000 | Active |
| 15 | `two_wheeler_share_green` is **HIGH** AND `q_red` is **LOW** | `extension` is **LONG** | 1.000 | Active |
| 16 | `green_elapsed` is **HIGH** AND `q_red` is **HIGH** | `extension` is **ZERO** | 1.000 | Active |
| 17 | `green_elapsed` is **HIGH** AND `q_red` is **MEDIUM** | `extension` is **SHORT** | 1.000 | Active |
| 18 | `green_elapsed` is **LOW** AND `q_green` is **HIGH** | `extension` is **LONG** | 1.000 | Active |
