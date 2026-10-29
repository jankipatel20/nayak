# NAYAK Evolved Fuzzy Controller (Nayak-Balanced) Rule Base & Membership Report

## 1. Membership Function Breakpoints (Natural Units)

| Variable | Term | Type | Breakpoints (Natural Units) |
| :--- | :--- | :--- | :--- |
| `q_green` | **LOW** | TrapezoidalMF | `(0.00, 0.00, 1.61, 3.71)` |
| `q_green` | **MEDIUM** | TriangularMF | `(1.61, 7.43, 45.40)` |
| `q_green` | **HIGH** | TrapezoidalMF | `(11.15, 45.40, 50.00, 50.00)` |
| `q_red` | **LOW** | TrapezoidalMF | `(0.00, 0.00, 1.35, 2.60)` |
| `q_red` | **MEDIUM** | TriangularMF | `(1.35, 8.58, 17.12)` |
| `q_red` | **HIGH** | TrapezoidalMF | `(14.56, 17.12, 50.00, 50.00)` |
| `bus_share_red` | **LOW** | TrapezoidalMF | `(0.00, 0.00, 0.19, 0.26)` |
| `bus_share_red` | **MEDIUM** | TriangularMF | `(0.19, 0.52, 0.96)` |
| `bus_share_red` | **HIGH** | TrapezoidalMF | `(0.78, 0.96, 1.00, 1.00)` |
| `two_wheeler_share_green` | **LOW** | TrapezoidalMF | `(0.00, 0.00, 0.10, 0.16)` |
| `two_wheeler_share_green` | **MEDIUM** | TriangularMF | `(0.10, 0.18, 0.74)` |
| `two_wheeler_share_green` | **HIGH** | TrapezoidalMF | `(0.20, 0.74, 1.00, 1.00)` |
| `green_elapsed` | **LOW** | TrapezoidalMF | `(0.00, 0.00, 15.54, 23.61)` |
| `green_elapsed` | **MEDIUM** | TriangularMF | `(15.54, 31.10, 52.12)` |
| `green_elapsed` | **HIGH** | TrapezoidalMF | `(38.60, 52.12, 60.00, 60.00)` |
| `ambulance_red` | `0` / `1` | Binary Crisp | `ambulance_red in [0, 1]` |
| `ambulance_green` | `0` / `1` | Binary Crisp | `ambulance_green in [0, 1]` |
| `extension` | **ZERO** | TrapezoidalMF | `(0.00, 0.00, 0.51, 0.87)` |
| `extension` | **SHORT** | TriangularMF | `(0.51, 2.29, 4.36)` |
| `extension` | **MEDIUM** | TriangularMF | `(3.71, 5.30, 7.58)` |
| `extension` | **LONG** | TrapezoidalMF | `(6.24, 7.58, 15.00, 15.00)` |

## 2. Active & Pruned Fuzzy Rule Base

> Rules with weight $< 0.10$ are flagged as effectively pruned/inactive.

| # | Antecedents (IF) | Consequent (THEN) | Weight | Status |
| :--- | :--- | :--- | :---: | :---: |
| 1 | `ambulance_red` is **1** | `extension` is **ZERO** | 0.177 | Active |
| 2 | `ambulance_green` is **1** | `extension` is **LONG** | 0.151 | Active |
| 3 | `ambulance_green` is **1** AND `ambulance_red` is **1** | `extension` is **SHORT** | 0.379 | Active |
| 4 | `q_green` is **LOW** AND `q_red` is **HIGH** | `extension` is **ZERO** | 0.751 | Active |
| 5 | `q_green` is **HIGH** AND `q_red` is **LOW** | `extension` is **LONG** | 0.318 | Active |
| 6 | `q_green` is **HIGH** AND `q_red` is **HIGH** | `extension` is **SHORT** | 0.910 | Active |
| 7 | `q_green` is **LOW** AND `q_red` is **LOW** | `extension` is **SHORT** | 0.112 | Active |
| 8 | `q_green` is **MEDIUM** AND `q_red` is **MEDIUM** | `extension` is **MEDIUM** | 0.644 | Active |
| 9 | `q_green` is **MEDIUM** AND `q_red` is **HIGH** | `extension` is **SHORT** | 0.096 | Pruned (<0.10) |
| 10 | `q_green` is **HIGH** AND `q_red` is **MEDIUM** | `extension` is **MEDIUM** | 0.318 | Active |
| 11 | `q_green` is **LOW** AND `q_red` is **MEDIUM** | `extension` is **ZERO** | 0.516 | Active |
| 12 | `bus_share_red` is **HIGH** AND `q_red` is **HIGH** | `extension` is **ZERO** | 0.298 | Active |
| 13 | `bus_share_red` is **HIGH** AND `q_red` is **MEDIUM** | `extension` is **SHORT** | 0.741 | Active |
| 14 | `two_wheeler_share_green` is **HIGH** AND `q_green` is **MEDIUM** | `extension` is **MEDIUM** | 0.796 | Active |
| 15 | `two_wheeler_share_green` is **HIGH** AND `q_red` is **LOW** | `extension` is **LONG** | 0.139 | Active |
| 16 | `green_elapsed` is **HIGH** AND `q_red` is **HIGH** | `extension` is **ZERO** | 0.963 | Active |
| 17 | `green_elapsed` is **HIGH** AND `q_red` is **MEDIUM** | `extension` is **SHORT** | 0.412 | Active |
| 18 | `green_elapsed` is **LOW** AND `q_green` is **HIGH** | `extension` is **LONG** | 0.265 | Active |
