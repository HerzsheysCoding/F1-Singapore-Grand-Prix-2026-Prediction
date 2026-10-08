# F1 Win Probability Engine — Singapore Grand Prix 

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LambdaMART%20%2F%20LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning ranking and win-probability pipeline for Formula 1 races, calibrated specifically for high-downforce street circuits (Marina Bay Street Circuit).

---

## Problem Formulation

Predicting a Grand Prix outcome is formulated as a Learning-to-Rank (LTR) task rather than independent classification:
* Race outcomes are zero-sum and ordered ($y \in \{1, \dots, N\}$).
* The objective optimizes Normalized Discounted Cumulative Gain (NDCG) via LambdaMART.
* Driver utility scores $\hat{s}$ are calibrated into win probabilities using a temperature-scaled Softmax:

$$P(\text{Win}_i) = \frac{e^{\hat{s}_i / T}}{\sum_{k=1}^{N} e^{\hat{s}_k / T}}$$

---

## Feature Store & Domain Priors

| Feature | Extraction Logic | Hypothesis / Circuit Sensitivity |
| :--- | :--- | :--- |
| `grid_pos` | Official Qualifying Classification | ~70% historical pole-to-win conversion rate at Singapore due to high dirty-air penalty. |
| `street_win_rate` | Bayesian prior across Monaco, Baku, Singapore | High traction sensitivity and barrier proximity tolerance. |
| `rolling_pace_ewma` | Exponentially Weighted Moving Average ($\alpha=0.3$) of race stint pace | In-season aerodynamic upgrade momentum. |
| `low_speed_traction` | GPS apex delta (60–120 km/h) from Free Practice sessions | Corner exit acceleration efficiency. |
| `incident_risk` | Career safety-car/DNF frequency on street circuits | Stochastic penalty for Marina Bay's ~100% Safety Car probability. |

---

## Predicted Distribution

```text
Driver                    Grid    Win Probability
-----------------------------------------------------------
Andrea Kimi Antonelli     P1      27.4%  ██████████████
Max Verstappen            P2      23.8%  ████████████
Lando Norris              P3      19.1%  ██████████
George Russell            P4      14.5%  ███████
Charles Leclerc           P5       9.2%  █████
Lewis Hamilton            P6       4.5%  ██
Other Grid Entries        P7+      1.5%  █
