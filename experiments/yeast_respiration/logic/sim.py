"""
Core simulation engine for the Yeast Respiration experiment.

Provides:
- simulate(inputs: ExperimentInputs) -> SimulationResult
- simulate_at_time(inputs: ExperimentInputs, t_min: float) -> FinalState
- generate_teacher_dataset(n: int, param_sampler: Callable) -> List[TrainingRow]

All domain logic (Monod-like kinetics, yield conversions, CO2/ethanol production
and visual hint generation) is contained here. Helpers are used for small utilities.
"""

from __future__ import annotations
import json
import math
import os
from typing import List, Callable, Optional, Dict

import numpy as np

from experiments.yeast_respiration.logic.types import (
    ExperimentInputs,
    Timeline,
    FinalState,
    VisualHints,
    SimulationResult,
    TrainingRow,
)
from experiments.yeast_respiration.logic.helpers import (
    set_random_seed,
    temperature_factor,
    pH_factor,
    sugar_limitation,
    aeration_multiplier,
    strain_growth_multiplier,
    noisy_array,
    exp_growth_curve,
    ensure_same_length,
)

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "sim_config.json")

def _load_config() -> dict:
    try:
        with open(_CFG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "default_sampling_interval_min": 1.0,
            "default_total_time_min": 120,
            "noise": {"measurement_sigma": 0.02, "cell_noise_factor": 0.01, "enable_noise": True},
            "constants": {
                "optimal_temperature_c": 30,
                "optimal_pH": 5.5,
                "ks_glucose_g_per_L": 0.5,
                "mu_max_per_min": 0.03,
                "yield_cells_per_g_sugar": 1e6,
                "co2_per_g_sugar": 1.0,
                "ethanol_per_g_sugar": 0.51,
                "oxygen_factor": {"aerobic": 1.0, "microaerobic": 0.5, "anaerobic": 0.05},
            },
            "visual": {"base_bubble_multiplier": 10, "max_bubbles": 120, "max_particles": 400},
        }


_CFG = _load_config()

def cfg_const(key: str, default):
    return _CFG.get("constants", {}).get(key, default)


def cfg_noise(key: str, default):
    return _CFG.get("noise", {}).get(key, default)


def cfg_visual(key: str, default):
    return _CFG.get("visual", {}).get(key, default)

def simulate(inputs: ExperimentInputs) -> SimulationResult:
    """
    Run a full simulation and return a SimulationResult (FinalState + Timeline + VisualHints).
    """
    set_random_seed(inputs.seed)

    mu_max = float(cfg_const("mu_max_per_min", 0.03))  
    Ks = float(cfg_const("ks_glucose_g_per_L", 0.5))  
    yield_cells_per_g = float(cfg_const("yield_cells_per_g_sugar", 1e6))  
    co2_per_g = float(cfg_const("co2_per_g_sugar", 1.0))  
    ethanol_per_g = float(cfg_const("ethanol_per_g_sugar", 0.51))  
    oxygen_map = cfg_const("oxygen_factor", {"aerobic": 1.0, "microaerobic": 0.5, "anaerobic": 0.05})

    base_bubble_multiplier = float(cfg_visual("base_bubble_multiplier", 10.0))
    max_bubbles_cfg = int(cfg_visual("max_bubbles", 120))
    max_particles = int(cfg_visual("max_particles", 400))

    meas_sigma = float(cfg_noise("measurement_sigma", 0.02))
    cell_noise_factor = float(cfg_noise("cell_noise_factor", 0.01))
    enable_noise_cfg = bool(cfg_noise("enable_noise", True))
    enable_noise = inputs.enable_noise and enable_noise_cfg

    dt = float(inputs.sampling_interval_min)
    total_time = float(inputs.total_time_min)
    steps = max(1, int(math.ceil(total_time / dt)))
    t_series = [round(i * dt, 6) for i in range(steps + 1)]

    cells = float(inputs.initial_cells or 0.0)  
    sugar_conc = float(inputs.initial_sugar_g_per_L or 0.0)  
    media_vol = float(inputs.media_volume_L or 0.05)  

    sugar_mass_g = sugar_conc * media_vol  

    temp_factor = temperature_factor(inputs.temperature_c, cfg_const("optimal_temperature_c", 30.0))
    phf = pH_factor(inputs.pH, cfg_const("optimal_pH", 5.5))
    oxy_factor = oxygen_map.get(inputs.aeration, 0.05)
    strain_mult = strain_growth_multiplier(inputs.strain)

    cells_t = []
    sugar_t = []
    co2_rate_t = []
    bubble_rate_t = []
    ethanol_t = []

    cumulative_co2 = 0.0

    eps = 1e-9

    for t in t_series:
        sugar_conc = sugar_mass_g / max(media_vol, eps)
        substrate_term = sugar_limitation(sugar_conc, Ks)
        mu = mu_max * substrate_term * temp_factor * phf * oxy_factor * strain_mult

        d_cells = mu * cells * dt
        sugar_consumed_g = (d_cells / yield_cells_per_g) if yield_cells_per_g > 0 else 0.0

        sugar_mass_g = max(0.0, sugar_mass_g - sugar_consumed_g)
        cells = max(0.0, cells + d_cells)

        co2_produced_g = sugar_consumed_g * co2_per_g
        co2_rate = co2_produced_g / max(dt, eps)  
        ethanol_produced_g = sugar_consumed_g * ethanol_per_g
        ethanol_conc_g_per_L = ethanol_produced_g / max(media_vol, eps)

        cumulative_co2 += co2_produced_g

        bubble_rate = co2_rate * base_bubble_multiplier

        measured_cells = cells
        if enable_noise:
            sigma_cells = max(cell_noise_factor * cells, 1.0)
            measured_cells = float(np.random.normal(cells, sigma_cells))

        cells_t.append(float(measured_cells))
        sugar_t.append(float(sugar_mass_g / max(media_vol, eps)))  
        co2_rate_t.append(float(co2_rate))
        bubble_rate_t.append(float(bubble_rate))
        ethanol_t.append(float(ethanol_conc_g_per_L))

    try:
        if cells_t[0] > 0 and cells_t[-1] > cells_t[0]:
            ratio = cells_t[-1] / cells_t[0]
            doubling_time = total_time * math.log(2) / math.log(ratio) if ratio > 1 else None
        else:
            doubling_time = None
    except Exception:
        doubling_time = None
      
    if enable_noise and meas_sigma > 0:
        sugar_t = noisy_array(sugar_t, meas_sigma)
        co2_rate_t = noisy_array(co2_rate_t, meas_sigma)
        bubble_rate_t = noisy_array(bubble_rate_t, meas_sigma)
        ethanol_t = noisy_array(ethanol_t, meas_sigma)

    timeline = Timeline(
        t_min=list(t_series),
        cells=[float(x) for x in cells_t],
        sugar_g_per_L=[float(x) for x in sugar_t],
        co2_rate_g_per_min=[float(x) for x in co2_rate_t],
        bubble_rate=[float(x) for x in bubble_rate_t],
        ethanol_g_per_L=[float(x) for x in ethanol_t],
    )

    final_state = FinalState(
        time_min=total_time,
        cells=float(cells_t[-1]) if cells_t else 0.0,
        sugar_g_per_L=float(sugar_t[-1]) if sugar_t else 0.0,
        co2_generated_g=float(cumulative_co2),
        co2_rate_g_per_min=float(co2_rate_t[-1]) if co2_rate_t else 0.0,
        ethanol_g_per_L=float(ethanol_t[-1]) if ethanol_t else 0.0,
        bubble_rate=float(bubble_rate_t[-1]) if bubble_rate_t else 0.0,
        doubling_time_min_estimate=doubling_time,
        notes=f"Simulated with Monod-like kinetics; noise enabled: {bool(enable_noise)}",
    )

    avg_bubble_rate = float(np.mean(bubble_rate_t)) if bubble_rate_t else 0.0
    visual = VisualHints(
        max_bubbles=min(max_bubbles_cfg, int(max(1, avg_bubble_rate * 1.5))),
        particle_density=min(max_particles, int(avg_bubble_rate * 2)),
        bubble_color="#F2C94C",
        animation_speed_multiplier=1.0,
        bubble_positions=None,
    )

    ensure_same_length(timeline.t_min, timeline.cells, timeline.sugar_g_per_L, timeline.co2_rate_g_per_min, timeline.bubble_rate, timeline.ethanol_g_per_L)

    result = SimulationResult(final_state=final_state, timeline=timeline, visual=visual, meta={"version": "1.0.0"})

    return result

def simulate_at_time(inputs: ExperimentInputs, t_min: float) -> FinalState:
    """
    Run the simulation truncated at t_min and return the FinalState at that time.
    """
    inputs_short = inputs.copy(update={"total_time_min": float(t_min)})
    sim = simulate(inputs_short)
    return sim.final_state

def generate_teacher_dataset(
    n: int,
    param_sampler: Optional[Callable[[int], ExperimentInputs]] = None,
    seed: Optional[int] = None,
) -> List[TrainingRow]:
    """
    Generate N training rows for ML using the simulate() engine.

    param_sampler: optional callable(index) -> ExperimentInputs
      - If None, uses random sampling around default values from config.
    seed: optional integer to make generation deterministic.
    Returns: List[TrainingRow]
    """
    if seed is not None:
        set_random_seed(seed)

    rows: List[TrainingRow] = []
    for i in range(n):
        if param_sampler is None:
            inp = ExperimentInputs(
                initial_cells=None,
                inoculum_volume_ml=float(max(0.01, np.random.lognormal(mean=0.0, sigma=0.5))),
                inoculum_cfu_per_ml=float(max(100.0, np.random.lognormal(mean=11.5, sigma=0.5))),
                initial_sugar_g_per_L=float(max(0.1, np.random.normal(loc=20.0, scale=8.0))),
                media_volume_L=float(max(0.001, np.random.normal(loc=0.05, scale=0.02))),
                headspace_fraction=float(min(0.9, max(0.0, np.random.normal(loc=0.2, scale=0.1)))),
                temperature_c=float(min(45.0, max(5.0, np.random.normal(loc=30.0, scale=4.0)))),
                pH=float(min(8.0, max(3.0, np.random.normal(loc=5.5, scale=0.5)))),
                aeration=str(np.random.choice(["aerobic", "microaerobic", "anaerobic"])),
                agitation_rpm=float(max(0.0, np.random.normal(loc=0.0, scale=50.0))),
                strain=str(np.random.choice(["lab_Saccharomyces_cerevisiae", "bakers_yeast", "wine_yeast", "wild_type"])),
                total_time_min=float(np.random.choice([60, 120, 240, 480])),
                sampling_interval_min=float(1.0),
                enable_noise=True,
                seed=None,
                model_mode="formula",
                target_day_for_prediction=1440.0,
                notes=None,
            )
        else:
            inp = param_sampler(i)

        sim = simulate(inp)
        target = {
            "final_cells": float(sim.final_state.cells),
            "co2_generated_g": float(sim.final_state.co2_generated_g),
            "ethanol_g_per_L": float(sim.final_state.ethanol_g_per_L),
        }
        rows.append(TrainingRow(inputs=inp, target=target, meta={"index": i}))
    return rows
