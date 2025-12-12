from __future__ import annotations
import numpy as np
import random
from typing import List, Optional, Dict

def set_random_seed(seed: Optional[int]):
    """Sets Python & NumPy RNG seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
  
def clamp(x: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, x))

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t

def smooth_curve(values: List[float], smoothing: float = 0.1) -> List[float]:
    """
    Smooths a noisy time-series using exponential moving average.
    smoothing = 0 → no smoothing
    smoothing = 1 → heavy smoothing
    """
    if smoothing <= 0:
        return values

    out = []
    prev = values[0]
    for v in values:
        current = prev + smoothing * (v - prev)
        out.append(current)
        prev = current
    return out

def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize to [0..1]."""
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def denormalize(norm: float, min_val: float, max_val: float) -> float:
    """Inverse normalization."""
    return min_val + norm * (max_val - min_val)

def add_gaussian_noise(value: float, sigma: float) -> float:
    """Adds Gaussian noise N(0, sigma)."""
    return float(value + np.random.normal(0, sigma))

def noisy_array(values: List[float], sigma: float) -> List[float]:
    """Apply Gaussian noise to each element."""
    return [add_gaussian_noise(v, sigma) for v in values]

def temperature_factor(temp_c: float, optimal_c: float, sigma: float = 10.0) -> float:
    """
    Gaussian-like temperature performance factor (0..1).
    Peaks at optimal temperature.
    """
    return float(np.exp(-((temp_c - optimal_c) ** 2) / (2 * sigma ** 2)))

def pH_factor(ph: float, optimal: float = 5.5, tolerance: float = 2.0) -> float:
    """
    Returns a pH-effect factor between 0..1.
    """
    return float(np.exp(-((ph - optimal) ** 2) / (2 * tolerance ** 2)))

def sugar_limitation(sugar: float, Ks: float = 0.5) -> float:
    """
    Monod-like substrate saturation curve: sugar / (Ks + sugar).
    """
    if sugar <= 0:
        return 0.0
    return float(sugar / (Ks + sugar))

def aeration_multiplier(aeration: str) -> float:
    """Multiplier for aerobic → fermentation pathways."""
    mapping = {
        "aerobic": 1.0,
        "microaerobic": 0.5,
        "anaerobic": 0.05,
    }
    return mapping.get(aeration, 0.2)

def strain_growth_multiplier(strain: str) -> float:
    """Different strains have slightly different growth efficiencies."""
    mapping = {
        "lab_Saccharomyces_cerevisiae": 1.0,
        "bakers_yeast": 0.9,
        "wine_yeast": 1.1,
        "wild_type": 0.8,
    }
    return mapping.get(strain, 1.0)

def exp_growth_curve(A: float, k: float, t: List[float]) -> List[float]:
    """A * (1 - exp(-k * t)) → common ML-parametric curve."""
    return [A * (1.0 - np.exp(-k * ti)) for ti in t]


def logistic_curve(K: float, r: float, t: List[float]) -> List[float]:
    """Standard logistic growth curve."""
    return [K / (1 + np.exp(-r * (ti - t[0]))) for ti in t]

def interpolate_time_series(x: List[float], y: List[float], new_x: List[float]) -> List[float]:
    """Interpolates a time series to new time points."""
    return list(np.interp(new_x, x, y))

def ensure_same_length(*arrays: List[List[float]]):
    """Raises an error if arrays don't have same length."""
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"Timeline arrays must have same lengths, got lengths: {lengths}")
