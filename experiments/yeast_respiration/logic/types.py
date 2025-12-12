"""
experiments/yeast_respiration/logic/types.py

Pydantic models for the Yeast Respiration experiment.

Contains:
- ExperimentInputs
- FinalState
- Timeline
- VisualHints
- SimulationResult
- ML helper types: TrainingRow, ModelPrediction, ModelTrainingResult
"""

from __future__ import annotations
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Optional, Literal, Any

class ExperimentInputs(BaseModel):
    initial_cells: Optional[float] = Field(
        None, description="Initial viable yeast cell count (CFU). Use this OR inoculum_volume_ml."
    )
    inoculum_volume_ml: float = Field(
        1.0, description="Volume of yeast inoculum added (mL).", ge=0.0
    )
    inoculum_cfu_per_ml: float = Field(
        100000.0, description="CFU per mL in inoculum.", ge=1.0
    )

    initial_sugar_g_per_L: float = Field(
        20.0, description="Initial glucose concentration (g/L).", ge=0.0
    )
    media_volume_L: float = Field(
        0.05, description="Liquid media volume (L).", ge=0.0001
    )
    headspace_fraction: float = Field(
        0.2, description="Fraction of vessel volume that is headspace (0..0.9).", ge=0.0, le=0.9
    )

    temperature_c: float = Field(
        30.0, description="Incubation temperature (°C)."
    )
    pH: float = Field(
        5.5, description="Medium pH."
    )
    aeration: Literal["aerobic", "microaerobic", "anaerobic"] = Field(
        "anaerobic", description="Oxygen availability mode."
    )
    agitation_rpm: float = Field(
        0.0, description="Shaking/stirring speed in rpm.", ge=0.0
    )

    strain: Literal["lab_Saccharomyces_cerevisiae", "bakers_yeast", "wine_yeast", "wild_type"] = Field(
        "lab_Saccharomyces_cerevisiae", description="Strain preset."
    )

    total_time_min: float = Field(
        120.0, description="Total simulation time (minutes).", ge=0.1
    )
    sampling_interval_min: float = Field(
        1.0, description="Sampling interval (minutes).", gt=0.0
    )

    enable_noise: bool = Field(
        True, description="Toggle measurement/process noise."
    )
    seed: Optional[int] = Field(
        None, description="Random seed for reproducible noise. Leave null for random behaviour."
    )

    model_mode: Literal["formula", "ml"] = Field(
        "formula",
        description="Simulation mode: 'formula' (scientific model) or 'ml' (ML surrogate)."
    )
    target_day_for_prediction: float = Field(
        1440.0, description="When using ML mode, predict state at this future time (minutes).", ge=1.0
    )

    notes: Optional[str] = Field(None, description="Optional user note (not used by simulation).")

    @validator("initial_cells", always=True)
    def compute_initial_cells_if_missing(cls, v, values):
        if v is None:
            inoc_vol = values.get("inoculum_volume_ml", 1.0)
            cfu_per_ml = values.get("inoculum_cfu_per_ml", 100000.0)
            try:
                return float(inoc_vol) * float(cfu_per_ml)
            except Exception:
                raise ValueError("Cannot infer initial_cells from inoculum values")
        return v

    @validator("headspace_fraction")
    def clamp_headspace(cls, v):
        if v < 0:
            return 0.0
        if v > 0.9:
            return 0.9
        return v

    @validator("sampling_interval_min")
    def positive_sampling(cls, v):
        if v <= 0:
            raise ValueError("sampling_interval_min must be > 0")
        return v

    class Config:
        orm_mode = True

class FinalState(BaseModel):
    time_min: float = Field(..., description="Simulation end time (minutes).")
    cells: float = Field(..., description="Final cell count (CFU).")
    sugar_g_per_L: float = Field(..., description="Remaining sugar concentration (g/L).")
    co2_generated_g: float = Field(..., description="Cumulative CO2 generated (g).")
    co2_rate_g_per_min: float = Field(..., description="CO2 production rate at final time (g/min).")
    ethanol_g_per_L: float = Field(..., description="Ethanol concentration (g/L).")
    bubble_rate: float = Field(..., description="Visual bubble spawn rate at final time (events/min).")
    doubling_time_min_estimate: Optional[float] = Field(None, description="Doubling time estimate in minutes (if computable).")
    notes: Optional[str] = Field(None, description="Optional human-readable notes.")

    class Config:
        orm_mode = True

class Timeline(BaseModel):
    t_min: List[float] = Field(..., description="Time points (minutes).")
    cells: List[float] = Field(..., description="Cell counts at each time point.")
    sugar_g_per_L: List[float] = Field(..., description="Sugar concentration at each time point (g/L).")
    co2_rate_g_per_min: List[float] = Field(..., description="CO2 production rate time series (g/min).")
    bubble_rate: List[float] = Field(..., description="Bubble rate time series (events/min).")
    ethanol_g_per_L: List[float] = Field(..., description="Ethanol concentration time series (g/L).")

    units: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {
            "t_min": "minutes",
            "cells": "count",
            "sugar_g_per_L": "g/L",
            "co2_rate_g_per_min": "g/min",
            "bubble_rate": "events/min",
            "ethanol_g_per_L": "g/L",
        },
        description="Units for each timeline field (frontend use)."
    )

    @root_validator
    def check_lengths(cls, values):
        t = values.get("t_min")
        if not t or len(t) == 0:
            raise ValueError("t_min must be a non-empty list")
        length = len(t)
        for key in ("cells", "sugar_g_per_L", "co2_rate_g_per_min", "bubble_rate", "ethanol_g_per_L"):
            arr = values.get(key)
            if arr is None:
                raise ValueError(f"{key} is required in the timeline")
            if len(arr) != length:
                raise ValueError(f"All timeline arrays must have the same length as t_min (mismatch: {key})")
        return values

    class Config:
        orm_mode = True

class BubblePosition(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized x position (0..1)")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized y position (0..1)")
    size: float = Field(..., ge=0.0, description="Relative bubble size")

class VisualHints(BaseModel):
    max_bubbles: Optional[int] = Field(None, description="Maximum bubbles allowed for animation")
    particle_density: Optional[int] = Field(None, description="Particle density hint")
    bubble_color: Optional[str] = Field(None, description="Color hex for bubbles")
    animation_speed_multiplier: Optional[float] = Field(1.0, description="Animation speed multiplier")
    bubble_positions: Optional[List[BubblePosition]] = Field(
        None, description="Optional preset bubble coordinates (normalized)."
    )

    class Config:
        orm_mode = True

class SimulationResult(BaseModel):
    final_state: FinalState
    timeline: Timeline
    visual: VisualHints
    meta: Optional[Dict[str, Any]] = Field(None, description="Optional metadata (version, notes, etc.)")

    class Config:
        orm_mode = True

class TrainingRow(BaseModel):
    inputs: ExperimentInputs
    target: Dict[str, float] = Field(..., description="Target values for training (e.g., {'final_cells': float, 'co2': float})")
    meta: Optional[Dict[str, Any]] = Field(None, description="Optional metadata useful for training (seed, run_id)")

class ModelPrediction(BaseModel):
    model_version: Optional[str] = Field(None, description="Model version or identifier")
    predicted_params: Dict[str, float] = Field(..., description="Compact parameters predicted by the model (e.g., mu, Ks, A, k)")
    timeline: Optional[Timeline] = Field(None, description="If the surrogate can return a reconstructed timeline")
    final_state: Optional[FinalState] = Field(None, description="If model directly predicts final state")

class ModelTrainingResult(BaseModel):
    model_version: str = Field(..., description="Saved model identifier/path")
    history: Optional[Dict[str, List[float]]] = Field(None, description="Training history dict (loss, val_loss, metrics, etc.)")
    trained_on_rows: Optional[int] = Field(None, description="Number of training samples used")
    metrics: Optional[Dict[str, float]] = Field(None, description="Final metrics summary on validation set")

    class Config:
        orm_mode = True

Inputs = ExperimentInputs
Result = SimulationResult
