# Yeast Respiration Virtual Lab

This backend powers the **Yeast Respiration Virtual Laboratory**, providing:

- Deterministic biological simulation  
- Machine-learning surrogate prediction  
- Training and dataset generation workflows  
- Strongly typed schemas  
- FastAPI HTTP API  
- Developer utilities for research and classroom teaching  


##  Table of Contents

- Scientific Overview 
- API Input & Output Examples 
- Timeline Explanation
- Machine Learning System  
- Developer Guide
- Frontend Integration Guide 

## 1. Scientific Overview

### What the Experiment Simulates

This virtual lab models batch fermentation of *Saccharomyces cerevisiae*, tracking how environmental variables influence respiration and fermentation.

| Controlled Variable | Effect |
|----------------------|--------|
| Glucose (g/L) | Drives growth & CO₂/Ethanol production |
| Temperature (°C) | Modifies reaction kinetics |
| pH | Affects enzyme activity |
| Aeration (aerobic / microaerobic / anaerobic) | Determines respiration vs fermentation |
| Agitation (rpm) | Influences oxygen transfer |
| Strain | Changes μmax and yield |

The bioprocess is modeled using:
- Monod-like growth kinetics  
- Temperature factor  
- pH factor  
- Oxygen availability multiplier  
- Strain-specific multipliers  

### Simulated Outputs

| Variable | Meaning |
|-----------|----------|
| cells | Viable yeast cell count (CFU) |
| sugar_g_per_L | Remaining glucose concentration |
| co2_rate_g_per_min | Instantaneous CO₂ production |
| co2_generated_g | Total CO₂ produced |
| ethanol_g_per_L | Ethanol concentration |
| bubble_rate | Bubble events/min (for animation) |
| doubling_time_min_estimate | Growth rate estimate |


### Simulation Output Structure

The simulator returns:

1. **final_state** — Snapshot at the end of the simulation.  
2. **timeline** — Time-aligned arrays describing dynamics across `t_min`.  
3. **visual** — Frontend animation hints (bubble count, particle density, etc.)

All fields conform to `schema/outputs.json`.

## 2. API Input & Output Examples

### Simulate

**Request**
```json
POST /simulate
{
  "inputs": {
    "initial_cells": null,
    "inoculum_volume_ml": 1.0,
    "inoculum_cfu_per_ml": 100000,
    "initial_sugar_g_per_L": 20,
    "temperature_c": 30,
    "pH": 5.5,
    "aeration": "anaerobic",
    "total_time_min": 120,
    "sampling_interval_min": 1
  }
}
```

**Response (truncated)**
```json
{
  "final_state": {
    "time_min": 120.0,
    "cells": 241884.3,
    "sugar_g_per_L": 13.92,
    "co2_generated_g": 6.08,
    "co2_rate_g_per_min": 0.054,
    "ethanol_g_per_L": 2.31,
    "bubble_rate": 0.54
  },
  "timeline": {
    "t_min": [0, 1, 2, ...],
    "cells": [...],
    "sugar_g_per_L": [...],
    "co2_rate_g_per_min": [...],
    "ethanol_g_per_L": [...]
  },
  "visual": {
    "max_bubbles": 60,
    "particle_density": 120
  }
}
```


### Train Model

**Request**
```json
POST /train
{
  "n_samples": 64,
  "save_model": true,
  "seed": 42
}
```

**Response**
```json
{
  "model_version": "./models/yeast_respiration_model.keras",
  "trained_on_rows": 64,
  "history": { "loss": [...], "val_loss": [...] },
  "metrics": { "loss": 0.008, "val_loss": 0.011 },
  "meta": { "<scaler>": "<scaler>" }
}
```

### Predict

**Request**
```json
POST /predict
{
  "inputs": {
    "initial_cells": 50000,
    "initial_sugar_g_per_L": 20,
    "total_time_min": 120,
    "sampling_interval_min": 1
  }
}
```

**Response (truncated)**
```json
{
  "model_version": "yeast_surrogate_mlp",
  "predicted_params": {
    "A_cells": 311000,
    "k_cells": 0.031,
    "A_co2": 0.23,
    "k_co2": 0.019
  },
  "timeline": {
    "t_min": [...],
    "cells": [...],
    "co2_rate_g_per_min": [...],
    "ethanol_g_per_L": [...]
  }
}
```


## 3. Timeline Explanation

### Array Alignment

All arrays share the same indices:

| Index | Meaning |
|--------|----------|
| t_min[i] → cells[i] | Population at time i |
| t_min[i] → sugar_g_per_L[i] | Remaining glucose |
| t_min[i] → co2_rate_g_per_min[i] | Instant CO₂ rate |
| t_min[i] → ethanol_g_per_L[i] | Ethanol concentration |


### Noise Behavior

If `enable_noise = true`:

| Variable | Noise Type |
|-----------|-------------|
| cells | Log-normal (biological variation) |
| sugar, CO₂, ethanol | Gaussian measurement noise |

`seed` → fully deterministic simulation.

### Frontend Animation Mapping

| Value | Visual Effect |
|--------|----------------|
| co2_rate_g_per_min | Bubble size or intensity |
| bubble_rate | Number of bubbles per second |
| cells | Liquid turbidity / opacity |
| ethanol_g_per_L | Color shift |
| sugar_g_per_L | Substrate depletion bar |

**Bubble Probability Formula**
```text
events_per_sec = bubble_rate / 60
prob_per_frame = events_per_sec / fps
```


## 4. Machine Learning System

The ML surrogate predicts fermentation curves instantly.

### Training Flow

`generate_teacher_dataset(n)` runs the real simulator N times.  
For each timeline, parameters are extracted:

| Parameter | Meaning |
|------------|----------|
| A | Plateau (max value) |
| k | Growth/decay rate constant |

The MLP learns:
```text
inputs → [A_cells, k_cells, A_co2, k_co2, A_eth, k_eth]
```

Runtime predictions reconstruct curves using:

\[
y(t) = A \times (1 - e^{-k t})
\]


## 5. Developer Guide

### Run a Simulation
```python
from experiments.yeast_respiration.logic.types import ExperimentInputs
from experiments.yeast_respiration.logic.sim import simulate

inp = ExperimentInputs(initial_cells=100000, total_time_min=120)
res = simulate(inp)
print(res.final_state)

```

### Start FastAPI Server
```bash

uvicorn experiments.yeast_respiration.api:app --reload --port 8001

```

### Train a Model
```python

from experiments.yeast_respiration.logic.sim import generate_teacher_dataset
from experiments.yeast_respiration.logic.ai import train_model
rows = generate_teacher_dataset(128)
result = train_model(rows)

```


### Run an ML Prediction
```python

from experiments.yeast_respiration.logic.ai import load_model, predict
model = load_model()
pred = predict(model, inp)
print(pred.timeline)

```

### Generate Teacher Dataset
```python

rows = generate_teacher_dataset(64, seed=10)

```


## 6. Frontend Integration Guide

### Mapping Outputs to Visuals

| Simulation Value | Visual Representation |
|-------------------|------------------------|
| bubble_rate | Bubble frequency |
| co2_rate_g_per_min | Bubble size |
| cells | Turbidity/opacity |
| ethanol_g_per_L | Color shift towards golden |
| sugar_g_per_L | Substrate-progress indicator |


### Frontend Best Practices

-  **Interpolate values between timeline samples** — Use linear interpolation or EMA for smooth animation.  
-  **Cap bubble count** — Follow:
 `visual.max_bubbles` and `visual.particle_density`.  
-  **Use provided bubble_positions if present** — Ensures stable bubble origin points.  
