# Yeast Respiration Experiment


## Input Parameters

| Parameter | Units | Description |
|---------|------|-------------|
| `initial_cells` | cells (CFU) | Initial viable yeast cell count |
| `initial_sugar_g_per_L` | g/L | Initial glucose concentration |
| `temperature_c` | °C | Incubation temperature |
| `pH` | pH | Acidity of the medium |
| `aeration` | category | Oxygen condition (`aerobic`, `microaerobic`, `anaerobic`) |
| `inoculum_volume_ml` | mL | Volume of yeast inoculum added |
| `total_time_min` | minutes | Total duration of simulation |
| `sampling_interval_min` | minutes | Time resolution of outputs |
| `seed` | integer / null | Random seed for reproducibility |

---

## Output Parameters

### 1. Final State (End Snapshot)

| Parameter | Units | Description |
|--------|------|-------------|
| `time_min` | minutes | End time of simulation |
| `cells` | cells (CFU) | Final yeast cell count |
| `sugar_g_per_L` | g/L | Remaining glucose concentration |
| `co2_generated_g` | grams | Total CO₂ produced |
| `ethanol_g_per_L` | g/L | Final ethanol concentration |
| `doubling_time_min_estimate` | minutes | Estimated cell doubling time |

---

### 2. Timeline (Time-Series Data)

All arrays are aligned with `t_min`.

| Parameter | Units | Description |
|--------|------|-------------|
| `t_min` | minutes | Time points |
| `cells` | cells | Cell count over time |
| `sugar_g_per_L` | g/L | Sugar concentration over time |
| `co2_rate_g_per_min` | g/min | CO₂ production rate |
| `bubble_rate` | events/min | Bubble generation rate |
| `ethanol_g_per_L` | g/L | Ethanol concentration over time |

---

### 3. Visual Hints (For Frontend Animation)

| Parameter | Description |
|--------|-------------|
| `max_bubbles` | Maximum bubbles allowed on screen |
| `particle_density` | Density of particles for visualization |
| `bubble_positions` | Optional preset bubble coordinates |

---

**Experiment Folder:**  
`experiments/yeast_respiration/`
