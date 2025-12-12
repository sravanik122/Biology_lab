from __future__ import annotations
import json
import os
import time
from typing import Tuple, Dict, Any, List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from experiments.yeast_respiration.logic.types import (
    ExperimentInputs,
    TrainingRow,
    ModelPrediction,
    ModelTrainingResult,
    Timeline,
)
from experiments.yeast_respiration.logic.sim import simulate

_MODEL_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "model_config.json")


def _load_model_config() -> dict:
    try:
        with open(_MODEL_CFG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "epochs": 40,
            "batch_size": 16,
            "validation_split": 0.2,
            "optimizer": {"type": "adam", "learning_rate": 0.001},
            "model_layers": [128, 64, 32],
            "loss": "mse",
            "metrics": ["mae"],
            "tensorboard_logdir": "./logs/model_training",
            "model_save_path": "./models/yeast_respiration_model.keras",
            "random_seed": 42,
        }


_MODEL_CFG = _load_model_config()


class MinMaxScaler:
    def __init__(self, min_: Optional[np.ndarray] = None, max_: Optional[np.ndarray] = None):
        self.min_ = min_
        self.max_ = max_

    def fit(self, X: np.ndarray):
        self.min_ = X.min(axis=0, keepdims=True)
        self.max_ = X.max(axis=0, keepdims=True)
        self.range_ = np.where(self.max_ - self.min_ <= 0, 1.0, (self.max_ - self.min_))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler not fitted")
        return (X - self.min_) / self.range_

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return X_scaled * self.range_ + self.min_


def build_model(input_dim: int, output_dim: int, cfg: Optional[dict] = None) -> tf.keras.Model:
    if cfg is None:
        cfg = _MODEL_CFG

    layers_config = cfg.get("model_layers", [128, 64, 32])
    lr = float(cfg.get("optimizer", {}).get("learning_rate", 0.001))
    opt_type = cfg.get("optimizer", {}).get("type", "adam")

    inputs = layers.Input(shape=(input_dim,), name="inputs")
    x = inputs
    for i, size in enumerate(layers_config):
        x = layers.Dense(int(size), activation="relu", name=f"dense_{i}")(x)
        x = layers.BatchNormalization()(x)
    x = layers.Dense(max(32, output_dim * 4), activation="relu", name="dense_head")(x)
    outputs = layers.Dense(output_dim, activation="linear", name="params")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="yeast_surrogate_mlp")

    if opt_type.lower() == "adam":
        opt = optimizers.Adam(learning_rate=lr)
    elif opt_type.lower() == "sgd":
        opt = optimizers.SGD(learning_rate=lr)
    else:
        opt = optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt, loss=cfg.get("loss", "mse"), metrics=cfg.get("metrics", ["mae"]))
    return model


_INPUT_ORDER = [
    "initial_cells",
    "inoculum_volume_ml",
    "inoculum_cfu_per_ml",
    "initial_sugar_g_per_L",
    "media_volume_L",
    "headspace_fraction",
    "temperature_c",
    "pH",
    "aeration",
    "agitation_rpm",
    "strain",
    "total_time_min",
    "sampling_interval_min",
    "enable_noise",
]


def _encode_inputs(inp: ExperimentInputs) -> np.ndarray:
    aer_options = ["aerobic", "microaerobic", "anaerobic"]
    strain_options = ["lab_Saccharomyces_cerevisiae", "bakers_yeast", "wine_yeast", "wild_type"]

    vals: List[float] = []
    vals.append(float(inp.initial_cells or 0.0))
    vals.append(float(inp.inoculum_volume_ml))
    vals.append(float(inp.inoculum_cfu_per_ml))
    vals.append(float(inp.initial_sugar_g_per_L))
    vals.append(float(inp.media_volume_L))
    vals.append(float(inp.headspace_fraction))
    vals.append(float(inp.temperature_c))
    vals.append(float(inp.pH))

    for a in aer_options:
        vals.append(1.0 if inp.aeration == a else 0.0)

    vals.append(float(inp.agitation_rpm))

    for s in strain_options:
        vals.append(1.0 if inp.strain == s else 0.0)

    vals.append(float(inp.total_time_min))
    vals.append(float(inp.sampling_interval_min))
    vals.append(1.0 if inp.enable_noise else 0.0)
    return np.array(vals, dtype=np.float32)


def _fit_A_k(t: np.ndarray, y: np.ndarray, k_grid=None) -> Tuple[float, float]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size == 0 or y.size == 0:
        return 0.0, 0.001
    if k_grid is None:
        k_grid = np.logspace(-4, 0, 50)
    best_A, best_k, best_err = 0.0, 0.01, float("inf")
    A_guess = max(y.max(), 1e-6)
    for k in k_grid:
        denom = (1.0 - np.exp(-k * t))
        if np.allclose(denom, 0.0):
            continue
        A = np.sum(y * denom) / (np.sum(denom * denom) + 1e-12)
        pred = A * denom
        err = np.mean((pred - y) ** 2)
        if err < best_err:
            best_err = err
            best_A = float(A)
            best_k = float(k)
    # local refine around best_k
    k0 = best_k
    kgrid = np.linspace(max(1e-6, k0 * 0.5), k0 * 1.5 + 1e-6, 30)
    for k in kgrid:
        denom = (1.0 - np.exp(-k * t))
        if np.allclose(denom, 0.0):
            continue
        A = np.sum(y * denom) / (np.sum(denom * denom) + 1e-12)
        pred = A * denom
        err = np.mean((pred - y) ** 2)
        if err < best_err:
            best_err = err
            best_A = float(A)
            best_k = float(k)
    if not np.isfinite(best_A):
        best_A = float(y.max() if y.size else 0.0)
    if not np.isfinite(best_k) or best_k <= 0:
        best_k = 0.001
    return best_A, best_k


def _rows_to_xy(rows: List[TrainingRow]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X = []
    y = []
    for r in rows:
        X.append(_encode_inputs(r.inputs))
        # Prefer precomputed compact params in meta
        meta = r.meta or {}
        if "predicted_params" in meta:
            p = meta["predicted_params"]
            y.append([p.get("A_cells", 0.0), p.get("k_cells", 0.01),
                      p.get("A_co2", 0.0), p.get("k_co2", 0.01),
                      p.get("A_eth", 0.0), p.get("k_eth", 0.01)])
            continue

        # If target contains timeline arrays, use them
        t_arr = None
        cells_ts = None
        co2_ts = None
        eth_ts = None
        if isinstance(r.target, dict) and "timeline_t" in r.target:
            t_arr = np.asarray(r.target.get("timeline_t", []), dtype=float)
            cells_ts = np.asarray(r.target.get("timeline_cells", []), dtype=float)
            co2_ts = np.asarray(r.target.get("timeline_co2", []), dtype=float)
            eth_ts = np.asarray(r.target.get("timeline_eth", []), dtype=float)
        else:
            # fallback: run simulate to produce timelines
            simres = simulate(r.inputs)
            t_arr = np.asarray(simres.timeline.t_min, dtype=float)
            cells_ts = np.asarray(simres.timeline.cells, dtype=float)
            co2_ts = np.asarray(simres.timeline.co2_rate_g_per_min, dtype=float)
            eth_ts = np.asarray(simres.timeline.ethanol_g_per_L, dtype=float)

        A_c, k_c = _fit_A_k(t_arr, cells_ts)
        A_co2, k_co2 = _fit_A_k(t_arr, co2_ts)
        A_eth, k_eth = _fit_A_k(t_arr, eth_ts)
        y.append([A_c, k_c, A_co2, k_co2, A_eth, k_eth])

    X = np.vstack(X).astype(np.float32)
    y = np.vstack(y).astype(np.float32)
    return X, y, []


def save_scalers(path_prefix: str, x_scaler: MinMaxScaler, y_scaler: MinMaxScaler) -> None:
    d = os.path.dirname(path_prefix)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    np.save(path_prefix + "_x_min.npy", x_scaler.min_)
    np.save(path_prefix + "_x_max.npy", x_scaler.max_)
    np.save(path_prefix + "_y_min.npy", y_scaler.min_)
    np.save(path_prefix + "_y_max.npy", y_scaler.max_)


def load_scalers(path_prefix: str) -> Tuple[MinMaxScaler, MinMaxScaler]:
    x_min = np.load(path_prefix + "_x_min.npy")
    x_max = np.load(path_prefix + "_x_max.npy")
    y_min = np.load(path_prefix + "_y_min.npy")
    y_max = np.load(path_prefix + "_y_max.npy")
    x = MinMaxScaler(min_=x_min, max_=x_max)
    x.range_ = np.where(x.max_ - x.min_ <= 0, 1.0, (x.max_ - x.min_))
    y = MinMaxScaler(min_=y_min, max_=y_max)
    y.range_ = np.where(y.max_ - y.min_ <= 0, 1.0, (y.max_ - y.min_))
    return x, y


def params_to_timeline(predicted_params: Dict[str, float], total_time_min: float, sampling_interval_min: float) -> Timeline:
    dt = float(sampling_interval_min)
    steps = max(1, int(round(total_time_min / dt)))
    t = np.linspace(0.0, total_time_min, steps + 1).tolist()

    def _curve(A: float, k: float, t_list: List[float]) -> List[float]:
        return [float(A * (1.0 - np.exp(-k * ti))) for ti in t_list]

    cells_ts = _curve(predicted_params.get("A_cells", 0.0), predicted_params.get("k_cells", 0.0), t)
    co2_ts = _curve(predicted_params.get("A_co2", 0.0), predicted_params.get("k_co2", 0.0), t)
    eth_ts = _curve(predicted_params.get("A_eth", 0.0), predicted_params.get("k_eth", 0.0), t)

    timeline = Timeline(
        t_min=[float(x) for x in t],
        cells=[float(x) for x in cells_ts],
        sugar_g_per_L=[0.0 for _ in t],
        co2_rate_g_per_min=[float(x) for x in co2_ts],
        bubble_rate=[float(x) for x in co2_ts],
        ethanol_g_per_L=[float(x) for x in eth_ts],
    )
    return timeline


def train_model(
    rows: List[TrainingRow],
    model: Optional[tf.keras.Model] = None,
    cfg: Optional[dict] = None,
    save_model: bool = True,
    model_save_path: Optional[str] = None,
) -> ModelTrainingResult:
    if cfg is None:
        cfg = _MODEL_CFG

    seed_val = int(cfg.get("random_seed", 42))
    tf.random.set_seed(seed_val)
    np.random.seed(seed_val)

    X, y, _ = _rows_to_xy(rows)
    input_dim = X.shape[1]
    output_dim = y.shape[1]

    x_scaler = MinMaxScaler()
    x_scaler.fit(X)
    Xs = x_scaler.transform(X)

    y_scaler = MinMaxScaler()
    y_scaler.fit(y)
    ys = y_scaler.transform(y)

    if model is None:
        model = build_model(input_dim, output_dim, cfg)

    logdir = cfg.get("tensorboard_logdir", "./logs/model_training")
    timestamp = int(time.time())
    tb_dir = os.path.join(logdir, f"run_{timestamp}")
    os.makedirs(tb_dir, exist_ok=True)
    cb_list = [
        callbacks.TensorBoard(log_dir=tb_dir),
        callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
    ]

    if model_save_path is None:
        model_save_path = cfg.get("model_save_path", "./models/yeast_respiration_model.keras")
    checkpoint_dir = os.path.dirname(model_save_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    cb_list.append(callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor="val_loss"))

    history = model.fit(
        Xs,
        ys,
        epochs=int(cfg.get("epochs", 40)),
        batch_size=int(cfg.get("batch_size", 16)),
        validation_split=float(cfg.get("validation_split", 0.2)),
        callbacks=cb_list,
        verbose=1,
    )

    if save_model:
        model.save(model_save_path)
        # persist scalers next to model using same prefix
        save_scalers(model_save_path, x_scaler, y_scaler)

    result = ModelTrainingResult(
        model_version=model_save_path,
        history={k: [float(v) for v in vals] for k, vals in history.history.items()},
        trained_on_rows=int(X.shape[0]),
        metrics={k: float(v[-1]) for k, v in history.history.items() if k.startswith("val_") or k == "loss" or k == "mae"},
    )
    result.meta = {"x_scaler": x_scaler, "y_scaler": y_scaler}
    return result


def load_model(path: Optional[str] = None) -> tf.keras.Model:
    if path is None:
        path = _MODEL_CFG.get("model_save_path", "./models/yeast_respiration_model.keras")
    return tf.keras.models.load_model(path)


def predict(
    model: tf.keras.Model,
    inputs: Union[ExperimentInputs, List[ExperimentInputs]],
    x_scaler: Optional[MinMaxScaler] = None,
    y_scaler: Optional[MinMaxScaler] = None,
    cfg: Optional[dict] = None,
    run_simulate: bool = False,
) -> Union[ModelPrediction, List[ModelPrediction]]:
    if cfg is None:
        cfg = _MODEL_CFG

    single = False
    if isinstance(inputs, ExperimentInputs):
        inputs = [inputs]
        single = True

    X = np.vstack([_encode_inputs(inp) for inp in inputs]).astype(np.float32)

    if x_scaler is None:
        # try to load saved scalers next to model if available
        model_path = cfg.get("model_save_path")
        if model_path and os.path.exists(model_path + "_x_min.npy"):
            try:
                x_s, y_s = load_scalers(model_path)
                x_scaler, y_scaler = x_s, y_s
            except Exception:
                x_scaler = MinMaxScaler()
                x_scaler.fit(X)
        else:
            x_scaler = MinMaxScaler()
            x_scaler.fit(X)

    Xs = x_scaler.transform(X)

    preds_scaled = model.predict(Xs)
    if y_scaler is not None:
        preds = y_scaler.inverse_transform(preds_scaled)
    else:
        preds = preds_scaled

    results: List[ModelPrediction] = []
    for i, p in enumerate(preds):
        A_cells, k_cells, A_co2, k_co2, A_eth, k_eth = [float(x) for x in p.tolist()]
        predicted_params = {
            "A_cells": A_cells,
            "k_cells": k_cells,
            "A_co2": A_co2,
            "k_co2": k_co2,
            "A_eth": A_eth,
            "k_eth": k_eth,
        }
        total_time = float(inputs[i].total_time_min)
        sampling_interval = float(inputs[i].sampling_interval_min)
        timeline = params_to_timeline(predicted_params, total_time, sampling_interval)
        final_state = None
        if run_simulate:
            try:
                final_state = simulate(inputs[i]).final_state
            except Exception:
                final_state = None

        results.append(ModelPrediction(model_version=str(getattr(model, "name", "unknown")), predicted_params=predicted_params, timeline=timeline, final_state=final_state))

    return results[0] if single else results
