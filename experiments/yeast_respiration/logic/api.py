from __future__ import annotations
import os
import uuid
from typing import List, Optional, Union, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from experiments.yeast_respiration.logic.types import (
    ExperimentInputs,
    SimulationResult,
    FinalState,
    TrainingRow,
    ModelPrediction,
    ModelTrainingResult,
)
from experiments.yeast_respiration.logic.sim import simulate, simulate_at_time, generate_teacher_dataset
from experiments.yeast_respiration.logic.ai import train_model, load_model, predict, load_scalers

app = FastAPI(title="Yeast Respiration API", description="Thin HTTP wrapper for simulate / train / predict", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOB_STORE: Dict[str, Dict[str, Any]] = {}

class SimulateRequest(BaseModel):
    inputs: ExperimentInputs

class SimulateAtTimeRequest(BaseModel):
    inputs: ExperimentInputs
    t_min: float

class TrainRequest(BaseModel):
    n_samples: Optional[int] = 64
    rows: Optional[List[TrainingRow]] = None
    model_config_overrides: Optional[dict] = None
    save_model: bool = True
    seed: Optional[int] = None
    run_in_background: bool = False

class PredictRequest(BaseModel):
    inputs: Union[ExperimentInputs, List[ExperimentInputs]]
    model_path: Optional[str] = None
    run_simulate: Optional[bool] = False


@app.post("/simulate", response_model=SimulationResult)
def api_simulate(req: SimulateRequest):
    try:
        return simulate(req.inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {e}")


@app.post("/simulate_at_time", response_model=FinalState)
def api_simulate_at_time(req: SimulateAtTimeRequest):
    try:
        return simulate_at_time(req.inputs, req.t_min)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"simulate_at_time error: {e}")


def _background_train(job_id: str, rows: List[TrainingRow], cfg: Optional[dict], save_model: bool):
    JOB_STORE[job_id] = {"status": "running"}
    try:
        result = train_model(rows, cfg=cfg, save_model=save_model)
        out = result.dict()
        if result.meta:
            out["meta"] = {k: "<scaler>" for k in result.meta}
        JOB_STORE[job_id] = {"status": "finished", "result": out}
    except Exception as e:
        JOB_STORE[job_id] = {"status": "error", "error": str(e)}


@app.post("/train", response_model=Union[ModelTrainingResult, Dict[str, str]])
def api_train(req: TrainRequest, background_tasks: BackgroundTasks):
    try:
        rows = req.rows or generate_teacher_dataset(req.n_samples or 64, seed=req.seed)
        cfg = req.model_config_overrides or None

        if req.run_in_background:
            job_id = str(uuid.uuid4())
            JOB_STORE[job_id] = {"status": "queued"}
            background_tasks.add_task(_background_train, job_id, rows, cfg, req.save_model)
            return {"job_id": job_id}

        result = train_model(rows, cfg=cfg, save_model=req.save_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {e}")


@app.get("/train/status/{job_id}")
def train_status(job_id: str):
    rec = JOB_STORE.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    return rec


@app.post("/predict", response_model=Union[ModelPrediction, List[ModelPrediction]])
def api_predict(req: PredictRequest):
    try:
        model = load_model(req.model_path) if req.model_path else load_model()

        x_s = None
        y_s = None
        model_path = req.model_path
        if model_path and os.path.exists(model_path + "_x_min.npy"):
            try:
                x_s, y_s = load_scalers(model_path)
            except Exception:
                pass

        preds = predict(
            model,
            req.inputs,
            x_scaler=x_s,
            y_scaler=y_s,
            cfg=None,
            run_simulate=req.run_simulate,
        )
        return preds

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_config_path": os.path.join(os.path.dirname(__file__), "config", "model_config.json"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("experiments.yeast_respiration.api:app", host="127.0.0.1", port=8001, reload=True)
