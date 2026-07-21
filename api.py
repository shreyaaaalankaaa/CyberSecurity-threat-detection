from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from src.data import FEATURE_COLUMNS
from src.inference import predict_record
from src.modeling import load_model

app = FastAPI(title="Network Threat Detection API", version="1.0.0")
MODEL_PATH = Path("models/best_model.joblib")

class TrafficRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    duration: float
    protocol_type: str
    service: str
    flag: str
    src_bytes: float
    dst_bytes: float
    wrong_fragment: int
    urgent: int
    hot: int
    num_failed_logins: int
    logged_in: int
    count: int
    srv_count: int
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    dst_host_count: int
    dst_host_srv_count: int

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}

@app.get("/model")
def model_info() -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model not trained. Run python train.py first.")
    artifact = load_model(MODEL_PATH)
    return {"model_name": artifact["model_name"], "metrics": artifact["metrics"], "features": FEATURE_COLUMNS}

@app.post("/predict")
def predict(payload: TrafficRecord) -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model not trained. Run python train.py first.")
    try:
        return predict_record(load_model(MODEL_PATH)["pipeline"], payload.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
