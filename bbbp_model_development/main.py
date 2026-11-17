"""Minimal local REST API for the existing BBBP prediction pipeline."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predictor import predict_single_smiles


# FastAPI creates the HTTP server application and its automatic Swagger docs.
app = FastAPI(
    title="BBBP Prediction API",
    description="Predict BBB permeability from one SMILES string.",
)


class PredictionRequest(BaseModel):
    """JSON body accepted by POST /predict."""

    smiles: str


class PredictionResponse(BaseModel):
    """Structured JSON returned to the API client."""

    smiles: str
    prediction: int
    label: str
    probability: float


@app.get("/health")
def health() -> dict[str, str]:
    """Let a client check that the local server is running."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Accept JSON, call the existing ML pipeline, and return JSON.

    HTTP 400 means the client supplied a SMILES string RDKit cannot parse.
    Catching the error here prevents one bad request from crashing the server.
    """
    try:
        result = predict_single_smiles(request.smiles)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid SMILES") from None

    prediction = result["prediction"]
    return PredictionResponse(
        smiles=request.smiles,
        prediction=prediction,
        label="BBB permeable" if prediction == 1 else "BBB non-permeable",
        probability=result["probability_permeable"],
    )
