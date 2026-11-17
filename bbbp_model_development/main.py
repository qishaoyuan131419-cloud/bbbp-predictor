"""Local web interface and REST API for the BBBP prediction pipeline."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from predictor import predict_single_smiles


# FastAPI creates the HTTP server application and its automatic Swagger docs.
app = FastAPI(
    title="BBBP Prediction API",
    description="Predict BBB permeability from one SMILES string.",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionRequest(BaseModel):
    """JSON body accepted by POST /predict."""

    smiles: str


class PredictionResponse(BaseModel):
    """Structured JSON returned to the API client."""

    smiles: str
    prediction: int
    label: str
    probability: float
    confidence: float
    canonical_smiles: str
    molecular_formula: str
    descriptors: dict[str, float]
    structure_image: str


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the small browser interface."""
    return FileResponse(STATIC_DIR / "index.html")


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
        confidence=result["confidence"],
        canonical_smiles=result["canonical_smiles"],
        molecular_formula=result["molecular_formula"],
        descriptors=result["descriptors"],
        structure_image=result["structure_image"],
    )
