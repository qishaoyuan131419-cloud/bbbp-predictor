"""Load the saved Random Forest and expose one reusable prediction function.

This module is the bridge between the existing machine-learning code and the
web API. Importing it loads the saved model; it never trains a model.
"""

from pathlib import Path

import joblib
from rdkit.Chem import rdFingerprintGenerator

from bbbp_rf_descriptor import (
    MORGAN_FP_SIZE,
    MORGAN_RADIUS,
    molecule_summary,
    predict_bbb,
)


# Resolve the model relative to this file, so loading works from any directory.
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Saved BBBP model not found: {MODEL_PATH}")

# Load the fitted model once when the server starts, not once per request.
_artifact = joblib.load(MODEL_PATH)
_model = _artifact["model"]

# Use exactly the same Morgan fingerprint settings as the existing ML pipeline.
_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_FP_SIZE,
)


def predict_single_smiles(smiles: str) -> dict:
    """Run one SMILES through the existing BBBP prediction function.

    ``predict_bbb`` remains the single source of truth for feature generation,
    Random Forest prediction, and probability calculation.
    """
    prediction = predict_bbb(smiles, _model, _generator)
    return {**prediction, **molecule_summary(smiles)}
