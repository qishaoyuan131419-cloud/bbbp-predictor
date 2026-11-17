# BBBP Predictor

This project predicts blood-brain barrier permeability from a molecular SMILES
string. It uses RDKit molecular features and a scikit-learn Random Forest
classifier trained on the BBBP dataset.

The repository contains the original exploratory scripts, cross-validation and
SHAP analyses, and a small FastAPI application for interactive prediction.

## Web predictor

The web API loads the saved model in `bbbp_model_development/model.joblib`; it
does not retrain the model when the server starts.

Create an environment and install the API dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r bbbp_model_development\requirements.txt
```

Start the local server:

```powershell
cd bbbp_model_development
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs>, expand `POST /predict`, click **Try it
out**, and submit a SMILES string such as:

```json
{
  "smiles": "CCO"
}
```

The API also provides `GET /health` for a basic server health check.

## Model features

- 2,048-bit Morgan fingerprint with radius 2
- Molecular weight
- LogP
- Topological polar surface area
- Hydrogen-bond donors and acceptors
- Rotatable-bond count
- Ring count
- Heavy-atom count
- Fraction Csp3

## Project files

| Path | Purpose |
| --- | --- |
| `BBBP.csv` | BBB permeability dataset |
| `bbbp_cross_validation.py` | Random Forest cross-validation and evaluation |
| `bbbp_rf+descriptor.py` | Fingerprint and descriptor model experiment |
| `Descriptor Combination Test.py` | Descriptor-combination comparison |
| `SHAP Analysis.py` | SHAP feature-importance analysis |
| `bbbp_model_development/main.py` | FastAPI server |
| `bbbp_model_development/predictor.py` | Saved-model loading and prediction wrapper |
| `bbbp_model_development/model.joblib` | Fitted Random Forest artifact |

## Important limitation

This is a research and educational predictor, not a clinical decision tool.
Its probability is a Random Forest model score and should not be interpreted as
an experimentally validated probability for an individual compound. Predictions
are most reliable for molecules similar to the model's training domain.
