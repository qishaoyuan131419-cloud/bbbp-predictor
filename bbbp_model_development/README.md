# BBBP Model Development

This folder contains the reproducible development copy of the original
`bbbp_rf+descriptor.py`. The original file in the project root is unchanged.

Features used by the model:

- 2048-bit Morgan fingerprint (`radius=2`)
- 9 RDKit molecular descriptors
- 2057 total input features

Both the train/test split and Random Forest use `random_state=42`.

Run from the project root:

```powershell
python bbbp_model_development/bbbp_rf_descriptor.py
```

After training, enter one SMILES string at the prompt. The program returns:

- whether the molecule is predicted to cross the BBB
- predicted probability of BBB permeability
- confidence in the predicted class
- holdout test-set Accuracy and ROC-AUC

Press Enter on an empty prompt to exit.

The probability/confidence is a Random Forest model score. Accuracy is an
overall holdout test-set metric and cannot be calculated for a single unknown
molecule without its true label.

## Minimal FastAPI application

`model.joblib` contains the already-fitted Random Forest. Starting the API does
not read the training CSV and does not retrain the model. `predictor.py` loads
that model and calls the existing `predict_bbb()` function.

From this folder, install the dependencies and start the local server:

```powershell
pip install -r requirements.txt
uvicorn main:app --reload
```

The command means:

- `main` is the Python file `main.py`.
- `app` is the FastAPI server object inside that file.
- `127.0.0.1` is this computer's local IP address.
- `8000` is the network port used by the server.
- `--reload` restarts the development server when code changes.

Open `http://127.0.0.1:8000/docs` in a browser. Swagger UI acts as the client:
it sends an HTTP request containing JSON to the FastAPI server and displays the
JSON response.

Example request:

```json
{
  "smiles": "CCO"
}
```

Available endpoints:

- `GET /health` checks whether the server is running.
- `POST /predict` predicts one SMILES string.
