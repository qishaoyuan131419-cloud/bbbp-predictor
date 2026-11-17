# BBBP Predictor

A small web application that predicts whether a molecule can cross the blood-brain barrier from a SMILES string.

The result includes the predicted class, BBB permeability score, confidence, molecular structure, formula, and basic molecular descriptors.

## Run locally

```bash
git clone https://github.com/qishaoyuan131419-cloud/bbbp-predictor.git
cd bbbp-predictor
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages and start the server:

```bash
pip install -r bbbp_model_development/requirements.txt
cd bbbp_model_development
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), enter a SMILES string, and select **Predict BBB permeability**.
