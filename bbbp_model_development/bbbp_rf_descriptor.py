"""BBBP Random Forest using Morgan fingerprints and RDKit descriptors.

This is a reproducible development copy based on the original
``bbbp_rf+descriptor.py``. The original file is intentionally unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
MORGAN_RADIUS = 2
MORGAN_FP_SIZE = 2048

# The dataset remains in the project root, one level above this file.
DATA_PATH = Path(__file__).resolve().parent.parent / "BBBP.csv"


def molecule_features(smiles: str, generator) -> np.ndarray:
    """Create a 2048-bit Morgan fingerprint plus nine descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    morgan_fp = generator.GetFingerprintAsNumPy(mol)
    descriptors = np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.FractionCSP3(mol),
        ],
        dtype=float,
    )
    return np.concatenate([morgan_fp, descriptors])


def train_model():
    """Train the model and return it together with holdout metrics."""
    df = pd.read_csv(DATA_PATH)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_FP_SIZE,
    )

    features = []
    labels = []
    skipped = 0

    for smiles, label in zip(df["smiles"], df["p_np"]):
        try:
            features.append(molecule_features(smiles, generator))
            labels.append(label)
        except ValueError:
            skipped += 1

    X = np.asarray(features)
    y = np.asarray(labels)

    print(f"Dataset: {DATA_PATH}")
    print(f"X shape: {X.shape}")  # 2048 Morgan bits + 9 descriptors = 2057
    print(f"y shape: {y.shape}")
    print(f"Invalid molecules skipped: {skipped}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, generator, accuracy, roc_auc


def predict_bbb(smiles: str, model, generator) -> dict:
    """Predict whether one SMILES can cross the blood-brain barrier."""
    features = molecule_features(smiles.strip(), generator).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    prediction = int(model.predict(features)[0])

    permeable_probability = float(probabilities[1])
    confidence = float(probabilities[prediction])
    return {
        "prediction": prediction,
        "can_cross_bbb": prediction == 1,
        "probability_permeable": permeable_probability,
        "confidence": confidence,
    }


def main() -> None:
    model, generator, accuracy, roc_auc = train_model()

    print("\n" + "=" * 60)
    print("BBBP interactive prediction")
    print("Enter a SMILES string, or press Enter to exit.")

    while True:
        smiles = input("\nSMILES: ").strip()
        if not smiles:
            print("Prediction finished.")
            break

        try:
            result = predict_bbb(smiles, model, generator)
        except ValueError as exc:
            print(f"Error: {exc}")
            continue

        answer = "Yes (BBB permeable)" if result["can_cross_bbb"] else "No (BBB non-permeable)"
        print(f"Can cross BBB: {answer}")
        print(f"Probability of BBB permeability: {result['probability_permeable']:.2%}")
        print(f"Prediction confidence: {result['confidence']:.2%}")
        print(f"Model test accuracy: {accuracy:.2%}")
        print(f"Model test ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
