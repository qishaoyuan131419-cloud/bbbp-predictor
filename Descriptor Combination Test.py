import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score


df = pd.read_csv("BBBP.csv")

smiles_list = df["smiles"]
labels = df["p_np"]

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


descriptor_functions = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "FractionCSP3": Descriptors.FractionCSP3,
}


descriptor_combinations = {
    "baseline_no_descriptor": [],
    "MW_LogP": ["MolWt", "MolLogP"],
    "MW_LogP_TPSA": ["MolWt", "MolLogP", "TPSA"],
    "Hbond": ["NumHDonors", "NumHAcceptors"],
    "Flexibility": ["NumRotatableBonds", "FractionCSP3"],
    "Size_Shape": ["MolWt", "HeavyAtomCount", "RingCount"],
    "All_descriptors": list(descriptor_functions.keys()),
}


results = []

for combo_name, descriptor_names in descriptor_combinations.items():

    X = []
    y = []

    for smiles, label in zip(smiles_list, labels):

        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            fp = generator.GetFingerprintAsNumPy(mol)

            desc_values = np.array([
                descriptor_functions[name](mol)
                for name in descriptor_names
            ])

            feature = np.concatenate([fp, desc_values])

            X.append(feature)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    auc_scores = []
    acc_scores = []

    for train_idx, test_idx in cv.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc_scores.append(roc_auc_score(y_test, y_prob))
        acc_scores.append(accuracy_score(y_test, y_pred))

    results.append({
        "combination": combo_name,
        "descriptors_used": ", ".join(descriptor_names) if descriptor_names else "None",
        "num_descriptors": len(descriptor_names),
        "num_features": X.shape[1],
        "mean_auc": np.mean(auc_scores),
        "std_auc": np.std(auc_scores),
        "mean_accuracy": np.mean(acc_scores),
        "std_accuracy": np.std(acc_scores),
        "auc_scores": auc_scores,
        "accuracy_scores": acc_scores
    })


results_df = pd.DataFrame(results)

results_df.to_csv("descriptor_combination_results.csv", index=False)

print(results_df)
print("\nSaved to descriptor_combination_results.csv")