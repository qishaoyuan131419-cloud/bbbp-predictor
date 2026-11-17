# SHAP Analysis.py
# Author: Shaoyuan Qi
# =========================
# Import libraries
# =========================

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import shap
import matplotlib.pyplot as plt


# =========================
# 1. Read BBBP dataset
# =========================

df = pd.read_csv("BBBP.csv")

smiles_list = df["smiles"]
labels = df["p_np"]


# =========================
# 2. Generate RDKit Descriptors
# =========================

X = []
y = []

descriptor_names = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
    "FractionCSP3"
]

for smiles, label in zip(smiles_list, labels):

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:

        desc = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.FractionCSP3(mol)
        ]

        X.append(desc)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================
# 3. Cross Validation
# =========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

auc_scores = []
acc_scores = []

fold = 1


# =========================
# 4. Cross Validation Loop
# =========================

for train_idx, test_idx in cv.split(X, y):

    print(f"\n========== Fold {fold} ==========")

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

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

    auc = roc_auc_score(y_test, y_prob)

    acc = accuracy_score(y_test, y_pred)

    print("ROC-AUC =", round(auc, 4))
    print("Accuracy =", round(acc, 4))

    auc_scores.append(auc)
    acc_scores.append(acc)

    fold += 1


# =========================
# 5. Final CV Results
# =========================

print("\n==============================")
print("Cross Validation Results")
print("==============================")

print("AUC Scores:", auc_scores)
print("Accuracy Scores:", acc_scores)

print("\nMean ROC-AUC:",
      round(np.mean(auc_scores), 4))

print("Std ROC-AUC:",
      round(np.std(auc_scores), 4))

print("\nMean Accuracy:",
      round(np.mean(acc_scores), 4))

print("Std Accuracy:",
      round(np.std(acc_scores), 4))


# =========================
# 6. Train Final Model
# =========================

final_model = RandomForestClassifier(

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

final_model.fit(X, y)


# =========================
# 7. SHAP Analysis
# =========================

print("\nRunning SHAP analysis...")

explainer = shap.TreeExplainer(final_model)

shap_values = explainer.shap_values(X)


# compatibility with different SHAP versions

if isinstance(shap_values, list):

    shap_class1 = shap_values[1]

else:

    shap_class1 = shap_values[:, :, 1]


# =========================
# 8. SHAP Importance Table
# =========================

mean_shap = np.abs(shap_class1).mean(axis=0)

importance_df = pd.DataFrame({
    "Descriptor": descriptor_names,
    "Mean_SHAP": mean_shap
})

importance_df = importance_df.sort_values(
    by="Mean_SHAP",
    ascending=False
)

print("\n==============================")
print("SHAP Feature Importance")
print("==============================")
print(importance_df)


# save csv

importance_df.to_csv(
    "SHAP_importance.csv",
    index=False
)

print("\nSaved: SHAP_importance.csv")


# =========================
# 9. SHAP Bar Plot
# =========================

plt.figure(figsize=(8, 5))

shap.summary_plot(
    shap_class1,
    X,
    feature_names=descriptor_names,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "SHAP_barplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: SHAP_barplot.png")


# =========================
# 10. SHAP Summary Plot
# =========================

shap.summary_plot(
    shap_class1,
    X,
    feature_names=descriptor_names,
    show=False
)

plt.savefig(
    "SHAP_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: SHAP_summary.png")


# =========================
# 11. Top 5 Descriptors
# =========================

print("\nTop 5 Most Important Descriptors:")

for i in range(5):

    print(
        f"{i+1}. "
        f"{importance_df.iloc[i]['Descriptor']} "
        f"(SHAP={importance_df.iloc[i]['Mean_SHAP']:.4f})"
    )