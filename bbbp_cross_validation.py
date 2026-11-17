
# =========================
# Import libraries
# =========================

# pandas:
# used to read CSV files and manipulate tables
import pandas as pd

# numpy:
# used for numerical arrays and matrix operations
import numpy as np


# RDKit chemistry toolkit
from rdkit import Chem

# used to generate Morgan fingerprints
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors


# Random Forest model
from sklearn.ensemble import RandomForestClassifier


# StratifiedKFold:
# used for stratified cross validation
from sklearn.model_selection import StratifiedKFold


# ROC-AUC evaluation metric
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


# =========================
# 1. Read BBBP dataset
# =========================

# read BBBP.csv into a pandas dataframe
df = pd.read_csv("BBBP.csv")


# extract SMILES strings
# each row represents one molecule
smiles_list = df["smiles"]


# extract labels
# 1 = BBB permeable
# 0 = non-permeable
labels = df["p_np"]



# =========================
# 2. Generate Morgan fingerprints
# =========================

# create Morgan fingerprint generator
#
# radius=2:
# look at atom neighborhoods within 2 bonds
#
# fpSize=2048:
# generate 2048-bit fingerprint vector
generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


# empty list for features
X = []

# empty list for labels
y = []


# iterate through all molecules
for smiles, label in zip(smiles_list, labels):

    # convert SMILES string into RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)


    # skip invalid molecules
    if mol is not None:
        fp = generator.GetFingerprintAsNumPy(mol)
        desc = np.array([
            Descriptors.MolWt(mol),  # molecular weight
            Descriptors.MolLogP(mol),  # lipophilicity
            Descriptors.TPSA(mol),  # polar surface area
            Descriptors.NumHDonors(mol),  # H-bond donors
            Descriptors.NumHAcceptors(mol),  # H-bond acceptors
            Descriptors.NumRotatableBonds(mol),  # flexibility
            Descriptors.RingCount(mol),  # number of rings
            Descriptors.HeavyAtomCount(mol),  # non-H atoms
            Descriptors.FractionCSP3(mol),  # 3D character
        ])

        # concatenate fingerprint + descriptors
        # feature = np.concatenate([fp, desc])
        feature = np.concatenate([fp, desc])
        X.append(desc)
        y.append(label)


# convert Python lists into numpy arrays
#
# X shape:
# (number_of_molecules, number_of_features)
#
# y shape:
# (number_of_molecules,)
X = np.array(X)
y = np.array(y)


# print feature matrix shape
print("X shape:", X.shape)

# print label vector shape
print("y shape:", y.shape)



# =========================
# 3. Define Cross Validation
# =========================

# StratifiedKFold:
#
# split dataset into k folds
#
# "stratified" means:
# preserve class ratio in each fold
#
# Example:
#
# original dataset:
# 76% positive
# 24% negative
#
# every fold will maintain similar ratio
cv = StratifiedKFold(

    # 5-fold cross validation
    #
    # dataset divided into 5 parts
    n_splits=5,

    # shuffle data before splitting
    shuffle=True,

    # fix random seed
    # ensures reproducibility
    # random_state=42
)



# =========================
# 4. Cross Validation Loop
# =========================

# empty list to store AUC scores
auc_scores = []
acc_scores = []

# fold counter
fold = 1


# cv.split(X, y):
#
# automatically generates:
#
# train indices
# test indices
#
# for each fold
for train_idx, test_idx in cv.split(X, y):

    print(f"\n========== Fold {fold} ==========")


    # =========================
    # Split train/test data
    # =========================

    # training features
    X_train = X[train_idx]

    # testing features
    X_test = X[test_idx]


    # training labels
    y_train = y[train_idx]

    # testing labels
    y_test = y[test_idx]



    # =========================
    # 5. Build Random Forest
    # =========================

    model = RandomForestClassifier(

        # number of trees in forest
        #
        # more trees:
        # lower variance
        # more stable
        n_estimators=300,


        # maximum tree depth
        #
        # prevents trees from growing infinitely
        max_depth=15,


        # minimum samples required to split a node
        #
        # prevents overfitting
        min_samples_split=5,


        # minimum samples required in leaf node
        #
        # avoids leaves containing only 1 sample
        min_samples_leaf=3,


        # random feature selection
        #
        # sqrt(number_of_features)
        #
        # key mechanism of Random Forest
        max_features="sqrt",


        # bootstrap sampling
        #
        # each tree trained on random sampled data
        bootstrap=True,


        # handle class imbalance
        #
        # increase weight for minority class
        class_weight="balanced",


        # fixed random seed
        # random_state=42,


        # use all CPU cores
        n_jobs=-1
    )



    # =========================
    # 6. Train model
    # =========================

    # fit model on training data
    #
    # internally:
    #
    # 1. bootstrap sample
    # 2. build decision trees
    # 3. aggregate forest
    model.fit(X_train, y_train)



    # =========================
    # 7. Predict probabilities
    # =========================

    # predict probability of class 1
    #
    # [:,1]:
    # select probability for positive class
    y_prob = model.predict_proba(X_test)[:, 1]

    # predict class label
    y_pred = model.predict(X_test)


    # =========================
    # 8. Evaluate ROC-AUC
    # =========================

    # ROC-AUC measures:
    #
    # ranking ability of classifier
    #
    # 1.0 = perfect
    # 0.5 = random guessing
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    # print AUC fold result
    print("Fold ROC-AUC:", auc)

    # print Accuracy fold result
    print("Fold Accuracy:", acc)

    # store score
    auc_scores.append(auc)
    acc_scores.append(acc)


    # increment fold number
    fold += 1



# =========================
# 9. Final Results
# =========================

print("\n==============================")
print("Cross Validation Results")
print("==============================")


# print all fold scores
print("AUC scores:", auc_scores)
print("Accuracy", acc_scores)

# calculate mean performance
print("Mean ROC-AUC:", np.mean(auc_scores))
print("Mean Accuracy:", np.mean(acc_scores))

# calculate standard deviation
#
# lower std:
# more stable model
print("Std ROC-AUC:", np.std(auc_scores))
print("Std Accuracy:", np.std(acc_scores))
# =========================
# 10. SHAP Analysis
# =========================

import shap
import matplotlib.pyplot as plt


# descriptor names
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


# =========================
# Train final model
# =========================

final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features="sqrt",
    bootstrap=True,
    class_weight="balanced",
    n_jobs=-1
)

final_model.fit(X, y)


# =========================
# SHAP Explainer
# =========================

explainer = shap.TreeExplainer(final_model)

shap_values = explainer.shap_values(X)


# =========================
# Feature Importance Table
# =========================

# Different SHAP versions return different formats
if isinstance(shap_values, list):

    shap_class1 = shap_values[1]

else:

    # newer SHAP versions
    shap_class1 = shap_values[:, :, 1]


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


# =========================
# SHAP Bar Plot
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

plt.show()


# =========================
# SHAP Summary Plot
# =========================

shap.summary_plot(
    shap_class1,
    X,
    feature_names=descriptor_names
)