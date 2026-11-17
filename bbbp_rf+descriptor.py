import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# =========================
# 1. 读取数据
# =========================

df = pd.read_csv("BBBP.csv") #读取BBBP

print(df.head()) #检查前五行
print(df.shape) #显示多少行和列


# =========================
# 2. 提取 SMILES 和 label
# =========================

smiles_list = df["smiles"] #提取SMILES
labels = df["p_np"] #提取labels


# =========================
# 3. SMILES -> Morgan Fingerprint + Descriptors
# =========================

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# radius = 2查看每个原子周围两个邻居
# fpSize = 2048 每个分子被转化成为2048位特征

X = []
y = []

# x存分子特征
# y存标签

for smiles, label in zip(smiles_list, labels):
    # 遍历每个分子

    mol = Chem.MolFromSmiles(smiles)
    # 把smiles转化成rdkit分子对象

    if mol is not None:
        # Morgan fingerprint
        fp = generator.GetFingerprintAsNumPy(mol)

        # RDKit descriptors
        desc = np.array([
            Descriptors.MolWt(mol),              # molecular weight
            Descriptors.MolLogP(mol),            # lipophilicity
            Descriptors.TPSA(mol),               # polar surface area
            Descriptors.NumHDonors(mol),         # H-bond donors
            Descriptors.NumHAcceptors(mol),      # H-bond acceptors
            Descriptors.NumRotatableBonds(mol),  # flexibility
            Descriptors.RingCount(mol),          # number of rings
            Descriptors.HeavyAtomCount(mol),     # non-H atoms
            Descriptors.FractionCSP3(mol),       # 3D character
        ])

        # concatenate fingerprint + descriptors
        # feature = np.concatenate([fp, desc])
        feature = np.concatenate([fp, desc])
        X.append(feature)
        y.append(label)


#转化成numpy数组
X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================
# 4. 划分训练/测试集
# =========================

#划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #百分之二十作为test集
    # random_state=42, #固定随机种子
    stratify=y #数据分层，保证训练和测试的0/1比例一致
)


# =========================
# 5. 建立 Random Forest
# =========================

model = RandomForestClassifier(
    n_estimators=200, #200决策树
    max_features="sqrt",
    max_depth=None, #可以一直生长
    # random_state=42, #固定随机树
    n_jobs=-1 #使用所有CPU
)

model.fit(X_train, y_train)


# =========================
# 6. 预测
# =========================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# =========================
# 7. 评估模型
# =========================

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nAccuracy:", accuracy)
print("ROC-AUC:", auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))