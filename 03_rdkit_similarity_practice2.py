# 练习：做一个 mini molecule analyzer
# 目标
#
# 输入一个 molecule：
#
# 输出：
#
# 基本信息
# atom 信息
# bond 信息
# descriptors
# fingerprint 信息
from rdkit import Chem, DataStructs
from rdkit.Chem import rdChemDraw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.fmcs import nonempty_powerset
from rdkit.Chem import rdMolDescriptors

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

mol = Chem.MolFromSmiles(smiles)


# =========================
# 1. Basic Information
# =========================

print("===Basic Information===")
print("Canonical SMILES:", Chem.MolToSmiles(mol))
print("Atoms:", mol.GetNumAtoms())
print("Bonds", mol.GetNumBonds())

# =========================
# 2. Atoms
# =========================
print("===2. Atoms===")
for atom in mol.GetAtoms():
    Idx = atom.GetIdx()
    Symbol = atom.GetSymbol()
    AtomicNum = atom.GetAtomicNum()
    Degree = atom.GetDegree()
    Aromatic = atom.GetIsAromatic()
    print(f"{Symbol}{Idx}: {Aromatic} {AtomicNum}")

