from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# 1. SMILES -> Mol object
smiles = "OCC"  # ethanol
mol = Chem.MolFromSmiles(smiles)

print("Mol object:", mol)
print("Canonical SMILES:", Chem.MolToSmiles(mol))

# 2. 基本分子信息
print("Atoms:", mol.GetNumAtoms())
print("Bonds:", mol.GetNumBonds())
print("Molecular weight:", Descriptors.MolWt(mol))
print("LogP:", Descriptors.MolLogP(mol))

# 3. Morgan fingerprint
fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
fp_list = list(fp)

print("Fingerprint length:", len(fp_list))
print("First 20 bits:", fp_list[:20])

# 4. 画分子
img = Draw.MolToImage(mol)
img.save("ethanol.png")

print("Saved molecule image as ethanol.png")