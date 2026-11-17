from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.fmcs import nonempty_powerset

molecules = {
    "ethanol": "CCO",
    "propanol": "CCCO",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O"
}

fps = {}

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

for name, mol in molecules.items():
    mol = Chem.MolFromSmiles(mol)
    fp = generator.GetFingerprint(mol)
    fps[name] = fp

best_pair = None
best_score = 0

names = list(fps.keys())

for i in range(len(names)):
    for j in range(i+1, len(names)):
        sim = DataStructs.TanimotoSimilarity(fps[names[i]], fps[names[j]])
        print(names[i], names[j], sim)

        if sim > best_score:
            best_pair = (names[i], names[j])
            best_score = sim

print("Most similar pair:", best_pair)
print("Similarity:", best_score)