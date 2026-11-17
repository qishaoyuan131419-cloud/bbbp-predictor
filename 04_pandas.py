import pandas as pd

from rdkit import Chem


df = pd.read_csv("BBBP.csv")

smiles = df.loc[0, "smiles"]

print(smiles)

mol = Chem.MolFromSmiles(smiles)

print(mol)