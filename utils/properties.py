from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = {
        "Molecular Weight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "Number of H-Bond Donors": Descriptors.NumHDonors(mol),
        "Number of H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
    }
    return properties

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return Draw.MolToImage(mol)
