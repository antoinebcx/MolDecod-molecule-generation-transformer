from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "Number of H-Bond Donors": Descriptors.NumHDonors(mol),
        "Number of H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
        "Bertz Complexity Index": round(Descriptors.BertzCT(mol), 2),
        "Hall-Kier Alpha Value": round(Descriptors.HallKierAlpha(mol), 4),
        "Kappa1 Shape Index": round(Descriptors.Kappa1(mol), 4),
        "Kappa2 Shape Index": round(Descriptors.Kappa2(mol), 4),
        "Kappa3 Shape Index": round(Descriptors.Kappa3(mol), 4),
        "Balaban J Index": round(Descriptors.BalabanJ(mol), 4),
        "Molar Refractivity": round(Descriptors.MolMR(mol), 2),
        "Maximum Partial Charge": round(Descriptors.MaxPartialCharge(mol), 4),
        "Minimum Partial Charge": round(Descriptors.MinPartialCharge(mol), 4),
        "Topological Polar Surface Area": round(Descriptors.TPSA(mol), 2),
        "Approximate Surface Area": round(Descriptors.LabuteASA(mol), 2),
        "Number of Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "Number of Saturated Rings": Descriptors.NumSaturatedRings(mol),
        "Number of Aromatic Rings": Descriptors.NumAromaticRings(mol)
    }
    return properties

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return Draw.MolToImage(mol)
