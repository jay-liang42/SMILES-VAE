from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity




# -----------------------------
# Strict validity check
# -----------------------------
def is_valid_smiles_strict(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False




# -----------------------------
# Morgan fingerprint generator
# r=2, numbits=2048
# -----------------------------
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)




def morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _morgan_gen.GetFingerprint(mol)




# -----------------------------
# Similarity metric
# -----------------------------
def smiles_similarity(s1: str, s2: str) -> float:
    """
    Returns:
        0.0 if either SMILES invalid
        otherwise Tanimoto similarity between fingerprints
    """


    if not is_valid_smiles_strict(s1):
        return 0.0
    if not is_valid_smiles_strict(s2):
        return 0.0


    fp1 = morgan_fp(s1)
    fp2 = morgan_fp(s2)


    if fp1 is None or fp2 is None:
        return 0.0


    return float(TanimotoSimilarity(fp1, fp2))
