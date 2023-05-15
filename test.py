from openbabel import pybel
from rdkit import Chem

def correct_common_smiles_errors(smi: str) -> str:
    # no charge on tetravalent nitrogen
    if "[NH3]" in smi:
        smi = smi.replace('[NH3]', '[NH3+]')

    return smi

def correct_common_smiles_errors_rdkit_mol(mol: Chem.Mol) -> Chem.Mol:
    # charge on tetravalent nitrogen
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetExplicitValence() == 4:
            atom.SetFormalCharge(1)

    # charge on single-valent oxygen
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8 and atom.GetExplicitValence() == 1:
            atom.SetFormalCharge(-1)

    Chem.SanitizeMol(mol)
    return mol

xyz_string = """23
        -42.22074160
 N         -0.1533581637        0.9825804383        1.2739039051
 N         -0.9314672835        1.8721085000        0.6857694304
 N         -1.7844852926        1.2843725670       -0.1263217123
 O         -2.6469337238        1.9202577580       -0.8141829371
 C         -1.5643408264       -0.0715150082       -0.1096709587
 N         -2.1148123859       -1.0463058051       -0.8228164416
 C         -1.6126900798       -2.2411188846       -0.5950168831
 C         -0.6170689781       -2.5378299377        0.3524610930
 C         -0.0441163388       -1.5300207832        1.0959745405
 C         -0.5086394195       -0.2346414479        0.8308437824
 H         -2.0264777068       -3.0253583202       -1.2042225129
 H         -0.3223975079       -3.5661093430        0.4891505161
 H          0.6644444800       -1.7219814285        1.8903664078
 C          2.7396090389        1.1199412553        0.8102805998
 C          2.7792030617        0.3990428559       -0.4473799687
 O          2.7612448367        0.7561568062       -1.5778481698
 N          2.7922987702       -1.1447316980       -0.2304408461
 H          1.6705198241        1.1830332989        1.1343041867
 H          3.2861519240        0.6390553442        1.6129653403
 H          3.0680816907        2.1447567203        0.6678820797
 H          1.8974774735       -1.4308088947        0.2188311202
 H          3.5694814265       -1.4372851896        0.3787216036
 H          2.8717023863       -1.6122235109       -1.1466292738
"""

mol = pybel.readstring("xyz", xyz_string)
smi = mol.write(format="smi")
smi = smi.split()[0].strip()
smiles = smi.split('.')
for smi in smiles:
    smi = correct_common_smiles_errors(smi)
    mol = Chem.MolFromSmiles(smi)
    mol = correct_common_smiles_errors_rdkit_mol(mol)
    print(Chem.MolToSmiles(mol))