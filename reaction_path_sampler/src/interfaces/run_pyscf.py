import argparse

from typing import Literal, Optional

from pyscf import gto

SOLVENT_CONSTANT_DICT = {
    'Methanol': 32.613
}

def run_pscf(
    xyz_string: str,
    charge: int,
    spin: int,
    job: Literal["sp"] = "sp",
    solvent: Optional[str] = None,
    basis_set: str = "6-31G",
    xc_functional: str = "B3LYP",  
):  
    mol = gto.M(
        atom=xyz_string,
        basis=basis_set,
        verbose=0,
        charge=charge,
        spin=spin
    )

    mf = mol.RKS(xc=xc_functional)
    if solvent is not None:
        mf = mf.DDCOSMO()
        mf.with_solvent.eps = SOLVENT_CONSTANT_DICT[solvent]

    mf.run()
    energy = mf.e_tot
    return energy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xyz_file', type=str)
    parser.add_argument('--charge', type=int)
    parser.add_argument('--spin', type=int)
    parser.add_argument('--solvent', type=str)
    args = parser.parse_args()

    with open(args.xyz_file, 'r') as f:
        lines = f.readlines()

    energy = run_pscf(
        xyz_string="\n".join(lines[2:]),
        charge=args.charge,
        spin=args.spin,
        job="sp",
        solvent=args.solvent
    )

    with open('output.txt', 'w') as f:
        f.writelines(str(energy))
    