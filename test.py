from autode.solvent.solvents import solvents

def get_orca_solv(solvent):
    for solv in solvents:
        if solv.xtb == solvent:
            return solv.orca


print(get_orca_solv("CH2Cl2"))