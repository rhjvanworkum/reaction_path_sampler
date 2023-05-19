# reaction_path_sampler


### installation instructions
- install xtb + crest
- openbabel for linux
- clone geodesic_interpolate & python setup.py install
- pip install autode, pysisyphus, etc.
- conda install -c conda-forge openbabel

add xtb to path (for pysisyphus)


### notes on functionality
1. we specify the reactant & product reaction SMILES
2. we find a mapping between reactants & products atoms/graph
    - Currently we do this by computing RMSD's over different graph isomorphisms
    - In the future we can also imagine doing this by adjusting the bonding matrix of a reactant
    , optimizing with FF & checking if this SMILES corresponds to the product SMILES. Then we also have
    reactant & product conformer with same atom ordering
3. we need to sample different conformers of reactants/product complexes
    - we can just do metadynamics on both products & complexes
    - we can do metadynamics on products + BM-assisted reactant sampling
    - we can do autodE on products + BM-asssited reactant sampling
4. we need to select a set of most promising pairs to find reaction paths between
5. we need to optimize the reaction path & do TS optimization followed by IRC
    - geodesic interpolation + NEB-CI
    - RSMD-PP using metadynamics
    - GSM using pyGSM?
    