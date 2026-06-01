"""Reaction Path Sampler.

Sample reaction paths and locate transition states from reactant/product
SMILES using semi-empirical (xtb/CREST) and DFT (ORCA/PySCF) backends.

The package ``__init__`` is intentionally import-light: importing
``reaction_path_sampler`` does not pull in the heavy quantum-chemistry
dependencies (autodE, openbabel, ...). Import the concrete drivers from their
modules instead, e.g.::

    from reaction_path_sampler.base import ReactionSampler
    from reaction_path_sampler.reaction_path_sampler import ReactionPathSampler
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
