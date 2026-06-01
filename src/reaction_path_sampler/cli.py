"""Command-line entry points for Reaction Path Sampler.

Two drivers are exposed as console scripts (see ``[project.scripts]`` in
``pyproject.toml``):

* ``search-rxn-path``               -> :func:`search_rxn_path`
* ``search-rxn-path-from-template`` -> :func:`search_rxn_path_from_template`

Both read a single YAML settings file given as the only positional argument.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import yaml

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configure_logging() -> None:
    """Configure root logging from the ``RPS_LOG_LEVEL`` environment variable.

    Defaults to ``INFO`` when the variable is unset or holds an unrecognised
    value (the previous implementation raised ``KeyError``/``UnboundLocalError``
    in those cases).
    """
    level_name = os.environ.get("RPS_LOG_LEVEL", "INFO").upper()
    level = _LOG_LEVELS.get(level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _parse_settings(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse the CLI arguments, configure logging, and load the YAML settings."""
    parser = argparse.ArgumentParser(
        description="Run a Reaction Path Sampler workflow from a YAML settings file."
    )
    parser.add_argument(
        "settings_file_path",
        help="Path to the YAML file containing the run settings",
        type=str,
    )
    args = parser.parse_args(argv)
    configure_logging()
    with open(args.settings_file_path) as f:
        return yaml.load(f, Loader=yaml.Loader)


def search_rxn_path(argv: list[str] | None = None) -> None:
    """Search for a reaction path from reactant/product SMILES."""
    settings = _parse_settings(argv)

    from reaction_path_sampler.molecular_system import MolecularSystem, Reaction
    from reaction_path_sampler.reaction_path_sampler import ReactionPathSampler

    reactants = MolecularSystem.from_smiles(".".join(settings["reactant_smiles"]))
    products = MolecularSystem.from_smiles(".".join(settings["product_smiles"]))
    reaction = Reaction(reactants, products, settings["solvent"])
    reaction.map_reaction(n_workers=int(settings["n_processes"] * settings["xtb_n_cores"]))

    reaction_path_sampler = ReactionPathSampler(settings, reaction)
    rc_conformers, pc_conformers = reaction_path_sampler.sample_reaction_complex_conformers()
    conformer_pairs = reaction_path_sampler.select_promising_reactant_product_pairs(
        rc_conformers, pc_conformers, products.charge
    )

    for idx, conformer_pair in enumerate(conformer_pairs):
        print(f"Working on Reactant-Product Complex pair {idx}")

        output_dir = os.path.join(settings["output_dir"], f"{idx}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        success = reaction_path_sampler.find_reaction_path(
            rc_conformer=conformer_pair[0],
            pc_conformer=conformer_pair[1],
            output_dir=output_dir,
            final_dir=settings["output_dir"],
        )

        if success:
            return


def search_rxn_path_from_template(argv: list[str] | None = None) -> None:
    """Search for a reaction path using precomputed transition-state templates."""
    settings = _parse_settings(argv)

    from autode.conformers.conformer import Conformer
    from autode.conformers.conformers import atoms_from_rdkit_mol

    from reaction_path_sampler.template_sampler import TemplateSampler
    from reaction_path_sampler.ts_template import get_ts_templates

    template_reaction_sampler = TemplateSampler(settings)
    template_reaction_sampler.generate_reaction_complexes()

    ts_templates = get_ts_templates(folder_path=settings["ts_template_dir"])
    template_reaction_sampler.select_and_load_ts_template(ts_templates=ts_templates)
    n_guesses = template_reaction_sampler.embed_ts_guesses()

    for idx in range(n_guesses):
        print(f"Working on TS guess geometry {idx}")

        output_dir = os.path.join(settings["output_dir"], f"{idx}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        complex = [
            template_reaction_sampler.pc_complex,
            template_reaction_sampler.rc_complex,
        ][template_reaction_sampler.isomorphism_idx]
        atoms = atoms_from_rdkit_mol(complex.rdkit_mol_obj, idx)
        ts_guess = Conformer(atoms=atoms)

        success = template_reaction_sampler.optimize_ts_guess(
            ts_guess=ts_guess,
            output_dir=output_dir,
            final_dir=settings["output_dir"],
        )

        if success:
            return
