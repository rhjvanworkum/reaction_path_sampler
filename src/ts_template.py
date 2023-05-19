"""
Fork taken from autodE: https://github.com/duartegroup/autodE

Changes include:
- TStemplate now also saves cartesian coords of TS
"""

import os
import autode
import autode as ade
from datetime import date
from autode.mol_graphs import MolecularGraph
from autode.species.complex import Complex
from autode.bond_rearrangement import BondRearrangement
from autode.config import Config
from autode.log import logger
from autode.mol_graphs import is_isomorphic
from autode.exceptions import TemplateLoadingFailed
from autode.solvent.solvents import get_solvent
from autode.mol_graphs import (
    get_mapping_ts_template,
    get_truncated_active_mol_graph,
)


from autode.transition_states.templates import get_ts_template_folder_path, get_value_from_file, get_values_dict_from_file
import numpy as np
import networkx as nx

from src.xyz2mol import read_xyz_string


class TStemplate:
    def __init__(
        self,
        graph=None,
        charge=None,
        mult=None,
        solvent=None,
        species=None,
        filename=None,
    ):
        """
        TS template

        -----------------------------------------------------------------------
        Keyword Arguments:
            graph (nx.Graph): Active bonds in the TS are represented by the
                  edges with attribute active=True, going out to nearest bonded
                  neighbours

            solvent (autode.solvent.solvents.Solvent):

            charge (int):

            mult (int):

            species (autode.species.Species):

            filename (str): Saved template to load
        """

        self._filename = filename
        self.graph = graph
        self.solvent = solvent
        self.charge = charge
        self.mult = mult

        if species is not None:
            self.solvent = species.solvent
            self.charge = species.charge
            self.mult = species.mult

        if self._filename is not None:
            self.load(filename)

    def _save_to_file(self, file):
        """Save this template to a plain text .txt file with a ~yaml syntax"""

        title_line = (
            f"TS template generated by autode v.{autode.__version__}"
            f" on {date.today()}\n"
        )

        # Add nodes as a list, and their atom labels/symbols
        nodes_str = ""
        for i, data in self.graph.nodes(data=True):
            node_str = f'    {i}: atom_label={data["atom_label"]} '
            
            if "cartesian" in data.keys():
                for i in range(3):
                    node_str += f'cartesian_{i}={data["cartesian"][i]} '

            nodes_str += f"{node_str} \n"

        # Add edges as a list and their associated properties as a dict
        edges_str = ""
        for i, j, data in self.graph.edges(data=True):
            edge_str = f"    {i}-{j}: "

            if "pi" in data.keys():
                edge_str += f'pi={str(data["pi"])} '

            if "active" in data.keys():
                edge_str += f'active={str(data["active"])} '

            if "distance" in data.keys():
                edge_str += f'distance={data["distance"]:.4f} '

            edges_str += f"{edge_str}\n"

        print(
            title_line,
            f"solvent: {self.solvent}",
            f"charge: {self.charge}",
            f"multiplicity: {self.mult}",
            "nodes:",
            nodes_str,
            "edges:",
            edges_str,
            sep="\n",
            file=file,
        )

        return None

    def graph_has_correct_structure(self):
        """Check that the graph has some active edges and distances"""

        if self.graph is None:
            logger.warning("Incorrect TS template stricture - it was None!")
            return False

        n_active_edges = 0
        for edge in self.graph.edges:

            if "active" not in self.graph.edges[edge].keys():
                continue

            if not self.graph.edges[edge]["active"]:
                continue

            if (
                self.graph.edges[edge]["active"]
                and "distance" not in self.graph.edges[edge].keys()
            ):
                logger.warning("Active edge has no distance")
                return False

            n_active_edges += 1

        # A reasonably structured graph has at least 1 active edge
        if n_active_edges >= 1:
            return True

        else:
            logger.warning("Graph had no active edges")
            return False

    def save(self, basename="template", folder_path=None):
        """
        Save the TS template object in a plain text .txt file. With folder_path
        =None then the template will be saved to the default directory
        (see get_ts_template_folder_path). The name of the file will be
        basename.txt where i is an integer iterated until the file doesn't
        already exist.

        -----------------------------------------------------------------------
        Keyword Arguments:
            basename (str):

            folder_path (str or None):
        """

        folder_path = get_ts_template_folder_path(folder_path)
        logger.info(f"Saving TS template to {folder_path}")

        if not os.path.exists(folder_path):
            logger.info(f"Making directory {folder_path}")
            os.mkdir(folder_path)

        # Iterate i until the templatei.obj file doesn't exist
        name, i = basename + "0", 0
        while True:
            if not os.path.exists(os.path.join(folder_path, f"{name}.txt")):
                break
            name = basename + str(i)
            i += 1

        file_path = os.path.join(folder_path, f"{name}.txt")
        logger.info(f"Saving the template as {file_path}")

        with open(file_path, "w") as template_file:
            self._save_to_file(template_file)

        return None

    def load(self, filename):
        """
        Load a template from a saved file

        -----------------------------------------------------------------------
        Arguments:
            filename (str):

        Raise:
            (autode.exceptions.TemplateLoadingFailed):
        """
        try:
            template_lines = open(filename, "r").readlines()
        except (IOError, UnicodeDecodeError):
            raise TemplateLoadingFailed("Failed to read file lines")

        if len(template_lines) < 5:
            raise TemplateLoadingFailed("Not enough lines in the template")

        name = get_value_from_file("solvent", template_lines)

        if name.lower() == "none":
            self.solvent = None
        else:
            self.solvent = get_solvent(solvent_name=name, kind="implicit")

        self.charge = int(get_value_from_file("charge", template_lines))
        self.mult = int(get_value_from_file("multiplicity", template_lines))

        # Set the template graph by adding nodes and edges with atoms labels
        # and active/pi/distance attributes respectively
        self.graph = MolecularGraph()

        nodes = get_values_dict_from_file("nodes", template_lines)
        for idx, data in nodes.items():
            if 'cartesian_0' in data.keys():
                data['cartesian'] = np.array([data['cartesian_0'], data['cartesian_1'], data['cartesian_2']])
                del data['cartesian_0']
                del data['cartesian_1']
                del data['cartesian_2']
            self.graph.add_node(idx, **data)

        edges = get_values_dict_from_file("edges", template_lines)

        for pair, data in edges.items():
            self.graph.add_edge(*pair, **data)

        if not self.graph_has_correct_structure():
            raise TemplateLoadingFailed("Incorrect graph structure")

        return None

    @property
    def filename(self) -> str:
        return "unknown" if self._filename is None else self._filename
    


def get_ts_templates(folder_path=None):
    """Get all the transition state templates from a folder, or the default if
    folder path is None. Transition state templates should be .txt files with
    at least a charge, multiplicity, solvent, and a graph with some active
    edge including distances.

    ---------------------------------------------------------------------------
    Keyword Arguments:
        folder_path (str): e.g. '/path/to/the/ts/template/library'

    Returns:
        (list(autode.transition_states.templates.TStemplate)): List of
        templates
    """
    folder_path = get_ts_template_folder_path(folder_path)
    logger.info(f"Getting TS templates from {folder_path}")

    if not os.path.exists(folder_path):
        logger.error("Folder does not exist")
        return []

    templates = []

    # Attempt to form transition state templates for all the .txt files in the
    # TS template folder
    for filename in os.listdir(folder_path):

        if not filename.endswith(".txt"):
            continue

        try:
            template = TStemplate(filename=os.path.join(folder_path, filename))
            templates.append(template)

        except TemplateLoadingFailed:
            logger.warning(f"Failed to load a template for {filename}")

    logger.info(f"Have {len(templates)} TS templates")
    return templates


def save_ts_template(
    tsopt: str,
    complex: ade.Species,
    bond_rearr: BondRearrangement,
    output_dir: str
) -> TStemplate:
    _, coords = read_xyz_string(tsopt)
    complex.coordinates = np.array(coords)

    for bond in bond_rearr.all:
        complex.graph.add_active_edge(*bond)
    
    truncated_graph = get_truncated_active_mol_graph(graph=complex.graph, active_bonds=bond_rearr.all)
    
    # bonds
    for bond in bond_rearr.all:
        truncated_graph.edges[bond]["distance"] = complex.distance(*bond)
    
    # cartesians
    nx.set_node_attributes(truncated_graph, {node: complex.coordinates[node] for idx, node in enumerate(truncated_graph.nodes)}, 'cartesian')

    ts_template = TStemplate(truncated_graph, species=complex)
    ts_template.save(folder_path=f'{output_dir}/')

    return ts_template



def get_constraints_from_template(
    complex: Complex,
    bond_rearr: BondRearrangement,
    ts_template: TStemplate,
):
    truncated_graph = get_truncated_active_mol_graph(graph=complex.graph, active_bonds=bond_rearr.all)

    cartesian_constraints = {}
    mapping = get_mapping_ts_template(
        larger_graph=truncated_graph, smaller_graph=ts_template.graph
    )
    for node in truncated_graph.nodes:
        try:
            coords = ts_template.graph.nodes[mapping[node]]["cartesian"]
            cartesian_constraints[node] = coords
        except KeyError:
            print(f"Couldn't find a mapping for atom {node}")

    return cartesian_constraints