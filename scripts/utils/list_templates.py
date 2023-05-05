import os
import itertools

from src.ts_template import TStemplate

def get_substitute_groups_from_da_template(template):
    dienophile_sets, diene_sets = [], []

    active_nodes = []
    for edge in template.graph.edges(data=True):
        if edge[2]['active']:
            active_nodes.append(edge[0])
            active_nodes.append(edge[1])

    for comb in itertools.combinations(active_nodes, 2):
        for edge in template.graph.edges:
            if edge == comb or (edge[1], edge[0]) == comb:
                dienophile_nodes = list(comb)
                diene_nodes = list(filter(lambda x: x not in dienophile_nodes, active_nodes))

    for nodes, sets in zip(
        [dienophile_nodes, diene_nodes],
        [dienophile_sets, diene_sets]
    ):
        for node in nodes:
            neighbors = template.graph.neighbors(node)
            neighbors = [n for n in neighbors]
            set = [
                template.graph.nodes[neighbors[0]]["atom_label"],
                template.graph.nodes[neighbors[1]]["atom_label"],
                template.graph.nodes[neighbors[2]]["atom_label"],
                template.graph.nodes[neighbors[3]]["atom_label"],
            ]
            set.remove('C')
            set.remove('C')
            sets.append(set)

    return dienophile_sets, diene_sets

if __name__ == "__main__":
    dir = "./scratch/templates/da_cores/"
    for _, _, files in os.walk(dir):
        for file in files:
            template = TStemplate(filename=os.path.join(dir, file))
            print(get_substitute_groups_from_da_template(template))

    