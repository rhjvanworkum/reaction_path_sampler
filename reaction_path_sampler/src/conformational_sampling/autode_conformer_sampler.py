
from typing import List
import autode as ade

from reaction_path_sampler.src.conformational_sampling import ConformerSampler


class AutodEConformerSampler(ConformerSampler):

    def __init__(self) -> None:
        super().__init__()
        pass

    def sample_conformers(self, mol: ade.Species) -> List[str]:
        return None