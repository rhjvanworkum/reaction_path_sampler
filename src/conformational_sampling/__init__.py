
from typing import List
import autode as ade

class ConformerSampler:

    def __init__(self) -> None:
        pass
        
    def sample_conformers(self, mol: ade.Species) -> List[str]:
        raise NotImplementedError
    