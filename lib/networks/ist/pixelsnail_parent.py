from lib.networks.ist.pixelsnail import Network as BaseNetwork
from lib.config import cfg

class Network(BaseNetwork):
    # Wrapper
    def __init__(self, hier=None):
        if hier is None:
            hier = cfg.hier
        super().__init__(hier)
