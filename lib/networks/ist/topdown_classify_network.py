from builtins import breakpoint
from lib.networks.ist.pixelsnail import CondResNet
import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.networks.ist.resnext import resnext152

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = resnext152(num_classes=len(cfg.symbols))

    def forward(self, img, batch, use_pred_rule=False):
        # x: (batch_size, 3, 256, 256)
        prob = self.classifier(img)
        ret = {'prob': prob}
        return ret
