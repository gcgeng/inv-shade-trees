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
        NK = len(cfg.symbols) if cfg.pred_type == 'ops' else len(cfg.grammar)
        self.classifier = resnext152(num_classes=NK, in_channel=9)

    def forward(self, img, batch, use_pred_rule=False):
        # x: (batch_size, 3, 256, 256)
        component1 = batch['components'][:, 0]
        component2 = batch['components'][:, 1]


        x = torch.cat([img, component1, component2], dim=1)
        prob = self.classifier(x)
        ret = {'prob': prob}
        return ret
