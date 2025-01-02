from builtins import breakpoint
from lib.networks.ist.pixelsnail import CondResNet
import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.config import cfg

class encoderInitial(nn.Module):
    def __init__(self):
        super(encoderInitial, self).__init__()
        in_channel = 9
        in_channel = 3 * (cfg.condition_count + 1)
        if cfg.mapping_input == 'latent':
            in_channel += cfg.mapping_input_latent_dim - 3
        # Input should be segmentation, image with environment map, image with point light + environment map
        # self.conv1 = nn.Conv2d(in_channels=7, out_channels=32, kernel_size=6, stride=2, padding=2, bias=False)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)), True )
        x2 = F.relu(self.bn2(self.conv2(x1)), True )
        x3 = F.relu(self.bn3(self.conv3(x2)), True )
        x4 = F.relu(self.bn4(self.conv4(x3)), True )
        x5 = F.relu(self.bn5(self.conv5(x4)), True )
        x = F.relu(self.bn6(self.conv6(x5)), True )
        return x1, x2, x3, x4, x5, x

class decoderInitial(nn.Module):
    def __init__(self, mode='predict'):
        super(decoderInitial, self).__init__()
        # branch for normal prediction
        self.dconv0 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, padding=1, bias=False)
        self.dbn0 = nn.BatchNorm2d(256)
        self.dconv1 = nn.ConvTranspose2d(in_channels=256+256, out_channels=256, kernel_size=4, padding=1, bias=False)
        self.dbn1 = nn.BatchNorm2d(256)
        self.dconv2 = nn.ConvTranspose2d(in_channels=256+256, out_channels=128, kernel_size=4, padding=1, bias=False)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dconv3 = nn.ConvTranspose2d(in_channels=128+128, out_channels=64, kernel_size=4, padding=1, bias=False)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dconv4 = nn.ConvTranspose2d(in_channels=64+64,  out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn4 = nn.BatchNorm2d(32)
        self.dconv5 = nn.ConvTranspose2d(in_channels=32+32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn5 = nn.BatchNorm2d(64)

        self.convFinal = nn.Conv2d(in_channels=64, out_channels = 3, kernel_size = 5, stride=1, padding=2, bias=True)
        
        self.mode = mode
        assert self.mode in ['select', 'predict']

    def forward(self, x1, x2, x3, x4, x5, x):
        x_d1 = F.relu( self.dbn0(self.dconv0(x) ), True)
        x_d1_next = torch.cat( (x_d1, x5), dim = 1)
        x_d2 = F.relu( self.dbn1(self.dconv1(x_d1_next) ), True)
        x_d2_next = torch.cat( (x_d2, x4), dim = 1)
        x_d3 = F.relu( self.dbn2(self.dconv2(x_d2_next) ), True)
        x_d3_next = torch.cat( (x_d3, x3), dim = 1)
        x_d4 = F.relu( self.dbn3(self.dconv3(x_d3_next) ), True)
        x_d4_next = torch.cat( (x_d4, x2), dim = 1)
        x_d5 = F.relu( self.dbn4(self.dconv4(x_d4_next) ), True)
        x_d5_next = torch.cat( (x_d5, x1), dim = 1)
        x_d6 = F.relu( self.dbn5(self.dconv5(x_d5_next) ), True)
        x_orig = torch.clamp(torch.tanh(self.convFinal(x_d6)) * 1.05, -1.0, 1.0)
        ret = x_orig.view(x_orig.size(0), 3, int(256 * cfg.ratio / 0.5), int(256 * cfg.ratio / 0.5))
        return ret

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoderInitial() 
        self.component_predictor = decoderInitial(mode='predict')

        latent_dim = cfg.mapping_input_latent_dim
        total_stride = cfg.top_stride * cfg.bottom_stride
        if cfg.mapping_input == 'latent':
            self.n_class = cfg.vqvae.K
            if total_stride == 32:
                self.cond_conv = nn.Sequential(*[
                    nn.ConvTranspose2d(self.n_class, latent_dim, 4, stride=2, padding=1), 
                    nn.ReLU(),
                    nn.ConvTranspose2d(latent_dim, latent_dim, 4, stride=2, padding=1), 
                    nn.ReLU(),
                    nn.ConvTranspose2d(latent_dim, latent_dim, 4, stride=2, padding=1), 
                    nn.ReLU(),
                    nn.ConvTranspose2d(latent_dim, latent_dim, 4, stride=2, padding=1), 
                    nn.ReLU(),
                    nn.ConvTranspose2d(latent_dim, latent_dim, 4, stride=2, padding=1), 
                    nn.ReLU(),
                ])
            else:
                raise NotImplementedError

        if cfg.mapping_output == 'latent':
            self.n_class = cfg.vqvae.K
            latent_dim = 32
            if cfg.top_stride == 4:
                self.out_latent_conv1 = nn.Sequential(*[
                    nn.Conv2d(3 * cfg.component_pad_size, latent_dim, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(latent_dim, latent_dim, 3, stride=2, padding=1),
                    nn.ReLU()
                ])
                self.latent_conv = nn.Sequential(*[
                    nn.Conv2d(latent_dim, self.n_class * cfg.component_pad_size, 3, stride=2, padding=1),
                    nn.ReLU(),
                ])
            else:
                raise NotImplementedError

            if cfg.bottom_stride == 8:
                self.out_latent_conv2 = nn.Sequential(*[
                    nn.Conv2d(latent_dim, latent_dim, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(latent_dim, latent_dim, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(latent_dim, self.n_class * cfg.component_pad_size, 3, stride=2, padding=1),
                    nn.ReLU(),
                ])
            else:
                raise NotImplementedError

    def forward(self, img, batch, use_pred_rule=False):
        # x: (batch_size, 3, 256, 256)
        component1 = batch['components'][:, :cfg.condition_count] # (B, 3, 3, 256, 256)
        component1[:, 0] = component1[:, 0].mean((-1, -2))[:, :, None, None] # uniform color
        component1 = component1.reshape(component1.size(0), -1, component1.size(3), component1.size(4))
        x = torch.cat((img, component1), dim=1)
        ret = {}
        x1, x2, x3, x4, x5, x_p = self.encoder(x)
        components_pred = self.component_predictor(x1, x2, x3, x4, x5, x_p)
        breakpoint()
        components_pred[components_pred > 0.] = 1.0
        components_pred[components_pred <= 0.] = -1.0
        ret['param_pred'] = components_pred
        return ret
