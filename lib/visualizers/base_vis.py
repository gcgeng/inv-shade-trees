import numpy as np
from torchvision import utils
from lib.config import cfg
import cv2
from termcolor import colored
import os.path as osp
from lib.utils.base_utils import get_time
from pathlib import Path

class BaseVisualizer:
    def __init__(self, name = None):
        if name is None or name == 'default':
            name = get_time()
        self.name = name
        self.result_dir = osp.join(cfg.result_dir, name)
        Path(self.result_dir).mkdir(exist_ok=True, parents=True)
        print(
            colored('the results are saved at {}'.format(self.result_dir),
                    'yellow'))

    def exists(self, batch):
        return False

    def vis_image_pil(self, name, batch, img, range_i=(-1, 1), normalize=True, tgtdir=None, idx=None):
        getidx = lambda i: idx if idx is not None else batch['idx'][i].item()
        if tgtdir is None:
            tgtdir = self.result_dir
        if len(img.shape) == 4:
            N = batch['idx'].shape[0]
            for i in range(N):
                idx = getidx(i)
                utils.save_image(
                    img[i],
                    osp.join(tgtdir, '%06d_%s.png'%(idx, name)),
                    normalize=normalize,
                    range=range_i
                )
        else:
            idx = getidx(0)
            utils.save_image(
                img,
                osp.join(self.result_dir, '%06d_%s.png'%(idx, name)),
                normalize=normalize,
                range=range_i
            )

    def vis_image(self, name, batch, img, mask=False):
        for i in range(batch['idx'].shape[0]):
            try:
                index = batch['idx'][i].item()
            except:
                index = 0
            out_path = osp.join(self.result_dir, '%06d_%s.png'%(index, name))
            img = img[i].detach().cpu().numpy().squeeze()
            if mask:
                msk = batch['msk'][i].detach().cpu().numpy().squeeze().squeeze()
                img = img * msk
            if img.shape[0] < img.shape[1]: # (C, H, W)
                img = img.transpose(1, 2, 0)
            if img.max() < 2.0:
                img = (img * 255).astype(np.uint8)
            if len(img.shape) == 3:
                img = img[..., ::-1]
            cv2.imwrite(out_path, img)

    def dump_latent(self, name, batch, latent):
        for i in range(batch['idx'].shape[0]):
            try:
                index = batch['idx'][i].item()
            except:
                index = 0
            out_path = osp.join(self.result_dir, '%06d_%s.npy'%(index, name))
            np.save(out_path, latent.detach().cpu().numpy())

    def vis_latent(self, name, batch, latent, N=2048):
        for i in range(batch['idx'].shape[0]):
            index = batch['idx'][i].item()
            out_path = osp.join(self.result_dir, '%06d_%s.png'%(index, name))
            img = latent[i].detach().cpu().numpy().squeeze() / N
            if img.shape[0] < img.shape[1]: # (C, H, W)
                img = img.transpose(1, 2, 0)
            if img.max() < 2.0:
                img = (img * 255).astype(np.uint8)
            if len(img.shape) == 3:
                img = img[..., ::-1]
            cv2.imwrite(out_path, img)

    def visualize(self, output, batch, split='vis'):
        return

    def summarize(self):
        return

class Visualizer(BaseVisualizer):
    pass