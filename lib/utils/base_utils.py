import pickle
from lib.config import cfg
import os.path as osp
import os
import numpy as np
from pathlib import Path
from termcolor import colored
import torch
from glob import glob
from torchvision import transforms as T

def create_dir(name: os.PathLike):
    Path(name).mkdir(exist_ok=True, parents=True)

def create_link(src, tgt):
    new_link = os.path.basename(tgt)
    if osp.exists(src) and osp.islink(src):
        print("Found old latest dir link {} which link to {}, replacing it to {}".format(src, os.readlink(src), tgt))
        os.unlink(src)
    os.symlink(new_link, src)

def dump_cfg(cfg, tgt_path: os.PathLike):
    if os.path.exists(tgt_path):
        print(colored("Hey, there exists an experiment with same name before. Please make sure you are continuing.", "green"))
        return
    create_dir(Path(tgt_path).parent)
    cfg_str = cfg.dump()
    with open(tgt_path, "w") as f:
        f.write(cfg_str)

def git_committed():
    from git import Repo
    return len([x for x in Repo('.').index.diff(None)]) == 0

def git_hash():
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha

def get_time():
    from datetime import datetime
    now = datetime.now()
    str1 =  '_'.join(now.__str__().split(' ')).split('.')[0]
    str2 = str(now.timestamp()).split('.')[0]
    return str1 + '_' + str2

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def project_torch(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = torch.mm(xyz, RT[:, :3].T) + RT[:, 3:].T
    depth = xyz[..., -1]
    xyz = torch.mm(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy, depth


def write_K_pose_inf(K, poses, img_root):
    K = K.copy()
    K[:2] = K[:2] * 8
    K_inf = os.path.join(img_root, 'Intrinsic.inf')
    os.system('mkdir -p {}'.format(os.path.dirname(K_inf)))
    with open(K_inf, 'w') as f:
        for i in range(len(poses)):
            f.write('%d\n'%i)
            f.write('%f %f %f\n %f %f %f\n %f %f %f\n' % tuple(K.reshape(9).tolist()))
            f.write('\n')

    pose_inf = os.path.join(img_root, 'CamPose.inf')
    with open(pose_inf, 'w') as f:
        for pose in poses:
            pose = np.linalg.inv(pose)
            A = pose[0:3,:]
            tmp = np.concatenate([A[0:3,2].T, A[0:3,0].T,A[0:3,1].T,A[0:3,3].T])
            f.write('%f %f %f %f %f %f %f %f %f %f %f %f\n' % tuple(tmp.tolist()))

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG', '*.exr']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def load_rgb(path, ratio=1.0):
    import cv2
    try:
        ratio *= cfg.render_ratio
    except:
        pass
    if not osp.exists(path):
        print(colored("{} does not exist".format(path), "red"))
        return None
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., ::-1]
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)
    H, W = img.shape[:2]
    new_H, new_W = int(H * ratio), int(W * ratio)
    img = cv2.resize(img, (new_W, new_H))
    # img = imageio.imread(path)[:, :, :3]
    img = np.float32(img)
    if not path.endswith('.exr'):
        img = img / 255.

    img = img.transpose(2, 0, 1)     # [C, H, W]
    return img

def load_rgb_pil(path, ratio=1.0):
    from PIL import Image
    try:
        ratio *= cfg.render_ratio
    except:
        pass
    if not osp.exists(path):
        print(colored("{} does not exist".format(path), "red"))
        return None
    img = Image.open(path)
    size = img.size
    new_size = (int(size[0] * ratio), int(size[1] * ratio))
    resize_transform = T.Resize(new_size)
    img = resize_transform(img)

    return img


def load_mask(path, ratio=1.0):
    import cv2
    try:
        ratio *= cfg.render_ratio
    except:
        pass
    if not osp.exists(path):
        print(colored("{} does not exist".format(path), "red"))
        return None
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    try:
        if img.shape[2] == 3:
            img = img[..., 0]
    except:
        pass
    H, W = img.shape[:2]
    new_H, new_W = int(H * ratio), int(W * ratio)
    img = cv2.resize(img, (new_W, new_H))
    if len(img.shape) == 2:
        img = img[None]
    # img = imageio.imread(path)[:, :, :3]
    return img > 0

def load_dump(path):
    latent = np.load(path).squeeze()
    return latent


def linear2srgb(tensor_0to1):
    if isinstance(tensor_0to1, torch.Tensor):
        pow_func = torch.pow
        where_func = torch.where
    else:
        pow_func = np.power
        where_func = np.where

    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_0to1 = torch.clip(tensor_0to1, 0.0, 1.0)

    tensor_linear = tensor_0to1 * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (
        pow_func(tensor_0to1, 1 / srgb_exponent)
    ) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb

def getattr_default(cfg, attr, default):
    if hasattr(cfg, attr):
        return getattr(cfg, attr)
    else:
        return default

