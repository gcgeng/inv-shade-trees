from . import yacs
from .yacs import CfgNode as CN
import argparse
import os
import numpy as np
import pprint

cfg = CN()

cfg.world_size = 1

cfg.white_specular = True
cfg.geometry_init = True
cfg.pretrained_occ_network = ''

cfg.random_noise = False

cfg.voxel_size = [0.01, 0.01, 0.01]

cfg.parent_cfg = 'configs/default.yaml'
cfg.method = ''

# experiment name
cfg.exp_name = 'hello'
cfg.vis_name = 'default'

cfg.clip_hdr = False

# network
cfg.point_feature = 9
cfg.distributed = False
cfg.num_latent_code = -1

# data
cfg.training_view = [0, 6, 12, 18]
cfg.test_view = []

# task
cfg.task = 'nerf'

# gpus
cfg.gpus = list(range(8))
# if load the pretrained network
cfg.pretrained_model = "none"
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 100
cfg.save_latest_ep = 5
cfg.eval_ep = 100

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 10000
cfg.train.num_workers = 8
cfg.train.collator = ''
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.shuffle = True

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 0.

cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})

cfg.train.batch_size = 4

cfg.train.acti_func = 'relu'

cfg.train.use_vgg = False
cfg.train.vgg_pretrained = ''
cfg.train.vgg_layer_name = [0,0,0,0,0]

cfg.train.use_ssim = False
cfg.train.use_d = False

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.sampler = 'default'
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.test.frame_sampler_interval = 30
cfg.global_test_switch = False

# val
cfg.val = CN()
cfg.val.dataset = 'CocoVal'
cfg.val.batch_size = 1
cfg.val.epoch = -1
cfg.val.sampler = 'FrameSampler'
cfg.val.frame_sampler_interval = 20
cfg.val.batch_sampler = 'default'
cfg.val.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.val.collator = ''

cfg.vis = CN()
cfg.vis.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.vis.collator = ''
cfg.vis.term = 'whole'
cfg.vis.batch_sampler = 'default'

cfg.split = 'train'
cfg.num_light = 2

# recorder
cfg.record_dir = 'data/record'
cfg.log_interval = 20
cfg.record_interval = 20

# result
cfg.result_dir = 'exps'

# training
cfg.training_mode = 'default'

# evaluation
cfg.eval = False
cfg.skip_eval = False

cfg.fix_random = False

# cfg.vis = 'mesh'

# data
cfg.debug = False

cfg.mlp_weight_decay = 1.0

cfg.ratio = 0.5

cfg.test_overfit = False

cfg.sanity_check = False

cfg.train_overfit = False

## defination of grammar
# cfg.grammar = [
#     'M->M+M',
#     'M->M+g',
#     'M->g+g',
#     'M->M+t',
#     'M->d+d',
#     'M->M+d',
#     'g',
#     't',
#     'd'
# ]

cfg.grammar = [
    # 'M->M+M',
    'M->M+g',
    'M->g+g',
    # 'M->M+t',
    # 'M->d+d',
    # 'M->M+d',
    'g',
    # 't',
    # 'd'
]

cfg.component_pad_size = 2
cfg.concat_normal = False
cfg.selector_concat_normal = False
cfg.pred_residual = False
cfg.prev_result_path = ''
cfg.min_of_n_loss = False

cfg.top_stride = 2
cfg.bottom_stride = 4
cfg.n_embed = 512

cfg.vqvae = CN()
cfg.vqvae.K = 512
cfg.vqvae.top_shape = [32, 32]
cfg.vqvae.bottom_shape = [64, 64]

cfg.random_swap = False

cfg.dataset_name_rep = ''

cfg.mp = 1
cfg.pid = 0
cfg.res_tree_path = ''

cfg.bt = False

cfg.child_cfg = ''
cfg.inv_child = False
cfg.inherit_ckpt = ''
cfg.inherit = False
cfg.runmp = 1
cfg.runpid = 0
cfg.ntname = 'nts.json'
cfg.catmask = False
cfg.condition_count = 2
cfg.depth_thresh = 5
cfg.local_folder = ''
cfg.lstm_graph = False
cfg.ti_bs = 32 # tree infer batch size
cfg.ti_rp = 4 # tree infer repeat
cfg.ti_new = False # tree infer new
cfg.ti_demo = False
cfg.ti_edit = False
cfg.demo_path = ''
cfg.ti_edit_idx = 0
cfg.ti_edit_split = ''
cfg.ti_drm = False

def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)
    cfg.trained_model_dir = os.path.join(cfg.result_dir, 'trained_model')
    cfg.record_dir = os.path.join(cfg.result_dir, 'record')

    cfg.local_rank = args.local_rank
    cfg.distributed = cfg.distributed or args.launcher not in ['none']

    if cfg.debug:
        os.environ['PYTHONBREAKPOINT']="pudb.set_trace"
        cfg.train.num_workers = 0
    else:
        os.environ['PYTHONBREAKPOINT']="0"

def inherit_cfg(current_cfg):
    newcfg = CN()
    if 'parent_cfg' in current_cfg.keys():
        with open(current_cfg.parent_cfg, 'r') as f:
            parent_cfg = inherit_cfg(yacs.load_cfg(f))
            newcfg.merge_from_other_cfg(parent_cfg)
    newcfg.merge_from_other_cfg(current_cfg)
    return newcfg


def make_cfg(args):
    with open(args.cfg_file, 'r') as f:
        current_cfg = yacs.load_cfg(f)

    # if 'parent_cfg' in current_cfg.keys():
    #     with open(current_cfg.parent_cfg, 'r') as f:
    #         parent_cfg = yacs.load_cfg(f)
    #     cfg.merge_from_other_cfg(parent_cfg)
    current_cfg = inherit_cfg(current_cfg)
    if 'child_cfg' in args.opts:
        child_config = args.opts[args.opts.index('child_cfg') + 1]
        with open(child_config, 'r') as f:
            child_cfg = inherit_cfg(yacs.load_cfg(f))
            current_cfg.merge_from_other_cfg(child_cfg)
    cfg.merge_from_other_cfg(current_cfg)
    cfg.merge_from_list(args.opts)

    cfg.merge_from_list(args.opts)

    parse_cfg(cfg, args)
    # pprint.pprint(cfg)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="vis")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--launcher', type=str, default='none', choices=['none', 'pytorch'])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
