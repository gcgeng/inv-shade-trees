import os, os.path as osp
import numpy as np
import cv2
import torch
import torch.nn as nn

from pix2latent import VariableManager, save_variables
from pix2latent.optimizer import BasinCMAOptimizer, GradientOptimizer
from pix2latent.tree import TreeFolder
from pix2latent.utils import image, video

import pix2latent.loss_functions as LF
import pix2latent.utils.function_hooks as hook
import pix2latent.distribution as dist
from pix2latent.config import cfg
from pix2latent.renderers import AlbedoRenderer, HighlightRenderer, DiffuseRenderer, TDiffRenderer, RimRenderer, THLRenderer
from pix2latent.utils.net_utils import load_network

def regnp(x):
    if len(x.shape) == 2:
        x = np.stack([x, x, x], axis=-1)
    return x

def np2torch(x, batch=False):
    # convert an image represented in the format of numpy array to torch tensor
    # and normalize it to [-1, 1]
    # if batch is True, then add a dim in the front
    x = regnp(x)
    x = torch.Tensor(x).cuda()
    # x = x / 255.0 * 2 - 1
    x = x * 2 - 1
    try:
        x = x.permute(2, 0, 1)
    except:
        x = x.permute(0, 3, 1, 2)
        return x
    if batch:
        x = x[None]
    return x

def torch2np(x):
    x = x.permute(0, 2, 3, 1)
    x = (x + 1.0) * 0.5
    return x

def get_sphere_normal():
    normal = np.load('./normal.npy')
    normal = cv2.resize(normal, (256, 256))
    normal = normal.reshape(-1, 3)
    return torch.Tensor(normal).cuda()

normal = get_sphere_normal()
mask = (torch.norm(normal, dim=-1) > 0.1).reshape(256, 256)

renderers = {
    'albedo': AlbedoRenderer(),
    'diff': DiffuseRenderer(),
    'highlight': HighlightRenderer(),
    'tdiff': TDiffRenderer(),
    'rim': RimRenderer()
}

import torch.nn.functional as F

TINY_NUMBER = 1e-8

class StCode():
    def __init__(self, code):
        self.code = code
        self.n = len(code)
        self.z_begin = [-1 for i in range(self.n)]
        self.z_end = [-1 for i in range(self.n)]
        self.z_len = 0
        self.renderer = {
            'diff': DiffuseRenderer(),
            'highlight': HighlightRenderer(),
            'albedo': AlbedoRenderer(),
            'tdiff': TDiffRenderer(),
            'rim': RimRenderer(),
            'thl': THLRenderer()
        }
        for i in range(1, self.n):
            if self.code[i] > 0: # leaf
                self.z_begin[i] = self.z_len
                self.z_len += len(self.renderer[cfg.symbols[self.code[i]]])
                self.z_end[i] = self.z_len

    def evaluate(self, z):
        assert z.shape[-1] == self.z_len
        ret = [None for _ in range(self.n)]
        for i in range(1, self.n):
            if self.code[i] > 0: # leaf, eval
                this_renderer = self.renderer[cfg.symbols[self.code[i]]]
                this_z = z[..., self.z_begin[i]:self.z_end[i]]
                this_ret = this_renderer(this_z)
                ret[i] = this_ret
        return ret

    def eval_node(self, idx, z):
        assert self.code[idx] > 0
        this_renderer = self.renderer[cfg.symbols[self.code[idx]]]
        this_z = z[..., self.z_begin[idx]:self.z_end[idx]]
        if len(this_z.shape) == 1:
            this_z = this_z[None]
        this_ret = this_renderer(this_z.cuda())
        this_ret = this_ret.squeeze().permute(2, 0, 1).reshape(3, 256, 256)
        this_ret *= mask
        this_ret = this_ret * 2 - 1
        return this_ret

class CompositeRenderer():
    def __init__(self, stcode, param_save_path):
        super().__init__()
        self.code = StCode(stcode)
        self.composite = {
            'screen': self.screen,
            'multiply': self.multiply,
            'mix': self.mix
        }
        self.param_save_path = param_save_path

    def screen(self, a, b):
        return 1 - (1 - a) * (1 - b)

    def multiply(self, a, b):
        return a * b

    def mix(self, a, b, p):
        assert p.min() >= 0 and p.max() <= 1
        return p * b + (1 - p) * a

    def calc(self, idx, leaves, input):
        self.input = input
        if self.code.code[idx] > 0:
            return leaves[idx] 
        else:
            compose = self.composite[cfg.ops[-self.code.code[idx]]]
            return compose(self.calc(idx*2, leaves, input), self.calc(idx*2+1, leaves, input))

    def forward(self, z, img):
        img = torch.cat([img] * z.shape[0], dim=0)
        leaves = self.code.evaluate(z)
        return self.calc(1, leaves, img)

class Warper(nn.Module):
    def __init__(self, stcode, ppath):
        super().__init__()
        self.model = CompositeRenderer(stcode, ppath)

    def assign_img(self, img):
        self.input = img

    def forward(self, z):
        ret = self.model.forward(z, self.input) # of shape (N, P, 3)
        ret = torch.Tensor(ret).cuda()
        ret = ret.permute(0, 3, 1, 2).reshape(ret.shape[0], 3, 256, 256)
        ret = ret * mask
        assert ret.max() <= 1 and ret.min() >= 0
        ret = (ret - 0.5) * 2
        return ret

class TreeWarper(nn.Module):
    """
    used in whole tree optimization
    """
    def __init__(self, treefolder, rt_id):
        super().__init__()
        self.tf = TreeFolder(treefolder)
        self.default_z = self.tf.get_optim_info(rt_id, renderers)
        self.z_len = self.default_z.shape[0]
        self.rt_id = rt_id

    def forward(self, z):
        if len(z.shape) == 2:
            rets = []
            for i in range((z.shape[0])):
                self.tf.set_leaf_param(renderers, z[i], self.rt_id)
                ret = self.tf.btcalc(renderers, self.rt_id)
                ret = ret.reshape(3, 256, 256)
                rets.append(ret)
            rets = torch.stack(rets)
            return rets

loss_fn = LF.ProjectionLoss()

def optim(fp, name, meta_steps=30, optimizer='basincma'):
    target = image.read(fp, as_transformed_tensor=True, im_size=256)
    model.assign_img(target[None])
    save_dir = f'./results/{cfg.exp_name}/{name}'
    os.makedirs(save_dir, exist_ok=True)

    var_manager = VariableManager()
    z_len = model.model.code.z_len

    # (4) define input output variable structure. the variable name must match
    # the argument name of the model and loss function call

    var_manager.register(
                variable_name='z',
                shape=(z_len,),
                grad_free=True,
                # grad_free=False,
                distribution=dist.TruncatedNormalModulo(
                                    sigma=1.0,
                                    trunc=cfg.truncate
                                    ),
                var_type='input',
                learning_rate=cfg.lr,
                hook_fn=hook.Clamp(cfg.truncate),
                )


    var_manager.register(
                variable_name='target',
                shape=(3, 256, 256),
                requires_grad=False,
                default=target,
                var_type='output'
                )


    ### ---- optimize --- ###

    if optimizer == 'basincma':
        opt = BasinCMAOptimizer(
                model, var_manager, loss_fn,
                max_batch_size=cfg.max_minibatch,
                log=cfg.make_video,
                log_dir=save_dir
                )
        vars, out, loss = opt.optimize(meta_steps=meta_steps, grad_steps=30, last_grad_steps=300)
    elif optimizer == 'adam':
        opt = GradientOptimizer(
            model, var_manager, loss_fn,
            max_batch_size=cfg.max_minibatch,
            log=cfg.make_video,
            log_dir=save_dir
        )
        vars, out, loss = opt.optimize(num_samples=20, grad_steps=500)



    ### ---- save results ---- #

    image.save(osp.join(save_dir, 'target.jpg'), target)
    image.save(osp.join(save_dir, 'out.jpg'), out[-1])
    # cv2.imwrite(osp.join(save_dir, 'out.png'), out[-1].permute(1, 2, 0).cpu().numpy() * 255)
    np.save(osp.join(save_dir, 'tracked.npy'), opt.tracked)

    if cfg.make_video:
        video.make_video(osp.join(save_dir, 'out.mp4'), out)

    out = opt.tracked['z'][-1]
    vars.loss = loss
    save_variables(osp.join(save_dir, 'vars.npy'), vars)
    min_loss_idx = torch.Tensor(loss[-1][-1]['loss']).argmin()
    min_loss = loss[-1][-1]['loss'][min_loss_idx]
    return vars['input']['z']['data'][min_loss_idx], min_loss

def optim_tree(fp, name, meta_steps=30):
    target = image.read(fp, as_transformed_tensor=True, im_size=256)
    save_dir = f'./results/{cfg.exp_name}/{name}'
    os.makedirs(save_dir, exist_ok=True)

    var_manager = VariableManager()
    z_len = model.z_len

    # (4) define input output variable structure. the variable name must match
    # the argument name of the model and loss function call

    var_manager.register(
        variable_name='z',
        shape=(z_len,),
        grad_free=True,
        distribution=dist.TruncatedNormalModulo(
            sigma=1.0,
            trunc=cfg.truncate
        ),
        var_type='input',
        learning_rate=cfg.lr,
        hook_fn=hook.Clamp(cfg.truncate),
        # default=torch.Tensor(model.default_z).cuda()
    )


    var_manager.register(
        variable_name='target',
        shape=(3, 256, 256),
        requires_grad=False,
        default=target,
        var_type='output'
    )


    ### ---- optimize --- ###

    opt = BasinCMAOptimizer(
        model, var_manager, loss_fn,
        max_batch_size=cfg.max_minibatch,
        log=cfg.make_video,
        log_dir=save_dir
    )

    vars, out, loss = opt.optimize(meta_steps=meta_steps, grad_steps=30, last_grad_steps=300)


    ### ---- save results ---- #

    image.save(osp.join(save_dir, 'target.jpg'), target)
    image.save(osp.join(save_dir, 'out.jpg'), out[-1])
    # cv2.imwrite(osp.join(save_dir, 'out.png'), out[-1].permute(1, 2, 0).cpu().numpy() * 255)
    np.save(osp.join(save_dir, 'tracked.npy'), opt.tracked)

    if cfg.make_video:
        video.make_video(osp.join(save_dir, 'out.mp4'), out)

    out = opt.tracked['z'][-1]
    vars.loss = loss
    save_variables(osp.join(save_dir, 'vars.npy'), vars)
    min_loss_idx = torch.Tensor(loss[-1][-1]['loss']).argmin()
    min_loss = loss[-1][-1]['loss'][min_loss_idx]
    return vars['input']['z']['data'][min_loss_idx], min_loss


structures_human = [
    ['highlight'],
    ['albedo'],
    ['diff'],
    ['rim'],
    ['tdiff'],
    ['multiply', 'diff', 'albedo'],
    ['screen', 'rim', 'albedo']
]

msteps = {
    'albedo': 20,
    'diff': 15,
    'highlight': 20,
    'rim': 5,
    'tdiff': 15,
}

structures = []
for s in structures_human:
    ns = [-1]
    for ss in s:
        if ss in cfg.ops:
            ns.append(-cfg.ops.index(ss))
        elif ss in cfg.symbols:
            ns.append(cfg.symbols.index(ss))
        else:
            raise NotImplementedError
    structures.append(ns)


tree = TreeFolder(cfg.result)

if cfg.mode == 'inter':
    for st_id, st in enumerate(structures):
        gstcode = StCode(st)
        tree = TreeFolder(cfg.result)
        N = len(tree.nodes)
        leaf_type = st[-1]
        leaf_type_name = cfg.symbols[leaf_type]
        Msteps = 30
        if leaf_type_name in msteps and len(st) == 2:
            Msteps = msteps[leaf_type_name]
        print("Msteps = {}".format(Msteps))
        for i in range(N):
            if not i % cfg.np == cfg.pid:
                continue
            if tree.nodes[i].type == -100 or (not tree.nodes[i].is_leaf() and not tree.nodes[i].has_all_child()):
                print("Optimizing for node {} using structure {}".format(i, st_id))
                pre_path = osp.join('results', cfg.exp_name, '%d_%06d' % (st_id, i), 'vars.npy')
                ppath = osp.join('results', cfg.exp_name, '%d_%06d' % (st_id, i))
                if osp.exists(pre_path):
                    var = np.load(pre_path, allow_pickle=True).item()
                    loss = torch.Tensor(var['loss'][-1][1]['loss'])
                    min_loss_idx = loss.argmin()
                    error = loss[min_loss_idx]
                    z = var['input']['z']['data'][min_loss_idx]
                    print('loaded from %s' % pre_path)
                else:
                    model = Warper(st, ppath)
                    z, error = optim(
                        osp.join(cfg.result, 'images', '%06d.png'%i), '%s_%06d'%(st_id, i), meta_steps=Msteps
                    )
                if error < 1.6:
                    tree.set_structure(gstcode, z, 1, i, ppath)
                    print("Error={} is acceptable.".format(error))
                else:
                    print("Error={} is not acceptable.".format(error))
    if cfg.np == 1:
            try:
                if cfg.save_res:
                    tree.dump_json()
            except:
                import time
                time.sleep(1)
                if cfg.save_res:
                    tree.dump_json()
elif cfg.mode == 'leaf':
    for st_id, st in enumerate(structures):
    # for st_id in st_leaves_idx:
        st = structures[st_id]
        if len(st) != 2:
            continue
        gstcode = StCode(st)
        leaf_type = st[-1]
        leaf_type_name = cfg.symbols[leaf_type]
        Msteps = 30
        if leaf_type_name in msteps:
            Msteps = msteps[leaf_type_name]
        tree = TreeFolder(cfg.result)
        N = len(tree.nodes)
        for i in range(N):
            if i % cfg.np != cfg.pid:
                continue
            ppath = osp.join('results', cfg.exp_name, '%d_%06d' % (st_id, i))
            if tree.nodes[i].type == leaf_type and (tree.nodes[i].btimg is None or tree.nodes[i].z is None):
                print("Optimizing for node {} using structure {}".format(i, st_id))
                pre_path = osp.join('results', cfg.exp_name, '%d_%06d' % (st_id, i), 'vars.npy')
                if osp.exists(pre_path):
                    var = np.load(pre_path, allow_pickle=True).item()
                    loss = torch.Tensor(var['loss'][-1][1]['loss'])
                    min_loss_idx = loss.argmin()
                    error = loss[min_loss_idx]
                    z = var['input']['z']['data'][min_loss_idx]
                    print('loaded from %s' % pre_path)
                else:
                    model = Warper(st, ppath)
                    imgpath = osp.join(cfg.result, 'images', '%06d.png'%i)
                    if not osp.exists(imgpath):
                        imgpath = osp.join(cfg.result, 'bt_images', '%06d.png'%i)
                    z, error = optim(
                        imgpath,
                        '%d_%06d'%(st_id, i),
                        meta_steps=Msteps
                    )
                tree.set_structure(gstcode, z, 1, i, ppath)
        if cfg.np == 1:
            tree.dump_json()
        breakpoint()
elif cfg.mode == 'other_leaf':
    # choose the closest structure
    tree = TreeFolder(cfg.result)
    N = len(tree.nodes)
    for i in range(N):
        if not i % cfg.np == cfg.pid:
            continue
        if tree.nodes[i].type == -100 or (not tree.nodes[i].is_leaf() and not tree.nodes[i].has_all_child()):
            best_error = 10000
            best_z = None
            best_st = None
            ppaths = []
            best_id = None
            for st_id, st in enumerate(structures):
                ppath = osp.join('results', cfg.exp_name, '%d_%06d' % (st_id, i))
                if len(st) != 2:
                    continue
                pre_path = osp.join('results', cfg.exp_name, '%d_%06d' % (st_id, i), 'vars.npy')
                if osp.exists(pre_path):
                    var = np.load(pre_path, allow_pickle=True).item()
                    loss = torch.Tensor(var['loss'][-1][1]['loss'])
                    min_loss_idx = loss.argmin()
                    error = loss[min_loss_idx]
                    z = var['input']['z']['data'][min_loss_idx]
                    print('loaded from %s' % pre_path)
                else:
                    model = Warper(st, ppath)
                    imgpath = osp.join(cfg.result, 'images', '%06d.png'%i)
                    if not osp.exists(imgpath):
                        imgpath = osp.join(cfg.result, 'bt_images', '%06d.png'%i)
                    z, error = optim(
                        imgpath,
                        '%d_%06d'%(st_id, i),
                        meta_steps=30
                    )
                var = np.load(pre_path, allow_pickle=True).item()
                loss = torch.Tensor(var['loss'][-1][1]['loss'])
                min_loss_idx = loss.argmin()
                error = loss[min_loss_idx]
                z = var['input']['z']['data'][min_loss_idx]
                print('loaded from %s' % pre_path)
                if error < best_error:
                    best_error = error
                    best_z = z
                    best_st = st
                    best_id = st_id
                ppaths.append(ppath)
            gstcode = StCode(best_st)
            tree.set_structure(gstcode, best_z, 1, i, ppaths[best_id])
    if cfg.np == 1:
        tree.dump_json()
elif cfg.mode == 'bt':
    tree = TreeFolder(cfg.result)
    N = len(tree.nodes)
    breakpoint()
    for i in range(N):
        if tree.nodes[i].btimg is None:
            tree.calc_btimg(i, renderers = renderers)
    tree.dump_json()
    tree.render_graph()
else:
    raise NotImplementedError
