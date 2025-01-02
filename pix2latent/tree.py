import random

import numpy
import skimage
import torch
import json
from tqdm import tqdm
import os.path as osp
from termcolor import colored
import cv2
import json
from pathlib import Path
from glob import glob
from pix2latent.config import cfg
from torchvision import utils
from shutil import copyfile
import graphviz
import numpy as np

def img2tensor(img):
    if img.max() > 1:
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
    else:
        return torch.from_numpy(img).permute(2, 0, 1).float() * 2.0 - 1.0


def get_sphere_normal():
    normal = np.load('./normal.npy')
    normal = cv2.resize(normal, (256, 256))
    normal = normal.reshape(-1, 3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.Tensor(normal).to(device)

normal = get_sphere_normal()
mask = (torch.norm(normal, dim=-1) > 0.1).reshape(256, 256)

def dump_img(img, path):
    utils.save_image(
        img,
        path,
        normalize=True,
        range=(-1, 1)
    )

class DictWarper():
    def __init__(self, d) -> None:
        self.d = d

    def __getitem__(self, key):
        if key in self.d:
            return self.d[key]
        else:
            return None

DEPTH_THRESH = 5

def mix(s1, s2, tmask):
    return s1 * (tmask) + s2 * (1-tmask)

def hmi(s1, s2):
    return 0.5 * (s1 + s2)

def fresnel(s1, s2, fmask):
    return s1 * (fmask) + s2 * (1 - fmask)

def screen(s1, s2, dummy=None):
    img = 1 - (1-s1)*(1-s2)
    img = img.clip(0., 1.)
    return img

def multiply(s1, s2, dummy=None):
    img = s1 * s2
    img = img.clip(0., 1.)
    return img

def add(s1, s2):
    return s1 + s2

def cvtm11(img):
    return (img - 0.5) * 2

def cvt01(img):
    return (img + 1) / 2

def get_op(op_idx):
    op_name = cfg.ops[op_idx]
    if op_name == 'mix':
        return mix
    elif op_name == 'hmi':
        return hmi
    elif op_name == 'fresnel':
        return fresnel
    elif op_name == 'screen':
        return screen
    elif op_name == 'multiply':
        return multiply
    elif op_name == 'add':
        return add
    else:
        raise NotImplementedError

class TreeNode:
    def __init__(self, rootdir, id=None, img=None, btimg=None, childs=None, ntype=None, z=None, depth=0, ratio=1.0):
        self.rootdir = rootdir
        self.img = img
        self.btimg = btimg
        self.z = torch.Tensor(z) if z is not None else z
        self.id = id
        self.type = ntype
        self.ratio = ratio
        if childs is None:
            self.childs = [-1, -1, -1]
        else:
            self.childs = childs
        self.depth = depth

    def cvt_to_graph_img(self, path):
        img = cv2.imread(path)
        if img is None:
            print(path)
        import numpy as np
        nm = np.load('./normal.npy')
        mask = ((nm ** 2).sum(axis=-1) > 0).astype(np.uint8) * 255
        H, W = img.shape[:2]
        mask = cv2.resize(mask, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img[..., -1] = mask
        cv2.imwrite(path, img)


    def get_graph_img(self, troot):
        if cfg.bt and self.btimg is not None:
            pth = osp.join('..', 'bt_images', '%06d.png' % (self.id))
            abspath = osp.join(troot, 'bt_images', '%06d.png' % (self.id))
            self.cvt_to_graph_img(abspath)
            return pth
        elif self.img is not None:
            abspath = osp.join(troot, 'images', '%06d.png' % (self.id))
            self.cvt_to_graph_img(abspath)
            return osp.join('../images', '%06d.png' % (self.id))
        else:
            assert self.btimg is not None
            return osp.join('../bt_images', '%06d.png' % (self.id))
        # return osp.join('../images', '%06d.png'%(self.id))

    def set_child(self, ctype, cid):
        """
        :param ctype: 0: left, 1: right, 2: param
        """
        self.childs[ctype] = cid

    def is_leaf(self):
        if self.type == -100:
            return True
        if self.type is None:
            raise Exception('type is empty %d ' % self.id)
        return self.type > 0

    def fail(self):
        return self.type == -100

    def has_child(self, id):
        return self.childs[id] != -1

    def get_child_count(self):
        if self.type == -100:
            return 2
        assert self.type <= 0
        this_type = cfg.ops[-self.type]
        if this_type in ['mix', 'fresnel']:
            return 3
        else:
            return 2
    def has_all_child(self):
        for i in range(self.get_child_count()):
            if not self.has_child(i):
                return False
        return True

    def is_satisfied(self):
        return self.fail() or self.is_leaf() or self.has_all_child()

    def know_type(self):
        return self.type is not None

    def fetch_img(self, bt=False, strict=True, size=256):
        from PIL import Image
        from torchvision import transforms as T
        ratio = cfg.ratio if self.depth == 0 else 1
        # print(self.type)
        if self.type >= len(cfg.symbols):
            print(colored('%d has type error: %d'%(self.id, self.type), 'red'))
            fetch_bt = bt
        else:
            typename = cfg.symbols[self.type]
            fetch_bt = bt and not typename in ['mask', 'tmask', 'stmask', 'env']
        # fetch_bt = True
        path = osp.join(self.rootdir, 'images' if not fetch_bt else 'bt_images', '%06d.png'%(self.id))
        if not osp.exists(path) and not strict:
            print(colored('path not exists: %s'%(path), 'red'))
            path = osp.join(self.rootdir, 'bt_images' if not fetch_bt else 'images', '%06d.png'%(self.id))
        img = Image.open(path)
        osize = img.size
        # size = img.size
        # new_size = (int(size[0] * ratio), int(size[1] * ratio))
        new_size = (size, size)
        transform = T.Compose(
            [
                T.Resize(new_size),
                T.CenterCrop(new_size),
                T.ToTensor(),
            ]
        )
        transform2 = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img = transform(img)
        if img.shape[0] > 3:
            img = img[:3]
        img = transform2(img)
        if osize != (256, 256):
            dump_img(img, path)
        return img

    def depth_larger_than_thresh(self):
        return self.depth > DEPTH_THRESH

    def dump_json(self, folder):
        z_dump = self.z.cpu().numpy().tolist() if self.z is not None else None
        json_dict = {
            'id': self.id,
            'type': self.type,
            'childs': self.childs,
            'depth': self.depth,
            'img': self.img,
            'z': z_dump,
            'btimg': self.btimg,
        }
        tgt_file = osp.join(folder, '%06d.json'%(self.id))
        with open(tgt_file, 'w') as f:
            json.dump(json_dict, f, indent=4)


class TreeFolder:
    """
    Class implementing recursive algorithm
    """
    def __init__(self, root_folder, root_image_paths=None, bt_folder="bt_images"):
        self.root_folder = root_folder
        self.image_folder = osp.join(root_folder, 'images')
        self.bt_image_folder = osp.join(root_folder, bt_folder)
        self.config_folder = osp.join(root_folder, 'configs')
        # self.graph_folder = osp.join(root_folder, 'graph')
        self.graph_folder = osp.join(root_folder, 'graph' if not cfg.bt else 'btgraph')
        Path(self.image_folder).mkdir(exist_ok=True, parents=True)
        Path(self.bt_image_folder).mkdir(exist_ok=True, parents=True)
        Path(self.config_folder).mkdir(exist_ok=True, parents=True)
        Path(self.graph_folder).mkdir(exist_ok=True, parents=True)
        self.nodes = []

        self.global_id = 0
        self.load_folder(self.config_folder)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if root_image_paths is not None:
            for img_path in root_image_paths:
                self.add_root(img_path)

        self.composite = {
            'screen': screen,
            'multiply': multiply,
            'fresnel': fresnel,
            'mix': mix,
        }

    def calc_btimg(self, id, size=256, renderers=None):
        # if self.nodes[id].btimg is not None and (not cfg.rebt or self.nodes[id].is_leaf()):
        if id == -1:
            dummy = np.zeros((256, 256, 3), dtype=np.float32)
            dummy = img2tensor(dummy)
            return dummy
            raise NotImplementedError
        assert id != -1
        if self.nodes[id].is_leaf():
            if osp.exists(osp.join(self.bt_image_folder, '%06d.png'%(id))):
                self.nodes[id].btimg = osp.join(self.bt_image_folder, '%06d.png'%(id))
        if (self.nodes[id].btimg is not None and (not cfg.rebt)):
        # if self.nodes[id].btimg is not None:
            return self.nodes[id].fetch_img(bt=True)
        else:
            if self.nodes[id].is_leaf():
                type_name = cfg.symbols[self.nodes[id].type]
                if type_name in ['env', 'tmask', 'stmask', 'mask']:
                    # fixme
                    return self.nodes[id].fetch_img(bt=False, size=size)
                    # return torch.zeros_like(self.nodes[id].fetch_img(bt=False, size=size))
                else:
                    # print("failed")
                    # print('type={}'.format(self.nodes[id].type))
                    # print('id={}'.format(id))
                    if renderers is None:
                        assert False
                    res = self.btcalc(renderers, id)
                    res[:, ~mask] = 1.0
                    self.nodes[id].btimg = self.dump_bt_img(res, id)
                    return res
            else:
                if self.nodes[id].childs[0] == -1 or self.nodes[id].childs[1] == -1:
                    print(id)
                limg = self.calc_btimg(self.nodes[id].childs[0], renderers=renderers, size=size)
                rimg = self.calc_btimg(self.nodes[id].childs[1], renderers=renderers, size=size)
                cvt = lambda x: ((x + 1.0) * 0.5).cuda()
                limg = cvt(limg)
                rimg = cvt(rimg)
                opname = cfg.ops[-self.nodes[id].type]
                # print(opname)
                if opname in ['screen', 'multiply']:
                    res = self.composite[cfg.ops[-self.nodes[id].type]](limg, rimg)
                else:
                    print(id)
                    param = self.calc_btimg(self.nodes[id].childs[2], renderers=renderers, size=size)
                    param = cvt(param)
                    res = self.composite[opname](limg, rimg, param)
                res = (res - 0.5) * 2.0
                res[:, ~mask] = 1.0
                self.nodes[id].btimg = self.dump_bt_img(res, id)
                return res

    def generate_graph(self, id):
        if hasattr(self.nodes[id], 'graph'):
            return self.nodes[id].graph
        par_node = self.nodes[id]
        par_node.graph = graphviz.Digraph()
        par_node.graph.node(str(id), label='', image=par_node.get_graph_img(self.root_folder), shape='plaintext', xlabel=str(id))
        if not (par_node.is_leaf() or par_node.depth_larger_than_thresh()):
            for i in range(par_node.get_child_count()):
                if par_node.has_child(i):
                    child_graph = self.generate_graph(par_node.childs[i])
                    par_node.graph.subgraph(child_graph)
                    par_node.graph.edge(str(id), str(par_node.childs[i]))
                else:
                    print(par_node.depth)
                    raise Exception('child %d not found' % i)
        return par_node.graph

    def render_graph(self):
        for node in self.nodes:
            self.generate_graph(node.id)
            node.graph.render(osp.join(self.graph_folder, '%06d'%(node.id)))
            node.graph.render(osp.join(self.graph_folder, '%06d'%(node.id)), format='png')

    def load_folder(self, folder):
        configs = sorted(glob(osp.join(folder, '*.json')))
        for config_path in configs:
            with open(config_path) as f:
                try:
                    config = json.load(f)
                except:
                    print(f)
                    exit(-1)
            cconfig = DictWarper(config)
            this_node = TreeNode(
                rootdir = self.root_folder,
                id = cconfig['id'],
                img = cconfig['img'],
                childs = cconfig['childs'],
                ntype = cconfig['type'],
                depth = cconfig['depth'],
                btimg = cconfig['btimg'],
                z = cconfig['z'],
                ratio = 0.5
            )
            self.nodes.append(this_node)
        self.global_id = len(configs)

    def dump_img(self, img, id):
        path = osp.join(self.image_folder, '%06d.png'%(id))
        utils.save_image(
            img,
            path,
            normalize=True,
            range=(-1, 1)
        )
        return path

    def dump_bt_img(self, img, id):
        from torchvision import transforms as T
        transform = T.Resize((256, 256))
        path = osp.join(self.bt_image_folder, '%06d.png'%(id))
        img = transform(img)
        utils.save_image(
            img,
            path,
            normalize=True,
            range=(-1, 1)
        )
        return path

    def add_root(self, root_image_path):
        # add image
        tgt_path = osp.join(self.image_folder, '%06d.png'%(self.global_id))
        import cv2
        img = cv2.imread(root_image_path)
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(tgt_path, img)
        node = TreeNode(rootdir=self.root_folder, id=self.global_id, img=tgt_path, depth=0)
        self.nodes.append(node)
        self.global_id += 1

    def add_node(self, img, depth, type=None):
        if img is not None:
            imgpath = self.dump_img(img, self.global_id)
        else:
            imgpath = None
        new_node = TreeNode(rootdir=self.root_folder, id=self.global_id, img=imgpath, depth=depth)
        if type is not None:
            new_node.type = type
        assert(len(self.nodes) == self.global_id)
        self.nodes.append(new_node)
        self.global_id += 1
        return self.global_id - 1

    def get_images_with_unknown_type(self, max_cnt=32):
        ret_imgs = []
        ret_ids = []
        for node in self.nodes:
            if not node.know_type():
                ret_imgs.append(node.fetch_img())
                ret_ids.append(node.id)
                if len(ret_imgs) >= max_cnt:
                    return ret_imgs, ret_ids
        return ret_imgs, ret_ids

    def get_images_with_unknown_child(self, max_cnt=32):
        ret_imgs = []
        ret_ids = []
        for node in self.nodes:
            if (not node.is_satisfied()) and (not node.depth_larger_than_thresh()):
                ret_imgs.append(node.fetch_img())
                ret_ids.append(node.id)
                if len(ret_imgs) >= max_cnt:
                    return ret_imgs, ret_ids
        return ret_imgs, ret_ids

    def set_images_with_unknown_type(self, input_ids, ret_type):
        for i, id in enumerate(input_ids):
            self.nodes[id].type = ret_type[i].item()

    def set_images_with_predicted_ops(self, input_ids, ret_type):
        for i, id in enumerate(input_ids):
            self.nodes[id].type = -ret_type[i].item()

    def set_images_with_unknown_child(self, input_ids, lchilds, rchild, params, fmask=None):
        for i, id in enumerate(input_ids):
            if self.nodes[id].fail():
                continue
            lcid = self.add_node(lchilds[i], self.nodes[id].depth + 1)
            rcid = self.add_node(rchild[i], self.nodes[id].depth + 1)
            if fmask is not None and fmask[i] == True:
                self.nodes[rcid].type = 2
            self.nodes[id].set_child(0, lcid)
            self.nodes[id].set_child(1, rcid)
            if self.nodes[id].get_child_count() == 3:
                node_type = cfg.symbolss.index('fmask') if cfg.ops[-self.nodes[id].type] == 'fresnel' else cfg.symbolss.index('tmask')
                pid = self.add_node(params[i], self.nodes[id].depth + 1, type=node_type)
                self.nodes[id].set_child(2, pid)

    def dump_json(self):
        print(colored('dump json', 'green'))
        for node in tqdm(self.nodes):
            node.dump_json(self.config_folder)

    def set_structure(self, st_code, z, idx, node_id, ppath):
        if st_code.code[idx] > 0: # leaf
            # update param
            # update img
            img = st_code.eval_node(idx, z)
            self.nodes[node_id].btimg = self.dump_bt_img(img, node_id)
            self.nodes[node_id].type = st_code.code[idx]
            self.nodes[node_id].z = z[st_code.z_begin[idx]: st_code.z_end[idx]]
            return img
        elif st_code.code[idx] <= 0: # internal
            # first, add two childs
            lcid = self.add_node(None, self.nodes[node_id].depth + 1)
            rcid = self.add_node(None, self.nodes[node_id].depth + 1)
            # then, recurisively set two childs
            limg = self.set_structure(st_code, z, idx*2, lcid, ppath)
            rimg = self.set_structure(st_code, z, idx*2+1, rcid, ppath)
            limg = limg * 0.5 + 0.5
            rimg = rimg * 0.5 + 0.5
            # then, compose current node
            this_compose = self.composite[cfg.ops[-st_code.code[idx]]]
            if cfg.ops[-st_code.code[idx]] == 'mix':
                param = cv2.imread(osp.join(ppath, '%06d.png' % idx))[..., ::-1] / 255.
                param = torch.Tensor(param).permute(2, 0, 1).cuda()
                param = (param - 0.5) * 2
                pid = self.add_node(param, self.nodes[node_id].depth + 1, type=cfg.symbols.index('tmask'))
                this_img = this_compose(limg, rimg, param)
                self.nodes[node_id].set_child(2, pid)
            else:
                this_img = this_compose(limg, rimg)
            this_img = this_img * 2 - 1
            self.nodes[node_id].btimg = self.dump_bt_img(this_img, node_id)
            self.nodes[node_id].type = st_code.code[idx]
            self.nodes[node_id].set_child(0, lcid)
            self.nodes[node_id].set_child(1, rcid)
            return this_img

    def reset_op(self, id):
        nd = self.nodes[id]
        img = cvt01(nd.fetch_img(bt=False))
        limg = cvt01(self.nodes[nd.childs[0]].fetch_img(bt=False))
        rimg = cvt01(self.nodes[nd.childs[1]].fetch_img(bt=False))
        if nd.childs[2] == -1:
            return
        param = cvt01(self.nodes[nd.childs[2]].fetch_img())
        otype = -nd.type
        best_type = None
        best_error = None
        if nd.type <= 0: # internal node
            for i in range(len(cfg.ops)):
                this_compose = self.composite[cfg.ops[i]]
                this_res = this_compose(limg, rimg, param)
                this_error = torch.mean(torch.abs(this_res - img))
                if best_type is None or this_error < best_error:
                    best_type = i
                    best_error = this_error
        if best_type != otype:
            self.nodes[id].type = -best_type

    def reachable(self, id1, id2):
        if id1 == id2:
            return True
        if not self.nodes[id1].is_leaf():
            return self.reachable(self.nodes[id1].childs[0], id2) or self.reachable(self.nodes[id1].childs[1], id2)
        else:
            return False

    def get_optim_info(self, rt_id, renderers, freeze=[]):
        z_default = []
        for i in range(len(self.nodes)):
            if not self.reachable(rt_id, i) or not self.nodes[i].is_leaf():
                continue
            type_name = cfg.symbols[self.nodes[i].type]
            if type_name in renderers:
                z_default.extend(self.nodes[i].z.squeeze().detach().cpu().numpy().tolist())
                if i in freeze:
                    self.nodes[i].freeze = True
                else:
                    self.nodes[i].freeze = False
        return np.array(z_default)

    def set_leaf_param(self, renderers, z, rt_id):
        st = 0
        for i in range(len(self.nodes)):
            if not self.reachable(rt_id, i) or not self.nodes[i].is_leaf():
                continue
            type_name = cfg.symbols[self.nodes[i].type]
            if type_name in renderers:
                length = len(renderers[type_name])
                if not self.nodes[i].freeze:
                    self.nodes[i].z = z[st: st+length]
                st += length
            else:
                pass
        return

    def btcalc(self, renderers, id):
        if id == -1:
            dummy = np.zeros((256, 256, 3), dtype=np.float32)
            dummy = img2tensor(dummy)
            return dummy
            raise NotImplementedError
        if self.nodes[id].is_leaf():
            type_name = cfg.symbols[self.nodes[id].type]
            # print(f'btcalc {id} {self.nodes[id].is_leaf()} {type_name}')
            if type_name in ['env', 'mask', 'stmask', 'tmask']:
                return self.nodes[id].fetch_img(bt=False, size=256).to(self.device)
            else:
                if self.nodes[id].z is None:
                    print(type_name)
                    print(id)
                ret = renderers[type_name](torch.Tensor(self.nodes[id].z)[None].to(self.device))
                # ret = ret[0].reshape(512, 512, 3).detach().cpu().numpy()
                ret = ret[0].reshape(256, 256, 3)
                # ret = cv2.resize(ret, (256, 256))
                ret = torch.Tensor(ret).to(self.device).permute(2, 0, 1)
                ret = (ret - 0.5) * 2
                return ret
        else:
            type_name = cfg.ops[-self.nodes[id].type]
            # print(f'btcalc {id} {self.nodes[id].is_leaf()} {type_name}')
            limg = self.btcalc(renderers, self.nodes[id].childs[0])
            rimg = self.btcalc(renderers, self.nodes[id].childs[1])
            cvt = lambda x: ((x + 1.0) * 0.5).cuda()
            if type_name in ['screen', 'multiply']:
                res = self.composite[cfg.ops[-self.nodes[id].type]](cvt(limg), cvt(rimg))
            else:
                param = self.btcalc(renderers, self.nodes[id].childs[2])
                param = cvt(param)
                res = self.composite[cfg.ops[-self.nodes[id].type]](cvt(limg), cvt(rimg), param)
            res = (res - 0.5) * 2
            return res

    def set_optim_z(self, z, renderers, rt_id):
        st = 0
        for i in range(len(self.nodes)):
            if not self.reachable(rt_id, i) or not self.nodes[i].is_leaf():
                continue
            type_name = cfg.symbols[self.nodes[i].type]
            if type_name in renderers:
                length = len(renderers[type_name])
                self.nodes[i].z = z[st: st+length]
                st += length

    def find_type(self, rt_idx, tp):
        if rt_idx == -1:
            return None
        if self.nodes[rt_idx].type == tp:
            return rt_idx
        else:
            for i in range(3):
                cres = self.find_type(self.nodes[rt_idx].childs[i], tp)
                if cres:
                    return cres
        return None

    def find_type_zero(self, rt_idx, tp):
        ret = self.find_type(rt_idx, tp)
        if ret:
            return (self.nodes[ret].fetch_img() + 1.0) * 0.5
        else:
            return torch.ones_like(self.nodes[rt_idx].fetch_img()) * 0.5

def get_changed_type(tf, idx0, idx1, idx2):
    loss = []
    cands = [3, 5, 8]
    breakpoint()
    for type_find in cands:
        leaf1 = tf.find_type(idx1, type_find)
        leaf2 = tf.find_type(idx2, type_find)
        if leaf1 and leaf2:
            # leaf_img1 = tf.nodes[leaf1].z
            leaf_img1 = tf.nodes[leaf1].fetch_img()
            # leaf_img2 = tf.nodes[leaf2].z
            leaf_img2 = tf.nodes[leaf2].fetch_img()
            loss.append(torch.mean((leaf_img1 - leaf_img2) ** 2))
        else:
            loss.append(10000 + random.randint(1, 10))
    loss = np.array(loss)
    print(loss)
    return cands[np.argmin(loss)], loss.min()

def generate_using_type(tf, type, idx0, idx1, idx2):
    breakpoint()
    idx_highlight = idx0 if cfg.symbols[type] != 'highlight' else idx1
    idx_diff = idx0 if cfg.symbols[type] != 'diff' else idx1
    idx_albedo = idx0 if cfg.symbols[type] != 'albedo' else idx1
    highlight = tf.find_type_zero(idx_highlight, cfg.symbols.index('highlight'))
    diff = tf.find_type_zero(idx_diff, cfg.symbols.index('diff'))
    albedo = tf.find_type_zero(idx_albedo, cfg.symbols.index('albedo'))
    ret = tf.composite['screen'](highlight, tf.composite['multiply'](diff, albedo))
    ret = ret.permute(1, 2, 0).detach().cpu().numpy()
    cv2.imwrite('tmp.png', (ret[..., ::-1] * 255).astype(np.uint8))
    return ret
