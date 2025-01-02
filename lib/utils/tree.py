import numpy
import json
from tqdm import tqdm
import os.path as osp
from termcolor import colored
import cv2
import json
from pathlib import Path
from glob import glob
from lib.config import cfg
from torchvision import utils
from shutil import copyfile
import graphviz


class DictWarper():
    def __init__(self, d) -> None:
        self.d = d

    def __getitem__(self, key):
        if key in self.d:
            return self.d[key]
        else:
            return None


DEPTH_THRESH = cfg.depth_thresh

def mix(s1, s2, tmask):
    return s1 * (1 - tmask) + s2 * (tmask)

def hmi(s1, s2):
    return 0.5 * (s1 + s2)

def fresnel(s1, s2, fmask):
    return s1 * (1 - fmask) + s2 * (fmask)

def screen(s1, s2):
    return 1 - (1 - s1) * (1 - s2)

def multiply(s1, s2):
    return s1 * s2

def add(s1, s2):
    return s1 + s2

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
    def __init__(self, id=None, img=None, btimg=None, childs=None, ntype=None, z=None, depth=0, ratio=1.0):
        self.img = img
        self.btimg = btimg
        self.z = z
        self.id = id
        self.type = ntype
        self.ratio = ratio
        if childs is None:
            self.childs = [-1, -1, -1]
        else:
            self.childs = childs
        self.depth = depth
        self.z = None

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
        breakpoint()
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


    def set_child(self, ctype, cid):
        """
        :param ctype: 0: left, 1: right, 2: param
        """
        self.childs[ctype] = cid

    def is_leaf(self):
        if self.type == -100:
            return True
        if self.type is None:
            return True
            raise Exception('type is empty')
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

    def fetch_img(self):
        from PIL import Image
        from torchvision import transforms as T
        ratio = cfg.ratio if self.depth == 0 else 1
        if self.img is not None:
            img = Image.open(self.img)
        else:
            if self.btimg is None:
                print(str(self.id) + ' has no img')
                assert False
            img = Image.open(self.btimg)
        size = img.size
        new_size = (int(size[0] * ratio), int(size[1] * ratio))
        new_size = (256, 256)
        transform = T.Compose(
            [
                T.Resize(new_size),
                T.CenterCrop(new_size),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        img = transform(img)
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
        tgt_file = osp.join(folder, '%06d.json' % (self.id))
        with open(tgt_file, 'w') as f:
            json.dump(json_dict, f, indent=4)


class TreeFolder:
    """
    Class implementing recursive algorithm
    """

    def __init__(self, root_folder, root_image_paths=None):
        self.root_folder = root_folder
        self.image_folder = osp.join(root_folder, 'images')
        self.bt_image_folder = osp.join(root_folder, 'bt_images')
        self.config_folder = osp.join(root_folder, 'configs')
        self.graph_folder = osp.join(root_folder, 'graph' if not cfg.bt else 'btgraph')
        Path(self.image_folder).mkdir(exist_ok=True, parents=True)
        Path(self.bt_image_folder).mkdir(exist_ok=True, parents=True)
        Path(self.config_folder).mkdir(exist_ok=True, parents=True)
        Path(self.graph_folder).mkdir(exist_ok=True, parents=True)
        self.nodes = []

        self.global_id = 0
        self.load_folder(self.config_folder)

        if root_image_paths is not None:
            for img_path in root_image_paths:
                self.add_root(img_path)

        self.composite = {
            'screen': screen,
            'multiply': multiply,
        }

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
                    continue
                    raise Exception('child %d not found' % i)
        return par_node.graph

    def render_graph(self):
        for node in tqdm(self.nodes):
            self.generate_graph(node.id)
            node.graph.render(osp.join(self.graph_folder, '%06d' % (node.id)))
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
                id=cconfig['id'],
                img=cconfig['img'],
                childs=cconfig['childs'],
                ntype=cconfig['type'],
                depth=cconfig['depth'],
                btimg=cconfig['btimg'],
                z=cconfig['z'],
                ratio=0.5
            )
            self.nodes.append(this_node)
        self.global_id = len(configs)

    def dump_img(self, img, id):
        path = osp.join(self.image_folder, '%06d.png' % (id))
        utils.save_image(
            img,
            path,
            normalize=True,
            range=(-1, 1)
        )
        return path

    def dump_bt_img(self, img, id):
        path = osp.join(self.bt_image_folder, '%06d.png' % (id))
        utils.save_image(
            img,
            path,
            normalize=True,
            range=(-1, 1)
        )
        return path

    def add_root(self, root_image_path):
        # add image
        tgt_path = osp.join(self.image_folder, '%06d.png' % (self.global_id))
        import cv2
        img = cv2.imread(root_image_path)
        if img is None:
            print(root_image_path)
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(tgt_path, img)
        node = TreeNode(id=self.global_id, img=tgt_path, depth=0)
        self.nodes.append(node)
        self.global_id += 1

    def add_root_image(self, root_image):
        tgt_path = osp.join(self.image_folder, '%06d.png' % (self.global_id))
        img = cv2.resize(root_image, (256, 256))
        cv2.imwrite(tgt_path, img)
        node = TreeNode(id=self.global_id, img=tgt_path, depth=0)
        self.nodes.append(node)
        self.global_id += 1


    def add_node(self, img, depth, type=None):
        # assert type is not None
        if img is not None:
            imgpath = self.dump_img(img, self.global_id)
        else:
            imgpath = None
        new_node = TreeNode(id=self.global_id, img=imgpath, depth=depth)
        if type is not None:
            new_node.type = type
        assert (len(self.nodes) == self.global_id)
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
                node_type = cfg.symbols.index('fmask') if cfg.ops[
                                                              -self.nodes[id].type] == 'fresnel' else cfg.symbols.index(
                    'tmask')
                pid = self.add_node(params[i], self.nodes[id].depth + 1, type=node_type)
                self.nodes[id].set_child(2, pid)

    def dump_json(self):
        print(colored('dump json', 'green'))
        for node in tqdm(self.nodes):
            node.dump_json(self.config_folder)

    def set_structure(self, st_code, z, idx, node_id):
        if st_code.code[idx] > 0:  # leaf
            # update param
            self.nodes[node_id].z = z[st_code.z_begin[idx]: st_code.z_end[idx]]
            # update img
            img = st_code.eval_node(idx, z)
            self.nodes[node_id].bt_img = self.dump_bt_img(img, node_id)
            return img
        elif st_code.code[idx] <= 0:  # internal
            # first, add two childs
            lcid = self.add_node(None, self.nodes[node_id].depth + 1)
            rcid = self.add_node(None, self.nodes[node_id].depth + 1)
            # then, recurisively set two childs
            limg = self.set_structure(st_code, z, idx * 2, lcid)
            rimg = self.set_structure(st_code, z, idx * 2 + 1, rcid)
            limg = limg * 0.5 + 0.5
            rimg = rimg * 0.5 + 0.5
            # then, compose current node
            this_compose = self.composite[cfg.ops[-st_code.code[idx]]]
            # here we dont consider the param
            this_img = this_compose(limg, rimg)
            this_img = this_img * 2 - 1
            self.nodes[node_id].bt_img = self.dump_bt_img(this_img, node_id)
            self.nodes[node_id].type = st_code.code[idx]
            self.nodes[node_id].set_child(0, lcid)
            self.nodes[node_id].set_child(1, rcid)
            return this_img

