import torch
import torch.nn.functional as F
import cv2
import numpy as np
from glob import glob

def get_sphere_normal():
    normal = np.load('./normal.npy')
    normal = cv2.resize(normal, (256, 256))
    normal = normal.reshape(-1, 3)
    return torch.Tensor(normal).cuda()

TINY_NUMBER = 1e-8
normal = get_sphere_normal()
mask = (torch.norm(normal, dim=-1) > 0.1).reshape(256, 256)
mask = torch.Tensor(mask).cuda()

class BaseRenderer():
    def __init__(self, lobes = []):
        self.lobes = lobes

    def genz(self):
        z = torch.Tensor(np.random.uniform(0., 1., size=(self.__len__(),))).cuda()
        return z

    def forward(self, z):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def haslobe(self):
        return False

    def __str__(self):
        raise NotImplementedError

    def __call__(self, zs=None):
        imgs = []
        if len(zs.shape) == 2:
            for idx_z in range(zs.shape[0]):
                z = zs[idx_z]
                cnt = 0
                # print(self)
                if z is None or len(z) == 0:
                    while True:
                        z = self.genz()
                        if len(self.lobes) > 0:
                            random_idx_lobes = np.random.choice(len(self.lobes), size=1, replace=False)
                            lobe = self.lobes[random_idx_lobes.item()]
                            self.lobes = [self.lobes[i] for i in range(len(self.lobes)) if i != random_idx_lobes]
                            z[:3] = lobe
                        img = self.forward(z)
                        # print((img > 0).sum())
                        cnt += 1
                        if ((img > 0).sum() > 0):
                            break
                else:
                    z = torch.Tensor(z).cuda()
                    img = self.forward(z)
                # print(cnt)
                img = img.reshape(256, 256, 3)
                img *= mask[..., None]
                img = img.clamp(0., 1.)
                # img = (img * 255).astype(np.uint8)
                imgs.append(img)
        imgs = torch.stack(imgs)
        return imgs

class HighlightRenderer(BaseRenderer):
    def __init__(self, lobes = []):
        super().__init__(lobes)
        normal = get_sphere_normal()
        self.normal = normal
        self.hl = ''

    def reg_loss(self):
        return 0
        # return (torch.norm(self.lobe) - 1) ** 2 + torch.relu(-self.lobe[..., 2])

    def __len__(self):
        return 5

    def haslobe(self):
        return True

    def __str__(self):
        return "HighlightRenderer"

    def forward(self, z):
        # lobe = z[:3] * 2 - 1
        # lobe[..., 2] = torch.abs(lobe[..., 2]) + 1
        # z = (z + 2) / 4.
        lobe = F.normalize(z[:3], dim=-1)
        # lobe[..., 2] = torch.abs(lobe[..., 2]) + 0.2
        # lobe = F.normalize(z[:3], dim=-1)
        amplitude = z[3] * 0.5 + 1.0
        sharpness = z[4] * 70 + 30
        amplitude = torch.abs(amplitude)
        sharpness = torch.abs(sharpness)
        # return self.amplitude * torch.exp(self.sharpness * ((normal * lobe).sum(dim=-1, keepdim=True) - 1))
        ret = amplitude * torch.exp(sharpness * -((self.normal - lobe)**2).sum(dim=-1, keepdim=True))
        ret = ret.repeat(1, 3)
        ret = torch.clamp(ret, 0., 1.)
        return ret

class THLRenderer(BaseRenderer):
    def __init__(self, lobes = []):
        super().__init__(lobes)
        normal = get_sphere_normal()
        self.normal = normal

    def reg_loss(self):
        return 0
        # return (torch.norm(self.lobe) - 1) ** 2 + torch.relu(-self.lobe[..., 2])

    def __len__(self):
        return 7

    def haslobe(self):
        return True

    def __str__(self):
        print("THLRenderer")

    @staticmethod
    def clip(x, a, b):
        if isinstance(x, torch.Tensor):
            x[x < a] = a
            x[x > b] = b
            return x
        if x < a:
            return a
        elif x > b:
            return b
        else:
            return x

        # lobe[..., -1] = torch.abs(lobe[..., -1]) + 0.2
        # lobe = F.normalize(lobe, dim=-1)
        # amplitude = self.clip(z[3] * 0.3 + 0.5, 0.3, 0.8)
        # sharpness = self.clip(z[4] * 60 + 30, 30.0, 90.0)
        # fade = self.clip(z[6] * 0.2 + 0.5, 0.2, 0.7)
        # thresh = self.clip(z[5] * 0.4 + 0.1, 0.1, 0.5)
    def forward(self, z):
        # lobe = z[:3] * 2 - 1
        # lobe[..., 2] = torch.abs(lobe[..., 2]) + 1
        breakpoint()
        z = (z + 2) / 4.
        lobe = F.normalize(z[:3], dim=-1)
        amplitude = z[3] * 0.3 + 0.5
        sharpness = z[4] * 60 + 30
        fade = z[6] * 0.2 + 0.5
        thresh = z[5] * 0.4 + 0.1
        amplitude = torch.abs(amplitude)
        sharpness = torch.abs(sharpness)
        # return self.amplitude * torch.exp(self.sharpness * ((normal * lobe).sum(dim=-1, keepdim=True) - 1))
        ret = amplitude * torch.exp(sharpness * -((self.normal - lobe)**2).sum(dim=-1, keepdim=True))
        ret = ret.repeat(1, 3)
        ret = (ret > thresh).float()
        ret *= fade
        ret = torch.clamp(ret, 0., 1.)
        return ret

class AlbedoRenderer(BaseRenderer):
    def __init__(self):
        super().__init__()
        return

    def __len__(self):
        return 3

    def __str__(self):
        return "AlbedoRenderer"

    def forward(self, z):
        color = torch.Tensor(z)
        ret = torch.Tensor(np.zeros((256, 256, 3))).cuda()
        color = color - torch.floor(color)
        ret[mask] = color
        ret = ret.clip(0, 1)
        return ret

class DiffuseRenderer(BaseRenderer):
    def __init__(self, lobes = []):
        super().__init__(lobes)
        self.normal = normal
        pass

    def __len__(self):
        return 4

    def haslobe(self):
        return True

    def __str__(self):
        return "DiffuseRenderer"

    def forward(self, z):
        # lobe = z[:3] * 2 - 1
        # lobe[..., 2] = torch.abs(lobe[..., 2]) + 1.
        lobe = F.normalize(z[:3], dim=-1)
        ambient = z[3] * 0.3 + 0.2
        ret = ambient + (1-ambient) * torch.relu((self.normal * lobe).sum(dim=-1, keepdim=True))
        ret = ret.clip(0, 1)
        ret = ret.repeat(1, 3)
        return ret

def smoothstep(edge0, edge1, x, a=0, b=1):
    ret = torch.zeros_like(x)
    ret[x < edge0] = 0
    ret[x > edge1] = 1
    ret[(x >= edge0) & (x <= edge1)] = (x[(x >= edge0) & (x <= edge1)] - edge0) / (edge1 - edge0)
    ret = ret * ret * (3 - 2 * ret)
    ret = ret * (b - a) + a
    return ret

class TDiffRenderer(BaseRenderer):
    def __init__(self, lobes = []):
        super().__init__(lobes)
        self.normal = normal

    def __len__(self):
        return 6

    def haslobe(self):
        return True

    def __str__(self):
        return "TDiffRenderer"

    def forward(self, z):
        lobe = F.normalize(z[:3], dim=-1)
        # lobe[..., 2] = torch.abs(lobe[..., 2])
        ambient = z[3] * 0.5 + 0.2
        thresh = z[4] * 0.4 - 0.2
        dot = (lobe * self.normal).sum(dim=-1, keepdim=True)
        ret = smoothstep(thresh, thresh+0.1, dot, a=ambient, b=1.0)
        ret = ret.repeat(1, 3)
        # ret *= fade
        return ret

class CTWRenderer(BaseRenderer):
    def __init__(self, lobes = []):
        super().__init__(lobes)
        self.normal = normal

    def __len__(self):
        return 9

    def __str__(self):
        return "CTWRenderer"

    def haslobe(self):
        return True

    def forward(self, z):
        lobe = F.normalize(z[:3], dim=-1)[None]
        # lobe[..., 2] = torch.abs(lobe[..., 2])
        cold = z[3:6]
        warm = z[6:9]
        dot = (lobe * self.normal).sum(dim=-1, keepdim=True)
        dot = (1.0 + dot) * 0.5
        ret = dot * cold + (1-dot) * warm
        return ret

class RimRenderer(BaseRenderer):
    def __init__(self, lobes = []):
        super().__init__(lobes)
        self.normal = normal
        self.dir = torch.Tensor([0., 0., 1.]).cuda()

    def __len__(self):
        # 3 v
        return 3

    def __str__(self):
        return "RimRenderer"

    def haslobe(self):
        return True

    def forward(self, z):
        lobe = F.normalize(z[:3], dim=-1)[None]
        # lobe[..., 2] = torch.abs(lobe[..., 2])
        nv_dot = (self.normal * self.dir).sum(dim=-1, keepdim=True)
        nl_dot = (self.normal * lobe).sum(dim=-1, keepdim=True)
        nl_dot[nl_dot < 0] = 0
        rim = (1 - nv_dot) * nl_dot
        ret = rim.repeat(1, 3)
        return ret

class TRimRenderer(BaseRenderer):
    def __init__(self, lobes = []):
        super().__init__(lobes)
        self.normal = normal
        self.dir = torch.Tensor([0., 0., 1.]).cuda()

    def __len__(self):
        # 3 v
        return 5

    def __str__(self):
        return "TRimRenderer"

    def haslobe(self):
        return True

    def forward(self, z):
        lobe = F.normalize(z[:3] * 2 - 1, dim=-1)[None]
        lobe[..., 2] = torch.abs(lobe[..., 2])
        thresh = z[3] * 0.3 + 0.3
        fade = z[4] * 0.3 + 0.5
        nv_dot = (self.normal * self.dir).sum(dim=-1, keepdim=True)
        nl_dot = (self.normal * lobe).sum(dim=-1, keepdim=True)
        nl_dot[nl_dot < 0] = 0
        rim = (1 - nv_dot) * nl_dot
        rim = rim > thresh
        rim = (rim).float() * fade
        rim = rim.repeat(1, 3)
        return rim


def mix(s1, s2, tmask):
    tmask[tmask > 0.5] = 1
    tmask[tmask < 0.5] = 0
    return s1 * (1-tmask) + s2 * (tmask)

def hmi(s1, s2):
    return 0.5 * (s1 + s2)

def fresnel(s1, s2, fmask):
    return s1 * (1-fmask) + s2 * (fmask)

def screen(s1, s2):
    return 1 - (1-s1)*(1-s2)

def multiply(s1, s2):
    return s1 * s2

def add(s1, s2):
    return s1 + s2

class Composition:
    def __init__(self):
        self.func = {
            'mix': mix,
            'hmi': hmi,
            'fresnel': fresnel,
            'screen': screen,
            'multiply': multiply,
            'add': add,
        }

    def __getitem__(self, item):
        def func(*childs):
            img = self.func[item](*childs)
            img = np.clip(img, 0, 1)
            return img
        return func

if __name__ == "__main__":
    split = 'train'
    term_paths_train = {
        'albedo': AlbedoRenderer(),
        'highlight': HighlightRenderer(),
        'diff': DiffuseRenderer(),
        'tdiff': TDiffRenderer(),
        'ctw': CTWRenderer(),
        'rim': RimRenderer(),
        'trim': TRimRenderer(),
        'thl': THLRenderer(),
    }
    for name, render in term_paths_train.items():
        img, z = render()
        cv2.imwrite(f'test3_{name}.png', img[..., ::-1])


