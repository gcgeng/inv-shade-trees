# Copyright (c) Xi Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed from https://github.com/neocxi/pixelsnail-public and ported it to PyTorch

from lib.config import cfg
from math import sqrt
from functools import partial, lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def wn_linear(in_dim, out_dim):
    return nn.utils.weight_norm(nn.Linear(in_dim, out_dim))


class WNConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        activation=None,
    ):
        super().__init__()

        self.conv = nn.utils.weight_norm(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

        self.out_channel = out_channel

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.kernel_size = kernel_size

        self.activation = activation

    def forward(self, input):
        out = self.conv(input)

        if self.activation is not None:
            out = self.activation(out)

        return out


def shift_down(input, size=1):
    return F.pad(input, [0, 0, size, 0])[:, :, : input.shape[2], :]


def shift_right(input, size=1):
    return F.pad(input, [size, 0, 0, 0])[:, :, :, : input.shape[3]]


class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding='downright',
        activation=None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2

        self.kernel_size = kernel_size

        if padding == 'downright':
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]

        elif padding == 'down' or padding == 'causal':
            pad = kernel_size[1] // 2

            pad = [pad, pad, kernel_size[0] - 1, 0]

        self.causal = 0
        if padding == 'causal':
            self.causal = kernel_size[1] // 2

        self.pad = nn.ZeroPad2d(pad)

        self.conv = WNConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=0,
            activation=activation,
        )

    def forward(self, input):
        out = self.pad(input)

        if self.causal > 0:
            self.conv.conv.weight_v.data[:, :, -1, self.causal :].zero_()

        out = self.conv(out)

        return out


class GatedResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        conv='wnconv2d',
        activation=nn.ELU,
        dropout=0.1,
        auxiliary_channel=0,
        condition_dim=0,
    ):
        super().__init__()

        if conv == 'wnconv2d':
            conv_module = partial(WNConv2d, padding=kernel_size // 2)

        elif conv == 'causal_downright':
            conv_module = partial(CausalConv2d, padding='downright')

        elif conv == 'causal':
            conv_module = partial(CausalConv2d, padding='causal')

        self.activation = activation()
        self.conv1 = conv_module(in_channel, channel, kernel_size)

        if auxiliary_channel > 0:
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = conv_module(channel, in_channel * 2, kernel_size)

        if condition_dim > 0:
            # self.condition = nn.Linear(condition_dim, in_channel * 2, bias=False)
            self.condition = WNConv2d(condition_dim, in_channel * 2, 1, bias=False)

        self.gate = nn.GLU(1)

    def forward(self, input, aux_input=None, condition=None):
        out = self.conv1(self.activation(input))

        if aux_input is not None:
            out = out + self.aux_conv(self.activation(aux_input))

        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if condition is not None:
            condition = self.condition(condition)
            out += condition
            # out = out + condition.view(condition.shape[0], 1, 1, condition.shape[1])

        out = self.gate(out)
        out += input

        return out


@lru_cache(maxsize=64)
def causal_mask(size):
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0

    return (
        torch.from_numpy(mask).unsqueeze(0),
        torch.from_numpy(start_mask).unsqueeze(1),
    )


class CausalAttention(nn.Module):
    def __init__(self, query_channel, key_channel, channel, n_head=8, dropout=0.1):
        super().__init__()

        self.query = wn_linear(query_channel, channel)
        self.key = wn_linear(key_channel, channel)
        self.value = wn_linear(key_channel, channel)

        self.dim_head = channel // n_head
        self.n_head = n_head

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key):
        batch, _, height, width = key.shape

        def reshape(input):
            return input.view(batch, -1, self.n_head, self.dim_head).transpose(1, 2)

        query_flat = query.view(batch, query.shape[1], -1).transpose(1, 2)
        key_flat = key.view(batch, key.shape[1], -1).transpose(1, 2)
        query = reshape(self.query(query_flat))
        key = reshape(self.key(key_flat)).transpose(2, 3)
        value = reshape(self.value(key_flat))

        attn = torch.matmul(query, key) / sqrt(self.dim_head)
        mask, start_mask = causal_mask(height * width)
        mask = mask.type_as(query)
        start_mask = start_mask.type_as(query)
        attn = attn.masked_fill(mask == 0, -1e4)
        attn = torch.softmax(attn, 3) * start_mask
        attn = self.dropout(attn)

        out = attn @ value
        out = out.transpose(1, 2).reshape(
            batch, height, width, self.dim_head * self.n_head
        )
        out = out.permute(0, 3, 1, 2)

        return out


class PixelBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        n_res_block,
        attention=True,
        dropout=0.1,
        condition_dim=0,
    ):
        super().__init__()

        resblocks = []
        for i in range(n_res_block):
            resblocks.append(
                GatedResBlock(
                    in_channel,
                    channel,
                    kernel_size,
                    conv='causal',
                    dropout=dropout,
                    condition_dim=condition_dim,
                )
            )

        self.resblocks = nn.ModuleList(resblocks)

        self.attention = attention

        if attention:
            self.key_resblock = GatedResBlock(
                in_channel * 2 + 2, in_channel, 1, dropout=dropout
            )
            self.query_resblock = GatedResBlock(
                in_channel + 2, in_channel, 1, dropout=dropout
            )

            self.causal_attention = CausalAttention(
                in_channel + 2, in_channel * 2 + 2, in_channel // 2, dropout=dropout
            )

            self.out_resblock = GatedResBlock(
                in_channel,
                in_channel,
                1,
                auxiliary_channel=in_channel // 2,
                dropout=dropout,
            )

        else:
            self.out = WNConv2d(in_channel + 2, in_channel, 1)

    def forward(self, input, background, condition=None):
        out = input

        for resblock in self.resblocks:
            out = resblock(out, condition=condition)

        if self.attention:
            key_cat = torch.cat([input, out, background], 1)
            key = self.key_resblock(key_cat)
            query_cat = torch.cat([out, background], 1)
            query = self.query_resblock(query_cat)
            attn_out = self.causal_attention(query, key)
            out = self.out_resblock(out, attn_out)

        else:
            bg_cat = torch.cat([out, background], 1)
            out = self.out(bg_cat)

        return out


class CondResNet(nn.Module):
    def __init__(self, in_channel, channel, kernel_size, n_res_block):
        super().__init__()

        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)]

        for i in range(n_res_block):
            blocks.append(GatedResBlock(channel, channel, kernel_size))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class PixelSNAIL(nn.Module):
    def __init__(
        self,
        hier,
        shape,
        n_class,
        channel,
        kernel_size,
        n_block,
        n_res_block,
        res_channel,
        attention=True,
        dropout=0.1,
        n_cond_res_block=0,
        cond_res_channel=0,
        cond_res_kernel=3,
        n_out_res_block=0,
    ):
        super().__init__()

        self.hier = hier

        height, width = shape

        self.n_class = n_class

        if kernel_size % 2 == 0:
            kernel = kernel_size + 1

        else:
            kernel = kernel_size

        self.horizontal = CausalConv2d(
            n_class, channel, [kernel // 2, kernel], padding='down'
        )
        self.vertical = CausalConv2d(
            n_class, channel, [(kernel + 1) // 2, kernel // 2], padding='downright'
        )

        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('background', torch.cat([coord_x, coord_y], 1))

        self.blocks = nn.ModuleList()

        for i in range(n_block):
            self.blocks.append(
                PixelBlock(
                    channel,
                    res_channel,
                    kernel_size,
                    n_res_block,
                    attention=attention,
                    dropout=dropout,
                    condition_dim=cond_res_channel,
                )
            )

        if n_cond_res_block > 0:
            if self.hier == 'parent_top':
                self.cond_resnet = CondResNet(
                    n_class * 2, cond_res_channel, cond_res_kernel, n_cond_res_block
                )
            elif self.hier == 'parent_top_sibling':
                self.cond_resnet = CondResNet(
                    n_class * 4, cond_res_channel, cond_res_kernel, n_cond_res_block
                )
            elif self.hier == 'parent_bottom':
                self.cond_resnet = CondResNet(
                    n_class * 3, cond_res_channel, cond_res_kernel, n_cond_res_block
                )
            elif self.hier == 'parent_bottom_sibling':
                self.cond_resnet = CondResNet(
                    n_class * 5, cond_res_channel, cond_res_kernel, n_cond_res_block
                )
            else:
                self.cond_resnet = CondResNet(
                    n_class, cond_res_channel, cond_res_kernel, n_cond_res_block
                )

        if self.hier == 'parent_top' or self.hier == 'parent_top_sibling':
            if cfg.top_stride == 2:
                self.cond_conv = nn.Sequential(*[
                    nn.Conv2d(self.n_class, self.n_class, 3, stride=2, padding=1), 
                    nn.ReLU(),
                ])
            elif cfg.top_stride == 4:
                self.cond_conv = nn.Sequential(*[
                    nn.Conv2d(self.n_class, self.n_class, 3, stride=2, padding=1), 
                    nn.ReLU(),
                    nn.Conv2d(self.n_class, self.n_class, 3, stride=2, padding=1), 
                    nn.ReLU(),
                ])
            else:
                raise NotImplementedError
        elif self.hier == 'parent_bottom' or self.hier == 'parent_bottom_sibling':
            if cfg.top_stride == 2:
                self.cond_conv = nn.Sequential(*[
                    nn.ConvTranspose2d(self.n_class, self.n_class, 4, stride=2, padding=1), 
                    nn.ReLU(),
                ])
            elif cfg.top_stride == 4:
                self.cond_conv = nn.Sequential(*[
                    nn.ConvTranspose2d(self.n_class, self.n_class, 4, stride=2, padding=1), 
                    nn.ReLU(),
                    nn.ConvTranspose2d(self.n_class, self.n_class, 4, stride=2, padding=1), 
                    nn.ReLU(),
                ])
            else:
                raise NotImplementedError

        out = []

        for i in range(n_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))

        out.extend([nn.ELU(inplace=True), WNConv2d(channel, n_class, 1)])

        self.out = nn.Sequential(*out)

    def forward(self, input, condition=None, cache=None):
        if cache is None:
            cache = {}
        batch, height, width = input.shape
        input = (
            F.one_hot(input, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
        )
        horizontal = shift_down(self.horizontal(input))
        vertical = shift_right(self.vertical(input))
        out = horizontal + vertical


        background = self.background[:, :, :height, :].expand(batch, 2, height, width)

        if condition is not None:
            if 'condition' in cache:
                condition = cache['condition']
                condition = condition[:, :, :height, :]
            else:
                if self.hier == 'parent_top':
                    latent_top, latent_bottom = condition
                    condition_top = F.one_hot(latent_top, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_bottom = F.one_hot(latent_bottom, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_bottom = self.cond_conv(condition_bottom)
                    condition = torch.cat([condition_top, condition_bottom], 1)
                    condition = self.cond_resnet(condition)
                    cache['condition'] = condition.detach().clone()
                    condition = condition[:, :, :height, :]
                elif self.hier == 'parent_bottom':
                    latent_top, latent_bottom, gt_top = condition
                    condition_top = F.one_hot(latent_top, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_bottom = F.one_hot(latent_bottom, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_chitop = F.one_hot(gt_top, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_top = self.cond_conv(condition_top)
                    condition_chitop = self.cond_conv(condition_chitop)
                    condition = torch.cat([condition_top, condition_bottom, condition_chitop], 1)
                    condition = self.cond_resnet(condition)
                    cache['condition'] = condition.detach().clone()
                    condition = condition[:, :, :height, :]
                elif self.hier == 'parent_top_sibling':
                    latent_top, latent_bottom, sibling_top, sibling_bottom = condition
                    condition_latent_top = F.one_hot(latent_top, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_latent_bottom = F.one_hot(latent_bottom, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_sibling_top = F.one_hot(sibling_top, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_sibling_bottom = F.one_hot(sibling_bottom, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_latent_bottom = self.cond_conv(condition_latent_bottom)
                    condition_sibling_bottom = self.cond_conv(condition_sibling_bottom)
                    condition = torch.cat([condition_latent_top, condition_latent_bottom, condition_sibling_top, condition_sibling_bottom], 1)
                    condition = self.cond_resnet(condition)
                    cache['condition'] = condition.detach().clone()
                    condition = condition[:, :, :height, :]
                elif self.hier == 'parent_bottom_sibling':
                    latent_top, latent_bottom, sibling_top, sibling_bottom, gt_top = condition
                    condition_latent_top = F.one_hot(latent_top, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_latent_bottom = F.one_hot(latent_bottom, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_sibling_top = F.one_hot(sibling_top, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_sibling_bottom = F.one_hot(sibling_bottom, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_this_top = F.one_hot(gt_top, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
                    condition_latent_top = self.cond_conv(condition_latent_top)
                    condition_sibling_top = self.cond_conv(condition_sibling_top)
                    condition_this_top = self.cond_conv(condition_this_top)
                    condition = torch.cat([condition_latent_top, condition_latent_bottom, condition_sibling_top, condition_sibling_bottom, condition_this_top], 1)
                    condition = self.cond_resnet(condition)
                    cache['condition'] = condition.detach().clone()
                    condition = condition[:, :, :height, :]
                else:
                    condition = (
                        F.one_hot(condition, self.n_class)
                        .permute(0, 3, 1, 2)
                        .type_as(self.background)
                    )
                    condition = self.cond_resnet(condition)
                    condition = F.interpolate(condition, scale_factor=cfg.top_stride)
                    cache['condition'] = condition.detach().clone()
                    condition = condition[:, :, :height, :]

        for block in self.blocks:
            out = block(out, background, condition=condition)

        out = self.out(out)

        return out, cache
    
class Network(nn.Module):
    # Wrapper
    def __init__(self, hier):
        super().__init__()
        if hier == 'top':
            self.net = PixelSNAIL(
                hier,
                cfg.vqvae.top_shape,
                cfg.vqvae.K,
                cfg.pixelsnail.channel,
                5,
                4,
                cfg.pixelsnail.n_res_block,
                cfg.pixelsnail.n_res_channel,
                dropout=cfg.pixelsnail.dropout,
                n_out_res_block=cfg.pixelsnail.n_out_res_block,
            )
        elif hier == 'bottom':
            self.net = PixelSNAIL(
                hier,
                cfg.vqvae.bottom_shape,
                cfg.vqvae.K,
                cfg.pixelsnail.channel,
                5,
                4,
                cfg.pixelsnail.n_res_block,
                cfg.pixelsnail.n_res_channel,
                attention=False,
                dropout=cfg.pixelsnail.dropout,
                n_cond_res_block=cfg.pixelsnail.n_cond_res_block,
                cond_res_channel=cfg.pixelsnail.n_res_channel,
            )
        elif hier == 'parent_top' or hier == 'parent_top_sibling':
            self.net = PixelSNAIL(
                hier,
                cfg.vqvae.top_shape,
                cfg.vqvae.K,
                cfg.pixelsnail.channel,
                5,
                4,
                cfg.pixelsnail.n_res_block,
                cfg.pixelsnail.n_res_channel,
                dropout=cfg.pixelsnail.dropout,
                n_out_res_block=cfg.pixelsnail.n_out_res_block,
                n_cond_res_block=cfg.pixelsnail.n_cond_res_block,
                cond_res_channel=cfg.pixelsnail.n_res_channel,
            )
        elif hier == 'parent_bottom' or hier == 'parent_bottom_sibling':
            self.net = PixelSNAIL(
                hier,
                cfg.vqvae.bottom_shape,
                cfg.vqvae.K,
                cfg.pixelsnail.channel,
                5,
                4,
                cfg.pixelsnail.n_res_block,
                cfg.pixelsnail.n_res_channel,
                attention=False,
                dropout=cfg.pixelsnail.dropout,
                n_cond_res_block=cfg.pixelsnail.n_cond_res_block,
                cond_res_channel=cfg.pixelsnail.n_res_channel,
            )
        self.hier = hier

    def forward(self, batch):
        if self.hier == 'top' or self.hier == 'bottom':
            top = batch['top']
            bottom = batch['bottom']
            if self.hier == 'top':
                target = top
                out, _ = self.net(top)
            elif self.hier == 'bottom':
                target = bottom
                out, _ = self.net(bottom, condition=top)
            ret = {
                'gt': target,
                'pred': out,
            }
        elif self.hier == 'parent_top':
            latent_top = batch['latent_top']
            latent_bottom = batch['latent_bottom']
            target = batch['components_top'][:, 0]
            gt = target
            out, _ = self.net(target, condition=[latent_top, latent_bottom])
            ret = {
                'gt': gt,
                'pred': out
            }
        elif self.hier == 'parent_bottom':
            latent_top = batch['latent_top']
            latent_bottom = batch['latent_bottom']
            gt_top = batch['components_top'][:, 0]
            target = batch['components_bottom'][:, 0]
            gt = target
            out, _ = self.net(target, condition=[latent_top, latent_bottom, gt_top])
            ret = {
                'gt': gt,
                'pred': out
            }
        elif self.hier == 'parent_top_sibling':
            latent_top = batch['latent_top']
            latent_bottom = batch['latent_bottom']
            sibling_top = batch['components_top'][:, 0]
            sibling_bottom = batch['components_bottom'][:, 0]
            target = batch['components_top'][:, 1]
            out, _ = self.net(target, condition=[latent_top, latent_bottom, sibling_top, sibling_bottom])
            ret = {
                'gt': target,
                'pred': out
            }
        elif self.hier == 'parent_bottom_sibling':
            latent_top = batch['latent_top']
            latent_bottom = batch['latent_bottom']
            sibling_top = batch['components_top'][:, 0]
            sibling_bottom = batch['components_bottom'][:, 0]
            gt_top = batch['components_top'][:, 1]
            target = batch['components_bottom'][:, 1]
            out, _ = self.net(target, condition=[latent_top, latent_bottom, sibling_top, sibling_bottom, gt_top])
            ret = {
                'gt': target,
                'pred': out
            }
        return ret
