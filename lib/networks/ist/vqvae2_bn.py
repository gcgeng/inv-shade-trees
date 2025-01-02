import torch
from lib.config import cfg
from torch import nn
from torch.nn import functional as F

import torch.distributed as dist_fn

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim) # (N, D)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype) # (N, K)
        embed_ind = embed_ind.view(*input.shape[:-1]) # (N,)
        quantize = self.embed_code(embed_ind)

        if self.training:
            # Update the cluster embedding using equation described in the paper
            # "exponential moving average updates"
            embed_onehot_sum = embed_onehot.sum(0) # (K,)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot # (D, N) @ (N, K) = (D, K)
            # The sum of every input vectors in each cluster

            # TODO: use distributed training
            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channel // 2),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channel),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channel // 2),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        elif stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channel // 2),
                nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channel // 2),
                nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channel // 2),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        elif stride == 16:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channel // 2),
                nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channel // 2),
                nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channel // 2),
                nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channel // 2),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        else:
            raise NotImplementedError

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class UpsampleDeconv(nn.Module):
    """
    Implement deconv operation using nearest neighbor
    upsampling and normal convolution.
    """ 
    def __init__(self, in_channel, channel, out_channel, stride):
        super().__init__()

        self.blocks = []

        upsample_method = cfg.upsample_method

        if stride == 2:
            self.blocks.append(nn.Upsample(scale_factor=2, mode=upsample_method))
            self.blocks.append(nn.Conv2d(in_channel, out_channel, 3, padding=1))
            self.blocks.append(nn.ReLU(inplace=True))
        elif stride == 4:
            self.blocks.append(nn.Upsample(scale_factor=2, mode=upsample_method))
            self.blocks.append(nn.Conv2d(in_channel, channel, 3, padding=1))
            self.blocks.append(nn.ReLU(inplace=True))
            self.blocks.append(nn.BatchNorm2d(channel))
            self.blocks.append(nn.Upsample(scale_factor=2, mode=upsample_method))
            self.blocks.append(nn.Conv2d(channel, out_channel, 3, padding=1))
            self.blocks.append(nn.ReLU(inplace=True))
        elif stride == 8:
            self.blocks.append(nn.Upsample(scale_factor=2, mode=upsample_method))
            self.blocks.append(nn.Conv2d(in_channel, channel, 3, padding=1))
            self.blocks.append(nn.ReLU(inplace=True))
            self.blocks.append(nn.BatchNorm2d(channel))
            self.blocks.append(nn.Upsample(scale_factor=2, mode=upsample_method))
            self.blocks.append(nn.Conv2d(channel, channel, 3, padding=1))
            self.blocks.append(nn.ReLU(inplace=True))
            self.blocks.append(nn.BatchNorm2d(channel))
            self.blocks.append(nn.Upsample(scale_factor=2, mode=upsample_method))
            self.blocks.append(nn.Conv2d(channel, out_channel, 3, padding=1))
            self.blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))


        if not cfg.use_upsample:
            if stride == 4:
                blocks.extend(
                    [
                        nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(channel // 2),
                        nn.ConvTranspose2d(
                            channel // 2, out_channel, 4, stride=2, padding=1
                        ),
                    ]
                )

            elif stride == 2:
                blocks.append(
                    nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
                )

            elif stride == 8:
                blocks.extend(
                    [
                        nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(channel // 2),
                        nn.ConvTranspose2d(channel // 2, channel // 2, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(channel // 2),
                        nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
                    ]
                )

            elif stride == 16:
                blocks.extend(
                    [
                        nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(channel // 2),
                        nn.ConvTranspose2d(channel // 2, channel // 2, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(channel // 2),
                        nn.ConvTranspose2d(channel // 2, channel // 2, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(channel // 2),
                        nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
                    ]
                )

            else:
                raise NotImplementedError
        else:
            if cfg.more_resnet:
                self.upsampler = UpsampleDeconv(channel, channel // 2, channel, stride)
                self.bn = nn.BatchNorm2d(channel)
                self.post_res = nn.Sequential(*[ResBlock(channel, n_res_channel) for _ in range(5)])
                self.final_conv = nn.Conv2d(channel, out_channel, 3, padding=1)
            else:
                self.upsampler = UpsampleDeconv(channel, channel // 2, out_channel, stride)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        if not cfg.use_upsample:
            x = self.blocks(input)
        else:
            x = self.blocks(input)
            x = self.upsampler(x)
            if cfg.more_resnet:
                x = self.bn(x)
                x = self.post_res(x)
                x = self.final_conv(x)
        x = torch.clamp(torch.tanh(x) * 1.05, -1, 1)
        return x

class Network(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=cfg.n_embed,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=cfg.bottom_stride)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=cfg.top_stride)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=cfg.top_stride
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        if cfg.top_stride == 2:
            self.upsample_t = nn.ConvTranspose2d(
                embed_dim, embed_dim, 4, stride=2, padding=1
            )
        elif cfg.top_stride == 4:
            self.upsample_t = nn.Sequential(
                *[
                    nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(embed_dim),
                    nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1),
                ]
            )
        else:
            raise NotImplementedError
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=cfg.bottom_stride,
        )

    def forward(self, input, batch):
        quant_t, quant_b, diff, id_t, id_b = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        ret = {
            'latent_t': id_t,
            'latent_b': id_b,
            'quant_t': quant_t,
            'quant_b': quant_b,
            'x_tiled': dec,
            'diff': diff
        }

        return ret

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
