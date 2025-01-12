# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.modules.attention import LinearAttention
from ldm.modules.mlp import MLP
from ldm.modules.diffusionmodules.util import make_coord_cell, make_coord

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb, scale_ratio=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)

        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.out_dim = block_in


    def forward(self, z, scale_ratio=None):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, scale_ratio)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, scale_ratio)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb, scale_ratio)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

        # end
        return h
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class Decoder_gs(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, tanh_out=False, use_linear_attn=False, training=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.out_dim = block_in
        self.segmentation_head = SegmentationHead(in_channels=self.out_dim, out_channels=100)
        self.training = training
    def forward(self, z, scale_ratio=None):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, scale_ratio)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, scale_ratio)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb, scale_ratio)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

        # end
        logits = self.segmentation_head(h)
        B, Class, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).contiguous().view(B * H * W, Class)
        if self.training:
            logits = F.gumbel_softmax(logits, tau=1, hard=False)
        if not self.training:
            logits = F.gumbel_softmax(logits, tau=1, hard=True)
        logits = logits.view(B, H, W, Class).permute(0, 3, 1, 2).contiguous()
        return h, logits
    
class DecoderOrigin(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     nn.Conv2d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LatentRescaler(nn.Module):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2):
        super().__init__()
        # residual block, interpolate, residual block
        self.factor = factor
        self.conv_in = nn.Conv2d(in_channels,
                                 mid_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.res_block1 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])
        self.attn = AttnBlock(mid_channels)
        self.res_block2 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])

        self.conv_out = nn.Conv2d(mid_channels,
                                  out_channels,
                                  kernel_size=1,
                                  )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x, None)
        x = torch.nn.functional.interpolate(x, size=(int(round(x.shape[2]*self.factor)), int(round(x.shape[3]*self.factor))))
        x = self.attn(x)
        for block in self.res_block2:
            x = block(x, None)
        x = self.conv_out(x)
        return x


class MergedRescaleEncoder(nn.Module):
    def __init__(self, in_channels, ch, resolution, out_ch, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 ch_mult=(1,2,4,8), rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(in_channels=in_channels, num_res_blocks=num_res_blocks, ch=ch, ch_mult=ch_mult,
                               z_channels=intermediate_chn, double_z=False, resolution=resolution,
                               attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv,
                               out_ch=None)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=intermediate_chn,
                                       mid_channels=intermediate_chn, out_channels=out_ch, depth=rescale_module_depth)

    def forward(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x


class MergedRescaleDecoder(nn.Module):
    def __init__(self, z_channels, out_ch, resolution, num_res_blocks, attn_resolutions, ch, ch_mult=(1,2,4,8),
                 dropout=0.0, resamp_with_conv=True, rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        tmp_chn = z_channels*ch_mult[-1]
        self.decoder = Decoder(out_ch=out_ch, z_channels=tmp_chn, attn_resolutions=attn_resolutions, dropout=dropout,
                               resamp_with_conv=resamp_with_conv, in_channels=None, num_res_blocks=num_res_blocks,
                               ch_mult=ch_mult, resolution=resolution, ch=ch)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=z_channels, mid_channels=tmp_chn,
                                       out_channels=tmp_chn, depth=rescale_module_depth)

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size//in_size))+1
        factor_up = 1.+ (out_size % in_size)
        print(f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}")
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, mid_channels=2*in_channels,
                                       out_channels=in_channels)
        self.decoder = Decoder(out_ch=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2,
                               attn_resolutions=[], in_channels=None, ch=in_channels,
                               ch_mult=[ch_mult for _ in range(num_blocks)])

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Resize(nn.Module):
    def __init__(self, in_channels=None, learned=False, mode="bilinear"):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

    def forward(self, x, scale_factor=1.0):
        if scale_factor==1.0:
            return x
        else:
            x = torch.nn.functional.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x

class FirstStagePostProcessor(nn.Module):

    def __init__(self, ch_mult:list, in_channels,
                 pretrained_model:nn.Module=None,
                 reshape=False,
                 n_channels=None,
                 dropout=0.,
                 pretrained_config=None):
        super().__init__()
        if pretrained_config is None:
            assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.instantiate_pretrained(pretrained_config)

        self.do_reshape = reshape

        if n_channels is None:
            n_channels = self.pretrained_model.encoder.ch

        self.proj_norm = Normalize(in_channels,num_groups=in_channels//2)
        self.proj = nn.Conv2d(in_channels,n_channels,kernel_size=3,
                            stride=1,padding=1)

        blocks = []
        downs = []
        ch_in = n_channels
        for m in ch_mult:
            blocks.append(ResnetBlock(in_channels=ch_in,out_channels=m*n_channels,dropout=dropout))
            ch_in = m * n_channels
            downs.append(Downsample(ch_in, with_conv=False))

        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)


    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        # self.pretrained_model.train = False
        for param in self.pretrained_model.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def encode_with_pretrained(self,x):
        c = self.pretrained_model.encode(x)
        if isinstance(c, DiagonalGaussianDistribution):
            c = c.mode()
        return  c

    def forward(self,x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)

        for submodel, downmodel in zip(self.model,self.downsampler):
            z = submodel(z,temb=None)
            z = downmodel(z)

        if self.do_reshape:
            z = rearrange(z,'b c h w -> b (h w) c')
        return z


class LIIF(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, cell_decode=True, local_ensemble=True):
        super().__init__()
        self.cell_decode = cell_decode
        self.local_ensemble = local_ensemble

        in_dim += 2 # attach coord
        if self.cell_decode:
            in_dim += 2
        self.imnet = MLP(in_dim, out_dim, hidden_list)
    
    def query_rgb(self, feat, coord, cell):
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def batched_predict(self, feat, coord, cell, bsize):
        with torch.no_grad():
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = self.query_rgb(feat, coord[:, ql: qr, :], cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        return pred

    def forward(self, feat, coord=None, cell=None, output_size=None, return_img=True, bsize=65536):
        if return_img:
            assert output_size is not None

        if self.training:
            bsize = 0
            
        if coord is None:
            assert output_size is not None
            coord, cell = make_coord_cell(feat.shape[0], output_size, output_size)
        if cell is None:
            assert output_size is not None
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / output_size
            cell[:, 1] *= 2 / output_size

        if bsize > 0:
            out = self.batched_predict(feat, coord, cell, bsize)
        else:
            out = self.query_rgb(feat, coord, cell)

        if return_img:
            out = rearrange(out, 'b (h w) c -> b c h w', h=output_size, w=output_size)

        return out



class GaussianSplatter(nn.Module):
    """A module that applies 2D Gaussian splatting to input features."""

    def __init__(self, in_dim, out_dim, hidden_list, kernel_size=5, hidden_dim=256, unfold_row=7, unfold_column=7,
                 num_points=100):
        """
        Initialize the 2D Gaussian Splatter module.
        Args:
            kernel_size (int): The size of the kernel to convert rasterization.
            unfold_row (int): The number of points in the row dimension of the Gaussian grid.
            unfold_column (int): The number of points in the column dimension of the Gaussian grid.
        """
        super(GaussianSplatter, self).__init__()
        self.feat, self.logits = None, None

        self.coef = nn.Conv2d(in_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(in_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)
        self.fc = MLP(hidden_dim, out_dim, hidden_list)

        # Key parameter in 2D Gaussian Splatter
        self.kernel_size = kernel_size
        self.row = unfold_row
        self.column = unfold_column
        self.num_points = num_points

        # Initialize Trainable Parameters
        sigma_x, sigma_y = torch.meshgrid(torch.linspace(0.2, 3.0, 10), torch.linspace(0.2, 3.0, 10))
        self.sigma_x = sigma_x.reshape(-1)
        self.sigma_y = sigma_y.reshape(-1)
        self.opacity = torch.sigmoid(torch.ones(self.num_points, 1, requires_grad=True))
        self.rho = torch.clamp(torch.zeros(self.num_points, 1, requires_grad=True), min=-1, max=1)
        self.sigma_x = nn.Parameter(self.sigma_x)  # Standard deviation in x-axis
        self.sigma_y = nn.Parameter(self.sigma_y)  # Standard deviation in y-axis
        self.opacity = nn.Parameter(self.opacity)  # Transparency of feature, shape=[num_points, 1]
        self.rho = nn.Parameter(self.rho)

    def weighted_gaussian_parameters(self, logits):
        """
        Computes weighted Gaussian parameters based on logits and the Gaussian kernel parameters (sigma_x, sigma_y, opacity).
        The logits tensor is used as a weight to compute a weighted sum of the Gaussian kernel parameters for each spatial
        location across the batch dimension. The resulting weighted parameters are then averaged across the batch dimension.
        Args:
            logits (torch.Tensor): Logits tensor of shape [batch, class, height, width].
        Returns:
            tuple: A tuple containing the weighted Gaussian parameters:
                - weighted_sigma_x (torch.Tensor): Tensor of shape [height * width] representing the weighted x-axis standard deviations.
                - weighted_sigma_y (torch.Tensor): Tensor of shape [height * width] representing the weighted y-axis standard deviations.
                - weighted_opacity (torch.Tensor): Tensor of shape [height * width] representing the weighted opacities.
        Description:
            This function computes weighted Gaussian parameters based on the input tensor, logits, and the provided Gaussian kernel parameters (sigma_x, sigma_y, and opacity). The logits tensor is used as a weight to compute a weighted sum of the Gaussian kernel parameters for each spatial location (height and width) across the batch dimension. The resulting weighted parameters are then averaged across the batch dimension, yielding tensors of shape [height * width] for the weighted sigma_x, sigma_y, and opacity.
        """
        batch_size, num_classes, height, width = logits.size()
        logits = logits.permute(0, 2, 3, 1)  # Reshape logits to [batch, height, width, class]

        # Compute weighted sum of Gaussian parameters across class dimension
        weighted_sigma_x = (logits * self.sigma_x.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_sigma_y = (logits * self.sigma_y.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_opacity = (logits * self.opacity[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_rho = (logits * self.rho[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        # Reshape and average across batch dimension
        weighted_sigma_x = weighted_sigma_x.reshape(batch_size, -1).mean(dim=0)
        weighted_sigma_y = weighted_sigma_y.reshape(batch_size, -1).mean(dim=0)
        weighted_opacity = weighted_opacity.reshape(batch_size, -1).mean(dim=0)
        weighted_rho = weighted_rho.reshape(batch_size, -1).mean(dim=0)

        return weighted_sigma_x, weighted_sigma_y, weighted_opacity, weighted_rho

    def gen_feat(self, inp):
        """Generate feature and logits by encoder."""
        self.inp = inp
        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).cuda().permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])
        return self.feat, self.logits

    def query_rgb(self, inp, logits, coord, scale, cell=None):
        """
        Continuous sampling through 2D Gaussian Splatting.
        Args:
            coord (torch.Tensor): [Batch, Sample_q, 2]. The normalized coordinates of HR space (of range [-1, 1]).
            cell (torch.Tensor): [Batch, Sample_q, 2]. The normalized cell size of HR space.
            scale (torch.Tensor): [Batch]. The magnification scale of super-resolution. (1, 4) during training.
        Returns:
            torch.Tensor: The output features after Gaussian splatting, of the same shape as the input.
        """
        # 1. Get LR feature and logits
        feat, lr_feat, logits = inp[:, :8, :, :], inp[:, 8:, :, :], logits  # Channel decoupling
        feat_size, feat_device = feat.shape, feat.device

        # 2. Calculate the high-resolution image size
        hr_h = round(feat.shape[-2] * scale)  # shape: [batch size]
        hr_w = round(feat.shape[-1] * scale)

        # 3. Unfold the feature / logits to many small patches to avoid extreme GPU memory consumption
        num_kernels_row = math.ceil(feat_size[-2] / self.row)
        num_kernels_column = math.ceil(feat_size[-1] / self.column)
        upsampled_size = (num_kernels_row * self.row, num_kernels_column * self.column)
        upsampled_inp = F.interpolate(feat, size=upsampled_size, mode='bicubic', align_corners=False)
        upsampled_logits = F.interpolate(logits, size=upsampled_size, mode='bicubic', align_corners=False)
        unfold = nn.Unfold(kernel_size=(self.row, self.column), stride=(self.row, self.column))
        unfolded_feature = unfold(upsampled_inp)
        unfolded_logits = unfold(upsampled_logits)
        # Unfolded_feature dimension becomes [Batch, C*K*K, L], where L is the number of columns after unfolding
        L = unfolded_feature.shape[-1]
        unfolded_feature_reshaped = unfolded_feature.transpose(1, 2). \
            reshape(feat_size[0] * L, feat_size[1], self.row, self.column)
        unfold_feat = unfolded_feature_reshaped  # shape: [num of patch * batch, channel, self.row, self.column]
        unfolded_logits_reshaped = unfolded_logits.transpose(1, 2). \
            reshape(logits.shape[0] * L, logits.shape[1], self.row, self.column)
        unfold_logits = unfolded_logits_reshaped  # shape: [num of patch * batch, channel, self.row, self.column]

        # 4. Generate colors_(features) and coords_norm
        coords_ = generate_meshgrid(unfold_feat.shape[-2], unfold_feat.shape[-1])
        num_LR_points = unfold_feat.shape[-2] * unfold_feat.shape[-1]
        colors_, coords_norm = fetching_features_from_tensor(unfold_feat, coords_)

        # 5. Rasterization: Generating grid
        # 5.1. Spread Gaussian points over the whole feature map
        batch_size, channel, _, _ = unfold_feat.shape
        weighted_sigma_x, weighted_sigma_y, weighted_opacity, weighted_rho = self.weighted_gaussian_parameters(
            unfold_logits)
        sigma_x = weighted_sigma_x.view(num_LR_points, 1, 1)
        sigma_y = weighted_sigma_y.view(num_LR_points, 1, 1)
        rho = weighted_rho.view(num_LR_points, 1, 1)

        # 5.2. Gaussian expression
        covariance = torch.stack(
            [torch.stack([sigma_x ** 2 + 1e-5, rho * sigma_x * sigma_y], dim=-1),
             torch.stack([rho * sigma_x * sigma_y, sigma_y ** 2 + 1e-5], dim=-1)], dim=-2
        )  # when correlation rou is set to zero, covariance will always be positive semi-definite
        inv_covariance = torch.inverse(covariance).to(feat_device)

        # 5.3. Choosing a broad range for the distribution [-5,5] to avoid any clipping
        start = torch.tensor([-5.0], device=feat_device).view(-1, 1)
        end = torch.tensor([5.0], device=feat_device).view(-1, 1)
        base_linspace = torch.linspace(0, 1, steps=self.kernel_size, device=feat_device)
        ax_batch = start + (end - start) * base_linspace
        # Expanding dims for broadcasting
        ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, self.kernel_size)
        ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, self.kernel_size, -1)

        # 5.4. Creating a batch-wise meshgrid using broadcasting
        xx, yy = ax_batch_expanded_x, ax_batch_expanded_y
        xy = torch.stack([xx, yy], dim=-1)
        z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy)
        kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=feat_device) *
                                 torch.sqrt(torch.det(covariance)).to(feat_device).view(num_LR_points, 1, 1))
        kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
        kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
        kernel_normalized = kernel / kernel_max_2
        kernel_reshaped = kernel_normalized.repeat(1, channel, 1).contiguous(). \
            view(num_LR_points * channel, self.kernel_size, self.kernel_size)
        kernel_color = kernel_reshaped.unsqueeze(0).reshape(num_LR_points, channel, self.kernel_size, self.kernel_size)

        # 5.5. Adding padding to make kernel size equal to the image size
        pad_h = round(unfold_feat.shape[-2] * scale) - self.kernel_size
        pad_w = round(unfold_feat.shape[-1] * scale) - self.kernel_size
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Kernel size should be smaller or equal to the image size.")
        padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
        kernel_color_padded = torch.nn.functional.pad(kernel_color, padding, "constant", 0)

        # 5.6. Create a batch of 2D affine matrices
        b, c, h, w = kernel_color_padded.shape  # num_LR_points, channel, hr_h, hr_w
        theta = torch.zeros(batch_size, b, 2, 3, dtype=torch.float32, device=feat_device)
        theta[:, :, 0, 0] = 1.0
        theta[:, :, 1, 1] = 1.0
        theta[:, :, :, 2] = coords_norm
        grid = F.affine_grid(theta.view(-1, 2, 3), size=[batch_size * b, c, h, w], align_corners=True).contiguous()
        kernel_color_padded_expanded = kernel_color_padded.repeat(batch_size, 1, 1, 1).contiguous()
        kernel_color_padded_translated = F.grid_sample(kernel_color_padded_expanded.contiguous(), grid.contiguous(),
                                                       align_corners=True)
        kernel_color_padded_translated = kernel_color_padded_translated.view(batch_size, b, c, h, w)

        # 6. Apply Gaussian splatting
        # colors_.shape = [batch, num_LR_points, channel], colors.shape = [batch, num_LR_points, channel]
        colors = colors_ * weighted_opacity.to(feat_device).unsqueeze(-1).expand(batch_size, -1, -1)
        color_values_reshaped = colors.unsqueeze(-1).unsqueeze(-1)
        final_image_layers = color_values_reshaped * kernel_color_padded_translated
        final_image = final_image_layers.sum(dim=1)
        final_image = torch.clamp(final_image, 0, 1)

        # 7. Fold the input back to the original size
        # Calculate the number of kernels needed to cover each dimension.
        kernel_h, kernel_w = round(self.row * scale), round(self.column * scale)
        fold = nn.Fold(output_size=(kernel_h * num_kernels_row, kernel_w * num_kernels_column),
                       kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        final_image = final_image.reshape(feat_size[0], L, feat_size[1] * kernel_h * kernel_w).transpose(1, 2)
        final_image = fold(final_image)
        final_image = F.interpolate(final_image, size=(hr_h, hr_w), mode='bicubic', align_corners=False)
        # Combine channel
        lr_feat = F.interpolate(lr_feat, size=(hr_h, hr_w), mode='bicubic', align_corners=False)
        final_image = torch.concat((final_image, lr_feat), dim=1)

        # 8. Augmentation (Useful for improving out-of-distribution scale performance)
        coef = self.coef(final_image)
        freq = self.freq(final_image)
        feat_coord = self.feat_coord
        coord_ = coord.clone()
        q_coef = F.grid_sample(coef, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_freq = F.grid_sample(freq, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                  :] \
            .permute(0, 2, 1)
        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]
        bs, q = coord.shape[:2]
        q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
        q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
        q_freq = torch.sum(q_freq, dim=-2)
        q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
        q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)

        inp = torch.mul(q_coef, q_freq)

        pred = self.fc(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)

        return pred
    def batched_predict(self, inp, logits, coord, scale, cell, bsize):
        with torch.no_grad():
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = self.query_rgb(inp, logits, coord[:, ql: qr, :], scale, cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        return pred
    def forward(self, inp, logits, coord, cell=None, output_size=None, return_img=True, bsize=65536):
        scale = output_size / inp.shape[-1]
        self.gen_feat(inp)
        if return_img:
            assert output_size is not None

        if self.training:
            bsize = 0
            
        if coord is None:
            assert output_size is not None
            coord, cell = make_coord_cell(inp.shape[0], output_size, output_size)
        if cell is None:
            assert output_size is not None
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / output_size
            cell[:, 1] *= 2 / output_size

        if bsize > 0:
            out = self.batched_predict(inp, logits, coord, scale, cell, bsize)
        else:
            out = self.query_rgb(inp, logits, coord, scale, cell)

        if return_img:
            out = rearrange(out, 'b (h w) c -> b c h w', h=output_size, w=output_size)



        return out
    

def generate_meshgrid(height, width):
    """
    Generate a meshgrid of coordinates for a given image dimensions.
    Args:
        height (int): Height of the image.
        width (int): Width of the image.
    Returns:
        torch.Tensor: A tensor of shape [height * width, 2] containing the (x, y) coordinates for each pixel in the image.
    """
    # Generate all pixel coordinates for the given image dimensions
    y_coords, x_coords = torch.arange(0, height), torch.arange(0, width)
    # Create a grid of coordinates
    yy, xx = torch.meshgrid(y_coords, x_coords)
    # Flatten and stack the coordinates to obtain a list of (x, y) pairs
    all_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return all_coords


def fetching_features_from_tensor(image_tensor, input_coords):
    """
    Extracts pixel values from a tensor of images at specified coordinate locations.
    Args:
        image_tensor (torch.Tensor): A 4D tensor of shape [batch, channel, height, width] representing a batch of images.
        input_coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the (x, y) coordinates at which to extract pixel values.
    Returns:
        color_values (torch.Tensor): A 3D tensor of shape [batch, N, channel] containing the pixel values at the specified coordinates.
        coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the normalized coordinates in the range [-1, 1].
    """
    # Normalize pixel coordinates to [-1, 1] range
    input_coords = input_coords.to(image_tensor.device)
    coords = input_coords / torch.tensor([image_tensor.shape[-2], image_tensor.shape[-1]],
                                         device=image_tensor.device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=image_tensor.device).float()
    coords = (center_coords_normalized - coords) * 2.0

    # Fetching the colour of the pixels in each coordinates
    batch_size = image_tensor.shape[0]
    input_coords_expanded = input_coords.unsqueeze(0).expand(batch_size, -1, -1)

    y_coords = input_coords_expanded[..., 0].long()
    x_coords = input_coords_expanded[..., 1].long()
    batch_indices = torch.arange(batch_size).view(-1, 1).to(input_coords.device)

    color_values = image_tensor[batch_indices, :, x_coords, y_coords]

    return color_values, coords