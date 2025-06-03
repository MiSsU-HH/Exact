import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.nn import Parameter


# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# class PreNormLocal(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         x = x.permute(0, 2, 3, 1)
#         x = self.norm(x)
#         x = x.permute(0, 3, 1, 2)
#         # print('before fn: ', x.shape)
#         x = self.fn(x, **kwargs)
#         # print('after fn: ', x.shape)
#         return x


# class Conv1x1Block(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(dim, hidden_dim, kernel_size=1),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Conv2d(hidden_dim, dim, kernel_size=1),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., return_att=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.return_att = return_att

    def forward(self, x):
        # print(x.shape)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # print(q.shape, k.shape, v.shape)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        if self.return_att:
            weights = attn

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out + x
        if self.return_att:
            return out, weights
        else:
            return out


# class ReAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

#         self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

#         self.reattn_norm = nn.Sequential(
#             Rearrange('b h i j -> b i j h'),
#             nn.LayerNorm(heads),
#             Rearrange('b i j h -> b h i j')
#         )

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

#         # attention

#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
#         attn = dots.softmax(dim=-1)

#         # re-attention

#         attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
#         attn = self.reattn_norm(attn)

#         # aggregate and out

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         return out


# class LeFF(nn.Module):

#     def __init__(self, dim=192, scale=4, depth_kernel=3):
#         super().__init__()

#         scale_dim = dim * scale
#         self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
#                                      Rearrange('b n c -> b c n'),
#                                      nn.BatchNorm1d(scale_dim),
#                                      nn.GELU(),
#                                      Rearrange('b c (h w) -> b c h w', h=14, w=14)
#                                      )

#         self.depth_conv = nn.Sequential(
#             nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
#             nn.BatchNorm2d(scale_dim),
#             nn.GELU(),
#             Rearrange('b c h w -> b (h w) c', h=14, w=14)
#             )

#         self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
#                                        Rearrange('b n c -> b c n'),
#                                        nn.BatchNorm1d(dim),
#                                        nn.GELU(),
#                                        Rearrange('b c n -> b n c')
#                                        )

#     def forward(self, x):
#         x = self.up_proj(x)
#         x = self.depth_conv(x)
#         x = self.down_proj(x)
#         return x


# class LCAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
#         q = q[:, :, -1, :].unsqueeze(2)  # Only Lth element use as query

#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

#         attn = dots.softmax(dim=-1)

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         return out



class TemporalAwareAffinityPropagationModule(nn.Module):
    """
    Temporal Aware Affinity Propagation Module(TAAP).
    """

    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = self.get_kernel()
        self.register_buffer('kernel', kernel)
        self.alpha = 0.3

    def get_kernel(self):

        kernel = torch.zeros(8, 1, 3, 3)
        kernel[0, 0, 0, 0] = 1
        kernel[1, 0, 0, 1] = 1
        kernel[2, 0, 0, 2] = 1
        kernel[3, 0, 1, 0] = 1
        kernel[4, 0, 1, 2] = 1
        kernel[5, 0, 2, 0] = 1
        kernel[6, 0, 2, 1] = 1
        kernel[7, 0, 2, 2] = 1

        return kernel

    def get_dilated_neighbors(self, x):
        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)
        return torch.cat(x_aff, dim=2)
    
    def forward(self, imgs, pred_phis):
        _imgs = self.get_dilated_neighbors(imgs)
        _imgs_rep = imgs.unsqueeze(2).repeat(1, 1, _imgs.shape[2], 1, 1)
        _imgs_cos = F.cosine_similarity(_imgs, _imgs_rep, dim=1)
        _imgs_std = torch.std(_imgs, dim=2, keepdim=True)
        aff = _imgs_cos.unsqueeze(1) / (_imgs_std.mean(dim=1, keepdim=True) + 1e-8) / self.alpha 
        aff = F.softmax(aff, dim=2)
        for _ in range(self.num_iter):
            _pred_phis = self.get_dilated_neighbors(pred_phis)
            pred_phis = (_pred_phis * aff).sum(2)
        return pred_phis