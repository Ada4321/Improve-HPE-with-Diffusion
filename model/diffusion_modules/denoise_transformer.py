import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack
from rotary_embedding_torch import RotaryEmbedding

import sys
sys.path.append('/root/Improve-HPE-with-Diffusion/model/diffusion_modules')
from utils import *


def l2norm(t):
    return F.normalize(t, dim = -1)

# layer norm
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5, fp16_eps = 1e-3, stable = False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim = -1, keepdim = True).detach()

        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g  # input.rsqrt() = 1 / sqrt(input)

# relative positional bias for causal transformer
class RelPosBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        num_buckets = 32,
        max_distance = 128
    ):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large)

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

class SwiGLU(nn.Module):
    """ used successfully in https://arxiv.org/abs/2204.0231 """
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.silu(gate)

def FeedForward(
    dim,
    mult = 4,
    dropout = 0.,
    post_activation_norm = False
):
    """ post-activation norm https://arxiv.org/abs/2110.09456 """

    inner_dim = int(mult * dim)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = True,
        rotary_emb = None,
        cosine_sim = True,
        cosine_sim_scale = 16
    ):
        super().__init__()
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))  # learnable
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)  # q -- (b,h,n,d)  # k,v -- (b,n,d)
        q = q * self.scale

        # rotary embeddings

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # add null key / value for classifier free guidance in prior net

        # nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        # k = torch.cat((nk, k), dim = -2)  # k,v -- (b,n+1,d)
        # v = torch.cat((nv, v), dim = -2)

        # whether to use cosine sim

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        pose_dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, pose_dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        #attn_bias = self.rel_pos_bias(n, n + 1, device = device)
        attn_bias = self.rel_pos_bias(n, n, device = device)

        for attn, ff in self.layers:  # transformer encoder layers
            x = attn(x, attn_bias = attn_bias) + x  # self-attention layer
            x = ff(x) + x                           # ff layer

        out = self.norm(x)
        #return self.project_out(out[..., -1, :])
        return self.project_out(out)

class SimpleAttention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            dim_head = 64,
            heads = 8,
            dropout = 0.,
            cosine_sim = True,
            cosine_sim_scale = 16
            ) -> None:
        super().__init__()
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # self.null_kv = nn.Parameter(torch.randn(2, dim_head))  # learnable
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, mask = None):
        # b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)  # q -- (b,h,n,d)  # k,v -- (b,n,d)
        q = q * self.scale

        # add null key / value for classifier free guidance in prior net

        # nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        # k = torch.cat((nk, k), dim = -2)  # k,v -- (b,n+1,d)
        # v = torch.cat((nv, v), dim = -2)

        # whether to use cosine sim

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SimpleTransformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            pose_dim,
            depth,
            dim_head = 64,
            heads = 8,
            ff_mult = 4,
            norm_in = False,
            norm_out = True,
            attn_dropout = 0.,
            ff_dropout = 0.,
            final_proj = True,
            normformer = False
            ) -> None:
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SimpleAttention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, pose_dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        # n, device = x.shape[1], x.device

        x = self.init_norm(x)

        for attn, ff in self.layers:  # transformer encoder layers
            x = attn(x) + x  # self-attention layer
            x = ff(x) + x                           # ff layer

        out = self.norm(x)
        return self.project_out(out[..., -1, :])

class DenoiseTransformer(nn.Module):
    def __init__(
        self,
        opt
    ):
        super().__init__()

        self.dim = opt['dim']  # dim = dim of image channels
        self.input_dim = opt['input_dim']
        self.use_coord_type_embeds = opt['use_coord_type_embeds']
        self.use_kp_type_embeds = opt['use_kp_type_embeds']
        self.use_cur_preds_type_embed = opt['use_cur_preds_type_embed']
        self.use_noisy_res_type_embed = opt['use_noisy_res_type_embed']

        self.num_keypoints = opt['num_keypoints']
        # self.num_image_embeds = opt['num_image_embeds']

        # self.coord_type_embeds = nn.Embedding(self.input_dim, int(self.dim // 2))
        self.kp_type_embeds = nn.Embedding(self.num_keypoints, self.dim)
        # self.cur_preds_type_embed = nn.Embedding(1, self.dim)
        # self.noisy_res_type_embed = nn.Embedding(1, self.dim)
        # self.z_embed = nn.Embedding(1, self.dim)
        # self.not_z_embed = nn.Embedding(1, self.dim)

        self.to_time_embeds = nn.Sequential(
            PositionalEncoding(self.dim),
            nn.Linear(self.dim, self.dim * 2),
            Swish(),
            nn.Linear(self.dim * 2, self.dim),
        )
        self.to_image_embeds = nn.Sequential(
            nn.Linear(opt['image_dim'], self.num_image_embeds * self.dim),
            Rearrange('b (n d) -> b n d', n = self.num_image_embeds)
        )
        self.to_pose_embeds = nn.Sequential(
            PositionalEncoding(self.dim, 
                               use_coord_type_embeds=self.use_coord_type_embeds, 
                               coord_type_embeds=[self.z_embed.weight, self.not_z_embed.weight]),
            nn.Linear(self.dim, self.dim * 2),
            Swish(),
            nn.Linear(self.dim * 2, self.dim),
        )
        self.to_res_embeds = nn.Sequential(
            PositionalEncoding(self.dim, 
                               use_coord_type_embeds=self.use_coord_type_embeds,
                                coord_type_embeds=[self.z_embed.weight, self.not_z_embed.weight]),
            nn.Linear(self.dim, self.dim * 2),
            Swish(),
            nn.Linear(self.dim * 2, self.dim),
        )

        self.learned_query = nn.Embedding(self.num_keypoints, self.dim)
        self.transformer = CausalTransformer(dim = self.dim, pose_dim=self.input_dim, **opt['transformer'])

        # torch.nn.init.constant_(self.sigma_linear.weight, 1.)
        # torch.nn.init.constant_(self.sigma_linear.bias, 0.)

    def forward(
        self,
        x,               # noisy resisual values (B,n_kp)
        image_embed,     # image embedding extracted from regressor (B,dim_img)
        time,            # noise level (B,1)
        cur_preds=None,  # current pose predictions from the regressor (B,n_kp)
        ):               # output: the denoised residual values

        image_embed = self.to_image_embeds(image_embed)  #  (B,num_image_embeds,dim)
        b = image_embed.shape[0]
        time_embed = self.to_time_embeds(time)           #  (B,num_time_embeds,dim)
        if cur_preds is not None:
            pose_embed = self.to_pose_embeds(cur_preds)  #  (B,num_pose_embeds,dim)
            if self.use_cur_preds_type_embed:
                pose_embed = pose_embed + self.cur_preds_type_embed.weight.unsqueeze(0).expand(b, self.num_keypoints, -1)
            if self.use_kp_type_embeds:
                pose_embed = pose_embed + self.kp_type_embeds.weight.unsqueeze(0).expand(b, -1, -1)

        noisy_res_embed = self.to_res_embeds(x)          #  (B,num_pose_embeds,dim)
        if self.use_noisy_res_type_embed:
            noisy_res_embed = noisy_res_embed + self.noisy_res_type_embed.weight.unsqueeze(0).expand(b, self.num_keypoints, -1)
        if self.use_kp_type_embeds:
            noisy_res_embed = noisy_res_embed + self.kp_type_embeds.weight.unsqueeze(0).expand(b, -1, -1)

        learned_queries = self.learned_query.weight                                          # (n_kps, dim)
        learned_queries = learned_queries.unsqueeze(0).expand(b, -1, -1)  # (b, n_kps, dim)
        if self.use_kp_type_embeds:
            learned_queries = learned_queries + self.kp_type_embeds.weight.unsqueeze(0).expand(b, -1, -1)

        # all condition tokens
        if cur_preds is not None:
            tokens = torch.cat((
                image_embed,
                time_embed,
                pose_embed,
                noisy_res_embed,
                learned_queries
            ), dim = -2)
        else:
            tokens = torch.cat((
                image_embed,
                time_embed,
                noisy_res_embed,
                learned_queries
            ), dim = -2)

        # attention
        # get learned query, which should predict the denoised pose
        pred_res = self.transformer(tokens)  # pred_pose should be the same size as input x
        pred_res = pred_res[..., -self.num_keypoints:, :].reshape(b, -1)

        return pred_res