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

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

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
            x = ff(x) + x    # ff layer

        out = self.norm(x)
        return self.project_out(out[..., -1, :])
    
class DenoiseTransformer(nn.Module):
    def __init__(
        self,
        opt
    ):
        super().__init__()

        self.dim = opt['dim']              # dim = dim of the model
        self.input_dim = opt['input_dim']  # dim of input coords(==3)
        self.st_dim = opt["st_dim"]        # dim of st-feats
        self.kps_dim = opt["kps_dim"]
        self.use_kp_type_embeds = opt['use_kp_type_embeds']
        self.num_keypoints = opt['num_keypoints']

        self.kp_type_embeds = nn.Embedding(self.num_keypoints, self.dim)

        self.null_condition_embeds = nn.Embedding(1, self.dim)
        self.null_noisy_res_embeds = nn.Embedding(1, self.dim)

        self.to_time_embeds = nn.Sequential(
            PositionalEncoding(self.dim),
            nn.Linear(self.dim, self.dim * 2),
            Swish(),
            nn.Linear(self.dim * 2, self.dim),
        )

        self.to_res_embeds = nn.Linear(3, self.dim)
        self.to_kps_embeds_3d = nn.Linear(3, self.kps_dim)  # in case using current preds from the regressor
        self.to_kps_embeds_2d = nn.Linear(2, self.kps_dim)  # in case using 2d keypoint inputs
        self.to_condition_embeds = nn.Linear(self.st_dim, self.dim) if not self.st_dim == self.dim else nn.Identity()

        self.learned_query = nn.Embedding(self.num_keypoints, self.dim)
        self.transformer = CausalTransformer(dim = self.dim, pose_dim=self.input_dim, **opt['transformer'])

    def forward(
        self,
        x,                 # noisy resisual values (B, num_kps, 3)
        st_feats,          # spatial-temporal features extracted from regressor (B, num_kps, st_dim)
        time,              # noise level (B,1)
        cond_drop_prob,     # condition drop probability for cf guidance
        cur_preds=None,    # current pose predictions from the regressor (B, n_kp, 3)
        kps_2d=None,       # 2d pose inputs (B, n_kp, 2)
        ):                 # output: the denoised residual values

        b = st_feats.shape[0]
        time_embed = self.to_time_embeds(time)   # (B, 1, dim)
        noisy_res_embed = self.to_res_embeds(x)  # (B, num_kps, dim)
        
        if cur_preds is not None and kps_2d is not None:
            assert self.kps_dim * 2 == self.st_dim
            kps_embed_3d = self.to_kps_embeds_3d(cur_preds)
            kps_embed_2d = self.to_kps_embeds_2d(kps_2d)
            final_conditions = st_feats + torch.cat([kps_embed_3d, kps_embed_2d], dim=-1)
        elif cur_preds is not None:
            assert self.kps_dim == self.st_dim
            kps_embed_3d = self.to_kps_embeds_3d(cur_preds)
            final_conditions = st_feats + kps_embed_3d
        elif kps_2d is not None:
            assert self.kps_dim == self.st_dim
            kps_embed_2d = self.to_kps_embeds_2d(kps_2d)
            final_conditions = st_feats + kps_embed_2d
        else:
            final_conditions = st_feats

        final_conditions = self.to_condition_embeds(final_conditions)

        if self.use_kp_type_embeds:
            final_conditions = final_conditions + self.kp_type_embeds.weight.unsqueeze(0).expand(b, -1, -1)
            noisy_res_embed = noisy_res_embed + self.kp_type_embeds.weight.unsqueeze(0).expand(b, -1, -1)
        
        # stachastic condition dropping for cf-guidance
        condition_keep_mask = prob_mask_like((b,self.num_keypoints,1), 1-cond_drop_prob, device=st_feats.device)
        noisy_res_keep_mask = prob_mask_like((b,self.num_keypoints,1), 1-cond_drop_prob, device=st_feats.device)
        
        null_condition_embeds = self.null_condition_embeds.weight.to(final_conditions.dtype)
        null_noisy_res_embeds = self.null_noisy_res_embeds.weight.to(noisy_res_embed.dtype)
        final_conditions = torch.where(
            condition_keep_mask, 
            final_conditions,
            null_condition_embeds
        )
        noisy_res_embed = torch.where(
            noisy_res_keep_mask,
            noisy_res_embed,
            null_noisy_res_embeds
        )

        learned_queries = self.learned_query.weight                       # (n_kps, dim)
        learned_queries = learned_queries.unsqueeze(0).expand(b, -1, -1)  # (b, n_kps, dim)
        if self.use_kp_type_embeds:
            learned_queries = learned_queries + self.kp_type_embeds.weight.unsqueeze(0).expand(b, -1, -1)

        # concat all tokens
        tokens = torch.cat((
            time_embed,
            final_conditions,
            noisy_res_embed,
            learned_queries
        ), dim = -2)

        # attention
        # get learned query, which should predict the denoised pose
        pred_res = self.transformer(tokens)[...,-self.num_keypoints:,:]  # (b, n_kps, dim)

        return pred_res
    
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=1.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale