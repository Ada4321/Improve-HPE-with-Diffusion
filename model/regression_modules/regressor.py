import torch
import torch.nn as nn
from functools import partial
from einops import rearrange

import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')
sys.path.append('/root/Improve-HPE-with-Diffusion/model/regression_modules')

from core.registry import Registry
MODEL_REGISTRY = Registry('model')

from mixste.blocks import MixSTEBlock

    
@MODEL_REGISTRY.register()
class ConvRegressor(nn.Module):
    def __init__(
            self,
            backbone,
            neck
            ):
        super(ConvRegressor, self).__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, x):
        BATCH_SIZE = x.shape[0]
        feat = self.backbone(x)
        output, feat = self.neck(BATCH_SIZE, feat)
        return output, feat


@MODEL_REGISTRY.register()
class MixSTE2(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, **kwargs):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(int(in_chans), embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            MixSTEBlock(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            MixSTEBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        # self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=num_frame, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )

        self.head_sigma = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def STE_forward(self, x):
        b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size
        x = rearrange(x, 'b f n c  -> (b f) n c', )
        ### now x is [batch_size*receptive frames, joint_num, 2 channels]
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        blk = self.STEblocks[0]
        x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

    def TTE_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        blk = self.TTEblocks[0]
        x = blk(x)

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        
        return x

    def forward(self, x):
        """return st-features and predicted 3d pose sequence
            st_feats: spatial-temporal features of pose sequence, (b,f,n,cw=512)
            x: predicted 3d pose sequence, (b,f,n,3)
        """
        b, f, n, c = x.shape
        ### now x is [batch_size, receptive frames, joint_num, 2 channels]
        # x shape:(b f n c)
        x = self.STE_forward(x)
        # now x shape is (b n) f cw
        x = self.TTE_foward(x)
        # now x shape is (b n) f cw
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.ST_foward(x)
        # now x shape is (b f n cw)
        st_feats = x
        mu = self.head(st_feats)
        #sigma = self.head_sigma(st_feats).sigmoid() + 0.5
        sigma = self.head_sigma(st_feats).sigmoid()
        # now x shape is (b f n 3)
        st_feats = st_feats.view(b, f, n, -1)
        mu = mu.view(b, f, n, -1)
        sigma = sigma.view(b, f, n, -1)

        return mu, sigma, st_feats
    
@MODEL_REGISTRY.register()
class STMO(nn.Module):
    pass

@MODEL_REGISTRY.register()
class DSTFormer(nn.Module):
    pass

@MODEL_REGISTRY.register()
class SRNet(nn.Module):
    pass