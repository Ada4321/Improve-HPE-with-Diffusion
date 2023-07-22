import torch
import torch.nn as nn

import sys
sys.path.append('/home/ubuntu/Improve-HPE-with-Diffusion/model/regression_modules')
from Resnet import ResNet


class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)  # initialize weights of self.linear

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y
    
class Regressor(nn.Module):
    def __init__(
            self,
            opt
            ):
        super(Regressor, self).__init__()
        # self.fc_dim = opt['NUM_FC_FILTERS']
        self.preset_opt = opt['preset']
        self.norm_layer = nn.BatchNorm2d
        self.num_joints = self.preset_opt['num_joints']
        self.height_dim = self.preset_opt['image_size'][0]
        self.width_dim = self.preset_opt['image_size'][1]

        # ResNet layers
        assert opt['num_layers'] in [18, 34, 50, 101, 152]
        self.preact = ResNet(f"resnet{opt['num_layers']}") 

        # Imagenet pretrain model
        import torchvision.models as tm  # noqa: F401,F403
        x = eval(f"tm.resnet{opt['num_layers']}(pretrained=True)")
        self.feature_channel = {
            18: 512,
            34: 512,
            50: 2048,
            101: 2048,
            152: 2048
        }[opt['num_layers']] 
        # self.hidden_list = opt['HIDDEN_LIST'] 

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        # average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # fc layer: global feature => key points
        self.fc_layer = Linear(self.feature_channel, self.num_joints * 2)

    def forward(self, x):
        BATCH_SIZE = x.shape[0]

        # obtain global image features
        feat = self.preact(x)

        _, _, f_h, f_w = feat.shape  # (B,C,H,W)
        feat = self.avg_pool(feat).reshape(BATCH_SIZE, -1) # global ResNet feature (B,C,1,1) => (B,C)

        # regress key points from global features
        pred_jts = self.fc_layer(feat).reshape(BATCH_SIZE, self.num_joints, -1)
        assert pred_jts.shape[-1] == 2
        pred_jts = pred_jts.reshape(BATCH_SIZE, -1)       # produce mu
        assert pred_jts.shape[-1] == self.num_joints * 2

        return pred_jts, feat