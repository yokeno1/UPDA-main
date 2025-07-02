import torch
import torch.nn as nn
import numpy as np

from .swin_transformer import SwinTransformer as swint_tiny
from .adaptation import AdversarialNetwork
from .adaptation import fusion_net
from .adaptation import rank_net


import torch.nn as nn
# import torch
# from torchvision.ops import roi_pool, roi_align
# from torch.nn import functional as F
import numpy as np
import math


class IQAHead(nn.Module):
    """MLP Regression Head for IQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
        self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc_last = nn.Linear(self.hidden_channels, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
        return qlt_score


class GMS_3DQA(nn.Module):
    def __init__(self, pretrained=True, checkpoint='path_to_checkpoint'):
        super().__init__()
        self.backbone = swint_tiny()
        self.iqa_head = IQAHead()
        self.mapping = mapping()
        self.rank_ad_net = rank_net()
        self.ad_net = AdversarialNetwork()
        if pretrained:
            self.load(self.backbone, checkpoint)
        # print(self.backbone.state_dict())

    def load(self, model, checkpoint):
        state_dict = torch.load(checkpoint)
        state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)

    def forward(self, image, t_image, mos, if_train=True):
        if if_train:
            image_size = image.shape
            image = image.view(-1, image_size[2], image_size[3], image_size[4])
            t_image = t_image.view(-1, image_size[2], image_size[3], image_size[4])
            feat = self.backbone(image)  ##(batch_size,49,768)
            t_feat = self.backbone(t_image)

            temp_feat, temp_t_feat = torch.mean(feat, dim=1), torch.mean(t_feat, dim=1)
            rank_loss = self.rank_ad_net(temp_feat, temp_t_feat, mos)

            feat_map, t_feat_map = self.mapping(feat, t_feat, if_fusion=False)

            feat = self.iqa_head(feat)
            avg_feat = torch.mean(feat, dim=1)
            avg_feat = avg_feat.view(image_size[0], image_size[1])
            score = torch.mean(avg_feat, dim=1)

            feat = self.iqa_head(feat_map)
            avg_feat = torch.mean(feat, dim=1)
            avg_feat = avg_feat.view(image_size[0], image_size[1])
            score_map = torch.mean(avg_feat, dim=1)


            feat_map, t_feat_map = torch.mean(feat_map, dim=1), torch.mean(t_feat_map, dim=1)
            domain_out = self.ad_net(feat_map)
            domain2_out = self.ad_net(t_feat_map)
            # print(score1, score2,)

            return domain_out, domain2_out, score, score_map, rank_loss[0], rank_loss[1]

        else:
            image_size = image.shape
            image = image.view(-1, image_size[2], image_size[3], image_size[4])
            # image2 = image2.view(-1, image_size[2], image_size[3], image_size[4])
            feat1 = self.backbone(image)  ##(batch_size,49,768)

            feat = self.iqa_head(feat1)
            avg_feat = torch.mean(feat, dim=1)
            avg_feat = avg_feat.view(image_size[0], image_size[1])
            score2 = torch.mean(avg_feat, dim=1)

            return score2

