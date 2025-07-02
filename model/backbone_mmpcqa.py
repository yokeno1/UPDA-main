import torch
import torch.nn as nn
from model.backbones import pointnet2, resnet50
from model.transformer import TransformerEncoderLayer_CMA
from .adaptation import AdversarialNetwork
from .adaptation import fusion_net
from .adaptation import rank_net

class CMA_fusion(nn.Module):
    def __init__(self, img_inplanes, pc_inplanes, cma_planes=1024):
        super(CMA_fusion, self).__init__()
        self.encoder = TransformerEncoderLayer_CMA(d_model=cma_planes, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.linear1 = nn.Linear(img_inplanes, cma_planes)
        self.linear2 = nn.Linear(pc_inplanes, cma_planes)
        # self.quality1 = nn.Linear(cma_planes * 4, cma_planes * 2)
        # self.quality2 = nn.Linear(cma_planes * 2, 1)
        self.img_bn = nn.BatchNorm1d(cma_planes)
        self.pc_bn = nn.BatchNorm1d(cma_planes)

    def forward(self, img, pc):
        # linear mapping and batch normalization
        img = self.linear1(img)
        img = self.img_bn(img)
        pc = self.linear2(pc)
        pc = self.pc_bn(pc)
        # cross modal attention and feature fusion
        img = img.unsqueeze(0)
        pc = pc.unsqueeze(0)
        img_a, pc_a = self.encoder(img, pc)
        output = torch.cat((img, img_a, pc_a, pc), dim=2)
        feature = output.squeeze(0)

        return feature


class MM_PCQAnet(nn.Module):
    def __init__(self):
        super(MM_PCQAnet, self).__init__()
        self.img_inplanes = 2048
        self.pc_inplanes = 1024
        self.cma_planes = 1024
        self.img_backbone = resnet50(pretrained=True)
        self.pc_backbone = pointnet2()
        self.cma = CMA_fusion(img_inplanes=self.img_inplanes, pc_inplanes=self.pc_inplanes,
                                     cma_planes=self.cma_planes)
        self.fusion = fusion_net()
        self.rank_ad_net = rank_net()
        self.ad_net = AdversarialNetwork()

        self.quality1 = nn.Linear(self.cma_planes * 4, self.cma_planes * 2)
        self.quality2 = nn.Linear(self.cma_planes * 2, 1)

    def forward(self, img, pc, t_img, t_pc, mos, if_train=True):
        if if_train:
            # extract features from the projections
            img_size = img.shape
            img = img.view(-1, img_size[2], img_size[3], img_size[4])
            img = self.img_backbone(img)
            img = torch.flatten(img, 1)
            # average the projection features
            img = img.view(img_size[0], img_size[1], self.img_inplanes)
            img = torch.mean(img, dim=1)

            # extract features from patches
            pc_size = pc.shape
            pc = pc.view(-1, pc_size[2], pc_size[3])
            pc = self.pc_backbone(pc)
            # average the patch features
            pc = pc.view(pc_size[0], pc_size[1], self.pc_inplanes)
            pc = torch.mean(pc, dim=1)

            ##目标域特征
            t_img = t_img.view(-1, img_size[2], img_size[3], img_size[4])
            t_img = self.img_backbone(t_img)
            t_img = torch.flatten(t_img, 1)
            # average the projection features
            t_img = t_img.view(img_size[0], img_size[1], self.img_inplanes)
            t_img = torch.mean(t_img, dim=1)

            t_pc = t_pc.view(-1, pc_size[2], pc_size[3])
            t_pc = self.pc_backbone(t_pc)
            # average the patch features
            t_pc = t_pc.view(pc_size[0], pc_size[1], self.pc_inplanes)
            t_pc = torch.mean(t_pc, dim=1)

            # attention, fusion
            feat1 = self.cma(img, pc)
            # feat_map = self.mapping(feat1)
            feat2 = self.cma(t_img, t_pc)

            rank_loss = self.rank_ad_net(feat1, feat2, mos)
            feat_map, t_feat_map = self.fusion(feat1, feat2, if_train)

            # dist_class1 = self.classification(feat1)
            output = self.quality1(feat1)
            score1 = self.quality2(output)

            # dist_class2 = self.classification(feat_map)
            output = self.quality1(feat_map)
            score2 = self.quality2(output)

            domain_out = self.ad_net(feat_map)
            t_domain_out = self.ad_net(t_feat_map)
            return domain_out, t_domain_out, score1, score2, rank_loss[0], rank_loss[1]

        else:
            # extract features from the projections
            img_size = img.shape
            img = img.view(-1, img_size[2], img_size[3], img_size[4])
            img = self.img_backbone(img)
            img = torch.flatten(img, 1)
            # average the projection features
            img = img.view(img_size[0], img_size[1], self.img_inplanes)
            img = torch.mean(img, dim=1)

            # extract features from patches
            pc_size = pc.shape
            pc = pc.view(-1, pc_size[2], pc_size[3])
            pc = self.pc_backbone(pc)
            # average the patch features
            pc = pc.view(pc_size[0], pc_size[1], self.pc_inplanes)
            pc = torch.mean(pc, dim=1)

            # attention, fusion
            feat1 = self.cma(img, pc)

            output = self.quality1(feat1)
            score2 = self.quality2(output)

            return score2


class Feature_extraction(nn.Module):
    def __init__(self):
        super(Feature_extraction, self).__init__()
        self.img_inplanes = 2048
        self.pc_inplanes = 1024
        self.cma_planes = 1024
        self.img_backbone = resnet50(pretrained=True)
        self.pc_backbone = pointnet2()
        self.cma = CMA_fusion(img_inplanes=self.img_inplanes, pc_inplanes=self.pc_inplanes,
                                     cma_planes=self.cma_planes)


    def forward(self, img, pc):
        # extract features from the projections
        img_size = img.shape
        img = img.view(-1, img_size[2], img_size[3], img_size[4])
        img = self.img_backbone(img)
        img = torch.flatten(img, 1)
        # average the projection features
        img = img.view(img_size[0], img_size[1], self.img_inplanes)
        img = torch.mean(img, dim=1)

        # extract features from patches
        pc_size = pc.shape
        pc = pc.view(-1, pc_size[2], pc_size[3])
        pc = self.pc_backbone(pc)
        # average the patch features
        pc = pc.view(pc_size[0], pc_size[1], self.pc_inplanes)
        pc = torch.mean(pc, dim=1)

        # attention
        feat1 = self.cma(img, pc)

        return feat1

class quality_regression(nn.Module):
    def __init__(self, args, final_channels=1):
        super(quality_regression, self).__init__()
        self.args = args

        self.img_inplanes = 2048
        self.pc_inplanes = 1024
        self.cma_planes = 1024

        self.quality1 = nn.Linear(self.cma_planes * 4, self.cma_planes * 2)
        self.quality2 = nn.Linear(self.cma_planes * 2, 1)

    def forward(self, feat):

        score = self.quality2(self.quality1(feat))
        
        return score

