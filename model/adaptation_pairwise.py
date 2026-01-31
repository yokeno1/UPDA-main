import torch
import torch.nn as nn
from torch.nn import NLLLoss
from torch.autograd import Function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer import TransformerEncoderLayer_CMA,TransformerEncoderLayer


class WeightedMMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的加权MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    weight_mode: 权重计算模式 ['quality_diff', 'rank_confident', 'hybrid']
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, weight_mode='quality_diff', **kwargs):
        super(WeightedMMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.weight_mode = weight_mode

    def compute_quality_weights(self, sample1_scores, sample2_scores):
        """
        根据质量差异计算权重
        
        Args:
            source_scores: 源域点云质量分数 [batch_size]
            target_scores: 目标域点云质量分数 [batch_size] (可为None)
        """
        # 域内样本对的质量差异
        weights = torch.sigmoid(torch.abs(sample1_scores - sample2_scores))
        weight_matrix = torch.outer(weights, weights)
        return weight_matrix

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target, sample1_scores=None, sample2_scores=None):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            if self.weight_mode is None:
                loss = torch.mean(XX + YY - XY - YX)
                return loss
            else:
                src_weights = self.compute_quality_weights(sample1_scores, sample2_scores)
                XX_weighted = (XX * src_weights).sum() / (src_weights.sum() + 1e-8)
                
                # 2. 目标域内部均匀权重（因为没有目标域分数）
                YY_weighted = YY.mean()
                
                # 3. 跨域部分均匀权重（因为没有目标域分数）
                XY_weighted = XY.mean()
                YX_weighted = YX.mean()
                
                loss = XX_weighted + YY_weighted - XY_weighted - YX_weighted
                return loss


class rank_net(nn.Module):
    def __init__(self, config, in_feature=768):
        super(rank_net, self).__init__()
        self.config = config
        self.mmd = WeightedMMDLoss()
        self.rank_loss = NLLLoss()
        self.x_rank = nn.Sequential()
        self.x_rank.add_module('d_fc1', nn.Linear(config.in_feature, 512))
        self.x_rank.add_module('d_bn1', nn.BatchNorm1d(512))
        self.x_rank.add_module('d_relu1', nn.ReLU(True))
        self.x_rank.add_module('d_fc2', nn.Linear(512, 256))
        self.x_rank.add_module('d_bn2', nn.BatchNorm1d(256))
        self.x_rank.add_module('d_relu2', nn.ReLU(True))
        self.x_rank.add_module('d_fc3', nn.Linear(256, 2))
        self.x_rank.add_module('d_logsoftmax', nn.LogSoftmax(dim=1))



    def forward(self, sample1_feat, sample2_feat, sample1_t_feat, sample2_t_feat, sample1_labels_source, sample2_labels_source):


        # x, t_x = torch.mean(x, dim=1), torch.mean(t_x, dim=1)
        s1_x, s2_x = sample1_feat, sample2_feat
        t1_x, t2_x = sample1_t_feat, sample2_t_feat
        s1_mos, s2_mos = sample1_labels_source, sample2_labels_source


        bs, x0 = s1_x.size(0), s1_x.size(1)

        F_rank = []
        f_rank_batch = []
        t_f_rank_batch =[]
        for i in range(bs):
            rank_mos = s1_mos[i] - s2_mos[i]
            f_rank = (s1_x[i] - s2_x[i]).view(1,-1)
            t_f_rank = (t1_x[i] - t2_x[i]).view(1,-1)
            f_rank_batch.append(f_rank)
            t_f_rank_batch.append(t_f_rank)
            # F_rank_pred.append(f_rank_pred)
            if rank_mos > 0:
                F_rank.append(1)
            else:
                F_rank.append(0)

        f_rank_batch = torch.stack(f_rank_batch).view(bs, -1)
        F_rank_pred = self.x_rank(f_rank_batch)
        F_rank = torch.tensor(F_rank).cuda()
        F_rank_pred = F_rank_pred.view(-1,2)

        # 按分数差异加权的损失：
        # 1) 计算每个样本的 NLLLoss（不做 reduction）
        loss1_per = F.nll_loss(F_rank_pred, F_rank, reduction='none')  # shape [bs]
        # 2) 计算权重：使用分数差的 sigmoid 作为权重（可调整因子，比如 5.0）
        s1_scores = s1_mos.view(-1).float()
        s2_scores = s2_mos.view(-1).float()
        weights = torch.sigmoid(torch.abs(s1_scores - s2_scores))  # shape [bs]
        # 3) 加权并求均值
        loss1 = (loss1_per * weights).mean()
        
        t_f_rank_batch = torch.stack(t_f_rank_batch).view(bs, -1)
        loss2 = self.mmd(f_rank_batch, t_f_rank_batch, s1_mos, s2_mos)
        # loss2 = self.coral(f_rank_batch, t_f_rank_batch)

        return loss1, loss2



class fusion_net(nn.Module):
    def __init__(self, config, in_feature=768):
        super(fusion_net, self).__init__()
        self.config = config
        self.att = TransformerEncoderLayer_CMA(config.in_feature, nhead=8, dim_feedforward=2048, dropout=0.1,)

    def forward(self, x, t_x, if_fusion=True):
        if if_fusion:

            x1, t_x1 = self.att(x, t_x)
            x1, t_x1 = t_x1, x1

            return x1 + x, t_x1 + t_x
        else:
            return x, t_x


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class AdversarialNetwork(nn.Module):

    def __init__(self, config, in_feature=768, pretrained=False):
        super(AdversarialNetwork, self).__init__()

        self.in_size = config.in_feature
        self.pretrained = pretrained
        self.device = torch.device("cuda")

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.in_size, 256))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(256))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(256, 2))
        # self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def forward(self, feature, alpha=1):

        # loss_domain = torch.nn.NLLLoss()
        bs = feature.size(0)
        feature = feature.reshape(bs, -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        return domain_output

if __name__ == "__main__":
    inputs = torch.rand(16, 49, 768).cuda()
    # inputs2 = torch.rand(16, 49, 768).cuda()
    # mos= torch.rand(16, 1).cuda()
    # # print(inputs)
    # model = rank_net().cuda()
    # model.eval()
    # domain_output = model.forward(inputs, inputs2, mos)

    # print(domain