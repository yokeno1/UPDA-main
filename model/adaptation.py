import torch
import torch.nn as nn
from torch.nn import NLLLoss
from torch.autograd import Function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer import TransformerEncoderLayer_CMA,TransformerEncoderLayer


class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

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

    def forward(self, source, target):
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
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class rank_net(nn.Module):
    def __init__(self, config, in_feature=768):
        super(rank_net, self).__init__()
        self.config = config
        self.mmd = MMDLoss()
        self.rank_loss = NLLLoss()
        self.x_rank = nn.Sequential()
        self.x_rank.add_module('d_fc1', nn.Linear(config.in_feature, 512))
        self.x_rank.add_module('d_fc2', nn.Linear(512, 256))
        self.x_rank.add_module('d_relu1', nn.ReLU(True))
        self.x_rank.add_module('d_drop1', nn.Dropout(0.5))
        self.x_rank.add_module('d_fc3', nn.Linear(256, 2))
        self.x_rank.add_module('d_logsoftmax', nn.LogSoftmax(dim=1))



    def forward(self, x, t_x, mos):


        # x, t_x = torch.mean(x, dim=1), torch.mean(t_x, dim=1)

        bs, x0 = x.size(0), x.size(1)

        F_rank = []
        f_rank_batch = []
        t_f_rank_batch =[]
        for i in range(bs):
            for j in range(bs):
                if i != j:
                    rank_mos = mos[i] - mos[j]
                    f_rank = (x[i] - x[j]).view(1,-1)
                    t_f_rank = (t_x[i] - t_x[j]).view(1,-1)
                    f_rank_batch.append(f_rank)
                    t_f_rank_batch.append(t_f_rank)
                    # F_rank_pred.append(f_rank_pred)
                    if rank_mos > 0:
                        F_rank.append(1)
                    else:
                        F_rank.append(0)

        # print( F_rank, F_rank_pred)

        f_rank_batch = torch.stack(f_rank_batch).view(bs*(bs-1), -1)

        F_rank_pred = self.x_rank(f_rank_batch)
        F_rank = torch.tensor(F_rank).cuda()
        F_rank_pred = F_rank_pred.view(-1,2)

        loss1 = self.rank_loss(F_rank_pred, F_rank)


        t_f_rank_batch = torch.stack(t_f_rank_batch).view(bs*(bs-1), -1)

        loss2 = self.mmd(f_rank_batch, t_f_rank_batch)
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
    inputs2 = torch.rand(16, 49, 768).cuda()
    mos= torch.rand(16, 1).cuda()
    # print(inputs)
    model = rank_net().cuda()
    model.eval()
    domain_output = model.forward(inputs, inputs2, mos)

    print(domain_output)
