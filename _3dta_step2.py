import os
import argparse
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataload.dataload_3dta import WPC_SD
# from model.backbone_3dta import Pct_3DTA
from model.backbone_3dta import Feature_extraction,quality_regression
from model.adaptation import AdversarialNetwork
from model.adaptation import fusion_net
# from model.adaptation import rank_net
import numpy as np
from torch.utils.data import DataLoader
from util import IOStream
import torch.nn.functional as F
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy import stats
from datetime import datetime
import time
from scipy.stats import pearsonr, spearmanr, kendalltau


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
                        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic

def train(args, i_fold):

    if args.source_database == 'WPC':
        source_data_dir = r'/media/vcg8009/DATA/xbx/data/wpc'
        source_txtfile = os.path.join(source_data_dir, f'patch_data_list_train.txt')

        if args.target_database == 'SJTU':
            target_data_dir = r'/media/vcg8009/DATA/xbx/data/sjtu'
            target_txtfile = os.path.join(target_data_dir, f'patch_data_list_train_{i_fold}.txt')
            test_txtfile = os.path.join(target_data_dir, f'patch_data_list_test_{i_fold}.txt')
        elif args.target_database == 'WPC2.0':
            target_data_dir = r'/media/vcg8009/DATA/xbx/data/wpc2.0'
            target_txtfile = os.path.join(target_data_dir, f'patch_data_list_train_{i_fold}.txt')
            test_txtfile = os.path.join(target_data_dir, f'patch_data_list_test_{i_fold}.txt')

    elif args.source_database == 'SJTU':
        source_data_dir = r'/media/vcg8009/DATA/xbx/data/sjtu'
        source_txtfile = os.path.join(source_data_dir, f'patch_data_list_train.txt')
        if args.target_database == 'WPC':
            target_data_dir = r'/media/vcg8009/DATA/xbx/data/wpc'
            target_txtfile = os.path.join(target_data_dir, f'patch_data_list_train_{i_fold}.txt')
            test_txtfile = os.path.join(target_data_dir, f'patch_data_list_test_{i_fold}.txt')
        elif args.target_database == 'WPC2.0':
            target_data_dir = r'/media/vcg8009/DATA/xbx/data/wpc2.0'
            target_txtfile = os.path.join(target_data_dir, f'patch_data_list_train_{i_fold}.txt')
            test_txtfile = os.path.join(target_data_dir, f'patch_data_list_test_{i_fold}.txt')

    elif args.source_database == 'WPC2.0':
        source_data_dir =r'/media/vcg8009/DATA/xbx/data/wpc2.0'
        source_txtfile = os.path.join(source_data_dir, f'patch_data_list_train.txt')
        if args.target_database == 'WPC':
            target_data_dir = r'/media/vcg8009/DATA/xbx/data/wpc'
            target_txtfile = os.path.join(target_data_dir, f'patch_data_list_train_{i_fold}.txt')
            test_txtfile = os.path.join(target_data_dir, f'patch_data_list_test_{i_fold}.txt')

        elif args.target_database == 'SJTU':
            target_data_dir = r'/media/vcg8009/DATA/xbx/data/sjtu'
            target_txtfile = os.path.join(target_data_dir, f'patch_data_list_train_{i_fold}.txt')
            test_txtfile = os.path.join(target_data_dir, f'patch_data_list_test_{i_fold}.txt')

    print("source_txtfile:", source_txtfile)
    print('target_txtfile:', target_txtfile)
    print("test_txtfile:", test_txtfile)

    train_data = WPC_SD(args, data_dir=source_data_dir, pattern='train', txtfile=source_txtfile)
    source_loader = DataLoader(train_data, num_workers=4,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    target_train_data = WPC_SD(args, data_dir=target_data_dir, pattern='train', txtfile=target_txtfile)
    target_loader = DataLoader(target_train_data, num_workers=4,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_data = WPC_SD(args, data_dir=target_data_dir, pattern='test', txtfile=test_txtfile)
    test_loader = DataLoader(test_data, num_workers=4,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = Feature_extraction(args).to(device).float()
    fusion = fusion_net(args).to(device).float()
    ad_net = AdversarialNetwork(args).to(device).float()
    qa_net = quality_regression(args).to(device).float()
    model = [backbone, fusion, ad_net, qa_net]
    # print(str(model))

    # optimizer_backbone = optim.Adam(model[0].parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer_backbone = optim.SGD(model[0].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer_fusion = optim.Adam(model[1].parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer_ad_net = optim.Adam(model[2].parameters(), lr=args.lr, weight_decay=1e-4)
    # optimizer_qa_net = optim.Adam(model[3].parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer_qa_net = optim.SGD(model[3].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    optimizer = [optimizer_backbone, optimizer_fusion, optimizer_ad_net, optimizer_qa_net]
    # print("Use Adam")
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler_backbone = CosineAnnealingLR(optimizer[0], args.epochs, eta_min=args.lr)
    # scheduler_fusion = CosineAnnealingLR(optimizer[1], args.epochs, eta_min=args.lr)
    # scheduler_ad_net = CosineAnnealingLR(optimizer[2], args.epochs, eta_min=args.lr)
    scheduler_qa_net = CosineAnnealingLR(optimizer[3], args.epochs, eta_min=args.lr)
    # scheduler_backbone = torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=args.decay_interval,
    #                                                      gamma=args.decay_ratio)
    scheduler_fusion = torch.optim.lr_scheduler.StepLR(optimizer[1], step_size=args.decay_interval,
                                                         gamma=args.decay_ratio)
    scheduler_ad_net = torch.optim.lr_scheduler.StepLR(optimizer[2], step_size=args.decay_interval,
                                                         gamma=args.decay_ratio)
    # scheduler_qa_net = torch.optim.lr_scheduler.StepLR(optimizer[3], step_size=args.decay_interval,
    #                                                      gamma=args.decay_ratio)
    scheduler = [scheduler_backbone, scheduler_fusion, scheduler_ad_net, scheduler_qa_net]

    mse_criterion = nn.MSELoss()
    loss_domain = nn.CrossEntropyLoss()

    if args.resume:
        checkpoint = torch.load(os.path.join(args.resume, args.source_database + "_to_" +
                                             args.target_database + '_fold_' + str(i_fold) + '_loss.pth'))
        model[0].load_state_dict(checkpoint['model_0_state_dict'], strict=True)

    begin_time = time.time()
    for epoch in range(args.epochs):
        # ?###################
        # ? Train
        train_total_loss = 0.0
        train_count = 0.0
        model[0].train()
        model[1].train()
        model[2].train()
        model[3].train()

        len_dataloader = min(len(source_loader), len(target_loader))
        data_source_iter = iter(source_loader)
        data_target_iter = iter(target_loader)

        for id in tqdm(range(len_dataloader), colour='blue'):

            data, mos, filenum = data_source_iter.__next__()
            target_data, _, _ = data_target_iter.__next__()

            data, mos = data.permute(0, 2, 1), mos.to(torch.float64).to(device).squeeze()
            data = data.type(torch.FloatTensor).to(device)

            target_data = target_data.permute(0, 2, 1)
            target_data = target_data.type(torch.FloatTensor).to(device)

            feat = model[0](data)
            feat2 = model[0](target_data)
            feat = F.adaptive_max_pool1d(feat, 1).view(args.batch_size, -1)
            feat2 = F.adaptive_max_pool1d(feat2, 1).view(args.batch_size, -1)

            feat_fusion, t_feat_fusion = model[1](feat, feat2,  if_fusion=True)

            source_domain = model[2](feat_fusion)
            target_domain = model[2](t_feat_fusion)

            source_score1 = model[3](feat)
            source_score2 = model[3](feat_fusion)

            batch_size = data.size()[0]

            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            optimizer[2].zero_grad()
            optimizer[3].zero_grad()

            source_score1 = source_score1.to(torch.float64).view(batch_size)
            source_score2 = source_score2.to(torch.float64).view(batch_size)
            # pre_mos_cpu = (pre_mos).detach().cpu().numpy()
            labels_source = mos

            s_domain_label = torch.zeros(batch_size).long()
            t_domain_label = torch.ones(batch_size).long()
            s_domain_label = s_domain_label.cuda()
            t_domain_label = t_domain_label.cuda()


            dif = torch.abs(source_score1.detach().cpu() - labels_source.detach().cpu())
            dif2 = torch.abs(source_score2.detach().cpu() - labels_source.detach().cpu())
            source_domain_copy = source_domain.clone()
            for index in range(dif.size(0)):
                if dif[index] < dif2[index]:
                    source_domain_copy[index] = 1 - source_domain[index]

            # print(pre_mos, mos)
            loss_ad = loss_domain(source_domain_copy, s_domain_label) + loss_domain(target_domain, t_domain_label)
            loss_mse = mse_criterion(source_score2, mos)

            loss = loss_mse + 0.8 * loss_ad
            # print(loss1, loss_ad)
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()
            optimizer[2].step()
            optimizer[3].step()
            train_count += batch_size
            train_total_loss += loss.detach().item() * batch_size

        scheduler[0].step()
        scheduler[1].step()
        scheduler[2].step()
        scheduler[3].step()

        record = f'Train {epoch:3d},  loss:{train_total_loss * 1.0 / train_count:.4f}'
        print(record)

    # *###################
    # * Test
    # *###################

    test_ply_num = int(len(test_data) / args.patch_num)
    test_total_loss = 0.0
    test_count = 0.0
    model[0].eval()  # training turn off
    model[1].eval()
    model[2].eval()
    model[3].eval()
    filenum_mos_true = [0] * test_ply_num
    filenum_mos_pred = [0] * test_ply_num

    for id, (data, mos, filenum) in tqdm(enumerate(test_loader, 0),
                                         total=len(test_loader), smoothing=0.9, desc=f'test  epoch：{epoch}',
                                         colour='green'):
        data, mos = data.permute(0, 2, 1), mos.to(torch.float64).to(device).squeeze()
        data = data.type(torch.FloatTensor).to(device)
        batch_size = data.size()[0]
        feat = model[0](data)
        feat = F.adaptive_max_pool1d(feat, 1).view(batch_size, -1)
        pre_mos = model[3](feat)
        pre_mos = pre_mos.to(torch.float64).view(batch_size)
        pre_mos_cpu = (pre_mos).detach().cpu().numpy()
        true_mos_cpu = (mos).cpu().numpy()

        loss = mse_criterion(pre_mos, mos)

        # preds = logits.max(dim=1)[1]            # for classfication
        test_count += batch_size
        test_total_loss += loss.item() * batch_size

        for i in range(batch_size):
            filenum_mos_pred[int(filenum[i])] += pre_mos_cpu[i]
            filenum_mos_true[int(filenum[i])] = true_mos_cpu[i]

    filenum_mos_true = np.array(filenum_mos_true)  # list2Tensor
    filenum_mos_pred = np.array(filenum_mos_pred)
    filenum_mos_pred = filenum_mos_pred / args.patch_num

    filenum_mos_pred = fit_function(filenum_mos_true, filenum_mos_pred)
    ply_test_PLCC = stats.mstats.pearsonr(filenum_mos_true, filenum_mos_pred)[0]  # calculate corelation
    ply_test_SRCC = stats.mstats.spearmanr(filenum_mos_true, filenum_mos_pred)[0]
    ply_test_KRCC = stats.mstats.kendalltau(filenum_mos_true, filenum_mos_pred)[0]
    ply_test_rmse = np.sqrt(((filenum_mos_true - filenum_mos_pred) ** 2).mean())

    record = f'Test  {epoch:3d},  loss:{test_total_loss * 1.0 / test_count:.4f}, PLCC:{ply_test_PLCC:.4f}, SRCC:{ply_test_SRCC:.4f}, KRCC:{ply_test_KRCC:.4f},rmse:{ply_test_rmse:.4f}'
    print(record)

    if not os.path.exists(args.ckpt_path + "/_3dta/step2"):
        os.makedirs(args.ckpt_path + "/_3dta/step2")
    filenamenew = os.path.join('checkpoint/_3dta/step2',
                               args.source_database + "_to_" + args.target_database + '_fold_' + str(
                                   i_fold) + '_final.pth')
    torch.save({
        'model_0_state_dict': model[0].state_dict(),
        'model_1_state_dict': model[1].state_dict(),
        'model_2_state_dict': model[2].state_dict(),
        'model_3_state_dict': model[3].state_dict(),
    }, filenamenew)

    print('Saving model to', filenamenew)

    return [ply_test_SRCC, ply_test_PLCC, ply_test_KRCC, ply_test_rmse]

def test(args):
    print('start test...')


    test_data = WPC_SD(args, pattern='test')
    test_loader = DataLoader(test_data, num_workers=4,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    backbone = Feature_extraction(args).to(device).float()
    qa_net = quality_regression(args).to(device).float()
    model = [backbone, qa_net]

    checkpoint = torch.load(os.path.join(args.resume, args.source_database + "_to_" +
                                         args.target_database + '_fold_' + str(i_fold) + '_loss.pth'))
    model[0].load_state_dict(checkpoint['model_0_state_dict'], strict=True)
    model[1].load_state_dict(checkpoint['model_3_state_dict'], strict=True)
    model[0].eval()  # training turn off
    model[1].eval()

    test_ply_num = int(len(test_data) / args.patch_num)
    test_count = 0.0
    filenum_mos_true = [0] * test_ply_num
    filenum_mos_pred = [0] * test_ply_num
    show_all_mos = torch.zeros([test_ply_num, int(args.patch_num)])
    for id, (data, mos, filenum) in tqdm(enumerate(test_loader, 0),
                                         total=len(test_loader), smoothing=0.9, desc=f'Just test', colour='green'):
        data, mos = data.to(device), mos.to(device).squeeze()
        data = data.permute(0, 2, 1)
        data = data.type(torch.FloatTensor)
        batch_size = data.size()[0]

        feat = model[0](data)
        feat = F.adaptive_max_pool1d(feat, 1).view(batch_size, -1)
        pre_mos = model[1](feat)
        pre_mos = pre_mos.to(torch.float64).view(batch_size)
        pre_mos_cpu = pre_mos.detach().cpu().numpy()
        true_mos_cpu = mos.cpu().numpy()
        # preds = logits.max(dim=1)[1]
        test_count += batch_size
        for i in range(batch_size):
            filenum_mos_pred[int(filenum[i])] += pre_mos_cpu[i]
            filenum_mos_true[int(filenum[i])] = true_mos_cpu[i]

    filenum_mos_true = torch.tensor(filenum_mos_true)
    filenum_mos_pred = torch.tensor(filenum_mos_pred)
    filenum_mos_pred = filenum_mos_pred / args.patch_num

    filenum_mos_pred = fit_function(filenum_mos_true, filenum_mos_pred)
    ply_test_PLCC = stats.mstats.pearsonr(filenum_mos_true, filenum_mos_pred)[0]  # calculate corelation
    ply_test_SRCC = stats.mstats.spearmanr(filenum_mos_true, filenum_mos_pred)[0]
    ply_test_KRCC = stats.mstats.kendalltau(filenum_mos_true, filenum_mos_pred)[0]
    ply_test_rmse = torch.sqrt(((filenum_mos_true - filenum_mos_pred) ** 2).mean())
    print(f'/033[1;35mTest (ply) {test_ply_num},    PLCC:{ply_test_PLCC:.4f},  SRCC:{ply_test_SRCC:.4f}', end='')
    print(f', KRCC:{ply_test_KRCC:.4f}, rmse:{ply_test_rmse:.4f}/n')

    print(f'Time now: {time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}')
    print(f'filenum_mos_true:{filenum_mos_true}')
    print(f'filenum_mos_pred:{filenum_mos_pred}/033[0m')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Our 3DTA')

    parser.add_argument('--exp_name', type=str, default='3DTA_patch_mos', metavar='N', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False, help='Use SGD')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=float, default=5)
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    parser.add_argument('--point_num', type=int, default=1024, help='num of points to use')

    parser.add_argument('--pre_train', type=bool, default=False, help='evaluate the model?')
    parser.add_argument('--eval', type=bool, default=False, help='evaluate the model?')

    parser.add_argument('--patch_dir', type=str, default='patch_72_10000', help='Where does patches exist?')
    parser.add_argument('--patch_num', type=int, default=72, metavar='N', help='How many patchs each PC have?')
    parser.add_argument('--in_feature', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, default='checkpoint')
    parser.add_argument('--resume', type=str, default='checkpoint/_3dta/step1', help='path for loading the checkpoint')

    parser.add_argument('--source_database', type=str, default='WPC', metavar='N')
    parser.add_argument('--target_database', type=str, default='SJTU', metavar='N')
    parser.add_argument('--k_fold', type=int, default=9)
    args = parser.parse_args()

    print(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        print(f'Using GPU :{torch.cuda.current_device()} from {torch.cuda.device_count()}devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')

    results = np.empty(shape=[0, 4])

    for i_fold in range(1, args.k_fold + 1):
        best_test = train(args, i_fold)
        print(
            '--------------------------------------------The {}-th Fold-----------------------------------------'.format(
                i_fold))
        print('Training completed.')
        print('The best training result SRCC: {:.4f}, PLCC: {:.4f}, KRCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test[0], best_test[1], best_test[2], best_test[3]))
        results = np.concatenate((results, np.array([best_test])), axis=0)
        print('-------------------------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------------------------')

    print('==============================done==============================================')
    print('The mean best result:', np.mean(results, axis=0))
    with open('_3dta_wpc2sjtu.txt', 'w') as f:
        f.write(str(np.mean(results, axis=0)))
    print('The median best result:', np.median(results, axis=0))