import os
import argparse
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR
from dataload.data_load import WPC_SD
# from model.backbone_3dta import Pct_3DTA
from model.backbone_3dta import Feature_extraction
# from model.adaptation import AdversarialNetwork
# from model.adaptation import fusion_net
from model.adaptation_inbatch import rank_net
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from util import IOStream
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy import stats
from datetime import datetime
import time
from scipy.stats import pearsonr, spearmanr, kendalltau

def train(args, i_fold):

    if args.target_database == 'SJTU':

        if args.source_database == 'WPC':
            # source_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            source_filename_list = '../datainfo/wpc_data_info/total.csv'
            target_filename_list = '../datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/sjtu_data_info/test_' + str(i_fold) + '.csv'

            source_data_dir = r'/mnt/d/xbx/data/wpc'
            target_data_dir = r'/mnt/d/xbx/data/sjtu'
            test_data_dir = r'/mnt/d/xbx/data/sjtu'

        elif args.source_database == 'WPC2.0':
            source_filename_list = '../datainfo/wpc2.0_data_info/total.csv'
            target_filename_list = '../datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/sjtu_data_info/test_' + str(i_fold) + '.csv'

            source_data_dir = r'/mnt/d/xbx/data/wpc2.0'
            target_data_dir = r'/mnt/d/xbx/data/sjtu'
            test_data_dir = r'/mnt/d/xbx/data/sjtu'

        elif args.source_database == 'SJTU':
            source_filename_list = '../datainfo/sjtu_data_info/total.csv'
            target_filename_list = '../datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/sjtu_data_info/test_' + str(i_fold) + '.csv'

            source_data_dir = r'/mnt/d/xbx/data/sjtu'
            target_data_dir = r'/mnt/d/xbx/data/sjtu'
            test_data_dir = r'/mnt/d/xbx/data/sjtu'

    elif args.target_database == 'WPC':
        if args.source_database == 'SJTU':
            source_filename_list = '../datainfo/sjtu_data_info/total.csv'
            # target_filename_list = 'datainfo/wpc_data_info/total.csv'
            # test_filename_list = 'datainfo/wpc_data_info/total.csv'
            # source_filename_list = 'datainfo/sjtu_data_info/test_1.csv'
            target_filename_list = '../datainfo/wpc_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'

            source_data_dir = r'/mnt/d/xbx/data/sjtu'
            target_data_dir = r'/mnt/d/xbx/data/wpc'
            test_data_dir = r'/mnt/d/xbx/data/wpc'

        elif args.source_database == 'WPC2.0':
            source_filename_list = '../datainfo/wpc2.0_data_info/total.csv'
            target_filename_list = '../datainfo/wpc_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'

            source_data_dir = r'/mnt/d/xbx/data/wpc2.0'
            target_data_dir = r'/mnt/d/xbx/data/wpc'
            test_data_dir = r'/mnt/d/xbx/data/wpc'

    elif args.target_database == 'WPC2.0':

        if args.source_database == 'SJTU':
            # source_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            source_filename_list = '../datainfo/sjtu_data_info/total.csv'
            # target_filename_list = 'datainfo/wpc_data_info/train_' + str(i_fold) + '.csv'
            # test_filename_list = 'datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'
            target_filename_list = '../datainfo/wpc2.0_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/wpc2.0_data_info/test_' + str(i_fold) + '.csv'
            source_data_dir = r'/mnt/d/xbx/data/sjtu'
            target_data_dir = r'/mnt/d/xbx/data/wpc2.0'
            test_data_dir = r'/mnt/d/xbx/data/wpc2.0'

        elif args.source_database == 'WPC':
            source_filename_list = '../datainfo/wpc_data_info/total.csv'
            target_filename_list = '../datainfo/wpc2.0_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/wpc2.0_data_info/test_' + str(i_fold) + '.csv'
            source_data_dir = r'/mnt/d/xbx/data/wpc'
            target_data_dir = r'/mnt/d/xbx/data/wpc2.0'
            test_data_dir = r'/mnt/d/xbx/data/wpc2.0'

    print("source_filename_list:", source_filename_list)
    print('target_filename_list:', target_filename_list)
    print("test_filename_list:", test_filename_list)

    train_data = WPC_SD(args, data_dir=source_data_dir, filename_list=source_filename_list)
    source_loader = DataLoader(train_data, num_workers=4,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    target_train_data = WPC_SD(args, data_dir=target_data_dir, filename_list=target_filename_list)
    target_loader = DataLoader(target_train_data, num_workers=4,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_data = WPC_SD(args, data_dir=test_data_dir, filename_list=test_filename_list)
    test_loader = DataLoader(test_data, num_workers=4,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = Feature_extraction(args).to(device)
    rank_ad_net = rank_net(args).to(device)
    model = [backbone, rank_ad_net]
    # print(str(model))

    optimizer_backbone = optim.Adam(model[0].parameters(),lr=args.lr, weight_decay=1e-4)
    # optimizer_backbone = optim.SGD(model[0].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer_rank_net = optim.Adam(model[1].parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = [optimizer_backbone, optimizer_rank_net]
    # print("Use Adam")
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # scheduler_backbone = CosineAnnealingLR(optimizer[0], args.epochs, eta_min=args.lr)
    scheduler_backbone = StepLR(optimizer[0], step_size=args.decay_interval, gamma=args.decay_ratio)
    scheduler_rank_net = StepLR(optimizer[1], step_size=args.decay_interval, gamma=args.decay_ratio)
    scheduler = [scheduler_backbone, scheduler_rank_net]
    min_loss = float('inf')

    for epoch in range(args.epochs):
        # ?###################
        # ? Train
        train_total_loss = 0.0
        train_count = 0.0
        model[0].train()
        model[1].train()  # training turn on

        len_dataloader = min(len(source_loader), len(target_loader))
        data_source_iter = iter(source_loader)
        data_target_iter = iter(target_loader)

        for id in tqdm(range(len_dataloader), colour='blue'):

            data, mos, filenum = data_source_iter.__next__()
            target_data, _, _ = data_target_iter.__next__()

            data, mos = data.to(torch.float32).permute(0, 2, 1), mos.to(torch.float32).to(device).squeeze()
            data = data.to(device)

            target_data = target_data.to(torch.float32).permute(0, 2, 1)
            target_data = target_data.to(device)

            optimizer[0].zero_grad()
            optimizer[1].zero_grad()

             # ?@@@@@@@@@@@@@@@@@@@@@@@  train forward
            feat = model[0](data)
            feat2 = model[0](target_data)

            feat = F.adaptive_max_pool1d(feat, 1).view(args.batch_size, -1)
            feat2 = F.adaptive_max_pool1d(feat2, 1).view(args.batch_size, -1)

            rank_loss = model[1](feat, feat2, mos)
            loss = rank_loss[0] + rank_loss[1]
            # print(loss1, loss_ad)

            loss.backward()

            optimizer[0].step()
            optimizer[1].step()
            train_count += args.batch_size
            train_total_loss += loss.detach().item() * args.batch_size
            if (id + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, args.epochs, id + 1, len_dataloader,
                loss.detach().item()))

        scheduler[0].step()
        scheduler[1].step()

        record = f'Train {epoch:3d},  loss:{train_total_loss * 1.0 / train_count:.4f}'
        print(record)

        # time_now = f'{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}'
        # with open(args.results_dir + '/train_log.txt', 'a+') as txt:
        #     txt.write(f'/n{time_now}    {record}')

        if (train_total_loss * 1.0 / train_count) < min_loss:
            min_loss = train_total_loss * 1.0 / train_count

            print('Saving min loss model...')
            if not os.path.exists(args.ckpt_path + "/_3dta/step1"):
                os.makedirs(args.ckpt_path + "/_3dta/step1")

            filenamenew = os.path.join('checkpoint/_3dta/step1',
                                       args.source_database + "_to_" + args.target_database + '_fold_' + str(
                                           i_fold) + '_loss.pth')

            torch.save({
                'model_0_state_dict': model[0].state_dict(),
                'model_1_state_dict': model[1].state_dict(),
            }, filenamenew)


    print('Saving model...')
    if not os.path.exists(args.ckpt_path + "/_3dta/step1"):
        os.makedirs(args.ckpt_path + "/_3dta/step1")

    filenamenew = os.path.join('checkpoint/_3dta/step1',
                               args.source_database + "_to_" + args.target_database + '_fold_' + str(
                                   i_fold) + '_final.pth')

    torch.save({
        'model_0_state_dict': model[0].state_dict(),
        'model_1_state_dict': model[1].state_dict(),
    }, filenamenew)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Our 3DTA')

    parser.add_argument('--exp_name', type=str, default='3DTA_patch_mos', metavar='N', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=float, default=5)
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    parser.add_argument('--point_num', type=int, default=1024, help='num of points to use')

    parser.add_argument('--in_feature', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, default='checkpoint')
    # parser.add_argument('--data_dir', type=str, default=r'/mnt/d/xbx/data/wpc', metavar='N',
    #                     help='Where does dataset exist?')
    parser.add_argument('--patch_dir', type=str, default='patch_72_10000', help='Where does patches exist?')
    parser.add_argument('--patch_num', type=int, default=72, metavar='N', help='How many patchs each PC have?')

    parser.add_argument('--source_database', type=str, default='WPC', metavar='N')
    parser.add_argument('--target_database', type=str, default='SJTU', metavar='N')
    parser.add_argument('--k_fold', type=int, default=1)
    args = parser.parse_args()

    print(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        print(f'Using GPU :{torch.cuda.current_device()} from {torch.cuda.device_count()}devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')


    for i_fold in range(1, args.k_fold + 1):
       train(args, i_fold)