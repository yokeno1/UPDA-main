import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import random
import time
from scipy import stats
from scipy.optimize import curve_fit
import torch.nn.functional as F
# from model.evaluator import GMS_3DQA
from dataload.dataload_gms import QMM_Dataset
from tqdm import tqdm

from model.swin_transformer import SwinTransformer as swint_tiny
from model.adaptation import rank_net

from scipy.stats import pearsonr, spearmanr, kendalltau
# from caculate_PLCC import corr_value
torch.cuda.set_per_process_memory_fraction(1.0, 0)



def set_rand_seed(seed=1998):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(config, i_fold):
    config_dict = vars(config)
    for key in config_dict:
        print(key, ':', config_dict[key])

    print('-' * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    backbone = swint_tiny().cuda()
    rank_ad_net = rank_net(config).cuda()

    model = [backbone, rank_ad_net]

    state_dict = torch.load('checkpoint/swin_tiny_patch4_window7_224_22k.pth')
    state_dict = state_dict["model"]
    model[0].load_state_dict(state_dict, strict=False)

    # database configuration
    if config.target_database == 'SJTU':

        if config.source_database == 'WPC':
            # source_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            source_filename_list = 'datainfo/wpc_data_info/total.csv'
            target_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/sjtu_data_info/test_' + str(i_fold) + '.csv'

            source_images_dir = r'/media/vcg8009/DATA/xbx/data/wpc/img_4'
            target_images_dir = r'/media/vcg8009/DATA/xbx/data/sjtu/img_4'
            test_images_dir = r'/media/vcg8009/DATA/xbx/data/sjtu/img_4'

        elif config.source_database == 'WPC2.0':
            source_filename_list = 'datainfo/wpc2.0_data_info/total.csv'
            target_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/sjtu_data_info/test_' + str(i_fold) + '.csv'

            source_images_dir = r'/media/vcg8009/DATA/xbx/data/wpc2.0/img_4'
            target_images_dir = r'/media/vcg8009/DATA/xbx/data/sjtu/img_4'
            test_images_dir = r'/media/vcg8009/DATA/xbx/data/sjtu/img_4'

        elif config.source_database == 'SJTU':
            source_filename_list = 'datainfo/sjtu_data_info/total.csv'
            target_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/sjtu_data_info/test_' + str(i_fold) + '.csv'

            source_images_dir = r'/media/vcg8009/DATA/xbx/data/sjtu/img_4'
            target_images_dir = r'/media/vcg8009/DATA/xbx/data/sjtu/img_4'
            test_images_dir = r'/media/vcg8009/DATA/xbx/data/sjtu/img_4'

    elif config.target_database == 'WPC':
        if config.source_database == 'SJTU':
            source_filename_list = 'datainfo/sjtu_data_info/total.csv'
            # target_filename_list = 'datainfo/wpc_data_info/total.csv'
            # test_filename_list = 'datainfo/wpc_data_info/total.csv'
            # source_filename_list = 'datainfo/sjtu_data_info/test_1.csv'
            target_filename_list = 'datainfo/wpc_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'

            source_images_dir = r'/media/vcg8009/DATA/xbx/data/sjtu/img_4'
            target_images_dir = r'/media/vcg8009/DATA/xbx/data/wpc/img_4'
            test_images_dir = r'/media/vcg8009/DATA/xbx/data/wpc/img_4'

        elif config.source_database == 'WPC2.0':
            source_filename_list = 'datainfo/wpc2.0_data_info/total.csv'
            target_filename_list = 'datainfo/wpc_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'

            source_images_dir = r'/media/vcg8009/DATA/xbx/data/wpc2.0/img_4'
            target_images_dir = r'/media/vcg8009/DATA/xbx/data/wpc/img_4'
            test_images_dir = r'/media/vcg8009/DATA/xbx/data/wpc/img_4'

    elif config.target_database == 'WPC2.0':

        if config.source_database == 'SJTU':
            # source_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            source_filename_list = 'datainfo/sjtu_data_info/total.csv'
            # target_filename_list = 'datainfo/wpc_data_info/train_' + str(i_fold) + '.csv'
            # test_filename_list = 'datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'
            target_filename_list = 'datainfo/wpc2.0_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/wpc2.0_data_info/test_' + str(i_fold) + '.csv'
            source_images_dir = r'/media/vcg8009/DATA/xbx/data/sjtu/img_4'
            target_images_dir = r'/media/vcg8009/DATA/xbx/data/wpc2.0/img_4'
            test_images_dir = r'/media/vcg8009/DATA/xbx/data/wpc2.0/img_4'

        elif config.source_database == 'WPC':
            source_filename_list = 'datainfo/wpc_data_info/total.csv'
            target_filename_list = 'datainfo/wpc2.0_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/wpc2.0_data_info/test_' + str(i_fold) + '.csv'
            source_images_dir = r'/media/vcg8009/DATA/xbx/data/wpc/img_4'
            target_images_dir = r'/media/vcg8009/DATA/xbx/data/wpc2.0/img_4'
            test_images_dir = r'/media/vcg8009/DATA/xbx/data/wpc2.0/img_4'


    print('using source train datainfo: ' + source_filename_list)
    print('using target train datainfo: ' + target_filename_list)
    print('using target test datainfo: ' + test_filename_list)



    # dataloader configuration
    source_trainset = QMM_Dataset(csv_file=source_filename_list, data_prefix=source_images_dir, img_length_read=config.img_length_read)
    target_trainset = QMM_Dataset(csv_file=target_filename_list, data_prefix=target_images_dir, img_length_read=config.img_length_read)
    testset = QMM_Dataset(csv_file=test_filename_list, data_prefix=test_images_dir, img_length_read=config.img_length_read)
    source_loader = torch.utils.data.DataLoader(source_trainset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.num_workers,drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_trainset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.num_workers,drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=config.num_workers)


    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0000001)

    optimizer_backbone = torch.optim.Adam(model[0].parameters(), lr=config.lr, weight_decay=0.0000001)
    optimizer_rank_net = torch.optim.Adam(model[1].parameters(), lr=config.lr, weight_decay=0.0000001)
    # optimizer_mapping = torch.optim.SGD(model[1].parameters(), lr=args.lr, weight_decay=0.0005, momentum=args.momentum)
    # optimizer_regression = torch.optim.SGD(model[2].parameters(), lr=args.lr, weight_decay=0.0005, momentum=args.momentum)
    # optimizer_adnet = torch.optim.SGD(model[3].parameters(), lr=args.lr, weight_decay=0.0005, momentum=args.momentum)

    optimizer = [optimizer_backbone, optimizer_rank_net]

    scheduler_backbone = torch.optim.lr_scheduler.StepLR(optimizer_backbone, step_size=config.decay_interval, gamma=config.decay_ratio)
    scheduler_rank_net = torch.optim.lr_scheduler.StepLR(optimizer_rank_net, step_size=config.decay_interval, gamma=config.decay_ratio)

    scheduler = [scheduler_backbone, scheduler_rank_net]
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    min_loss = 100

    print('Starting training:')
    len_dataloader = min(len(source_loader), len(target_loader))
    pbar = tqdm(total=len_dataloader * config.epochs)

    for epoch in range(config.epochs):

        model[0].train()
        model[1].train()
        total_loss = 0
        start = time.time()

        data_source_iter = iter(source_loader)
        data_target_iter = iter(target_loader)

        for i in range(len_dataloader):
            pbar.update()

            data_source = data_source_iter.__next__()
            data_target = data_target_iter.__next__()
            image_source = data_source['image'].to(device)
            labels_source = data_source['gt_label'].float().detach().to(device)
            image_target = data_target['image'].to(device)

            image_size = image_source.shape
            image_source = image_source.view(-1, image_size[2], image_size[3], image_size[4])
            image_target = image_target.view(-1, image_size[2], image_size[3], image_size[4])
            feat = model[0](image_source)  ##(batch_size,49,768)
            t_feat = model[0](image_target)

            feat, t_feat = torch.mean(feat, dim=1), torch.mean(t_feat, dim=1)
            rank_loss = model[1](feat, t_feat, labels_source)

            loss = rank_loss[0] + rank_loss[1]
            total_loss += loss.detach().item()

            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, config.epochs, i + 1, len_dataloader,
                    loss.detach().item()))

        scheduler[0].step()
        scheduler[1].step()
        lr = scheduler[0].get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))
        end = time.time()
        print('Epoch %d training time cost: %.4f seconds' % (epoch + 1, end - start))


        if total_loss/len_dataloader < min_loss:
            min_loss = total_loss/len_dataloader

            print('Saving min loss model...')
            if not os.path.exists(config.ckpt_path + "/gms/step1"):
                os.makedirs(config.ckpt_path + "/gms/step1")

            filenamenew = os.path.join('checkpoint/gms/step1',
                                       config.source_database + "_to_" + config.target_database + '_fold_' + str(
                                           i_fold) + '_loss.pth')

            torch.save({
                'model_0_state_dict': model[0].state_dict(),
                'model_1_state_dict': model[1].state_dict(),
            }, filenamenew)

    if not os.path.exists(config.ckpt_path + "/gms/step1"):
        os.makedirs(config.ckpt_path + "/gms/step1")
    filenamenew = os.path.join('checkpoint/gms/step1',
                               config.source_database + "_to_" + config.target_database + '_fold_' + str(
                                   i_fold) + '_final.pth')
    torch.save({
        'model_0_state_dict': model[0].state_dict(),
        'model_1_state_dict': model[1].state_dict(),
    }, filenamenew)
    print('Saving model to',
          os.path.join('checkpoint/gms/step1',
                       config.source_database + "_to_" + config.target_database + '_fold_' + str(i_fold) + '_final.pth'))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters

    parser.add_argument('--source_database', type=str, default='WPC')
    parser.add_argument('--target_database', type=str, default='SJTU')
    parser.add_argument('--k_fold', type=int, default=9)

    parser.add_argument('--model_type', type=str, default='swin')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_ids', type=list, default=0)
    parser.add_argument('--ckpt_path', type=str, default='checkpoint')
    parser.add_argument('--load_path', type=str, default='checkpoint/swin_tiny_patch4_window7_224_22k.pth')
    # parser.add_argument('--load_path', type=str, default='trained_ckpt/SJTU_fold_1_best.pth')
    # training parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=float, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--in_feature', type=int, default=768)
    parser.add_argument('--img_length_read', type=int, default=4)

    config = parser.parse_args()
    set_rand_seed(seed=1)
    results = np.empty(shape=[0, 4])
    for i_fold in range(1, config.k_fold + 1):
        best_test = main(config, i_fold)
