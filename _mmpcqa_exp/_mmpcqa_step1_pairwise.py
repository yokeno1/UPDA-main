import os, argparse, time

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR
from torchvision import transforms
import random
import torch.backends.cudnn as cudnn
import scipy
from scipy import stats
from scipy.optimize import curve_fit
# from model.backbone_mmpcqa import MM_PCQAnet
from model.backbone_mmpcqa import Feature_extraction
from model.adaptation_pairwise import rank_net
from dataload.dataload_mmpcqa_pair import MMDataset
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from scipy.stats import pearsonr, spearmanr, kendalltau
torch.cuda.set_per_process_memory_fraction(1.0, 0)


def set_rand_seed(seed=1998):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # fix the random seed


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


def train(args, k_fold_id):
    print(
        '*************************************************************************************************************************')

    cudnn.enabled = True
    num_epochs = args.epochs
    batch_size = args.batch_size
    source_database = args.source_database
    target_database = args.target_database
    patch_length_read = args.patch_length_read
    img_length_read = args.img_length_read


    print('The current k_fold_id is ' + str(k_fold_id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     os.environ["CUDA_VISIBLE_DEVICES"] = 0

    if source_database == 'SJTU':
        s_train_filename_list = '../datainfo/sjtu_pair_info/total.csv'
        s_data_dir_2d = r"E:\xbx\data\mm_pcqa_data/sjtu/img"
        s_data_dir_pc = r"E:\xbx\data\mm_pcqa_data/sjtu/patch"
        if target_database == 'WPC':
            t_train_filename_list = f'../datainfo/wpc_pair_info/train_{k_fold_id}.csv'
            t_test_filename_list = f'../datainfo/wpc_pair_info/test_{k_fold_id}.csv'
            t_data_dir_2d = r'E:\xbx\data\mm_pcqa_data/wpc/img'
            t_data_dir_pc = r'E:\xbx\data\mm_pcqa_data/wpc/patch'
        elif target_database == 'WPC2.0':
            t_train_filename_list = f'../datainfo/wpc2.0_pair_info/train_{k_fold_id}.csv'
            t_test_filename_list = f'../datainfo/wpc2.0_pair_info/test_{k_fold_id}.csv'
            t_data_dir_2d = r"E:\xbx\data\mm_pcqa_data/wpc2.0/img"
            t_data_dir_pc = r"E:\xbx\data\mm_pcqa_data/wpc2.0/patch"

    elif source_database == 'WPC':
        s_train_filename_list = '../datainfo/wpc_pair_info/total.csv'
        s_data_dir_2d = r'E:\xbx\data\mm_pcqa_data/wpc/img'
        s_data_dir_pc = r'E:\xbx\data\mm_pcqa_data/wpc/patch'
        if target_database == 'SJTU':
            t_train_filename_list = f'../datainfo/sjtu_pair_info/train_{k_fold_id}.csv'
            t_test_filename_list = f'../datainfo/sjtu_pair_info/test_{k_fold_id}.csv'
            t_data_dir_2d = r"E:\xbx\data\mm_pcqa_data/sjtu/img"
            t_data_dir_pc = r"E:\xbx\data\mm_pcqa_data/sjtu/patch"
        elif target_database == 'WPC2.0':
            t_train_filename_list = f'../datainfo/wpc2.0_pair_info/train_{k_fold_id}.csv'
            t_test_filename_list = f'../datainfo/wpc2.0_pair_info/test_{k_fold_id}.csv'
            t_data_dir_2d = r"E:\xbx\data\mm_pcqa_data/wpc2.0/img"
            t_data_dir_pc = r"E:\xbx\data\mm_pcqa_data/wpc2.0/patch"

    elif source_database == 'WPC2.0':
        s_train_filename_list = '../datainfo/wpc2.0_pair_info/total.csv'
        s_data_dir_2d = r'E:\xbx\data\mm_pcqa_data/wpc2.0/img'
        s_data_dir_pc = r'E:\xbx\data\mm_pcqa_data/wpc2.0/patch'
        if target_database == 'SJTU':
            t_train_filename_list = f'../datainfo/sjtu_pair_info/train_{k_fold_id}.csv'
            t_test_filename_list = f'../datainfo/sjtu_pair_info/test_{k_fold_id}.csv'
            t_data_dir_2d = r"E:\xbx\data\mm_pcqa_data/sjtu/img"
            t_data_dir_pc = r"E:\xbx\data\mm_pcqa_data/sjtu/patch"
        elif target_database == 'WPC':
            t_train_filename_list = f'../datainfo/wpc_pair_info/train_{k_fold_id}.csv'
            t_test_filename_list = f'../datainfo/wpc_pair_info/test_{k_fold_id}.csv'
            t_data_dir_2d = r'E:\xbx\data\mm_pcqa_data/wpc/img'
            t_data_dir_pc = r'E:\xbx\data\mm_pcqa_data/wpc/patch'

    print('s_train_filename_list:',s_train_filename_list)
    print('t_train_filename_list:', t_train_filename_list)
    print('t_test_filename_list:', t_test_filename_list)
    transformations_train = transforms.Compose(
        [transforms.RandomCrop(224, pad_if_needed=True, fill=0, padding_mode='constant'), transforms.ToTensor(), \
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformations_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), \
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])


    backbone = Feature_extraction().to(device).float()
    rank_ad_net = rank_net(args).to(device).float()
    model = [backbone, rank_ad_net]
    scaler = GradScaler("cuda")

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
    #                       momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay_rate)
    print('Using Adam optimizer, initial learning rate: ' + str(args.lr))

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.9)

    optimizer_backbone = optim.Adam(model[0].parameters(), lr=args.lr, weight_decay=args.decay_rate)
    optimizer_rank_net = optim.Adam(model[1].parameters(), lr=args.lr, weight_decay=args.decay_rate)
    optimizer = [optimizer_backbone, optimizer_rank_net]
    # print("Use Adam")
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler_backbone = StepLR(optimizer[0], step_size=6, gamma=0.9)
    # scheduler_backbone = StepLR(optimizer_backbone, step_size=args.decay_interval, gamma=args.decay_ratio)
    scheduler_rank_net = StepLR(optimizer[1], step_size=6, gamma=0.9)
    scheduler = [scheduler_backbone, scheduler_rank_net]
    print("Ready to train network")
    print(
        '*************************************************************************************************************************')
    best_test_criterion = -1  # SROCC min
    # best = np.zeros(4)

    s_train_dataset = MMDataset(data_dir_2d=s_data_dir_2d, data_dir_pc=s_data_dir_pc, datainfo_path=s_train_filename_list,
                              transform=transformations_train)
    # test_dataset = MMDataset(data_dir_2d=s_data_dir_2d, data_dir_pc=s_data_dir_pc, datainfo_path=test_filename_list,
    #                          transform=transformations_test, is_train=False)
    t_train_dataset = MMDataset(data_dir_2d=t_data_dir_2d, data_dir_pc=t_data_dir_pc, datainfo_path=t_train_filename_list,
                              transform=transformations_train)
    t_test_dataset = MMDataset(data_dir_2d=t_data_dir_2d, data_dir_pc=t_data_dir_pc, datainfo_path=t_test_filename_list,
                             transform=transformations_test, is_train=False)

    s_train_loader = torch.utils.data.DataLoader(dataset=s_train_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=0, drop_last=True)
    t_train_loader = torch.utils.data.DataLoader(dataset=t_train_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=t_test_dataset, batch_size=1, shuffle=False, num_workers=0)

    best = []
    min_loss = 10000
    len_dataloader = min(len(s_train_loader), len(t_train_loader))
    pbar = tqdm(total=len_dataloader * num_epochs)
    for epoch in range(num_epochs):
        # begin training, during each epoch, the crops and patches are randomly selected for the training set and fixed for the testing set
        # if you want to change the number of images or projections, load the parameters here 'img_length_read = img_length_read, patch_length_read = patch_length_read'

        data_source_iter = iter(s_train_loader)
        data_target_iter = iter(t_train_loader)

        model[0].train()
        model[1].train()
        start = time.time()
        train_total_loss = 0

        for i in range(len_dataloader):
            pbar.update()
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            sample1_imgs, sample1_pc, sample1_mos, sample2_imgs, sample2_pc, sample2_mos = data_source_iter.__next__()
            sample1_t_imgs, sample1_t_pc, sample1_t_mos, sample2_t_imgs, sample2_t_pc, sample2_t_mos = data_target_iter.__next__()

            sample1_imgs = sample1_imgs.to(device)
            sample1_pc = torch.Tensor(sample1_pc.float())
            sample1_pc = sample1_pc.to(device)
            sample1_mos = sample1_mos[:, np.newaxis]
            sample1_mos = sample1_mos.to(device)
            sample1_labels_source = sample1_mos.view(-1)

            sample2_imgs = sample2_imgs.to(device)
            sample2_pc = torch.Tensor(sample2_pc.float())   
            sample2_pc = sample2_pc.to(device)
            sample2_mos = sample2_mos[:, np.newaxis]
            sample2_mos = sample2_mos.to(device)
            sample2_labels_source = sample2_mos.view(-1)

            sample1_t_imgs = sample1_t_imgs.to(device)
            sample1_t_pc = torch.Tensor(sample1_t_pc.float())
            sample1_t_pc = sample1_t_pc.to(device)

            sample2_t_imgs = sample2_t_imgs.to(device)
            sample2_t_pc = torch.Tensor(sample2_t_pc.float())
            sample2_t_pc = sample2_t_pc.to(device)

            with autocast("cuda"):
                
                sample1_feat = model[0](sample1_imgs, sample1_pc)
                sample2_feat = model[0](sample2_imgs, sample2_pc)
                sample1_t_feat = model[0](sample1_t_imgs, sample1_t_pc)
                sample2_t_feat = model[0](sample2_t_imgs, sample2_t_pc)

                rank_loss = model[1](sample1_feat, sample2_feat, sample1_t_feat, sample2_t_feat, sample1_labels_source, sample2_labels_source)
                loss = rank_loss[0] + rank_loss[1]
                train_total_loss += loss.detach().item()

            # compute loss
            # loss.backward()
            # optimizer[0].step()
            # optimizer[1].step()
            scaler.scale(loss).backward()
            scaler.step(optimizer[0])
            scaler.step(optimizer[1])
            scaler.update()

            if (i + 1) % 10 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch + 1, num_epochs, i + 1, len_dataloader, loss.detach().item()))

            # loss.backward()
            # optimizer.step()

        scheduler[0].step()
        scheduler[1].step()
        lr_current = scheduler[0].get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr_current[0]))

        end = time.time()
        print('Epoch %d training time cost: %.4f seconds' % (epoch + 1, end - start))
        if (train_total_loss) < min_loss:
            min_loss = train_total_loss

            print('Saving min loss model...')
            if not os.path.exists(args.ckpt_path + "/_mmpcqa/step1"):
                os.makedirs(args.ckpt_path + "/_mmpcqa/step1")

            filenamenew = os.path.join(args.ckpt_path + "/_mmpcqa/step1",
                                       args.source_database + "_to_" + args.target_database + '_fold_' + str(
                                           i_fold) + '_loss.pth')

            torch.save({
                'model_0_state_dict': model[0].state_dict(),
                'model_1_state_dict': model[1].state_dict(),
            }, filenamenew)


    print('Saving model...')
    if not os.path.exists(args.ckpt_path + "/_mmpcqa/step1"):
        os.makedirs(args.ckpt_path + "/_mmpcqa/step1")

    filenamenew = os.path.join(args.ckpt_path + "/_mmpcqa/step1",
                               args.source_database + "_to_" + args.target_database + '_fold_' + str(
                                   i_fold) + '_final.pth')

    torch.save({
        'model_0_state_dict': model[0].state_dict(),
        'model_1_state_dict': model[1].state_dict(),
    }, filenamenew)

    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--model', default='MM_PCQA', type=str)
    parser.add_argument('--patch_length_read', default=6, type=int, help='number of the using patches')
    parser.add_argument('--img_length_read', default=6, type=int, help='number of the using images')
    parser.add_argument('--in_feature', type=int, default=4096)
    parser.add_argument('--loss', default='mse', type=str)
    parser.add_argument('--ckpt_path', type=str, default='checkpoint')

    parser.add_argument('--epochs', help='Maximum number of training epochs.', default=30, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=2, type=int)
    parser.add_argument('--source_database', default='WPC', type=str)
    parser.add_argument('--target_database', default='SJTU', type=str)
    parser.add_argument('--k_fold_num', default=1, type=int,
                        help='9 for the SJTU-PCQA, 5 for the WPC, 4 for the WPC2.0')

    args = parser.parse_args()

    for i_fold in range(1, args.k_fold_num + 1):
        train(args,  i_fold)