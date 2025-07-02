import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import random
import time
from scipy import stats
from scipy.optimize import curve_fit
from dataload.dataload_gms import QMM_Dataset
from tqdm import tqdm

from model.swin_transformer import SwinTransformer as swint_tiny
from model.backbone_gms import IQAHead
from model.adaptation import AdversarialNetwork
from model.adaptation import fusion_net

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

def main(config, i_fold):
    config_dict = vars(config)
    for key in config_dict:
        print(key, ':', config_dict[key])

    print('-' * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = swint_tiny().cuda()
    fusion = fusion_net(config).cuda()
    ad_net = AdversarialNetwork(config).cuda()
    iqa_head = IQAHead().cuda()

    model = [backbone, fusion, ad_net, iqa_head]

    if config.resume:
        checkpoint = torch.load(os.path.join(config.resume, config.source_database + "_to_" + config.target_database + '_fold_' + str(i_fold) + '_loss.pth'))
        model[0].load_state_dict(checkpoint['model_0_state_dict'], strict=True)
    # state_dict = torch.load('checkpoint/swin_tiny_patch4_window7_224_22k.pth')
    # state_dict = state_dict["model"]
    # model[0].load_state_dict(state_dict, strict=False)

    # database configuration

    if config.target_database == 'SJTU':

        if config.source_database == 'WPC':
            # source_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            source_filename_list = 'datainfo/wpc_data_info/total.csv'
            target_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/sjtu_data_info/test_' + str(i_fold) + '.csv'

            source_images_dir = r'media/vcg8009/DATA/xbx/data/wpc/img_4'
            target_images_dir = r'media/vcg8009/DATA/xbx/data/sjtu/img_4'
            test_images_dir = r'media/vcg8009/DATA/xbx/data/sjtu/img_4'

        elif config.source_database == 'WPC2.0':
            source_filename_list = 'datainfo/wpc2.0_data_info/total.csv'
            target_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/sjtu_data_info/test_' + str(i_fold) + '.csv'

            source_images_dir = r'media/vcg8009/DATA/xbx/data/wpc2.0/img_4'
            target_images_dir = r'media/vcg8009/DATA/xbx/data/sjtu/img_4'
            test_images_dir = r'media/vcg8009/DATA/xbx/data/sjtu/img_4'

        elif config.source_database == 'SJTU':
            source_filename_list = 'datainfo/sjtu_data_info/total.csv'
            target_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/sjtu_data_info/test_' + str(i_fold) + '.csv'

            source_images_dir = r'media/vcg8009/DATA/xbx/data/sjtu/img_4'
            target_images_dir = r'media/vcg8009/DATA/xbx/data/sjtu/img_4'
            test_images_dir = r'media/vcg8009/DATA/xbx/data/sjtu/img_4'

    elif config.target_database == 'WPC':
        if config.source_database == 'SJTU':
            source_filename_list = 'datainfo/sjtu_data_info/total.csv'
            # target_filename_list = 'datainfo/wpc_data_info/total.csv'
            # test_filename_list = 'datainfo/wpc_data_info/total.csv'
            # source_filename_list = 'datainfo/sjtu_data_info/test_1.csv'
            target_filename_list = 'datainfo/wpc_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'

            source_images_dir = r'media/vcg8009/DATA/xbx/data/sjtu/img_4'
            target_images_dir = r'media/vcg8009/DATA/xbx/data/wpc/img_4'
            test_images_dir = r'media/vcg8009/DATA/xbx/data/wpc/img_4'

        elif config.source_database == 'WPC2.0':
            source_filename_list = 'datainfo/wpc2.0_data_info/total.csv'
            target_filename_list = 'datainfo/wpc_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'

            source_images_dir = r'media/vcg8009/DATA/xbx/data/wpc2.0/img_4'
            target_images_dir = r'media/vcg8009/DATA/xbx/data/wpc/img_4'
            test_images_dir = r'media/vcg8009/DATA/xbx/data/wpc/img_4'

    elif config.target_database == 'WPC2.0':

        if config.source_database == 'SJTU':
            # source_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            source_filename_list = 'datainfo/sjtu_data_info/total.csv'
            # target_filename_list = 'datainfo/wpc_data_info/train_' + str(i_fold) + '.csv'
            # test_filename_list = 'datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'
            target_filename_list = 'datainfo/wpc2.0_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/wpc2.0_data_info/test_' + str(i_fold) + '.csv'
            source_images_dir = r'media/vcg8009/DATA/xbx/data/sjtu/img_4'
            target_images_dir = r'media/vcg8009/DATA/xbx/data/wpc2.0/img_4'
            test_images_dir = r'media/vcg8009/DATA/xbx/data/wpc2.0/img_4'

        elif config.source_database == 'WPC':
            source_filename_list = 'datainfo/wpc_data_info/total.csv'
            target_filename_list = 'datainfo/wpc2.0_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = 'datainfo/wpc2.0_data_info/test_' + str(i_fold) + '.csv'
            source_images_dir = r'media/vcg8009/DATA/xbx/data/wpc/img_4'
            target_images_dir = r'media/vcg8009/DATA/xbx/data/wpc2.0/img_4'
            test_images_dir = r'media/vcg8009/DATA/xbx/data/wpc2.0/img_4'

    print('using source train datainfo: ' + source_filename_list)
    print('using target train datainfo: ' + target_filename_list)
    print('using target test datainfo: ' + test_filename_list)

    criterion = nn.MSELoss().to(device)
    loss_domain = torch.nn.CrossEntropyLoss()

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
    optimizer_backbone = torch.optim.Adam(model[0].parameters(), lr=config.lr, weight_decay=0.0000001)
    optimizer_fusion = torch.optim.Adam(model[1].parameters(), lr=config.lr, weight_decay=0.0000001)
    # optimizer_ad_net = torch.optim.SGD(model[2].parameters(), lr=config.lr, weight_decay=0.0005, momentum=config.momentum)
    optimizer_ad_net = torch.optim.Adam(model[2].parameters(), lr=config.lr, weight_decay=0.0000001)
    optimizer_iqa_head = torch.optim.Adam(model[3].parameters(), lr=config.lr, weight_decay=0.0000001)
    optimizer = [optimizer_backbone, optimizer_fusion, optimizer_ad_net, optimizer_iqa_head]

    scheduler_backbone = torch.optim.lr_scheduler.StepLR(optimizer_backbone, step_size=config.decay_interval,
                                                         gamma=config.decay_ratio)
    scheduler_fusion = torch.optim.lr_scheduler.StepLR(optimizer_fusion, step_size=config.decay_interval,
                                                         gamma=config.decay_ratio)
    scheduler_ad_net = torch.optim.lr_scheduler.StepLR(optimizer_ad_net, step_size=config.decay_interval,
                                                         gamma=config.decay_ratio)
    scheduler_iqa_head = torch.optim.lr_scheduler.StepLR(optimizer_iqa_head, step_size=config.decay_interval,
                                                         gamma=config.decay_ratio)
    scheduler = [scheduler_backbone, scheduler_fusion, scheduler_ad_net, scheduler_iqa_head]


    print('Starting training:')
    len_dataloader = min(len(source_loader), len(target_loader))
    pbar = tqdm(total=len_dataloader * config.epochs)

    for epoch in range(config.epochs):

        model[0].train()
        model[1].train()
        model[2].train()
        model[3].train()

        batch_losses = []
        batch_losses_each_disp = []
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

            feat_fusion, t_feat_fusion = model[1](feat, t_feat, if_fusion=False)

            feat = model[3](feat)
            avg_feat = torch.mean(feat, dim=1)
            avg_feat = avg_feat.view(image_size[0], image_size[1])
            score = torch.mean(avg_feat, dim=1)

            feat = model[3](feat_fusion)
            avg_feat = torch.mean(feat, dim=1)
            avg_feat = avg_feat.view(image_size[0], image_size[1])
            score2 = torch.mean(avg_feat, dim=1)

            feat_fusion, t_feat_fusion = torch.mean(feat_fusion, dim=1), torch.mean(t_feat_fusion, dim=1)
            source_domain = model[2](feat_fusion)
            target_domain = model[2](t_feat_fusion)


            s_domain_label = torch.zeros(config.batch_size).long()
            t_domain_label = torch.ones(config.batch_size).long()
            s_domain_label = s_domain_label.cuda()
            t_domain_label = t_domain_label.cuda()


            dif = torch.abs(score.detach().cpu() - labels_source.detach().cpu())
            dif2 = torch.abs(score2.detach().cpu() - labels_source.detach().cpu())
            source_domain_copy = source_domain.clone()
            for index in range(dif.size(0)):
                if dif[index] < dif2[index]:
                    source_domain_copy[index] = 1 - source_domain[index]

            loss_mse = criterion(score2, labels_source)
            loss_ad = loss_domain(source_domain_copy, s_domain_label) + loss_domain(target_domain, t_domain_label)
            loss = loss_mse + 0.8 * loss_ad

            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            optimizer[2].zero_grad()
            optimizer[3].zero_grad()
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()
            optimizer[2].step()
            optimizer[3].step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Score_loss: {:.4f}, Domain_loss: {:.4f}'.format(
                    epoch + 1, config.epochs, i + 1, len_dataloader,
                    loss.detach().item(), loss_mse.detach().item(), loss_ad.detach().item()))

        scheduler[0].step()
        scheduler[1].step()
        scheduler[2].step()
        scheduler[3].step()
        lr = scheduler[0].get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))
        end = time.time()
        print('Epoch %d training time cost: %.4f seconds' % (epoch + 1, end - start))

    # do validation after last epoch
    start = time.time()
    with torch.no_grad():
        model[0].eval()
        model[1].eval()
        model[2].eval()
        model[3].eval()
        label = np.zeros([len(testset)])
        y_pred = np.zeros([len(testset)])
        for i, data in enumerate(test_loader):
            image = data['image'].to(device)
            label[i] = data['gt_label'].item()
            image_size = image.shape
            image = image.view(-1, image_size[2], image_size[3], image_size[4])
            feat = model[0](image)  ##(batch_size,49,768)
            feat = model[3](feat)
            avg_feat = torch.mean(feat, dim=1)
            avg_feat = avg_feat.view(image_size[0], image_size[1])
            score = torch.mean(avg_feat, dim=1)


            y_pred[i] = score.item()
        y_pred = fit_function(label, y_pred)
        test_PLCC = stats.pearsonr(y_pred, label)[0]
        test_SRCC = stats.spearmanr(y_pred, label)[0]
        test_KRCC = stats.kendalltau(y_pred, label)[0]
        test_RMSE = np.sqrt(((y_pred - label) ** 2).mean())
        end = time.time()

        print('Epoch %d testing time cost: %.4f seconds' % (epoch + 1, end - start))
        print('SRCC: {:.4f}, PLCC: {:.4f}, KRCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC, test_PLCC,
                                                                                           test_KRCC,
                                                                                           test_RMSE),
              flush=True)

        if not os.path.exists(config.ckpt_path + "/gms/step2"):
            os.makedirs(config.ckpt_path + "/gms/step2")
        filenamenew = os.path.join('checkpoint/gms/step2',
                         config.source_database + "_to_" + config.target_database + '_fold_' + str(
                             i_fold) + '_final.pth')
        torch.save({
            'model_0_state_dict': model[0].state_dict(),
            'model_1_state_dict': model[1].state_dict(),
            'model_2_state_dict': model[2].state_dict(),
            'model_3_state_dict': model[3].state_dict(),
        }, filenamenew)

        print('Saving model to', filenamenew)
    return [test_SRCC, test_PLCC, test_KRCC, test_RMSE]


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
    parser.add_argument('--resume', type=str, default='checkpoint/gms/step1', help='path for loading the checkpoint')
    # parser.add_argument('--load_path', type=str, default='trained_ckpt/SJTU_fold_1_best.pth')
    # training parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--momentum', type=int, default=0.9)
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
    with open('_gms_wpc2sjtu.txt', 'w') as f:
        f.write(str(np.mean(results, axis=0)))
    print('The median best result:', np.median(results, axis=0))