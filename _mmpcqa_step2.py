import os, argparse, time

import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms
import random
import torch.backends.cudnn as cudnn
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from model.backbone_mmpcqa import Feature_extraction,quality_regression
from model.adaptation import AdversarialNetwork
from model.adaptation import fusion_net
from dataload.dataload_mmpcqa import MMDataset
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
# torch.cuda.set_per_process_memory_fraction(1.0, 0)


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
        s_train_filename_list = 'datainfo/sjtu_data_info/total.csv'
        s_data_dir_2d = r"/media/vcg8009/DATA/xbx/data/sjtu/img_4"
        s_data_dir_pc = r"/media/vcg8009/DATA/xbx/data/sjtu/patch"
        if target_database == 'WPC':
            t_train_filename_list = f'datainfo/wpc_data_info/train_{k_fold_id}.csv'
            t_test_filename_list = f'datainfo/wpc_data_info/test_{k_fold_id}.csv'
            t_data_dir_2d = r'/media/vcg8009/DATA/xbx/data/wpc/img_4'
            t_data_dir_pc = r'/media/vcg8009/DATA/xbx/data/wpc/patch'
        elif target_database == 'WPC2.0':
            t_train_filename_list = f'datainfo/wpc2.0_data_info/train_{k_fold_id}.csv'
            t_test_filename_list = f'datainfo/wpc2.0_data_info/test_{k_fold_id}.csv'
            t_data_dir_2d = r"/media/vcg8009/DATA/xbx/data/wpc2.0/img_4"
            t_data_dir_pc = r"/media/vcg8009/DATA/xbx/data/wpc2.0/patch"

    elif source_database == 'WPC':
        s_train_filename_list = 'datainfo/wpc_data_info/total.csv'
        s_data_dir_2d = r'/media/vcg8009/DATA/xbx/data/wpc/img_4'
        s_data_dir_pc = r'/media/vcg8009/DATA/xbx/data/wpc/patch'
        if target_database == 'SJTU':
            t_train_filename_list = f'datainfo/sjtu_data_info/train_{k_fold_id}.csv'
            t_test_filename_list = f'datainfo/sjtu_data_info/test_{k_fold_id}.csv'
            t_data_dir_2d = r"/media/vcg8009/DATA/xbx/data/sjtu/img_4"
            t_data_dir_pc = r"/media/vcg8009/DATA/xbx/data/sjtu/patch"
        elif target_database == 'WPC2.0':
            t_train_filename_list = f'datainfo/wpc2.0_data_info/train_{k_fold_id}.csv'
            t_test_filename_list = f'datainfo/wpc2.0_data_info/test_{k_fold_id}.csv'
            t_data_dir_2d = r"/media/vcg8009/DATA/xbx/data/wpc2.0/img_4"
            t_data_dir_pc = r"/media/vcg8009/DATA/xbx/data/wpc2.0/patch"

    elif source_database == 'WPC2.0':
        s_train_filename_list = 'datainfo/wpc2.0_data_info/total.csv'
        s_data_dir_2d = r'/media/vcg8009/DATA/xbx/data/wpc2.0/img_4'
        s_data_dir_pc = r'/media/vcg8009/DATA/xbx/data/wpc2.0/patch'
        if target_database == 'SJTU':
            t_train_filename_list = f'datainfo/sjtu_data_info/train_{k_fold_id}.csv'
            t_test_filename_list = f'datainfo/sjtu_data_info/test_{k_fold_id}.csv'
            t_data_dir_2d = r"/media/vcg8009/DATA/xbx/data/sjtu/img_4"
            t_data_dir_pc = r"/media/vcg8009/DATA/xbx/data/sjtu/patch"
        elif target_database == 'WPC':
            t_train_filename_list = f'datainfo/wpc_data_info/train_{k_fold_id}.csv'
            t_test_filename_list = f'datainfo/wpc_data_info/test_{k_fold_id}.csv'
            t_data_dir_2d = r'/media/vcg8009/DATA/xbx/data/wpc/img_4'
            t_data_dir_pc = r'/media/vcg8009/DATA/xbx/data/wpc/patch'

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
    fusion = fusion_net(args).to(device).float()
    ad_net = AdversarialNetwork(args).to(device).float()
    qa_net = quality_regression(args).to(device).float()
    print('Using model: MM-PCQA')
    model = [backbone, fusion, ad_net, qa_net]

    if args.resume:
        checkpoint = torch.load(os.path.join(args.resume, args.source_database + "_to_" +
                                             args.target_database + '_fold_' + str(i_fold) + '_loss.pth'))
        model[0].load_state_dict(checkpoint['model_0_state_dict'], strict=True)

    criterion = nn.MSELoss().to(device)
    loss_domain = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
    #                       momentum=0.9, weight_decay=5e-4)
    optimizer_backbone = torch.optim.Adam(model[0].parameters(), lr=args.lr, weight_decay=args.decay_rate)
    optimizer_fusion = torch.optim.Adam(model[1].parameters(), lr=args.lr, weight_decay=args.decay_rate)
    optimizer_ad_net = torch.optim.Adam(model[2].parameters(), lr=args.lr, weight_decay=args.decay_rate)
    optimizer_qa_net = torch.optim.Adam(model[3].parameters(), lr=args.lr, weight_decay=args.decay_rate)
    optimizer = [optimizer_backbone, optimizer_fusion, optimizer_ad_net, optimizer_qa_net]

    print('Using Adam optimizer, initial learning rate: ' + str(args.lr))

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.9)
    scheduler_backbone = torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=args.decay_interval,
                                                         gamma=args.decay_ratio)
    scheduler_fusion = torch.optim.lr_scheduler.StepLR(optimizer[1], step_size=args.decay_interval,
                                                         gamma=args.decay_ratio)
    scheduler_ad_net = torch.optim.lr_scheduler.StepLR(optimizer[2], step_size=args.decay_interval,
                                                         gamma=args.decay_ratio)
    scheduler_qa_net = torch.optim.lr_scheduler.StepLR(optimizer[3], step_size=args.decay_interval,
                                                         gamma=args.decay_ratio)
    scheduler = [scheduler_backbone, scheduler_fusion, scheduler_ad_net, scheduler_qa_net]

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
                                                 num_workers=8, drop_last=True)
    t_train_loader = torch.utils.data.DataLoader(dataset=t_train_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=8, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=t_test_dataset, batch_size=1, shuffle=False, num_workers=8)

    best = []
    len_dataloader = min(len(s_train_loader), len(t_train_loader))
    pbar = tqdm(total=len_dataloader * num_epochs)
    for epoch in range(num_epochs):
        # begin training, during each epoch, the crops and patches are randomly selected for the training set and fixed for the testing set
        # if you want to change the number of images or projections, load the parameters here 'img_length_read = img_length_read, patch_length_read = patch_length_read'

        data_source_iter = iter(s_train_loader)
        data_target_iter = iter(t_train_loader)
        n_train = len(s_train_loader)
        n_test = len(test_loader)

        model[0].train()
        model[1].train()
        model[2].train()
        model[3].train()

        start = time.time()
        batch_losses = []
        batch_losses_each_disp = []
        x_output = np.zeros(n_train)
        x_test = np.zeros(n_train)

        for i in range(len_dataloader):
            pbar.update()
            imgs, pc, mos = data_source_iter.__next__()
            t_imgs, t_pc, t_mos = data_target_iter.__next__()

            imgs = imgs.to(device)
            pc = torch.Tensor(pc.float())
            pc = pc.to(device)
            mos = mos[:, np.newaxis]
            mos = mos.to(device)
            labels_source = mos

            t_imgs = t_imgs.to(device)
            t_pc = torch.Tensor(t_pc.float())
            t_pc = t_pc.to(device)
            # t_mos = t_mos[:, np.newaxis]
            # t_mos = t_mos.to(device)
            feat = model[0](imgs, pc)
            feat2 = model[0](t_imgs, t_pc)

            feat_fusion, t_feat_fusion = model[1](feat, feat2, if_fusion=False)

            source_domain = model[2](feat_fusion)
            target_domain = model[2](t_feat_fusion)

            source_score1 = model[3](feat)
            source_score2 = model[3](feat_fusion)

            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            optimizer[2].zero_grad()
            optimizer[3].zero_grad()

            batch_size = args.batch_size
            s_domain_label = torch.zeros(batch_size).long()
            t_domain_label = torch.ones(batch_size).long()
            s_domain_label = s_domain_label.cuda()
            t_domain_label = t_domain_label.cuda()
            # source_domain = torch.log(source_domain)
            # compute loss
            dif = torch.abs(source_score1.detach().cpu() - labels_source.detach().cpu())
            dif2 = torch.abs(source_score2.detach().cpu() - labels_source.detach().cpu())
            source_domain_copy = source_domain.clone()
            for index in range(dif.size(0)):
                if dif[index] < dif2[index]:
                    source_domain_copy[index] = 1 - source_domain[index]

            loss_mse = criterion(source_score2, mos)

            loss_ad = loss_domain(source_domain_copy, s_domain_label) + loss_domain(target_domain, t_domain_label)



            loss = loss_mse + 0.8 * loss_ad

            # compute loss
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()
            optimizer[2].step()
            optimizer[3].step()


            if (i + 1) % 10 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Score_loss: {:.4f}, Domain_loss: {:.4f}'.format(
                        epoch + 1, num_epochs, i + 1, len_dataloader,
                        loss.item(), loss_mse.item(), loss_ad.item()))
                # print(source_score2)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())


        avg_loss = sum(batch_losses) / (len(s_train_loader) // batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler[0].step()
        scheduler[1].step()
        scheduler[2].step()
        scheduler[3].step()
        lr_current = scheduler[0].get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr_current[0]))

        end = time.time()
        torch.cuda.empty_cache()
        print('Epoch %d training time cost: %.4f seconds' % (epoch + 1, end - start))

    # Test
    model[0].eval()  # training turn off
    model[1].eval()
    model[2].eval()
    model[3].eval()

    y_output = np.zeros(n_test)
    y_test = np.zeros(n_test)
    with torch.no_grad():
        for i, (imgs, pc, mos) in tqdm(enumerate(test_loader)):
            imgs = imgs.to(device)
            pc = torch.Tensor(pc.float())
            pc = pc.to(device)
            y_test[i] = (mos).item()
            feat = model[0](imgs, pc)
            outputs = model[3](feat)
            y_output[i] = outputs.item()

        y_output_logistic = fit_function(y_test, y_output)
        test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
        test_SROCC = stats.spearmanr(y_output_logistic, y_test)[0]
        test_RMSE = np.sqrt(((y_output_logistic - y_test) ** 2).mean())
        test_KROCC = scipy.stats.kendalltau(y_output_logistic, y_test)[0]
        print(
            "Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SROCC, test_KROCC,
                                                                                        test_PLCC, test_RMSE))

    if not os.path.exists(args.ckpt_path + "/_mmpcqa/step2"):
        os.makedirs(args.ckpt_path + "/_mmpcqa/step2")
    filenamenew = os.path.join('checkpoint/_mmpcqa/step2',
                               args.source_database + "_to_" + args.target_database + '_fold_' + str(
                                   i_fold) + '_final.pth')
    torch.save({
        'model_0_state_dict': model[0].state_dict(),
        'model_1_state_dict': model[1].state_dict(),
        'model_2_state_dict': model[2].state_dict(),
        'model_3_state_dict': model[3].state_dict(),
    }, filenamenew)

    print('Saving model to', filenamenew)
    return [test_SROCC, test_PLCC, test_KROCC, test_RMSE]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--epochs', help='Maximum number of training epochs.', default=30, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=4, type=int)
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--model', default='MM_PCQA', type=str)
    parser.add_argument('--patch_length_read', default=6, type=int, help='number of the using patches')
    parser.add_argument('--img_length_read', default=4, type=int, help='number of the using images')
    parser.add_argument('--in_feature', type=int, default=4096)
    parser.add_argument('--loss', default='mse', type=str)
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=float, default=5)
    parser.add_argument('--source_database', default='SJTU', type=str)
    parser.add_argument('--target_database', default='WPC', type=str)
    parser.add_argument('--k_fold_num', default=5, type=int,
                        help='9 for the SJTU-PCQA, 5 for the WPC, 4 for the WPC2.0')
    parser.add_argument('--ckpt_path', type=str, default='checkpoint')
    parser.add_argument('--resume', type=str, default='checkpoint/_mmpcqa/step1', help='path for loading the checkpoint')


    args = parser.parse_args()

    results = np.empty(shape=[0, 4])

    for i_fold in range(1, args.k_fold_num + 1):
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
    with open('_mmpcqa_wpc2sjtu.txt', 'w') as f:
        f.write(str(np.mean(results, axis=0)))
    print('The median best result:', np.median(results, axis=0))