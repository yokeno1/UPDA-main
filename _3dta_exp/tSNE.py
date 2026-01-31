import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from sklearn.manifold import TSNE
from tqdm import tqdm

from dataload.dataload_3dta import WPC_SD
from model.backbone_3dta import Feature_extraction,quality_regression
from torch.utils.data import DataLoader

from sklearn import preprocessing

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



def main(args, i_fold):

    print('-' * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = Feature_extraction(args).to(device).float()
    # fusion = fusion_net(args).to(device).float()
    # ad_net = AdversarialNetwork(args).to(device).float()
    # qa_net = quality_regression(args).to(device).float()
    model = backbone


    # model.load_state_dict(torch.load(r"/mnt/e/xbx/SFDA-main/GMS-3DQA-main/baseline_Dist/trained_ckpt/tta/SJTU_to_WPC_fold_1_best.pth"))

    if args.resume:
        checkpoint = torch.load('./checkpoint/_3dta/step2/WPC_to_WPC2.0_fold_1_final.pth')
        model.load_state_dict(checkpoint['model_0_state_dict'], strict=True)
        # model[1].load_state_dict(checkpoint['model_3_state_dict'], strict=True)
        print("load resume success")

    model = model.to(device)

    # database configuration
    if args.target_database == 'SJTU':

        if args.source_database == 'WPC':
            # source_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            source_filename_list = '../datainfo/wpc_data_info/total.csv'
            target_filename_list = '../datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/sjtu_data_info/test_' + str(i_fold) + '.csv'

            source_data_dir = r'D:\xbx\data/wpc'
            target_data_dir = r'D:\xbx\data/sjtu'
            test_data_dir = r'D:\xbx\data/sjtu'

        elif args.source_database == 'WPC2.0':
            source_filename_list = '../datainfo/wpc2.0_data_info/total.csv'
            target_filename_list = '../datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/sjtu_data_info/test_' + str(i_fold) + '.csv'

            source_data_dir = r'D:\xbx\data/wpc2.0'
            target_data_dir = r'D:\xbx\data/sjtu'
            test_data_dir = r'D:\xbx\data/sjtu'

        elif args.source_database == 'SJTU':
            source_filename_list = '../datainfo/sjtu_data_info/total.csv'
            target_filename_list = '../datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/sjtu_data_info/test_' + str(i_fold) + '.csv'

            source_data_dir = r'D:\xbx\data/sjtu'
            target_data_dir = r'D:\xbx\data/sjtu'
            test_data_dir = r'D:\xbx\data/sjtu'

    elif args.target_database == 'WPC':
        if args.source_database == 'SJTU':
            source_filename_list = '../datainfo/sjtu_data_info/total.csv'
            # target_filename_list = 'datainfo/wpc_data_info/total.csv'
            # test_filename_list = 'datainfo/wpc_data_info/total.csv'
            # source_filename_list = 'datainfo/sjtu_data_info/test_1.csv'
            target_filename_list = '../datainfo/wpc_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'

            source_data_dir = r'D:\xbx\data/sjtu'
            target_data_dir = r'D:\xbx\data/wpc'
            test_data_dir = r'D:\xbx\data/wpc'

        elif args.source_database == 'WPC2.0':
            source_filename_list = '../datainfo/wpc2.0_data_info/total.csv'
            target_filename_list = '../datainfo/wpc_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'

            source_data_dir = r'D:\xbx\data/wpc2.0'
            target_data_dir = r'D:\xbx\data/wpc'
            test_data_dir = r'D:\xbx\data/wpc'

    elif args.target_database == 'WPC2.0':

        if args.source_database == 'SJTU':
            # source_filename_list = 'datainfo/sjtu_data_info/train_' + str(i_fold) + '.csv'
            source_filename_list = '../datainfo/sjtu_data_info/total.csv'
            # target_filename_list = 'datainfo/wpc_data_info/train_' + str(i_fold) + '.csv'
            # test_filename_list = 'datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'
            target_filename_list = '../datainfo/wpc2.0_data_info/train_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/wpc2.0_data_info/test_' + str(i_fold) + '.csv'
            source_data_dir = r'D:\xbx\data/sjtu'
            target_data_dir = r'D:\xbx\data/wpc2.0'
            test_data_dir = r'D:\xbx\data/wpc2.0'

        elif args.source_database == 'WPC':
            source_filename_list = '../datainfo/wpc_data_info/test_' + str(i_fold) + '.csv'
            target_filename_list = '../datainfo/wpc2.0_data_info/test_' + str(i_fold) + '.csv'
            test_filename_list = '../datainfo/wpc2.0_data_info/test_' + str(i_fold) + '.csv'
            source_data_dir = r'D:\xbx\data/wpc'
            target_data_dir = r'D:\xbx\data/wpc2.0'
            test_data_dir = r'D:\xbx\data/wpc2.0'

    print("source_filename_list:", source_filename_list)
    print('target_filename_list:', target_filename_list)
    print("test_filename_list:", test_filename_list)


    # dataloader configuration
    train_data = WPC_SD(args, data_dir=source_data_dir, filename_list=source_filename_list)
    source_loader = DataLoader(train_data, num_workers=4,
                               batch_size=args.batch_size, shuffle=True, drop_last=True)

    target_train_data = WPC_SD(args, data_dir=target_data_dir, filename_list=target_filename_list)
    target_loader = DataLoader(target_train_data, num_workers=4,
                               batch_size=args.batch_size, shuffle=True, drop_last=True)

    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0000001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)

    best_test_criterion = -1  # accuracy
    best_test = []
    # print('Starting training:')
    pbar = tqdm(total=len(target_loader) + len(source_loader))
    for epoch in range(args.epochs):

        feature_source = []
        feature_target = []
        with torch.no_grad():
            model.eval()
            # label = np.zeros([len(testset)])
            # y_pred = np.zeros([len(testset)])
            # with open('feat.csv', 'w') as f:
            for id, (data, mos, filenum) in enumerate(source_loader):
                pbar.update()
                data, mos = data.permute(0, 2, 1), mos.to(torch.float64).to(device).squeeze()
                data = data.type(torch.FloatTensor).to(device)
                batch_size = data.size()[0]
                feat1 = model(data)
                # print(feat1.shape)
                # print(feat.shape)
                feat1 = feat1.cpu().numpy()
                features_reshaped = feat1.reshape(1,-1)
                # print(features_reshaped)
                feature_source.append(features_reshaped)

                #
                # csv_write = csv.writer(f)
                # csv_write.writerow([data['name'],feat1,feat,outputs])

                # y_pred[i] = outputs.item()


            for id, (data, mos, filenum) in enumerate(target_loader):
                pbar.update()
                data, mos = data.permute(0, 2, 1), mos.to(torch.float64).to(device).squeeze()
                # print(data, mos)
                data = data.type(torch.FloatTensor).to(device)
                batch_size = data.size()[0]
                feat1 = model(data)
                feat1 = feat1.cpu().numpy()
                features_reshaped = feat1.reshape(1,-1)
                # print(features_reshaped)
                feature_source.append(features_reshaped)

                #
                # csv_write = csv.writer(f)
                # csv_write.writerow([data['name'],feat1,feat,outputs])

            #         y_pred[i] = outputs.item()
            #
            #     y_pred = fit_function(label, y_pred)
            #     test_PLCC = stats.pearsonr(y_pred, label)[0]
            #     test_SRCC = stats.spearmanr(y_pred, label)[0]
            #     test_KRCC = stats.kendalltau(y_pred, label)[0]
            #     test_RMSE = np.sqrt(((y_pred - label) ** 2).mean())
            # end = time.time()
            # 使用 t-SNE 将特征降到 2 维

            tsne = TSNE(n_components=2, random_state=42)

            # feature_source.append(feature_target)

            feature = np.concatenate(feature_source)

            features_tsne = tsne.fit_transform(feature)
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            features_tsne = scaler.fit_transform(features_tsne)
            print(features_tsne.shape)
            # 绘制结果


            # fig.add_subplot(projection='3d')
            # plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=label)
            shapes = ['o', '^']  # 圆形和三角形
            # y = label
            # min_mos = np.min(y)
            # max_mos = np.max(y)
            # mos_norm = (y - min_mos) / (max_mos - min_mos)  # 归一化MOS分数到[0, 1]
            plt.figure(figsize=(8, 8))
            for i in range(len(feature_source)):
                # 根据模型标签选择形状
                if i < len(source_loader):
                    shape = shapes[0]
                    color = 'red'
                else:
                    shape = shapes[1]
                    color = 'blue'
                # shape = shapes[1]
                plt.scatter(features_tsne[i, 0], features_tsne[i, 1],
                            c=color)
            # for i in range(378):
            #                 # 根据模型标签选择形状
            #                 # if i % 42 == 0:
            #                 #     shape = shapes[i // 42]
            #                 shape = shapes[i % 6 ]
            #                 plt.scatter(features_tsne[i, 0], features_tsne[i, 1],
            #                             c=colors[i], marker=shape)

            plt.legend()
            plt.title('t-SNE visualization with different shapes and colors')
            plt.show()


            # print('Epoch %d testing time cost: %.4f seconds' % (epoch + 1, end - start))
            # if test_SRCC > best_test_criterion:
            #     print("Update best model using best_test_criterion in epoch {}".format(epoch + 1), flush=True)
            #     print('Updataed SRCC: {:.4f}, PLCC: {:.4f}, KRCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC, test_PLCC,
            #                                                                                        test_KRCC,
            #                                                                                        test_RMSE),
            #           flush=True)
            #     best_test_criterion = test_SRCC
            #     best_test = [test_SRCC, test_PLCC, test_KRCC, test_RMSE]


            # print('Saving model...')
            # if not os.path.exists(config.ckpt_path):
            #     os.makedirs(config.ckpt_path)
            # torch.save(model.state_dict(),
            #            os.path.join(config.ckpt_path, config.database + '_fold_' + str(i_fold) + '_best.pth'))
            # print('Saving model to',
            #       os.path.join(config.ckpt_path, config.database + '_fold_' + str(i_fold) + '_best.pth'))
    return best_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Our 3DTA')

    parser.add_argument('--exp_name', type=str, default='3DTA_patch_mos', metavar='N', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of episode to train ')
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
    parser.add_argument('--target_database', type=str, default='WPC2.0', metavar='N')
    parser.add_argument('--k_fold', type=int, default=1)
    args = parser.parse_args()

    print(args)
    # set_rand_seed(seed=2023)
    results = np.empty(shape=[0, 4])
    for i_fold in range(1, args.k_fold + 1):
        best_test = main(args, 1)
    #     print(
    #         '--------------------------------------------The {}-th Fold-----------------------------------------'.format(
    #             i_fold))
    #     print('Training completed.')
    #     print('The best training result SRCC: {:.4f}, PLCC: {:.4f}, KRCC: {:.4f}, and RMSE: {:.4f}'.format( \
    #         best_test[0], best_test[1], best_test[2], best_test[3]))
    #     results = np.concatenate((results, np.array([best_test])), axis=0)
    #     print('-------------------------------------------------------------------------------------------------------')
    #     print('-------------------------------------------------------------------------------------------------------')
    #
    # print('==============================done==============================================')
    # print('The mean best result:', np.mean(results, axis=0))
    # print('The median best result:', np.median(results, axis=0))
