import main_gms,main_3dta,main_mmpcqa
import argparse
import torch
import numpy as np
import random


def set_rand_seed(seed=1998):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()

if __name__ == '__main__':
    args = parser.parse_args()

    # input parameters
    parser.add_argument('--model', type=str, default='GMS_3DQA', help='Which model to use(GMS_3DQA, 3DTA, MM-PCQA)')
    parser.add_argument('--source_database', type=str, default='WPC')
    parser.add_argument('--target_database', type=str, default='SJTU')
    parser.add_argument('--k_fold', type=int, default=1, help='9 for SJTU, 5 for WPC, 4 for WPC2.0')

    # parser.add_argument('--load_path', type=str, default='trained_ckpt/SJTU_fold_1_best.pth')

    parser.add_argument('--batch_size', type=int, default=16, help='16 for gms, 16 for 3dta,')

    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    # 3dta parameters
    parser.add_argument('--point_num', type=int, default=1024, help='num of points to use')
    parser.add_argument('--pre_train', type=bool, default=False, help='evaluate the model?')
    parser.add_argument('--patch_dir', type=str, default='patch_72_10000', help='Where does patches exist?')
    parser.add_argument('--patch_num', type=int, default=72, metavar='N', help='How many patchs each PC have?')
    parser.add_argument('--patch_length_read', default=6, type=int, help='number of the using patches')
    parser.add_argument('--img_length_read', default=4, type=int, help='number of the using images')

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_ids', type=list, default=0)


    args = parser.parse_args()

    config = parser.parse_args()

    if config.model == 'GMS_3DQA':
        trainer = main_gms.test
    elif config.model == '3DTA':
        trainer = main_3dta.test
    elif config.model == 'MM_PCQA':
        trainer = main_mmpcqa.test

    set_rand_seed(seed=1)
    results = np.empty(shape=[0, 4])
    for i_fold in range(1, config.k_fold + 1):
        best_test = trainer(config, i_fold)
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
    print('The median best result:', np.median(results, axis=0))