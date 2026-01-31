import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import random
from torchvision import transforms
from torch.utils import data
from PIL import Image


class MMDataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir_2d, data_dir_pc, datainfo_path, transform, crop_size=224, img_length_read=4,
                 patch_length_read=6, npoint=2048, is_train=True):
        super(MMDataset, self).__init__()
        dataInfo = pd.read_csv(datainfo_path, header=0, sep=',', index_col=False, encoding="utf-8-sig")
        self.sample1_ply_name = dataInfo[['sample1_name']]
        self.sample1_ply_mos = dataInfo['sample1_mos']
        self.sample2_ply_name = dataInfo[['sample2_name']]
        self.sample2_ply_mos = dataInfo['sample2_mos']
        self.crop_size = crop_size
        self.data_dir_2d = data_dir_2d
        self.transform = transform
        self.img_length_read = img_length_read
        self.patch_length_read = patch_length_read
        self.npoint = npoint
        self.data_dir_pc = data_dir_pc
        self.length = len(self.sample1_ply_name)
        self.is_train = is_train

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        sample1_img_name = self.sample1_ply_name.iloc[idx, 0]
        sample1_frames_dir = os.path.join(self.data_dir_2d, sample1_img_name)
        sample2_img_name = self.sample2_ply_name.iloc[idx, 0]
        sample2_frames_dir = os.path.join(self.data_dir_2d, sample2_img_name)

        img_channel = 3
        img_height_crop = self.crop_size
        img_width_crop = self.crop_size

        img_length_read = self.img_length_read
        sample1_transformed_img = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])
        sample2_transformed_img = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])
        # read images
        img_read_index = 0
        for i in range(img_length_read):
            # load images
            sample1_imge_name = os.path.join(sample1_frames_dir, str(i) + '.png')
            sample2_imge_name = os.path.join(sample2_frames_dir, str(i) + '.png')
            if os.path.exists(sample1_imge_name) and os.path.exists(sample2_imge_name):
                sample1_read_frame = Image.open(sample1_imge_name)
                sample2_read_frame = Image.open(sample2_imge_name)
                # print(read_frame)
                sample1_read_frame = sample1_read_frame.convert('RGB')
                sample2_read_frame = sample2_read_frame.convert('RGB')
                # transform
                sample1_read_frame = self.transform(sample1_read_frame)
                sample1_transformed_img[i] = sample1_read_frame
                sample2_read_frame = self.transform(sample2_read_frame)
                sample2_transformed_img[i] = sample2_read_frame

                img_read_index += 1
            else:
                print(sample1_imge_name + ' or ' + sample2_imge_name)
                print('Image do not exist!')

        if img_read_index < img_length_read:
            for j in range(img_read_index, img_length_read):
                sample1_transformed_img[j] = sample1_transformed_img[img_read_index - 1]
                sample2_transformed_img[j] = sample2_transformed_img[img_read_index - 1]

        # read pc
        patch_length_read = self.patch_length_read
        npoint = self.npoint
        sample1_selected_patches = torch.zeros([patch_length_read, 3, npoint])
        sample2_selected_patches = torch.zeros([patch_length_read, 3, npoint])
        sample1_path = os.path.join(self.data_dir_pc, self.sample1_ply_name.iloc[idx, 0].split('.')[0] + '.npy')
        sample2_path = os.path.join(self.data_dir_pc, self.sample2_ply_name.iloc[idx, 0].split('.')[0] + '.npy')
        # load point clouds
        sample1_points = list(np.load(sample1_path))
        sample2_points = list(np.load(sample2_path))
        # randomly select patches during the training stage
        if self.is_train:
            sample1_random_patches = random.sample(sample1_points, patch_length_read)
            sample2_random_patches = random.sample(sample2_points, patch_length_read)
        else:
            sample1_random_patches = sample1_points
            sample2_random_patches = sample2_points
        for i in range(patch_length_read):
            sample1_selected_patches[i] = torch.from_numpy(sample1_random_patches[i]).transpose(0, 1)
            sample2_selected_patches[i] = torch.from_numpy(sample2_random_patches[i]).transpose(0, 1)

        sample1_y_mos = self.sample1_ply_mos.iloc[idx]
        sample1_y_label = torch.FloatTensor(np.array(sample1_y_mos))
        sample2_y_mos = self.sample2_ply_mos.iloc[idx]
        sample2_y_label = torch.FloatTensor(np.array(sample2_y_mos))

        return sample1_transformed_img, sample1_selected_patches, sample1_y_label, \
               sample2_transformed_img, sample2_selected_patches, sample2_y_label