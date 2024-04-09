from torch.utils.data import Dataset
import pandas as pd
import random
import numpy as np
import os
from utils import dict_train, data_path
import nibabel as nib


class dataProc(Dataset):
    def __init__(self, stage='train', fold=0, seed=2023, group='sMCI_pMCI', frame_path=r'T1_FGD.csv', version='v1'):

        frame = pd.read_csv(frame_path)
        self.frame = frame.copy()
        frame = frame[~pd.isna(frame['Data path.T1']) & ~pd.isna(frame['Data path.pet.fgd']) & frame['Group'].isin(
            group.split('_'))]
        index_ = frame.index.tolist()
        random.seed(seed)
        random.shuffle(index_)
        if stage == 'train':
            temp_ = [np.array_split(frame.index.tolist(), 5)[i] for i in range(5) if i != fold]
            self.index = np.concatenate(temp_, axis=0)
        else:
            self.index = np.array_split(frame.index.tolist(), 5)[fold]
        self.dict = {
            group.split('_')[0]: 0,
            group.split('_')[1]: 1,
        }
        self.version = version
        self.stage = stage
        self.normalization = dict_train[version]['normalization']
        self.modality_list = dict_train[version]['modality'].split('_')
        self.path = r'/media/shucheng/MyBook/DL_dataset/Conventional_CNN'

    def __len__(self):
        return self.index.shape[0]

    def to_label(self, group_list):
        label = []
        for i in group_list:
            cls = self.dict[i]
            label.append(cls)
        return label

    def get_sample_weights(self):
        weights = []
        count_nums = np.arange(0, 2).astype(np.int64)
        count = float(self.index.shape[0])
        label = self.to_label(self.frame.loc[self.index.tolist(), 'Group'])
        count_class_list = [float(label.count(i)) for i in count_nums]
        for i in label:
            for j in count_nums:
                if i == j:
                    weights.append(count / count_class_list[j])
        imbalanced_ratio = [count_class_list[0] / i_r for i_r in count_class_list]
        return weights, imbalanced_ratio

    def data_normal(self, npy):
        return (npy - np.mean(npy)) / (np.std(npy))

    def crop(self, npy):
        return npy[50: -50, 50:-50, 20: -50]

    # import matplotlib.pyplot as plt
    #
    # plt.imshow(data[80, ...])
    # plt.show()

    def __getitem__(self, item):
        subject_information = self.frame.loc[self.index[item], :]
        data_dict = {}
        for modality in self.modality_list:
            path = os.path.join(self.path, subject_information['Subject ID'], modality + '_crop.nii.gz')
            data = self.data_normal(nib.load(path).get_fdata())
            data_dict[modality] = np.expand_dims(data, axis=0)
        label = self.dict[subject_information['Group']]
        return data_dict, label


if __name__ == '__main__':
    dp = dataProc()
    for i in dp:
        print()
