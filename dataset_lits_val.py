from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
import glob
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset


class Val_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        self.files_A = sorted(glob.glob(os.path.join(args.val_path, 'CT_Reg') + '/*.nrrd'))
        self.files_B = sorted(glob.glob(os.path.join(args.val_path, 'MR_Reg') + '/*.nrrd'))
        self.files_C = sorted(glob.glob(os.path.join(args.val_path,  'SEG_hard') + '/*.nrrd'))
        # self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))



    def __getitem__(self, index):

        ct = sitk.ReadImage(self.files_A[index % len(self.files_A)])
        mr = sitk.ReadImage(self.files_B[index % len(self.files_B)])
        seg = sitk.ReadImage(self.files_C[index % len(self.files_C)])

        ct_array = sitk.GetArrayFromImage(ct)
        mr_array = sitk.GetArrayFromImage(mr)
        seg_array = sitk.GetArrayFromImage(seg)

        # a=np.unique(seg_array)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        mr_array = torch.FloatTensor(mr_array).unsqueeze(0)
        seg_array = torch.LongTensor(seg_array).unsqueeze(0)


        return ct_array,mr_array, seg_array

    def __len__(self):
        return len(self.files_A)



if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    train_ds = Train_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i,ct.size(),seg.size())