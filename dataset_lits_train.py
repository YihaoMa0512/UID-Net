from scipy import ndimage
from torch.utils.data import DataLoader
import os
import glob
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
import elasticdeform
import random
from .transforms import data_augmentation_scale, data_augmentation_rotate, data_augmentation_range, data_gamma

class Train_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        self.files_A = sorted(glob.glob(os.path.join(args.dataset_path, 'CT_Reg') + '/*.nrrd'))
        self.files_B = sorted(glob.glob(os.path.join(args.dataset_path, 'MR_Reg') + '/*.nrrd'))
        self.files_C = sorted(glob.glob(os.path.join(args.dataset_path, 'SEG_hard') + '/*.nrrd'))




    def __getitem__(self, index):

        ct = sitk.ReadImage(self.files_A[index % len(self.files_A)])
        mr = sitk.ReadImage(self.files_B[index % len(self.files_B)])
        seg = sitk.ReadImage(self.files_C[index % len(self.files_C)])



        ct_array = sitk.GetArrayFromImage(ct)
        mr_array = sitk.GetArrayFromImage(mr)
        seg_array = sitk.GetArrayFromImage(seg)


        seed=random.uniform(0,1)
        if random.randint(0, 2) == 1:
            [ct_array, mr_array, seg_array] = elasticdeform.deform_random_grid([ct_array, mr_array, seg_array],
                                                                               sigma=3 * seed,
                                                                               order=[0, 0,0])
        if random.randint(0, 2) == 1:
            ct_array, mr_array, seg_array= data_augmentation_scale(ct_array, mr_array, seg_array)
        # if random.randint(0, 2) == 1:
        #     ct_array, mr_array, seg_array= data_augmentation_rotate(ct_array, mr_array, seg_array )
        if random.randint(0, 2) == 1:
            ct_array, mr_array, seg_array = data_augmentation_range(ct_array, mr_array, seg_array)
        if random.randint(0, 2) == 1:
            ct_array, mr_array, seg_array = data_gamma(ct_array, mr_array, seg_array)
        seg_array2 = ndimage.zoom(seg_array, zoom=1 / 2, order=0)
        seg_array3 = ndimage.zoom(seg_array, zoom=1 / 4, order=0)
        seg_array4 = ndimage.zoom(seg_array, zoom=1 / 8, order=0)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        mr_array = torch.FloatTensor(mr_array).unsqueeze(0)
        seg_array1 = torch.LongTensor(seg_array)
        seg_array2 = torch.LongTensor(seg_array2)
        seg_array3 = torch.LongTensor(seg_array3)
        seg_array4 = torch.LongTensor(seg_array4)

        return ct_array, mr_array, seg_array1, seg_array2, seg_array3, seg_array4
    def __len__(self):
        return len(self.files_A)


if __name__ == "__main__":

    from config import args
    train_ds = Train_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 1, False, num_workers=1)

    for i, (ct, mr,seg) in enumerate(train_dl):
        print(i,ct.size(),mr.size(),seg.size())