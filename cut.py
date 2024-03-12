import os
import random
import SimpleITK as STK
import elasticdeform
import numpy as np
import pandas as pd
from pandas import ExcelWriter


path = '../set_1'
dir = 'origin_nor'
dir1 = 'data_train/all'

random.seed(2)
# path='../set_1'
# dir='origin_nor'
# dir1='data'

for CT_file in (os.listdir(os.path.join(path, dir, 'CT_Reg'))):
    for MR_file in (os.listdir(os.path.join(path, dir, 'MR_Reg'))):
        for SEG_file in (os.listdir(os.path.join(path, dir, 'SEG_all'))):

                if CT_file[0:7] in SEG_file[0:7] in MR_file[0:7] :
                    i=0
                    CT_img = STK.ReadImage(os.path.join(path, dir, 'CT_Reg', CT_file))
                    CT_array = STK.GetArrayFromImage(CT_img)
                    MR_img = STK.ReadImage(os.path.join(path, dir, 'MR_Reg', MR_file))
                    MR_array = STK.GetArrayFromImage(MR_img)
                    SEG_img = STK.ReadImage(os.path.join(path, dir, 'SEG_all', SEG_file))
                    SEG_array = STK.GetArrayFromImage(SEG_img)

                    for z in range(2):
                        for y in range(3):
                                split_ct = CT_array[z*96:(z+1)*96,y*96:(y+2)*96, 96:288]
                                split_mr = MR_array[z*96:(z+1)*96,y*96:(y+2)*96, 96:288]
                                split_seg = SEG_array[z*96:(z+1)*96,y*96:(y+2)*96, 96:288]
                                CT_img = STK.GetImageFromArray(split_ct)
                                MR_img = STK.GetImageFromArray(split_mr)
                                SEG_seg = STK.GetImageFromArray(split_seg.astype(np.uint8))
                                print(split_ct.shape)
                                if not os.path.exists(os.path.join(path, dir1, 'CT_Reg')):
                                    os.makedirs(os.path.join(path, dir1, 'CT_Reg'))
                                STK.WriteImage(CT_img, os.path.join(path, dir1, 'CT_Reg', str(i).zfill(2) + CT_file))

                                if not os.path.exists(os.path.join(path, dir1, 'MR_Reg')):
                                    os.makedirs(os.path.join(path, dir1, 'MR_Reg'))
                                STK.WriteImage(MR_img, os.path.join(path, dir1, 'MR_Reg', str(i).zfill(2) + MR_file))

                                if not os.path.exists(os.path.join(path, dir1, 'SEG_all')):
                                    os.makedirs(os.path.join(path, dir1, 'SEG_all'))
                                STK.WriteImage(SEG_seg, os.path.join(path, dir1, 'SEG_all', str(i).zfill(2) + SEG_file))


                                i=i+1

                    for z in range(2):
                        for x in range(3):
                            split_ct = CT_array[z * 96:(z + 1) * 96, 96:288,x * 96:(x + 2) * 96 ]
                            split_mr = MR_array[z * 96:(z + 1) * 96, 96:288,x * 96:(x + 2) * 96 ]
                            split_seg = SEG_array[z * 96:(z + 1) * 96, 96:288,x * 96:(x + 2) * 96 ]
                            CT_img = STK.GetImageFromArray(split_ct)
                            MR_img = STK.GetImageFromArray(split_mr)
                            SEG_seg = STK.GetImageFromArray(split_seg.astype(np.uint8))
                            print(split_ct.shape)
                            if not os.path.exists(os.path.join(path, dir1, 'CT_Reg')):
                                os.makedirs(os.path.join(path, dir1, 'CT_Reg'))
                            STK.WriteImage(CT_img, os.path.join(path, dir1, 'CT_Reg', str(i).zfill(2) + CT_file))

                            if not os.path.exists(os.path.join(path, dir1, 'MR_Reg')):
                                os.makedirs(os.path.join(path, dir1, 'MR_Reg'))
                            STK.WriteImage(MR_img, os.path.join(path, dir1, 'MR_Reg', str(i).zfill(2) + MR_file))

                            if not os.path.exists(os.path.join(path, dir1, 'SEG_all')):
                                os.makedirs(os.path.join(path, dir1, 'SEG_all'))
                            STK.WriteImage(SEG_seg,
                                           os.path.join(path, dir1, 'SEG_all', str(i).zfill(2) + SEG_file))

                            i = i + 1



                    for j in range(8):
                        cx = random.randint(0, CT_array.shape[1] - 192)
                        cy = random.randint(0, CT_array.shape[2] - 192)
                        cz = random.randint(0, CT_array.shape[0] - 96)

                        ct_array = CT_array[cz:cz + 96, cx:cx + 192, cy:cy + 192]
                        mr_array = MR_array[cz:cz + 96, cx:cx + 192, cy:cy + 192]
                        seg_array = SEG_array[cz:cz + 96, cx:cx + 192, cy:cy + 192]

                        print(ct_array.shape)

                        CT_img = STK.GetImageFromArray(ct_array)
                        MR_img = STK.GetImageFromArray(mr_array)
                        SEG_seg = STK.GetImageFromArray(seg_array.astype(np.uint8))

                        if not os.path.exists(os.path.join(path, dir1, 'CT_Reg')):
                            os.makedirs(os.path.join(path, dir1, 'CT_Reg'))
                        STK.WriteImage(CT_img, os.path.join(path, dir1, 'CT_Reg', str(i).zfill(2) + CT_file))

                        if not os.path.exists(os.path.join(path, dir1, 'MR_Reg')):
                            os.makedirs(os.path.join(path, dir1, 'MR_Reg'))
                        STK.WriteImage(MR_img, os.path.join(path, dir1, 'MR_Reg', str(i).zfill(2) + MR_file))

                        if not os.path.exists(os.path.join(path, dir1, 'SEG_all')):
                            os.makedirs(os.path.join(path, dir1, 'SEG_all'))
                        STK.WriteImage(SEG_seg, os.path.join(path, dir1, 'SEG_all', str(i).zfill(2) + SEG_file))
                        i=i+1
