import os
import random
import SimpleITK as sitk
import numpy as np

random.seed(2)
import matplotlib.pyplot as plt
import torch
import numpy as np
def centerCrop(image, output_size):
    # if image.shape[0] <= output_size[0] or image.shape[1] <= output_size[1] or image.shape[2] <= output_size[2]:
    #     pw = max((output_size[0] - image.shape[0]) // 2 + 3, 0)
    #     ph = max((output_size[1] - image.shape[1]) // 2 + 3, 0)
    #     pd = max((output_size[2] - image.shape[2]) // 2 + 3, 0)
    #     image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
    #     # label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    (w, h, d) = image.shape

    w1 = int(round((w - output_size[0]) / 2.))
    h1 = int(round((h - output_size[1]) / 2.))
    d1 = int(round((d - output_size[2]) / 2.))

    # print(image.shape, output_size, get_center(label), w1, h1, d1)
    image = image[ : , h1:h1 + output_size[1], d1:d1 + output_size[2]]
    # label = label[:, h1:h1 + output_size[1],d1:d1 + output_size[2]]

    return image
#
# def slide_and_sum(a, b):
#     a=a.copy()
#     b=b.copy()
#     a[a==0]=1000
#     b[b==0] = 1000
#     x, n, m = a.shape
#     y = b.shape[0]
#     # 计算可能的滑动次数
#     max_slides = x - y + 1
#     # 初始化最小差值绝对值和和滑动次数
#     min_sum = float('inf')
#     min_slides = 0
#     # 遍历所有可能的滑动位置
#     for i in range(max_slides):
#         # 根据滑动位置，截取 a 中与 b 大小相同的子数组
#         sub_a = a[i:i + y, :, :]
#         # 计算差值绝对值的和
#         diff_sum = np.sum(np.abs(sub_a - b))
#         # 更新最小差值绝对值和和滑动次数
#         if diff_sum < min_sum:
#             min_sum = diff_sum
#             min_slides = i
#     return  min_slides
size=[160, 384, 384]
def slide_and_sum(a, b, n=310):
    a = a.copy()
    b = b.copy()
    a[a == 0] = 1000
    b[b == 0] = 1000
    x,m = a.shape[0],b.shape[0]
    # 计算可能的滑动次数
    max_slides = x-m
    # 初始化最小差值绝对值和和滑动次数
    min_sum = float('inf')
    min_slides = 0
    # 遍历所有可能的滑动位置
    for i in range(max_slides):
        # 根据滑动位置，截取 a 和 b 中对应的[:,n,:]层
        sub_a = a[i:i+m, n, :]
        sub_b = b[:, n, :]
        # plt.imshow(sub_b, cmap='gray')
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()
        diff_sum = np.sum(np.abs(sub_a - sub_b))
        # 更新最小差值绝对值和和滑动次数
        if diff_sum < min_sum  :
            min_sum = diff_sum
            min_slides = i
    return min_slides

path = '../set_1/origin_nor'
for MR_file in (os.listdir(os.path.join(path, 'MR'))):
    for CT_file in (os.listdir(os.path.join(path, 'CT'))):
        if CT_file[0:7] in MR_file[0:7]:
            MR_img = sitk.ReadImage(os.path.join(path, 'MR', MR_file))
            MR_img_array=sitk.GetArrayFromImage(MR_img)
            CT_img = sitk.ReadImage(os.path.join(path, 'CT', CT_file))
            CT_img_array = sitk.GetArrayFromImage(CT_img)

            S=slide_and_sum(CT_img_array,MR_img_array)

            n = S  # 在前面补 5 个零
            print(CT_file,n)
            m = CT_img_array.shape[0]-n-MR_img_array.shape[0]  # 在后面补 3 个零

            padding_front = np.zeros((n, MR_img_array.shape[1], MR_img_array.shape[2]))
            padding_back = np.zeros((m, MR_img_array.shape[1], MR_img_array.shape[2]))
            padded_arr = np.vstack((padding_front, MR_img_array,padding_back))
            mr_array = centerCrop(padded_arr, size)

            mr_image=sitk.GetImageFromArray(mr_array)
            # print(padded_arr.shape)
            if not os.path.exists(os.path.join(path,'MR_Reg')):
                os.makedirs(os.path.join(path,'MR_Reg'))
            sitk.WriteImage(mr_image, os.path.join(path,'MR_Reg',MR_file))



            ct_array = centerCrop(CT_img_array, size)

            ct_image = sitk.GetImageFromArray(ct_array)
            # print(padded_arr.shape)

            if not os.path.exists(os.path.join(path, 'CT_Reg')):
                os.makedirs(os.path.join(path, 'CT_Reg'))
            sitk.WriteImage(ct_image, os.path.join(path, 'CT_Reg', CT_file))