
import os
import SimpleITK as sitk
from scipy import ndimage
import random
import elasticdeform
import numpy as np
import cv2


def data_augmentation_scale(image_CT, image_MR, image_SEG, scale_range=(0.8, 1.2)):
    # 生成随机缩放因子
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])

    height, width, channels = image_CT.shape

    # 根据缩放因子调整图像大小
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # 使用OpenCV进行图像缩放
    image_CT = cv2.resize(image_CT, (new_width, new_height))
    image_MR = cv2.resize(image_MR, (new_width, new_height))
    image_SEG = cv2.resize(image_SEG, (new_width, new_height),interpolation=cv2.INTER_NEAREST)



    # 如果缩放后的图像大小小于输入图像的大小，则在周围填充0
    if new_height < height or new_width < width:
        pad_height = max(0, height - new_height)
        pad_width = max(0, width - new_width)
        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad
        image_CT = cv2.copyMakeBorder(image_CT, top_pad, bottom_pad, left_pad, right_pad,
                                             cv2.BORDER_CONSTANT, value=0)
        image_MR = cv2.copyMakeBorder(image_MR, top_pad, bottom_pad, left_pad, right_pad,
                                      cv2.BORDER_CONSTANT, value=0)
        image_SEG = cv2.copyMakeBorder(image_SEG, top_pad, bottom_pad, left_pad, right_pad,
                                      cv2.INTER_NEAREST, value=0)



    # 如果缩放后的图像大小大于输入图像的大小，则裁剪图像
    if new_height > height or new_width > width:
        start_y = (new_height - height) // 2
        start_x = (new_width - width) // 2
        image_CT = image_CT[start_y:start_y + height, start_x:start_x + width]
        image_MR = image_MR[start_y:start_y + height, start_x:start_x + width]
        image_SEG = image_SEG[start_y:start_y + height, start_x:start_x + width]



    return image_CT, image_MR, image_SEG


def data_augmentation_rotate(image_CT, image_MR, image_SEG, rotate_range=(-30, 30)):
    angle = random.randint(rotate_range[0], rotate_range[1])
    image_CT = ndimage.rotate(image_CT, angle, axes=(1, 2), reshape=False)
    image_MR = ndimage.rotate(image_MR, angle, axes=(1, 2), reshape=False)
    image_SEG = ndimage.rotate(image_SEG, angle, axes=(1, 2), reshape=False)
    # 输出旋转后的数组形状

    return image_CT, image_MR, image_SEG

def data_augmentation_range(image_CT, image_MR, image_SEG,  translate_range=(-20, 20)):

    shift = random.randint(translate_range[0], translate_range[1])
    shift1 = random.randint(translate_range[0], translate_range[1])
    # 平移

    channels, height, width = image_CT.shape

    # 创建一个与输入图像相同的零矩阵
    new_image_CT = np.zeros_like(image_CT)

    # 在第1和2维度上进行平移操作
    new_image_CT[:, max(0, shift1):min(height, height + shift1), max(0, shift):min(width, width + shift)] = image_CT[:,
                                                                                                      max(0,
                                                                                                          -shift1):min(
                                                                                                          height,
                                                                                                          height - shift1),
                                                                                                      max(0,
                                                                                                          -shift):min(
                                                                                                          width,
                                                                                                          width - shift)]
    new_image_MR = np.zeros_like(image_MR)

    # 在第1和2维度上进行平移操作
    new_image_MR[:, max(0, shift1):min(height, height + shift1), max(0, shift):min(width, width + shift)] = image_MR[:,
                                                                                                            max(0,
                                                                                                                -shift1):min(
                                                                                                                height,
                                                                                                                height - shift1),
                                                                                                            max(0,
                                                                                                                -shift):min(
                                                                                                                width,
                                                                                                                width - shift)]
    new_image_SEG = np.zeros_like(image_SEG)

    # 在第1和2维度上进行平移操作
    new_image_SEG[:, max(0, shift1):min(height, height + shift1), max(0, shift):min(width, width + shift)] = image_SEG[:,
                                                                                                            max(0,
                                                                                                                -shift1):min(
                                                                                                                height,
                                                                                                                height - shift1),
                                                                                                            max(0,
                                                                                                                -shift):min(
                                                                                                                width,
                                                                                                                width - shift)]

    return new_image_CT ,new_image_MR,new_image_SEG


def data_gamma(image_CT, image_MR, image_SEG,gamma_range=(2, 0.8)):
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])

    image_CT = 255 * (image_CT / 255) ** gamma
    image_MR = 255 * (image_MR / 255) ** gamma
    return image_CT, image_MR, image_SEG
# def data_Gaussian(image_CT, image_MR, image_SEG,image_SEG1,image_SEG2, sigma_range=(0.02, 0.05)):
#     sigma = np.random.uniform(sigma_range[0], sigma_range[1])
#     print(sigma)
#     image_CT = image_CT + np.random.normal(loc=0, scale=sigma, size=image_CT.shape)
#     image_MR = image_MR + np.random.normal(loc=0, scale=sigma, size=image_MR.shape)
#     return image_CT, image_MR, image_SEG,image_SEG1,image_SEG2





#
# files_A = '../set_2/data_train/all/CT_Reg/00case_01_IMG_CT.nrrd'
# files_B = '../set_2/data_train/all/MR_Reg/00case_01_IMG_MR_T1.nrrd'
# files_C = '../set_2/data_train/all/SEG_1/00case_01_seg.nrrd'
# files_D = '../set_2/data_train/all/SEG_5/00case_01_seg.nrrd'
# files_E = '../set_2/data_train/all/SEG_all/00case_01_seg.nrrd'
# ct = sitk.ReadImage(files_A)
# mr = sitk.ReadImage(files_B)
# seg = sitk.ReadImage(files_C)
# seg1 = sitk.ReadImage(files_D)
# seg5 = sitk.ReadImage(files_E)
#
#
# ct_array = sitk.GetArrayFromImage(ct)
# mr_array = sitk.GetArrayFromImage(mr)
# seg_array = sitk.GetArrayFromImage(seg)
# seg1_array = sitk.GetArrayFromImage(seg1)
# seg5_array = sitk.GetArrayFromImage(seg5)
# seed = random.uniform(0, 1)
#
# [ct_array, mr_array, seg_array, seg1_array, seg5_array] = elasticdeform.deform_random_grid(
#     [ct_array, mr_array, seg_array, seg1_array, seg5_array],
#     sigma=3 * seed,
#     order=[0, 0, 0, 0, 0])
# #
# ct_array, mr_array, seg_array, seg1_array, seg5_array = data_augmentation_scale(ct_array, mr_array, seg_array,
#                                                                                 seg1_array, seg5_array)
#
# ct_array, mr_array, seg_array, seg1_array, seg5_array = data_augmentation_rotate(ct_array, mr_array, seg_array,
#                                                                                  seg1_array, seg5_array)
#
# ct_array, mr_array, seg_array, seg1_array, seg5_array = data_augmentation_range(ct_array, mr_array, seg_array,
#                                                                                 seg1_array, seg5_array)
#
# ct_array, mr_array, seg_array, seg1_array, seg5_array = data_gamma(ct_array, mr_array, seg_array, seg1_array,
#                                                                    seg5_array)
#
# # ct_array, mr_array, seg_array, seg1_array, seg5_array = data_Gaussian(ct_array, mr_array, seg_array, seg1_array,
# #                                                                       seg5_array)
#
# image = sitk.GetImageFromArray(ct_array)
# image_c = sitk.GetImageFromArray(seg_array)
# image_d = sitk.GetImageFromArray(seg1_array)
# image_e = sitk.GetImageFromArray(seg5_array)
# # print(padded_arr.shape)
#
# sitk.WriteImage(image, os.path.join('../jjj/aaa.nrrd'))
# sitk.WriteImage(image_c, os.path.join('../jjj/ccc.nrrd'))
# sitk.WriteImage(image_d, os.path.join('../jjj/ddd.nrrd'))
# sitk.WriteImage(image_e, os.path.join('../jjj/eee.nrrd'))
#
