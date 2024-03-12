import SimpleITK as STK
import numpy as np
import os
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
def img_resmaple(data, new_spacing):
    # data = STK.ReadImage(path)
    image = STK.GetArrayFromImage(data)
    # 获取图像
    print(image.shape)
    original_spacing = data.GetSpacing()
    print(original_spacing)
    original_size = data.GetSize()
    # print(data.GetOrigin(), data.GetDirection())
    new_shape = [
        int(np.round(original_spacing[0] * original_size[0] / new_spacing[0])),
        int(np.round(original_spacing[1] * original_size[1] / new_spacing[1])),
        int(np.round(original_spacing[2] * original_size[2] / new_spacing[2])),
    ]
    resmaple = STK.ResampleImageFilter()
    resmaple.SetInterpolator(STK.sitkLinear)
    resmaple.SetDefaultPixelValue(0)
    resmaple.SetOutputSpacing(new_spacing)
    resmaple.SetOutputOrigin(data.GetOrigin())
    resmaple.SetOutputDirection(data.GetDirection())
    resmaple.SetSize(new_shape)
    data = resmaple.Execute(data)
    # print(image.shape)
    return data

def slf(array):
    hu = np.array([-200, 200, 2000])
    nor = np.array([0, 0.6, 1.0])

    result = np.zeros(array.shape)

    result[array < hu[0]] = 0
    result[array > hu[2]] = 1

    mask1 = (hu[0] <= array) & (array <= hu[1])
    result[mask1] = ((array[mask1] - hu[0]) / (hu[1] - hu[0]))*0.6 + nor[0]

    mask2 = (hu[1] <= array) & (array <= hu[2])
    result[mask2] = ((array[mask2] - hu[1]) / (hu[2] - hu[1]))*0.4 + nor[1]
    # mask3 = (hu[2] <= array) & (array <= hu[3])
    # result[mask3] = ((array[mask3] - hu[2]) / (hu[3] - hu[2])) * 0.2 + nor[2]

    return result

root_dir='../set_1/origin'
mode='CT'
space=[0.7,0.7, 2.0]
size=[160, 600, 600]
for case in (os.listdir(root_dir)):
    for nrrd_file in (os.listdir(os.path.join(root_dir,case))):
        if mode in nrrd_file:
            ct_path = os.path.join(root_dir,case,nrrd_file)
            print('测试文件名为：', ct_path)
            data = STK.ReadImage(ct_path)
            spacing= data.GetSpacing()
            origin=data.GetOrigin()
            direction=data.GetDirection()
            array=STK.GetArrayFromImage(data)
            nor_array=slf(array)
            nor_img = STK.GetImageFromArray(nor_array)
            nor_img.SetSpacing(spacing)  # 设置像素间距
            nor_img.SetOrigin(origin)  # 设置原点
            nor_img.SetDirection(direction)  # 设置方向
            Resample_img=img_resmaple(nor_img,space)
            array = STK.GetArrayFromImage(Resample_img)
            if array.shape[0]>192:
                array=array[array.shape[0]-192:array.shape[0],:,:]
            elif array.shape[0] < 192:
                padding_back = np.zeros((192 - array.shape[0], array.shape[1], array.shape[1]))
                array = np.vstack((padding_back, array))

            image = centerCrop(array, size)
            image = STK.GetImageFromArray(image)

            saveroot=os.path.join(root_dir+'_nor',mode)
            if not os.path.exists(saveroot):
                os.makedirs(saveroot)
            STK.WriteImage(image,os.path.join(saveroot,nrrd_file) )
