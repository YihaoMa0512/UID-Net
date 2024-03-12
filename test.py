from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from models.Unet_Separate_3 import Unet_Separate
import ast
import os
from dataset.dataset_lits_test import Test_Dataset
import numpy as np
import SimpleITK as STK
import config_other




if __name__ == '__main__':
    args = config_other.args

    device = torch.device('cuda:0')
    # model info
    model = Unet_Separate(inc=1, n_classes=19).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0])  # multi-GPU
    ckpt = torch.load('experiments/sep2/latest_model.pth')
    model.load_state_dict(ckpt['net'])

    # test_log = logger.Test_Logger(save_path, "test_log")
    # data info


    datasets = Test_Dataset(args)
    dataloader = DataLoader(dataset=datasets, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    window_size = (64, 128, 128)

    with torch.no_grad():
        for ct_data, mr_data,file_name,spacing, origin,direction,bfcenterCrop_shape,bf192_shape,origin_shape in tqdm(dataloader, total=len(dataloader)):
            input = []
            test_results = []

            bfcenterCrop_shape=[int(ast.literal_eval(bfcenterCrop_shape[0])[0]),
                            int(ast.literal_eval(bfcenterCrop_shape[0])[1]),
                            int(ast.literal_eval(bfcenterCrop_shape[0])[2])]
            bf192_shape = [int(ast.literal_eval(bf192_shape[0])[0]),
                                  int(ast.literal_eval(bf192_shape[0])[1]),
                                  int(ast.literal_eval(bf192_shape[0])[2])]
            origin_shape = [ int(ast.literal_eval(origin_shape[0])[2]),
                            int(ast.literal_eval(origin_shape[0])[1]),
                           int(ast.literal_eval(origin_shape[0])[0])]
            spacing = np.array(ast.literal_eval(spacing[0]))
            origin = np.array(ast.literal_eval(origin[0]))
            direction = np.array(ast.literal_eval(direction[0]))
            num_crops = ct_data.size(2) // window_size[0]
            for i in range(num_crops):
                # 使用unfold函数进行滑动窗口裁剪
                unfolded_tensor_ct = ct_data[:, :, i * window_size[0]: (i + 1) * window_size[0], :, :]
                unfolded_tensor_mr = mr_data[:, :, i * window_size[0]: (i + 1) * window_size[0], :, :]
                input.append((unfolded_tensor_ct,unfolded_tensor_mr))

            # 对于最后一个滑动窗口，进行重叠裁剪和测试
            if ct_data.size(2) % window_size[0] != 0:
                last_unfolded_tensor_ct = ct_data[:, :, ct_data.size(2) - window_size[0]:  ct_data.size(2), :, :]

                last_unfolded_tensor_mr = mr_data[:, :, mr_data.size(2) - window_size[0]:  mr_data.size(2), :, :]
                input.append((last_unfolded_tensor_ct,last_unfolded_tensor_mr))
            for ct,mr in input:
                ct, mr= ct.to(device), mr.to(device)
                output = model(ct, mr)
                output = torch.argmax(output, dim=1)
                test_results.append(output)
            if ct_data.size(2) % window_size[0] != 0:
                last_tensor = test_results[num_crops][:,
                              window_size[0] - ct_data.size(2) % window_size[0]:  window_size[0], :, :]
                # 将测试结果拼接回原始形状
                output_tensor = torch.cat((test_results[0:num_crops]), dim=1)
                output_tensor = torch.cat((output_tensor, last_tensor), dim=1)
            else:
                output_tensor = torch.cat(test_results, dim=1)
            output_label = output_tensor.cpu().numpy().astype(np.uint8).squeeze(0)



            padding = tuple((max(bfcenterCrop_shape) - s) // 2 for s in output_label.shape)
            # if bfcenterCrop_shape[1]%2==0:
            # 使用 pad 函数进行填充
            padded_array = np.pad(output_label, ((0, 0), (padding[1], padding[1]), (padding[2], padding[2])),
                                  mode='constant')
            # else:
            #     padded_array = np.pad(output_label, ((0, 0), (padding[1], padding[1]+1), (padding[2], padding[2]+1)),
            #                           mode='constant')

            if bfcenterCrop_shape[0]<bf192_shape[0]:

                padding_back = np.zeros((bf192_shape[0]-bfcenterCrop_shape[0], padded_array.shape[1], padded_array.shape[2]))
                padded_array = np.vstack(( padding_back,padded_array))


            label = STK.GetImageFromArray(padded_array.astype(np.uint8))
            label.SetSpacing([1.4,1.4,2.0])  # 设置像素间距
            label.SetOrigin(origin)  # 设置原点
            label.SetDirection(direction)
            resmaple = STK.ResampleImageFilter()
            resmaple.SetInterpolator(STK.sitkNearestNeighbor)
            resmaple.SetOutputSpacing(spacing)
            resmaple.SetSize(origin_shape)
            resmaple.SetDefaultPixelValue(0)
            resmaple.SetOutputOrigin(origin)
            resmaple.SetOutputDirection(direction)
            label = resmaple.Execute(label)


            STK.WriteImage(label, os.path.join('results1', file_name[0]))

