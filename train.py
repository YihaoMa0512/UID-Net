
import random

from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset_other.dataset_lits_val import Val_Dataset
from dataset_other.dataset_lits_train import Train_Dataset
from lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config_1
from loss import AttentionExpDiceLoss, TestDiceLoss, FocalLoss, Focal_and_Dice_loss
from models.SegNet import mr_Unet_Separate

import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(3407)


def split_tensor_x(tensor):
    split_tensors = []
    for z in range(2):
        for x in range(3):
            split_tensor = tensor[:, :, z * 96:(z + 1) * 96, 96:288, x * 96:(x + 2) * 96]
            split_tensors.append(split_tensor)
    return split_tensors


def split_tensor_y(tensor):
    split_tensors = []
    for z in range(2):
        for y in range(3):
            split_tensor = tensor[:, :, z * 96:(z + 1) * 96, y * 96:(y + 2) * 96, 96:288]
            split_tensors.append(split_tensor)
    return split_tensors


def merge_tensors_x(tensors, tensor):
    list_1 = [0, 2, 1]
    for z in range(2):
        for x in list_1:
            tensor[:, :, z * 96:(z + 1) * 96, 96:288, x * 96:(x + 2) * 96] = tensors[z * 3 + x]
    return tensor


def merge_tensors_y(tensors, tensor):
    list_1 = [0, 2, 1]
    for z in range(2):
        for y in list_1:
            tensor[:, :, z * 96:(z + 1) * 96, y * 96:(y + 2) * 96, 96:288] = tensors[z * 3 + y]
    return tensor


def val(model, val_loader, num_class):
    num = 0
    model.eval()

    sum_dice2 = np.zeros([num_class])
    with torch.no_grad():
        for idx, (ct_data, mr_data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            ct_data, mr_data, target = ct_data.to(device), mr_data.to(device), target.to(device)

            ct_split_tensors_x = split_tensor_x(ct_data)
            mr_split_tensors_x = split_tensor_x(mr_data)
            ct_split_tensors_y = split_tensor_y(ct_data)
            mr_split_tensors_y = split_tensor_y(mr_data)
            output_tensors = []
            for t, m in zip(ct_split_tensors_x, mr_split_tensors_x):
                _, _, _, output1 = model(t, m)  # 将小tensor送入网络
                output_tensor = torch.argmax(output1, dim=1)  # 取得网络输出并使用torch.argmax处理
                output_tensors.append(output_tensor)
            tensor = torch.zeros([1, 1, 192, 384, 384], dtype=torch.float32).to(device)
            merged_tensor = merge_tensors_x(output_tensors, tensor)
            output_tensors = []
            for t, m in zip(ct_split_tensors_y, mr_split_tensors_y):
                _, _, _, output1 = model(t, m)  # 将小tensor送入网络
                output_tensor = torch.argmax(output1, dim=1)  # 取得网络输出并使用torch.argmax处理
                output_tensors.append(output_tensor)
            merged_tensor = merge_tensors_y(output_tensors, merged_tensor)

            val_dice2 = dice_eval(merged_tensor, target)
            val_dice2 = val_dice2.cpu().data.numpy()
            print('val dice:', ["{:.4f}".format(num) for num in val_dice2])
            sum_dice2 = (sum_dice2 + np.array(val_dice2))
            num = num + 1

        return np.sum(sum_dice2) / num


if __name__ == '__main__':
    args = config_1.args
    # seed_torch(args.seed)
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')

    # model info
    model = mr_Unet_Separate(inc=2, n_classes=args.n_labels, base_chns=6).to(device)

    # model = SwinUNETR(
    #     img_size=(96, 96, 96),
    #     in_channels=1,
    #     out_channels=args.n_labels,
    #     feature_size=24,
    #     use_checkpoint=True,
    # ).to(device)
    # model = torch.nn.DataParallel(model, device_ids=args.gpu_id).cuda()  # multi-GPU
    num_workers = 10
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=args.batch_size, num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, num_workers=num_workers, shuffle=False)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=150)

    # warmup_epochs = 20  # 设定预热 epoch 数
    # max_epochs = 200  # 设定最大 epoch 数
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs, max_epochs,warmup_start_lr=1e-5)

    # common.print_network(model)
    start_epoch = 1
    if args.continue_train:
        model_checkpoint = torch.load('experiments/sep2_2/latest_model.pth')  # 加载断点
        model.load_state_dict(model_checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(model_checkpoint['optimizer'])  # 加载优化器参数
        lr_scheduler.load_state_dict(model_checkpoint['lr_schedule'])  # 加载lr_scheduler
        start_epoch = model_checkpoint['epoch']
    ExpDiceLoss = AttentionExpDiceLoss(args.n_labels, 1).cuda()

    dice_eval = TestDiceLoss(n_class=args.n_labels)

    best = 0  # 初始化最优模型的epoch和performance
    alpha = 1  # 深监督衰减系数初始值

    train_loss = []
    train_acc = []
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, args.epochs + 1):

        print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.param_groups[0]['lr']
                                                   ))
        model.train()
        for idx, (ct_data, mr_data, target1, target2, target3, target4) in tqdm(enumerate(train_loader),
                                                                                total=len(train_loader)):
            ct_data, mr_data, target1, target2, target3, target4 = ct_data.to(device), mr_data.to(device), target1.to(device), target2.to(device), target3.to(device), target4.to(device)

            optimizer.zero_grad()
            output4, output3, output2, output1 = model(ct_data, mr_data)
            dcs_loss1 = ExpDiceLoss(output1, target1)
            dcs_loss2 = ExpDiceLoss(output2, target2)
            dcs_loss3 = ExpDiceLoss(output3, target3)
            dcs_loss4 = ExpDiceLoss(output4, target4)
            dcs_loss = dcs_loss1 + (dcs_loss2 + dcs_loss3 + dcs_loss4) * alpha
            dcs_loss.backward()

            loss = dcs_loss.cpu().data.numpy()
            loss1 = dcs_loss1.cpu().data.numpy()
            loss2 = dcs_loss2.cpu().data.numpy()
            loss3 = dcs_loss3.cpu().data.numpy()
            loss4 = dcs_loss4.cpu().data.numpy()

            train_loss.append(dcs_loss.item())

            optimizer.step()

            train_dice1 = dice_eval(torch.argmax(output1, dim=1), target1)
            train_dice1 = train_dice1.cpu().data.numpy()
            print('train dice2:', ["{:.4f}".format(num) for num in train_dice1])

            print('loss:', loss, '||', 'loss1:', loss1, '||', 'loss2:', loss2, '||', 'loss3:', loss3, '||', 'loss4:',
                  loss4)

            print('lr', optimizer.param_groups[0]['lr'])
        lr_scheduler.step()

        q = val(model, val_loader, args.n_labels)
        train_acc.append(q / args.n_labels)

        with open(os.path.join(save_path, "train_loss.txt"), 'w') as train_los:
            train_los.write(str(train_loss))

        with open(os.path.join(save_path, "train_acc.txt"), 'w') as train_ac:
            train_ac.write(str(train_acc))

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'lr_schedule': lr_scheduler.state_dict()}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        if epoch > 100:
            torch.save(state, os.path.join(save_path, str(epoch) + '_model.pth'))
        if q > best:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best = q
            best_epoch = epoch
            print('Best performance at Epoch: {} | {}'.format(best_epoch, best / args.n_labels))
        print(q / args.n_labels)

        if epoch > 100:
            alpha = 0
