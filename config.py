import argparse
parser = argparse.ArgumentParser(description='Hyper-parameters management')
# Hardware options
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--gpu_id', type=list,default=[0,1], help='use cpu only')
parser.add_argument('--seed', type=int, default=1013, help='random seed')
# Preprocess parameters
parser.add_argument('--n_labels', type=int, default=31,help='number of classes') # 分割肝脏则置为2（二类分割），分割肝脏和肿瘤则置为3（三类分割）

parser.add_argument('--continue_train',  default=False,help='number of classes')

# data in/out and dataset
parser.add_argument('--dataset_path',default = 'set_2/data_train1/all',help='fixed trainset root path')
parser.add_argument('--val_path',default = 'set_2/origin_nor_val',help='fixed trainset root path')
parser.add_argument('--test_path',default = 'test',help='fixed trainset root path')
parser.add_argument('--test_data_path',default = 'L30',help='Testset path')
parser.add_argument('--save',default='sep1',help='save path of trained model')
parser.add_argument('--batch_size', type=list, default=2,help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=150, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',help='learning rate (default: 0.0001)')




args = parser.parse_args()


