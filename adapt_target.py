import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

from dataset import get_dataset, ImageList
from network import ActiveBaseModel

from mining.utils import print_log
from mining.contrastive_sampling import CASampling

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def image_train(resize_size=256, crop_size=224, alexnet=False):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def build_strategy(args):

    t_dset_root = os.path.join(args.dset_root, args.t_folder)
    train_dataset = get_dataset(args.t_dset_file, t_dset_root)
    source_dataset = None

    # start experiment
    n_pool = len(train_dataset)
    idxs_lb = np.zeros(n_pool, dtype=bool)
    log_str = "Training Pool size: %d" % n_pool
    log_str += "\n------Active from %s------" % args.init_pool
    print_log(log_str, args)

    net = ActiveBaseModel(args)
    if args.init_pool == "source":
        net.load_from_file(args)
    else:
        raise NotImplementedError

    handler = ImageList
    train_transform = image_train()
    test_transform = image_test()

    strategy = CASampling(train_dataset, idxs_lb, net, handler, train_transform, test_transform, args, source_dataset)
    return strategy

def train_target(args, strategy):
    all_res = {}
    all_query_res = {}
    query_ratio = args.ratio_per_round
    num_round = args.num_round

    t_dset_root = os.path.join(args.dset_root, args.t_folder)
    test_dataset = get_dataset(args.t_dset_file, t_dset_root)
    log_str = 'Number of testing pool: {}'.format(len(test_dataset))
    print_log(log_str, args)

    acc = strategy.predict(test_dataset, round=0)
    log_str = 'Task: {}, Round: 0/{}, Accuracy = {:.2f}%'.format(args.name, num_round, acc)
    print_log(log_str, args)
    all_res[-1] = acc

    for i in range(num_round):
        # query
        if i == num_round - 1:
            query_n = round(len(test_dataset) * query_ratio * num_round) - sum(strategy.idxs_lb)
        else:
            query_n = round(len(test_dataset) * query_ratio)
        q_idxs = strategy.query(query_n)
        all_query_res[f"q_idxs_{i}"] = q_idxs

        strategy.update(q_idxs)
        strategy.train_one_round(i+1)
        
        # round accuracy
        acc = strategy.predict(test_dataset, round=i+1)
        log_str = 'Task: {}, Round: {}/{}, Accuracy = {:.2f}%'.format(args.name, i+1, num_round, acc)
        print_log(log_str, args)
        
        if args.issave == 1:
            strategy.save(i, args)
    return all_res, all_query_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='lftl for sfada by lyu etal')

    # data
    parser.add_argument('--dset', type=str, default='VISDA-C', choices=['VISDA-C', 'office', 'office-home'])
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--source_dir', type=str, default="", help="dir for source model")  

    # active learning
    parser.add_argument('--num_round', type=int, default=10)
    parser.add_argument('--ratio_per_round', type=float, default=0.001)
    parser.add_argument('--init_pool', type=str, choices=["random", "source"], default="random", help="how to select the first query set")
    
    # training
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations in each round")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--output', type=str, default="exps_sfada", help="dir for source model") 


    # model
    parser.add_argument('--net', type=str, default='resnet101', help="vgg16, resnet50, resnet101")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    
    parser.add_argument('--issave', type=int, default=0)
    parser.add_argument('--name', type=str, default="", help="model name")

    # SFADA
    parser.add_argument('--cd_ratio', type=float, default=0.03, help="lambda for contrastive decoding")
    parser.add_argument('--topk', type=int, default=200, help="lambda for dual hard")
    parser.add_argument('--lambda_topk', type=float, default=0.0, help="lambda for dual hard")

    parser.add_argument('--sfda_ubl', type=float, default=-1.0, help="amount of unlabeled data")
    parser.add_argument('--beta_im', type=float, default=0.0, help="lambda for im loss")
    parser.add_argument('--beta_vpa', type=float, default=0.0, help="lambda for attended_entropy loss")
    parser.add_argument('--lambda_mixup', type=float, default=1.0, help="lambda for mixup loss")
    parser.add_argument('--mem_momentum', type=float, default=0.9, help="momentum for memory update")
    parser.add_argument('--mixup_alpha', type=float, default=0.75, help="parameters for beta distribution")

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args.dset_root = "data/%s" % args.dset
    args.source_dir = args.source_dir if args.source_dir else "ckps/source/uda/%s" % args.dset
    if args.dset == "VISDA-C":
        names = ['train', 'validation']
        args.s_dset_file = "data/VISDA-C/train_list.txt"
        args.t_dset_file = "data/VISDA-C/validation_list.txt"
        args.s_folder = "train"
        args.t_folder = "validation"
        args.class_num = 12
        
    elif args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.s_dset_file = "./data/office-home/%s_alsfda.txt" % names[args.s]
        args.t_dset_file = "./data/office-home/%s_alsfda.txt" % names[args.t]
        args.s_folder = names[args.s]
        args.t_folder = names[args.t]
        args.source_dir = "%s/%s" % (args.source_dir, names[args.s][0])
        args.class_num = 65
        
    elif args.dset == "office":
        #dataset size: amazon: 2817, dslr: 498, webcam: 795
        names = ['amazon', 'dslr', 'webcam']
        args.s_dset_file = "%s/%s_alsfda.txt" % (args.dset_root, names[args.s])
        args.t_dset_file = "%s/%s_alsfda.txt" % (args.dset_root, names[args.t])
        args.s_folder = names[args.s]
        args.t_folder = names[args.t]
        args.source_dir = "%s/%s" % (args.source_dir, names[args.s][0].upper())
        args.class_num = 31

    args.freeze_head = 0

    args.output_dir = os.path.join(args.output, args.dset, f"b{args.ratio_per_round}x{args.num_round}", names[args.s][0] + names[args.t][0], args.name)
    os.makedirs(args.output_dir, exist_ok=True)
        
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.out_file = os.path.join(args.output_dir, "log_%d.txt" % args.seed)
    log_str = print_args(args)
    args.out_file = open(args.out_file, 'w')
    print_log(log_str, args)

    strategy = build_strategy(args)

    all_res, all_query_res = train_target(args, strategy)

    query_log_file = os.path.join(args.output_dir, "query.pth")
    torch.save(all_query_res, query_log_file)
    