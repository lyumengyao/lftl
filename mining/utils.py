import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X
    
def euclidean_metric(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_metric(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return cosine

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def print_log(log_str, args):
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)

def cal_acc(preds, labels):
    accuracy = torch.sum(preds == labels).item() / float(labels.size(0))
    # matrix = confusion_matrix(labels, preds)
    # acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    # aacc = acc.mean()
    # aa = [str(np.round(i, 2)) for i in acc]
    # acc = ' '.join(aa)
    # return aacc, acc, accuracy*100
    return accuracy*100

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cross_entropy_loss(inputs, targets, mask = None, reduction=False, use_gpu=True, label_smooth_epsilon=0.0):
    """
    Args:
        inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        targets: ground truth labels with shape (num_classes)
    """
    
    log_probs = nn.LogSoftmax(dim=1)(inputs)
    targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
    if use_gpu: 
        targets = targets.cuda()
    
    if label_smooth_epsilon > 0:
        num_classes = log_probs.size(1)
        targets = (1 - label_smooth_epsilon) * targets + label_smooth_epsilon / num_classes
    
    loss = (- targets * log_probs).sum(dim=1)
    if mask is not None:
        loss = loss * mask
    if reduction:
        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0.0).cuda()
            return loss.sum() / mask.sum()
        else:
            return loss.mean()
    else:
        return loss

def info_maximize_loss(inputs):
    softmax_out = nn.Softmax(dim=1)(inputs)
    entropy_loss = torch.mean(Entropy(softmax_out))
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    maxinfo_loss = entropy_loss - gentropy_loss
    return maxinfo_loss