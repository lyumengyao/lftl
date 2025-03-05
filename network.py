import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
import torch.optim as optim
import os

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

res_dict = {
    "resnet18": models.resnet18, 
    "resnet34": models.resnet34, 
    "resnet50": models.resnet50, 
    "resnet101": models.resnet101
}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class FeatureNeck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(FeatureNeck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class ClassifierHead(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(ClassifierHead, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        elif type == 'linear':
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        if not self.type in {'wn', 'linear'}:
            w = self.fc.weight
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            
            x = torch.nn.functional.normalize(x, dim=1, p=2)
            x = torch.nn.functional.linear(x, w)
        else:
            x = self.fc(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y

class ActiveBaseModel(nn.Module):
    def __init__(self, args):
        super(ActiveBaseModel, self).__init__()
        if args.net[0:3] == 'res':
            netF = ResBase(res_name=args.net)
        else:
            raise NotImplementedError("Network not supported: {}".format(args.net))

        netB = FeatureNeck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)
        netC = ClassifierHead(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck)

        self.args = args
        self.netF = netF
        self.netB = netB
        self.netC = netC
        self.emb_dim = args.bottleneck

        self.freeze_head = args.freeze_head == 1

    def get_embedding_dim(self):
        return self.emb_dim

    def train(self):
        self.netF.train()
        self.netB.train()
        if not self.freeze_head:
            self.netC.train()
        else:
            self.netC.eval()
            
    def eval(self):
        self.netF.eval()
        self.netB.eval()
        self.netC.eval()

    def load_from_file(self, args):
        print("--------Load model from %s--------" % args.source_dir)
        
        modelpath = os.path.join(args.source_dir, 'source_F.pt')
        self.netF.load_state_dict(torch.load(modelpath))
        
        modelpath = os.path.join(args.source_dir, 'source_B.pt')
        self.netB.load_state_dict(torch.load(modelpath))
        
        modelpath = os.path.join(args.source_dir, 'source_C.pt')
        self.netC.load_state_dict(torch.load(modelpath))

    def save(self, round, args):
        log_str = "save model in " + args.output_dir
        args.out_file.write(log_str+"\n")
        args.out_file.flush()
        print(log_str+"\n")
        torch.save(self.netF.state_dict(), os.path.join(args.output_dir, f"target_F_round_{round}.pt"))
        torch.save(self.netB.state_dict(), os.path.join(args.output_dir, f"target_B_round_{round}.pt"))
        torch.save(self.netC.state_dict(), os.path.join(args.output_dir, f"target_C_round_{round}.pt"))
    
    def named_parameters(self, args):
        param_group = []
        for _, v in self.netF.named_parameters():
            if args.lr_decay1 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
            else:
                v.requires_grad = False
                
        for _, v in self.netB.named_parameters():
            if args.lr_decay2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
            else:
                v.requires_grad = False

        if self.freeze_head:
            self.netC.eval()
            for _, v in self.netC.named_parameters():
                v.requires_grad = False
        else:
            for k, v in self.netC.named_parameters():
                if args.lr_decay2 > 0:
                    param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
                else:
                    v.requires_grad = False

        return param_group

    def forward(self, input):
        features_F = self.netF(input)
        features_B = self.netB(features_F)
        output = self.netC(features_B)
        return output, features_B
