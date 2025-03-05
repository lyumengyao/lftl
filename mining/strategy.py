import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from .utils import lr_scheduler, cal_acc, op_copy, cross_entropy_loss, info_maximize_loss, print_log
from .utils import euclidean_metric, cosine_metric, l1norm, l2norm, Entropy


class MixedDataset(Dataset):
    def __init__(self, input_a, input_b, target_a, target_b, args):
        self.input_a = input_a
        self.input_b = input_b
        self.target_a = target_a
        self.target_b = target_b
        self.args = args
        self.class_num = args.class_num

        assert input_a.size(0) == input_b.size(0) and target_a.size(0) == target_b.size(0)

    def __getitem__(self, index):
        #prepare mixed data
        l = np.random.beta(self.args.mixup_alpha, self.args.mixup_alpha)
        l = max(l, 1 - l)
        x1 = self.input_a[index]
        x2 = self.input_b[index]

        y1 = self.target_a[index].long()
        y2 = self.target_b[index].long()
        y1 = torch.zeros(self.class_num).scatter_(0, y1.cpu(), 1)
        y2 = torch.zeros(self.class_num).scatter_(0, y2.cpu(), 1)
        
        x = l * x1 + (1 - l) * x2
        y = l * y1 + (1 - l) * y2
        return x, y

    def __len__(self):
        return self.input_a.size(0)

class Strategy:
    def __init__(self, train_dataset, idxs_lb, net, handler, train_transform, test_transform, args, source_dataset):
        self.train_dataset = train_dataset
        self.idxs_lb = idxs_lb
        self.ori_net_dict = net.state_dict()
        self.net = net
        self.prev_pred = None
        self.handler = handler #List -> Dataset
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.args = args
        self.source_dataset = source_dataset

        self.n_pool = len(train_dataset)
        self.batch_size = args.batch_size

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.class_num = args.class_num
        self.sfada_ubl = -1
        if hasattr(args, "sfada_ubl"):
            self.sfada_ubl = args.sfada_ubl
        
        self.lbl_memory_embs = torch.zeros([self.n_pool, self.net.get_embedding_dim()])
        self.lbl_memory_probs = torch.zeros([self.n_pool, self.class_num])
        
        self.n_labeled = round(self.n_pool * args.ratio_per_round * args.num_round) 
    
    def print_log(self, log_str):
        self.args.out_file.write(log_str + '\n')
        self.args.out_file.flush()
        print(log_str)

    def save(self, round, args):
        self.net.save(round, args)

    def query(self, n):
        pass

    def update(self, q_idxs):
        self.idxs_lb[q_idxs] = True

    def init_net(self, training=True):
        self.net = self.net.to(self.device)

    def get_ubl_dataset(self):
        idxs_ubl = np.where(self.idxs_lb==0)[0]
        if self.sfada_ubl >= 0:
            idxs_ubl = np.random.choice(idxs_ubl, int(np.sum(self.idxs_lb) * self.sfada_ubl))
        ubl_datast = []
        for i in idxs_ubl:
            item = self.train_dataset[i]
            ubl_datast.append(item)
        
        return ubl_datast

    def train_with_mixup(self, data, optimizer, interval=100):

        input_a = torch.cat(data[0], dim=0)
        target_a = torch.cat(data[2], dim=0)

        rand_idxs = list(range(input_a.size(0)))
        random.shuffle(rand_idxs)
        
        # mix labeled samples
        input_b = input_a[rand_idxs]
        target_b = target_a[rand_idxs]

        mixed_dataset = MixedDataset(input_a, input_b, target_a, target_b, self.args)
        mixed_dataloader = DataLoader(mixed_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.args.worker, drop_last=False)

        for i, (input, target) in enumerate(mixed_dataloader):
            if input.size(0) == 1:
                continue
            mixed_inputs = input.to(self.device)
            mixed_targets = target.to(self.device)

            mixed_outputs, _ = self.net(mixed_inputs)
            mixed_outputs = nn.LogSoftmax(dim=1)(mixed_outputs)
            loss_mixed = (- mixed_targets * mixed_outputs).sum(dim=1).mean() * self.args.lambda_mixup

            optimizer.zero_grad()
            loss_mixed.backward()
            optimizer.step()

            if i % interval == 0:
                log_str = "[MIXUP], Loss: %.4f" % (loss_mixed.item())
                print_log(log_str, self.args)

    def train_one_round_with_unlabel(self, round):
        n_epoch = self.args.max_epoch
        self.init_net(training=True)
        param_group = self.net.named_parameters(self.args)

        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb].tolist()
        lbl_dataset = [self.train_dataset[i] for i in idxs_train]
        train_lbl_dataloader = DataLoader(self.handler(lbl_dataset, transform=self.train_transform),
                            batch_size=self.batch_size, shuffle=True, num_workers=self.args.worker, drop_last=False)
        
        if self.source_dataset is not None:
            train_src_dataloader = DataLoader(self.handler(self.source_dataset, transform=self.train_transform),
                                batch_size=self.batch_size, shuffle=True, num_workers=self.args.worker, drop_last=False)
        
        if self.sfada_ubl >= 0:
            self.sfada_ubl = math.ceil(self.n_labeled * 2 / np.sum(self.idxs_lb)) - 1
            print(f"getting {self.sfada_ubl} unlabeled samples")
            ubl_num = np.sum(self.idxs_lb) * self.sfada_ubl
        else:
            ubl_num = len(self.idxs_lb) - np.sum(self.idxs_lb)
        ubl_batch_num = int(ubl_num / self.batch_size) + int(ubl_num % self.batch_size != 0)
        max_iter = n_epoch * ubl_batch_num

        log_str = "------Train at Round %d with %d Annotated Data and %d Unlabeled Data-----" % (round, len(lbl_dataset), ubl_num)
        print_log(log_str, self.args)
        iter_num = 0
        interval = max_iter // 10
        for epoch in range(n_epoch):
            ubl_dataset = self.get_ubl_dataset()
            train_ubl_dataloader = DataLoader(self.handler(ubl_dataset, transform=self.train_transform),
                            batch_size=self.batch_size, shuffle=True, num_workers=self.args.worker, drop_last=False)
            assert len(train_ubl_dataloader) == ubl_batch_num, "%d!=%d" % (len(train_ubl_dataloader), ubl_batch_num)
            assert len(train_ubl_dataloader) >= len(train_lbl_dataloader)

            # update features in the memory
            momentum = self.args.mem_momentum
            _, lbl_embs = self.get_output(lbl_dataset)
            updated_lbl_embs = (1 - momentum) * self.lbl_memory_embs[idxs_train] + momentum * lbl_embs
            lbl_gnds = torch.tensor([item[1] for item in lbl_dataset])
            updated_lbl_probs = self.lbl_memory_probs[idxs_train]
            updated_lbl_probs = updated_lbl_probs.scatter_(1, lbl_gnds.unsqueeze(1).cpu(), 1)
            self.lbl_memory_embs[idxs_train] = updated_lbl_embs
            self.lbl_memory_probs[idxs_train] = updated_lbl_probs

            mixed_dataset = [[], [], [], []]
            
            for _ in range(len(train_ubl_dataloader)):
                self.net.train()
                try:
                    inputs_ubl, label_ubl, _ = next(iter_ubl)
                except:
                    iter_ubl = iter(train_ubl_dataloader)
                    inputs_ubl, label_ubl, _ = next(iter_ubl)
                
                try:
                    inputs_lbl, label_lbl, idx = next(iter_lbl)
                except:
                    iter_lbl = iter(train_lbl_dataloader)
                    inputs_lbl, label_lbl, idx = next(iter_lbl)

                if inputs_ubl.size(0) == 1 or inputs_lbl.size(0) == 1:
                    continue
                iter_num += 1

                lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
                inputs_ubl = inputs_ubl.to(self.device)
                inputs_lbl = inputs_lbl.to(self.device)

                outputs_ubl, embs_ubl = self.net(inputs_ubl)
                outputs_lbl, embs_lbl = self.net(inputs_lbl)

                loss_lbl = cross_entropy_loss(outputs_lbl, label_lbl, reduction=True, use_gpu=self.use_cuda)
                loss_im = info_maximize_loss(outputs_ubl) 

                self.lbl_memory_embs[idx] = (1 - momentum) * self.lbl_memory_embs[idx] + momentum * embs_lbl.data.cpu()
                updated_lbl_embs = self.lbl_memory_embs[idxs_train]
                updated_lbl_probs = self.lbl_memory_probs[idxs_train]

                attn = cosine_metric(embs_ubl, updated_lbl_embs.to(self.device))
                outputs_ubl_att = F.softmax(attn, dim=1)
                loss_att = torch.mean(Entropy(outputs_ubl_att))

                loss = loss_lbl + loss_im * self.args.beta_im + loss_att * self.args.beta_vpa

                mixed_dataset[0].append(inputs_lbl.float().cpu())
                mixed_dataset[1].append(inputs_ubl.float().cpu())
                mixed_dataset[2].append(label_lbl.data.float().cpu())
                mixed_dataset[3].append(label_ubl.data.float().cpu())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iter_num % interval == 0:
                    log_str = "[Round: %d, Epoch: %d, Iter: %d/%d], Loss: %.4f/%.4f/%.4f" % (round, epoch, iter_num, max_iter, loss_lbl.item(), loss_att.item(), loss_im.item())
                    print_log(log_str, self.args)

            if self.args.lambda_mixup > 0 and len(lbl_dataset) > 1:
                self.train_with_mixup(mixed_dataset, optimizer, interval) #mixup works. 87.03->87.27

    def train_one_round(self, round):
        if self.args.beta_im > 0:
            return self.train_one_round_with_unlabel(round)
        else:
            assert self.args.beta_im == 0 and self.args.beta_vpa == 0 and self.args.lambda_mixup == 0, "im, att and mixup are only for unlabeled data"
            
        n_epoch = self.args.max_epoch
        self.init_net(training=True)
        param_group = self.net.named_parameters(self.args)

        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb].tolist()
        round_dataset = [self.train_dataset[i] for i in idxs_train]
        train_dataloader = DataLoader(self.handler(round_dataset, transform=self.train_transform),
                            batch_size=self.batch_size, shuffle=True, num_workers=self.args.worker, drop_last=False)
        log_str = "------Train at Round %d with %d Annotated Data-----" % (round, len(round_dataset))
        print_log(log_str, self.args)

        max_iter = n_epoch * len(train_dataloader)
        iter_num = 0
        interval = max_iter // 10
        for epoch in range(n_epoch):
            self.net.train()
            for _, (inputs, label, _) in enumerate(train_dataloader):
                if inputs.size(0) == 1:
                    continue
                iter_num += 1
                lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
                inputs = inputs.to(self.device)
                outputs, _ = self.net(inputs)
                loss = cross_entropy_loss(outputs, label, reduction=True, use_gpu=self.use_cuda)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iter_num % interval == 0:
                    log_str = "[Round: %d, Epoch: %d, Iter: %d/%d], Loss: %.4f" % (round, epoch, iter_num, max_iter, loss.item())
                    print_log(log_str, self.args)

    def predict(self, test_dataset, round=-1):
        # test_dataset: list
        test_dataloader = DataLoader(self.handler(test_dataset, transform=self.test_transform),
                            batch_size=self.batch_size*3, shuffle=False, num_workers=self.args.worker, drop_last=False)
        log_str = "------Evaluate at Round %d on %d Data-----" % (round, len(test_dataset))
        print_log(log_str, self.args)
        self.init_net(training=False)
        self.net.eval()
        preds = torch.zeros(len(test_dataset))
        gnds = torch.zeros(len(test_dataset))
        with torch.no_grad():
            for x, y, idxs in test_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.net(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.data.cpu().float()
                gnds[idxs] = y.data.cpu().float()
        return cal_acc(preds, gnds)

    def predict_prob(self, test_dataset):
        # test_dataset: list
        test_dataloader = DataLoader(self.handler(test_dataset, transform=self.test_transform),
                            batch_size=self.batch_size*3, shuffle=False, num_workers=self.args.worker, drop_last=False)
        self.init_net(training=False)
        self.net.eval()
        probs = torch.zeros([len(test_dataset), self.class_num])
        with torch.no_grad():
            for x, y, idxs in test_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.net(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.data.cpu()
        
        return probs

    def predict_prob_dropout_split(self, test_dataset, n_drop):
        test_dataloader = DataLoader(self.handler(test_dataset, transform=self.test_transform),
                            batch_size=self.batch_size*3, shuffle=False, num_workers=self.args.worker, drop_last=False)
        self.init_net(training=False)
        self.net.train()
        probs = torch.zeros([n_drop, len(test_dataset), self.class_num])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in test_dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, _ = self.net(x)
                    probs[i][idxs] += F.softmax(out, dim=1).data.cpu()
        
        return probs

    def get_embedding(self, test_dataset):
        test_dataloader = DataLoader(self.handler(test_dataset, transform=self.test_transform),
                            batch_size=self.batch_size*3, shuffle=False, num_workers=self.args.worker, drop_last=False)
        self.init_net(training=False)
        self.net.eval()
        embedding = torch.zeros([len(test_dataset), self.net.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in test_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                _, e1 = self.net(x)
                embedding[idxs] = e1.data.cpu()
        
        return embedding
    
    def get_output(self, test_dataset):
        test_dataloader = DataLoader(self.handler(test_dataset, transform=self.test_transform),
                            batch_size=self.batch_size*3, shuffle=False, num_workers=self.args.worker, drop_last=False)
        self.init_net(training=False)
        self.net.eval()
        probs = torch.zeros([len(test_dataset), self.class_num])
        embedding = torch.zeros([len(test_dataset), self.net.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in test_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                out, emb = self.net(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.data.cpu()
                embedding[idxs] = emb.data.cpu()
        
        return probs, embedding

