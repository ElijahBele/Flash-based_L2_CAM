# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:36:37 2022

@author: Elijah
"""

import  torch
import  time
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from torch.autograd import Variable
from    learner import Learner
from Classifier import Classifier
from    copy import deepcopy
from function import *


class Proto(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config, load_model = False):
        """

        :param args:
        """
        super(Proto, self).__init__()

        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        
        self.net = Learner(config, args.n_way, args.k_spt, load_model)            
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def save_model(self):
        
        # torch.save(self.net, f'{acc*100}%_Proto.pth')
        self.net.save_model()

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter
    
    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        loss_fn = torch.nn.NLLLoss().cuda()

        
        for i in range(task_num):
            
            # print("x_spt", x_spt)
            spt_embed = self.net(x_spt[i],vars=None, bn_training=True)  # [25,64]
            qry_embed = self.net(x_qry[i],vars=None, bn_training=True)  # [75, 64]
            
            # print("spt_embed:", spt_embed)
            # print("spt_embed shape: ", spt_embed.shape)
            # print("qry_embed shape: ", qry_embed.shape)
            
            prototypes = spt_embed.reshape(self.n_way, self.k_spt, -1).mean(dim=1)  #[5,64]
            # print("prototypes shape: ", prototypes.shape)
            
            distances = pairwise_distances(qry_embed, prototypes, 'l2').cuda()  #[75,5]
            
            # print("distances shape: ", distances.shape)
            
            # print("distances:", distances)
            log_p_y = (-distances).log_softmax(dim=1)
            if i == 0:
                losses_q = loss_fn(log_p_y, y_qry[i]).cuda()
                corrects = torch.eq(torch.argmax(log_p_y, dim=1), y_qry[i]).sum().item()
            else:
                losses_q = losses_q + loss_fn(log_p_y, y_qry[i]).cuda()
                corrects = corrects + torch.eq(torch.argmax(log_p_y, dim=1), y_qry[i]).sum().item()
        # end of all tasks
        # sum over all losses on query set across all tasks
        losses_q = losses_q / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        losses_q.backward()
        self.meta_optim.step()

        accs = corrects / (querysz * task_num)

        return accs

       
    def test(self, x_spt, y_spt, x_qry, y_qry):
        
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        corrects = 0
        querysz = x_qry.size(0)
        i = 1;
        j = 1;
        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        with torch.no_grad():

            spt_embed = net(x_spt,vars=None, bn_training=False)  # [5,64]
            start_cnn = time.perf_counter_ns()
            while (i > 0):
                qry_embed = net(x_qry,vars=None, bn_training=False)  # [75, 64]
                i = i - 1;
            end_cnn = time.perf_counter_ns()
            latency_cnn = (end_cnn - start_cnn)
            # qry_embed = net(x_qry,vars=None, bn_training=False)  # [75, 64]
            # print("spt_embed shape: ", spt_embed.shape)
            # print("qry_embed shape: ", qry_embed.shape)
            


            start = time.perf_counter_ns()
            while(j>0):
              prototypes = spt_embed.reshape(self.n_way, self.k_spt, -1).mean(dim=1)  # [5,64]
              distances = pairwise_distances(qry_embed, prototypes, 'l2')  #[75,5]
              j=j-1;
            end = time.perf_counter_ns()
            latency = (end-start)
            y_pred = (-distances).softmax(dim=1)
            correct = torch.eq(torch.argmax(y_pred, dim=1), y_qry).sum().item()
            corrects = corrects + correct  
           

        del net

        accs = np.array(corrects) / querysz

        return accs
    def test_quantize(self, x_spt, y_spt, x_qry, y_qry, precision=3):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        corrects = 0
        querysz = x_qry.size(0)

        net = deepcopy(self.net)
        with torch.no_grad():
            spt_embed = net(x_spt,vars=None, bn_training=False)  # [5,64]
            qry_embed = net(x_qry,vars=None, bn_training=False)  # [75, 64]          
            
            prototypes = spt_embed.reshape(self.n_way, self.k_spt, -1).mean(dim=1)  #[5,64]
            
            max_value = 0.75*prototypes.max()
            min_value = prototypes.min()+0.3
            prototypes_quan = quantize(prototypes, precision, max_value, min_value)
            # qry_embed_quan  = quantize_qry(qry_embed, precision, max_value, min_value)
            qry_embed_quan = quantize(qry_embed, precision, max_value, min_value)

            torch.save(prototypes_quan, "spt.pth")
            torch.save(qry_embed_quan, "qry.pth")
            # torch.save(y_spt, "y_spt.pth")
            torch.save(y_qry, "y_qry.pth")

            distances = pairwise_distances(qry_embed_quan, prototypes_quan, 'cosine')  #[75,5]
            # distances = pairwise_distances_quantize(qry_embed_quan, prototypes_quan, 'l2', 1)  # [75,5],output precision
            y_pred = (-distances).softmax(dim=1)
            correct = torch.eq(torch.argmax(y_pred, dim=1), y_qry).sum().item()
            corrects = corrects + correct
        del net
        accs = np.array(corrects) / querysz
        return accs

    def test_quantize_RRAM(self, x_spt, y_spt, x_qry, y_qry, variation, precision=3):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        corrects = 0
        querysz = x_qry.size(0)

        net = deepcopy(self.net)
        with torch.no_grad():
            spt_embed = net(x_spt, vars=None, bn_training=False)  # [5,64]
            qry_embed = net(x_qry, vars=None, bn_training=False)  # [75, 64]

            prototypes = spt_embed.reshape(self.n_way, self.k_spt, -1).mean(dim=1)  # [5,64]

            max_value = 0.95 * prototypes.max()
            min_value = prototypes.min() + 0.05
            prototypes_quan = quantize(prototypes, precision, max_value, min_value)
            # qry_embed_quan  = quantize_qry(qry_embed, precision, max_value, min_value)
            qry_embed_quan = quantize(qry_embed, precision, max_value, min_value)

            torch.save(prototypes_quan, "spt.pth")
            torch.save(qry_embed_quan, "qry.pth")
            # torch.save(y_spt, "y_spt.pth")
            torch.save(y_qry, "y_qry.pth")

            #distances = pairwise_distances(qry_embed_quan, prototypes_quan, 'RRAM')  # [75,5]
            distances = pairwise_distances_RRAM(qry_embed_quan, prototypes_quan, 'RRAM', variation)  # [75,5]
            # distances = pairwise_distances_quantize(qry_embed_quan, prototypes_quan, 'l2', 1)  # [75,5],output precision
            y_pred = (-distances).softmax(dim=1)
            correct = torch.eq(torch.argmax(y_pred, dim=1), y_qry).sum().item()
            # correct = torch.eq(torch.argmax(y_pred, dim=1), y_qry)
            # correct = correct.sum()
            # correct = correct.item()
            corrects = corrects + correct
        del net
        accs = np.array(corrects) / querysz
        return accs

    def test_quantize_variation(self, x_spt, y_spt, x_qry, y_qry, variation, precision=3):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        corrects = 0
        querysz = x_qry.size(0)

        net = deepcopy(self.net)
        with torch.no_grad():
            spt_embed = net(x_spt, vars=None, bn_training=False)  # [5,64]
            qry_embed = net(x_qry, vars=None, bn_training=False)  # [75, 64]

            prototypes = spt_embed.reshape(self.n_way, self.k_spt, -1).mean(dim=1)  # [5,64]

            max_value = 0.75 * prototypes.max()
            min_value = prototypes.min() + 0.45

            prototypes_quan = quantize(prototypes, precision, max_value, min_value)
            qry_embed_quan = quantize_qry(qry_embed, precision, max_value, min_value)

            # torch.save(prototypes_quan, "spt.pth")
            # torch.save(qry_embed_quan, "qry.pth")
            # torch.save(y_spt, "y_spt.pth")
            # torch.save(y_qry, "y_qry.pth")

            distances = pairwise_distances_variation(qry_embed_quan, prototypes_quan, 'l2',variation)  # [75,5]
            #distances = pairwise_distances(qry_embed_quan, prototypes_quan, 'l2')  # [75,5]
            # distances = pairwise_distances_quantize(qry_embed, prototypes, 'l2', 4)  # [75,5],output precision
            y_pred = (-distances).softmax(dim=1)
            correct = torch.eq(torch.argmax(y_pred, dim=1), y_qry).sum().item()
            corrects = corrects + correct

        del net

        accs = np.array(corrects) / querysz

        return accs
    def test_lsh_variation(self, x_spt, y_spt, x_qry, y_qry, LSH_a, LSH_b, variation, precision=3):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        corrects = 0
        querysz = x_qry.size(0)
        net = deepcopy(self.net)
        with torch.no_grad():
            spt_embed = net(x_spt, vars=None, bn_training=False)  # [5,64]
            qry_embed = net(x_qry, vars=None, bn_training=False)  # [75, 64]
            prototypes = spt_embed.reshape(self.n_way, self.k_spt, -1).mean(dim=1)  # [5,64]

            spt_hash_values = (torch.mm(prototypes, LSH_a) + LSH_b) // torch.tensor(1)  #[5,80]
            qry_hash_values = (torch.mm(qry_embed, LSH_a) + LSH_b) // torch.tensor(1)   #[75,80]

            # spt_hash_values = torch.mm(prototypes, LSH_a)
            # qry_hash_values = torch.mm(qry_embed, LSH_a)
            # spt_hash_values_quan = torch.where(spt_hash_values < 0, 0.0, 29.4)
            # qry_hash_values_quan = torch.where(qry_hash_values < 0, 0.0, 29.4)

            max_value = torch.tensor(20);
            min_value = torch.tensor(-5);
            # # max_value = spt_hash_values.max();
            # # min_value = spt_hash_values.min();
            # # max_value = 0.5*torch.tensor(2**precision)
            # # min_value = 0.5*torch.tensor(-2**precision)
            spt_hash_values_quan = quantize(spt_hash_values, precision, max_value, min_value)
            qry_hash_values_quan = quantize(qry_hash_values, precision, max_value, min_value)

            distances = pairwise_distances_variation(qry_hash_values_quan, spt_hash_values_quan, 'l1', variation)  # [75,5]
            #distances = pairwise_distances(qry_hash_values_quan, spt_hash_values_quan, 'l1')  # [75,5]
            # distances = pairwise_distances(qry_embed, prototypes, 'l1')  # [75,5]


            y_pred = (-distances).softmax(dim=1)
            correct = torch.eq(torch.argmax(y_pred, dim=1), y_qry).sum().item()
            corrects = corrects + correct

        del net

        accs = np.array(corrects) / querysz

        return accs

    def test_lsh(self, x_spt, y_spt, x_qry, y_qry, LSH_a, LSH_b, precision=3):
        corrects = 0
        querysz = x_qry.size(0)
        net = deepcopy(self.net)
        with torch.no_grad():
            spt_embed = net(x_spt, vars=None, bn_training=False)  # [5,64]
            qry_embed = net(x_qry, vars=None, bn_training=False)  # [75, 64]
            prototypes = spt_embed.reshape(self.n_way, self.k_spt, -1).mean(dim=1)  # [5,64]

            # max_value = 0.6 * prototypes.max()
            # min_value = prototypes.min() + 0.45
            # prototypes = torch.clamp(prototypes, min= min_value, max= max_value)
            # qry_embed = torch.clamp(qry_embed, min=min_value,  max=max_value)

            # spt_hash_values = (torch.mm(prototypes, LSH_a) + LSH_b) // torch.tensor(1)  #[5,80]
            # qry_hash_values = (torch.mm(qry_embed, LSH_a) + LSH_b) // torch.tensor(1)   #[75,80]


            spt_hash_values = torch.mm(prototypes, LSH_a)
            qry_hash_values = torch.mm(qry_embed, LSH_a)
            a = torch.zeros(spt_hash_values.shape).cuda()
            b = torch.ones(spt_hash_values.shape).cuda()
            spt_hash_values_quan = torch.where(spt_hash_values < 0, a, b)
            a = torch.zeros(qry_hash_values.shape).cuda()
            b = torch.ones(qry_hash_values.shape).cuda()
            qry_hash_values_quan = torch.where(qry_hash_values < 0, a, b)
            # torch.save(spt_hash_values_quan,"spt_hash.pth");
            # torch.save(qry_hash_values_quan,"qry_hash.pth");

            # max_value = torch.tensor(10)
            # min_value = torch.tensor(-5)
            # spt_hash_values_quan = quantize(spt_hash_values, precision, max_value, min_value)
            # qry_hash_values_quan = quantize_qry(qry_hash_values, precision, max_value, min_value)

            torch.save(spt_hash_values_quan,"spt_hash.pth");
            torch.save(qry_hash_values_quan,"qry_hash.pth");
            torch.save(y_qry, "y_qry.pth")

            #distances = pairwise_distances_quantize(qry_hash_values_quan, spt_hash_values_quan, 'l2',  2)  # [75,5],output precision
            distances = pairwise_distances(qry_hash_values_quan, spt_hash_values_quan, 'l2')  # [75,5]
            #distances = pairwise_distances(qry_hash_values, spt_hash_values, 'l2')  # [75,5]
            # distances = pairwise_distances(qry_embed, prototypes, 'l1')  # [75,5]


            y_pred = (-distances).softmax(dim=1)
            prediction = torch.argmax(y_pred, dim=1)
            correct = torch.eq(torch.argmax(y_pred, dim=1), y_qry).sum().item()
            corrects = corrects + correct

        del net

        accs = np.array(corrects) / querysz

        return accs

def main():
    pass


if __name__ == '__main__':
    main()
