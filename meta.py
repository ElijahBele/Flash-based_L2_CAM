import  torch
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


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        
        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)




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
        losses_q = 0  # losses_q[i] is the loss on step i
        corrects = 0
        loss_fn = torch.nn.NLLLoss().cuda()
        
        for i in range(task_num):
            
            spt_embed = self.net(x_spt[i],vars=None, bn_training=True, linear=True)  # [25,64]
            qry_embed = self.net(x_qry[i],vars=None, bn_training=True, linear=True)  # [75, 64]
            
            # print("spt_embed shape: ", spt_embed.shape)
            # print("qry_embed shape: ", qry_embed.shape)
            
            distances = pairwise_distances(qry_embed, spt_embed, 'l2')  #[75,25]
            
            # print("distances shape: ", distances.shape)
            
            attention = (-distances).softmax(dim=1) #[75,25]
            
            # print("attention shape: ", attention.shape)
            
            y_pred = matching_net_predictions(attention, self.k_spt, self.n_way, self.k_qry, y_spt[i]) #[75,5]   
            
            # print("y_spt : ", y_spt[i])
            # print("y_qry : ", y_qry[i])
            y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)  #[75,5] 
            
            losses_q = losses_q + loss_fn(y_pred.log(), y_qry[i])  #NLLloss的输入为logsoftmax的预测和实际标签（无需转one-hot）
            # losses_q = losses_q + F.cross_entropy(y_pred, y_qry[i])  #corss_entropy的输入为logits的输出，即logsoftmax之前的结果
            
            correct = torch.eq(torch.argmax(y_pred, dim=1), y_qry[i]).sum().item()
            corrects = corrects + correct
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        return accs

        
        METRICS = {
        # 'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'cosine': lambda gallery, query: F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
        }
        return METRICS[metric_type]
    
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

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        with torch.no_grad():
            spt_embed = net(x_spt,vars=None, bn_training=False, linear=True)  # [5,64]
            qry_embed = net(x_qry,vars=None, bn_training=False, linear=True)  # [75, 64]
            # print("spt_embed shape: ", spt_embed.shape)
            # print("qry_embed shape: ", qry_embed.shape)
            
            distances = pairwise_distances(qry_embed, spt_embed, 'l2')  #[75,5]
            # print("distances shape: ", distances.shape)
            attention = (-distances).softmax(dim=1) #[75,5]
            # print("attention shape: ", attention.shape)
            y_pred = matching_net_predictions(attention, self.k_spt, self.n_way, self.k_qry, y_spt) #[75,5]     
            
            # y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)  #[75,5] 
            
            correct = torch.eq(torch.argmax(y_pred, dim=1), y_qry).sum().item()
            
            corrects = corrects + correct


        del net

        accs = np.array(corrects) / querysz

        return accs




def main():
    pass


if __name__ == '__main__':
    main()
