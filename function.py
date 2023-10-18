
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:21:22 2022

@author: Elijah
"""
import torch
import os
import shutil
from typing import Tuple, List

EPSILON = 1e-8
temp1 = [[0.2, 1.1, 3.8, 9.0, 15.0, 20.8, 26.3, 31.2],
       [1.0, 0.2, 1.0, 4.0, 9.1, 15.1, 21.0, 26.2],
       [3.8, 1.1, 0.2, 1.1, 4.0, 9.2, 15.3, 20.9],
       [9.0, 4.0, 1.1, 0.2, 1.1, 4.0, 9.4, 15.1],
       [15.1, 9.4, 4.0, 1.1, 0.2, 1.1, 4.0, 9.0],
       [20.9, 15.3, 9.2, 4.0, 1.1, 0.2, 1.1, 3.8],
       [26.2, 21.0, 15.1, 9.1, 4.0, 1.0, 0.2, 1.0],
       [31.2, 26.3, 20.8, 15.0, 9.0, 3.8, 1.1, 0.2]]

temp2 = [[0,0.6,2.4,3.6,9.6,15,21.6,29.4],
         [29.4,21.6,15,9.6,3.6,2.4,0.6,0]]

LUT = torch.tensor(temp1)

def pairwise_distances_variation(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str,
                       variation) -> torch.Tensor:
    n_x = x.shape[0]
    n_y = y.shape[0]
    if matching_fn == 'l2':
        # distances = (
        #         x.unsqueeze(1).expand(n_x, n_y, -1) -
        #         y.unsqueeze(0).expand(n_x, n_y, -1)
        # ).pow(2).sum(dim=2)

        distances = torch.abs((x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1))).pow(2)
        distances = distances + variation

        a = torch.zeros(distances.shape).cuda()
        distances = torch.where(distances < 0, a, distances)
        b = torch.ones((distances.shape)).cuda()
        distances = torch.where(distances>1000, a, distances)

        distances = distances.sum(dim=2)

        return distances
    
    elif matching_fn == 'idvg':
        
        x_ = x.unsqueeze(1).expand(n_x, n_y, -1).type(torch.int).to('cpu')        
        y_ = y.unsqueeze(0).expand(n_x, n_y, -1).type(torch.int).to('cpu')
        distances = torch.empty(x_.shape)
        # print(x_)
        # print(y_)
        for i in range(x_.shape[0]):
            for j in range(x_.shape[1]):
                for k in range(x_.shape[2]):
                    distances[i][j][k] = LUT[x_[i][j][k].item()][y_[i][j][k].item()]
        # print(distances)
        distances = distances.sum(dim=2)#.cuda()
        # print(distances)
        return distances
    
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON )
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON )

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return  -cosine_similarities

    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    elif matching_fn == 'l1':
        distances = torch.abs( x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1))

        # a = torch.zeros(distances.shape).cuda()
        # distances = torch.where(distances < 14.7, a, distances)
        a = torch.zeros(distances.shape).cuda()
        b = torch.ones((distances.shape)).cuda()
        distances = torch.where(distances > 1000, a, distances)
        distances = distances + variation
        distances = distances.sum(dim=2)

        return distances
    else:
        raise(ValueError('Unsupported similarity function'))


def pairwise_distances(x: torch.Tensor, y: torch.Tensor, matching_fn: str)-> torch.Tensor:
    n_x = x.shape[0]
    n_y = y.shape[0]
    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)

        # distances = torch.abs((x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1))).pow(2)
        #
        # a = torch.zeros(distances.shape).cuda()
        # b = torch.ones((distances.shape)).cuda()
        # distances = torch.where(distances > 1000, a, distances)
        #
        # distances = distances.sum(dim=2)

        return distances

    elif matching_fn == 'idvg':

        x_ = x.unsqueeze(1).expand(n_x, n_y, -1).type(torch.int).to('cpu')
        y_ = y.unsqueeze(0).expand(n_x, n_y, -1).type(torch.int).to('cpu')
        distances = torch.empty(x_.shape)
        # print(x_)
        # print(y_)
        for i in range(x_.shape[0]):
            for j in range(x_.shape[1]):
                for k in range(x_.shape[2]):
                    distances[i][j][k] = LUT[x_[i][j][k].item()][y_[i][j][k].item()]
        # print(distances)
        distances = distances.sum(dim=2)  # .cuda()
        # print(distances)
        return distances

    elif matching_fn == 'RRAM':

        x_ = x.unsqueeze(1).expand(n_x, n_y, -1).type(torch.int).to('cpu')
        y_ = y.unsqueeze(0).expand(n_x, n_y, -1).type(torch.int).to('cpu')
        distances = torch.zeros(x_.shape)
        # print(x_)
        # print(y_)
        for i in range(x_.shape[0]):
            for j in range(x_.shape[1]):
                for k in range(x_.shape[2]):
                    binary_input1 = dec2bin(7-x_[i][j][k].item())
                    binary_input2 = dec2bin(x_[i][j][k].item())
                    for bit in range(3):
                        distances[i][j][k] = distances[i][j][k] + (int(binary_input1[bit]) * (y_[i][j][k].item()**2 + 3 ) ) * (2**(2-bit))
                        distances[i][j][k] = distances[i][j][k] + (int(binary_input2[bit])*((7-y_[i][j][k].item())**2 + 3 ) ) * (2**(2-bit))

        # print(distances)
        distances = distances.sum(dim=2).cuda()
        # print(distances)
        return distances

    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return -cosine_similarities

    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    elif matching_fn == 'l1':
        # distances = torch.abs(
        #     x.unsqueeze(1).expand(n_x, n_y, -1) -
        #     y.unsqueeze(0).expand(n_x, n_y, -1)
        # ).sum(dim=2)

        distances = torch.abs((x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1)))

        a = torch.zeros(distances.shape).cuda()
        b = torch.ones((distances.shape)).cuda()
        distances = torch.where(distances > 1000, a, distances)

        distances = distances.sum(dim=2)

        return distances
    else:
        raise (ValueError('Unsupported similarity function'))
def pairwise_distances_RRAM(x: torch.Tensor, y: torch.Tensor, matching_fn: str, variation)-> torch.Tensor:
    n_x = x.shape[0]
    n_y = y.shape[0]
    if matching_fn == 'l2':
        # distances = (
        #         x.unsqueeze(1).expand(n_x, n_y, -1) -
        #         y.unsqueeze(0).expand(n_x, n_y, -1)
        # ).pow(2).sum(dim=2)

        distances = torch.abs((x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1))).pow(2)

        a = torch.zeros(distances.shape).cuda()
        b = torch.ones((distances.shape)).cuda()
        distances = torch.where(distances > 1000, a, distances)

        distances = distances.sum(dim=2)

        return distances
    elif matching_fn == 'RRAM':

        x_ = x.unsqueeze(1).expand(n_x, n_y, -1).type(torch.int).to('cpu')
        y_ = y.unsqueeze(0).expand(n_x, n_y, -1).type(torch.int).to('cpu')
        distances = torch.zeros(x_.shape)
        # print(x_)
        # print(y_)
        for i in range(x_.shape[0]):
            for j in range(x_.shape[1]):
                for k in range(x_.shape[2]):
                    binary_input1 = dec2bin(7-x_[i][j][k].item())
                    binary_input2 = dec2bin(x_[i][j][k].item())
                    for bit in range(3):
                        distances[i][j][k] = distances[i][j][k] + (int(binary_input1[bit]) * (y_[i][j][k].item()**2 + 3 + variation[j][k*2]) ) * (2**(2-bit))
                        distances[i][j][k] = distances[i][j][k] + (int(binary_input2[bit])*((7-y_[i][j][k].item())**2 + 3 + variation[j][k*2+1]) ) * (2**(2-bit))

        # print(distances)
        distances = distances.sum(dim=2).cuda()
        # print(distances)
        return distances


    else:
        raise (ValueError('Unsupported similarity function'))
def pairwise_distances_quantize(x, y, matching_fn, precision):
    n_x = x.shape[0]
    n_y = y.shape[0]
    if matching_fn == 'l2':
        # distances = (
        #         x.unsqueeze(1).expand(n_x, n_y, -1) -
        #         y.unsqueeze(0).expand(n_x, n_y, -1)
        # ).pow(2).sum(dim=2)

        distances = torch.abs((x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1))).pow(2)

        a = torch.zeros(distances.shape).cuda()
        b = torch.ones((distances.shape)).cuda()
        distances = torch.where(distances > 600, a, distances)

        distances = distances.sum(dim=2)
        # min_value = distances.min()
        # max_value = distances.max()
        # mean_value = distances.mean()
        min_value = torch.tensor(100)
        max_value = torch.tensor(900)
        distances_quan = quantize(distances, precision, max_value, min_value)
        return distances_quan

    elif matching_fn == 'idvg':

        x_ = x.unsqueeze(1).expand(n_x, n_y, -1).type(torch.int).to('cpu')
        y_ = y.unsqueeze(0).expand(n_x, n_y, -1).type(torch.int).to('cpu')
        distances = torch.empty(x_.shape)
        # print(x_)
        # print(y_)
        for i in range(x_.shape[0]):
            for j in range(x_.shape[1]):
                for k in range(x_.shape[2]):
                    distances[i][j][k] = LUT[x_[i][j][k].item()][y_[i][j][k].item()]
        # print(distances)
        distances = distances.sum(dim=2)  # .cuda()
        # print(distances)
        return distances

    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return -cosine_similarities

    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    elif matching_fn == 'l1':
        distances = torch.abs((x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1)))

        a = torch.zeros(distances.shape).cuda()
        b = torch.ones((distances.shape)).cuda()
        distances = torch.where(distances > 1000, a, distances)

        distances = distances.sum(dim=2)
        min_value = distances.min()
        max_value = distances.max()
        mean_value = distances.mean()
        distances_quan = quantize(distances, precision, max_value, min_value)
        return distances_quan
    else:
        raise (ValueError('Unsupported similarity function'))
def matching_net_predictions(attention: torch.Tensor, k: int, n: int, q: int, y_spt: torch.Tensor) -> torch.Tensor:
    """Calculates Matching Network predictions based on equation (1) of the paper.

    The predictions are the weighted sum of the labels of the support set where the
    weights are the "attentions" (i.e. softmax over query-support distances) pointing
    from the query set samples to the support set samples.

    # Arguments
        attention: torch.Tensor containing softmax over query-support distances.
            Should be of shape (q * k, k * n)


    # Returns
        y_pred: Predicted class probabilities
    """
    if attention.shape != (q * n, k * n):
        raise(ValueError(f'Expecting attention Tensor to have shape (q * k, k * n) = ({q * k, k * n})'))

    # Create one hot label vector for the support set
    y_onehot = torch.zeros(k * n, n)#.cuda()
    

    # Unsqueeze to force y to be of shape (K*n, 1) as this
    # is needed for .scatter()
    # y = torch.arange(0, k, 1 / n).long().unsqueeze(-1)
    y = y_spt.unsqueeze(-1)
    y_onehot = y_onehot.scatter(1, y, 1).cuda()

    y_pred = torch.mm(attention, y_onehot.cuda().float()).cuda()

    return y_pred        
        
def quantize(embed, precision, max_value, min_value):

    # y = (embed - min_value) * pow(2, precision) / (max_value - min_value)
    # result = torch.clamp(y, min=0, max=pow(2,precision)-0.1)
    # result = torch.clamp(y, min=4, max=11.9)
    # result.floor_()

    y = (embed-min_value)*pow(2,precision) / (max_value-min_value)
    result = torch.clamp(y, min=-0.49, max=pow(2,precision)-0.51)
    result.round_()
    # c = 1000 * torch.ones(result.shape).cuda()
    #result = torch.where(result == 3, c, result)
    #result = torch.where(result == 7, c, result)

    return result

def quantize_qry(embed, precision, max_value, min_value):

    y = (embed-min_value)*pow(2,precision) / (max_value-min_value)
    result = torch.clamp(y, min=-0.49, max=pow(2,precision)-0.51)
    result.round_()
    # a = 4 * torch.ones(embed.shape).cuda()
    # b = 11 * torch.ones(embed.shape).cuda()
    # result = torch.where(result < 7.5, a, b)

    a = torch.zeros(result.shape).cuda()
    b = (2**precision-1)*torch.ones(result.shape).cuda()
    c = 1000*torch.ones(result.shape).cuda()
    # #result = torch.where(result<2**(precision-1), a, b)
    result = torch.where(result<=2, a, result)
    result = torch.where(result>=5, b, result)
    # # result = torch.where(result == 2, c, result)
    # # result = torch.where(result == 5, c, result)
    # result = torch.where(result == 3, c, result)
    # result = torch.where(result == 4, c, result)
    result = torch.where((result>2) & (result<5), c, result)
    return result

def dec2bin(number):
    s = []
    binstring=''
    number=int(number)
    while number>0:
        rem=number%2
        s.append(rem)
        number=number//2
    while len(s)>0:
        binstring=binstring+str(s.pop())
    while len(binstring)<3:
        c=3-len(binstring)
        binstring='0'*c+binstring
    return binstring






















