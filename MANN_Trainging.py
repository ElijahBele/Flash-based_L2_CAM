# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:15:49 2022

@author: Elijah
"""

import  torch, os
import  numpy as np
from    omniglotNShot import OmniglotNShot
import  argparse

from   prototypical import Proto


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)


    args = argparser.parse_args()

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.enabled = True


    config = [
        ('conv2d', [64, 1, 3, 3, 1, 1]),   # [ch_out, ch_in, kernelsz, kernelsz, stride, padding]
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d',[2, 2, 0]),

        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d',[2, 2, 0]),

        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d',[2, 2, 0]),

        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d',[2, 2, 0]),
        ('flatten', []),

        # ('conv2d', [64, 1, 3, 3, 1, 1]),  # [ch_out, ch_in, kernelsz, kernelsz, stride, padding]
        # ('bn', [64]),
        # ('relu', [True]),
        # #('max_pool2d', [2, 2, 0]),
        #
        # ('conv2d', [64, 64, 3, 3, 1, 1]),
        # ('bn', [64]),
        # ('relu', [True]),
        # ('max_pool2d', [2, 2, 0]),
        #
        # ('conv2d', [128, 64, 3, 3, 1, 1]),
        # ('bn', [128]),
        # ('relu', [True]),
        # #('max_pool2d', [2, 2, 0]),
        #
        # ('conv2d', [128, 128, 3, 3, 1, 1]),
        # ('bn', [128]),
        # ('relu', [True]),
        # ('max_pool2d', [2, 2, 0]),
        # ('flatten', []),
        # ('linear', [128, 6272]),
        # ('linear', [64,128]),
        # ('flatten', [])
        
    ]
    
    device = torch.device('cuda')    
    # maml = Meta(args, config).to(device)
    maml = Proto(args, config, load_model=True).to(device)

    # tmp = filter(lambda x: x.requires_grad, maml.parameters())
    # num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(maml)
    # print('Total trainable tensors:', num)

    db_train = OmniglotNShot('omniglot',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)
    test_best_acc = 0
    for step in range(args.epoch):

        x_spt, y_spt, x_qry, y_qry = db_train.next()

        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        y_spt = y_spt.type(torch.long)
        y_qry = y_qry.type(torch.long)
   
        accs_train = maml(x_spt, y_spt, x_qry, y_qry)
        
        if step % 20 == 0:
            print('step:', step, '\ttraining acc:', accs_train)

        if (step+1) % 200 == 0:
            accs = []
            for _ in range(20):   # 1000 batches tasks 20*50
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                              torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
                y_spt = y_spt.type(torch.long)
                y_qry = y_qry.type(torch.long)
                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    #print(x_spt_one.shape, y_spt_one.shape, x_qry_one.shape, y_qry_one.shape)
                    #print(y_spt_one)
                    test_acc = maml.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
            if accs > test_best_acc:
                test_best_acc = accs
                maml.save_model()
                print("save model finished")

      
    
    
    
