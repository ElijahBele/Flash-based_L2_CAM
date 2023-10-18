# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 21:47:23 2022

@author: Elijah
"""

import  torch, os
import  numpy as np
from    omniglotNShot import OmniglotNShot
import  argparse

from   meta import Meta
from   prototypical import Proto


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=20)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
    #print(args)

    config = [
       ('conv2d', [8, 1, 3, 3, 1, 1]),
       ('bn', [8]),
       ('relu', [True]),
       ('max_pool2d',[2, 2, 0]),
       
       ('conv2d', [8, 8, 3, 3, 1, 1]),
       ('bn', [8]),
       ('relu', [True]),
       ('max_pool2d',[2, 2, 0]),
       
       ('conv2d', [8, 8, 3, 3, 1, 1]),
       ('bn', [8]),
       ('relu', [True]),
       ('max_pool2d',[2, 2, 0]),
       
       ('conv2d', [8, 8, 3, 3, 1, 1]),
       ('bn', [8]),
       ('relu', [True]),
       ('max_pool2d',[2, 2, 0]),
       ('flatten', []),
        #('linear', [args.n_way, 64])
    ]

    device = torch.device('cuda')
    # maml = Meta(args, config).to(device)
    maml = Proto(args, config, load_model=True)

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

        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt), torch.from_numpy(y_spt), \
                                     torch.from_numpy(x_qry), torch.from_numpy(y_qry)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        y_spt = y_spt.type(torch.long)
        y_qry = y_qry.type(torch.long)
   
        accs = []
        for _ in range(100):
            # test
            x_spt, y_spt, x_qry, y_qry = db_train.next('test')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt), torch.from_numpy(y_spt), \
                                          torch.from_numpy(x_qry), torch.from_numpy(y_qry)
            y_spt = y_spt.type(torch.long)
            y_qry = y_qry.type(torch.long)                             
                         
            # split to single task each time
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                #print(x_spt_one.shape, y_spt_one.shape, x_qry_one.shape, y_qry_one.shape)
                #print(y_spt_one)
                # test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                # test_acc = maml.test_quantize(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                test_acc = maml.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs.append( test_acc )

        # [b, update_step+1]
        accs = np.array(accs).mean(axis=0).astype(np.float16)  #总共随机生成了1000条任务来求平均准确率
        
        
        print('Test acc:', accs)    
        
        
        
        
        
        
        
        
        
      
    
    
    
