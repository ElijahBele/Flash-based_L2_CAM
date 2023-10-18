import torch, os
import numpy as np
from omniglotNShot import OmniglotNShot
import argparse
import random
from meta import Meta
from prototypical import Proto
from function import *

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10)

    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for querty set', default=15)

    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)

    args = argparser.parse_args()

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.enabled = True

    config = [
        ('conv2d', [64, 1, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),

        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),

        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),

        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', [])

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
    accs = []
    sigma_int = [1,3,5,9,12,14,29,43,57,71,86,100,114,129,143,157]
    hash_bit = 80
    LSH_a = np.random.normal(0, 1, size=(64, hash_bit))
    LSH_b = np.random.uniform(0, 1, size=(1, hash_bit))
    LSH_a, LSH_b = torch.from_numpy(LSH_a).to(device), torch.from_numpy(LSH_b).to(device)
    LSH_a = LSH_a.type(torch.float)
    LSH_b = LSH_b.type(torch.float)
    result = open("5way1shot_variation_L1CAM.txt","a+")
    for i in range(0, 16, 1):
        sigma = sigma_int[i]
        for cycle in range(0,10,1):
            accs = []
            # print(LSH_a, LSH_b)
            for _ in range(40):  # 200*50=10000 batches task
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
                y_spt = y_spt.type(torch.long)
                y_qry = y_qry.type(torch.long)
                variation = sigma * 0.01 * torch.randn(args.n_way, hash_bit).cuda()
                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    # print(x_spt_one.shape, y_spt_one.shape, x_qry_one.shape, y_qry_one.shape)
                    # print(y_spt_one)

                    test_acc = maml.test_lsh_variation(x_spt_one, y_spt_one, x_qry_one, y_qry_one, LSH_a, LSH_b, variation)
                    #test_acc = maml.test_quantize_variation(x_spt_one, y_spt_one, x_qry_one, y_qry_one, variation)
                    # test_acc = maml.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append(test_acc)
                # print('Test Acc:', np.array(accs).mean(axis=0).astype(np.float16))
            # [b, update_step+1]
            mean_accs = np.array(accs).mean(axis=0).astype(np.float16)
            #print('Mean Test Acc:', mean_accs)
            print('Sigma:', sigma*0.01, "Acc: ", mean_accs)
            print('Sigma:', sigma*0.01, "Acc: ", mean_accs, file = result)
            # if sigma==0:
            #     break
    result.close()

