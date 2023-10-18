# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:14:17 2022

@author: Elijah
"""

import  torch
import  numpy as np

AM = torch.load('AM.pth').cpu().numpy()
Query = torch.load('Query.pth').cpu().numpy()
Distacnes_L1 = torch.load( 'Distances_L1.pth').cpu().numpy()
Results_L1 = np.argmin(Distacnes_L1, axis=1)

Distacnes_L2 = torch.load( 'Distances_L2.pth').cpu().numpy()
Results_L2 = np.argmin(Distacnes_L2, axis=1)