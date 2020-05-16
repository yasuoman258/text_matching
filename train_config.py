# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:43:54 2020

@author: cccccccccc
"""
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#以下为超参数和路径的设置
##############################################################################
is_classweight=True
hidden_size=300                                                              #
dropout=0.3                                                                  #
lr=0.0002                                                                    #
embedding_matrix_dir='data/embedding_matrix_w2v100.npy'                      #
word2indices_dir='data/word_to_indices_w2v100.file'                          #   
train_dir='data/train.txt'                                                   #
target_dir='weights/w2v100-'                                                 # 
##############################################################################



embedding_matrix=np.load(embedding_matrix_dir)
embeddings = torch.tensor(embedding_matrix, dtype=torch.float).to(device)
parameters={'hidden_size':hidden_size,
            'dropout':dropout,
            'epochs':100,
            'lr':lr,
            'patience':5,
            'max_grad_norm':10,
            'checkpoint':False,
            'device':device,
            'embedding_matrix':embeddings,
            'save_flag':True,
            'target_dir':target_dir,
           }