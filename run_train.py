import numpy as np
import os
import argparse
import pickle
import torch
import json
from tqdm import tqdm
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from ESIM.esim.data import NLIDataset
from ESIM.esim.model import ESIM
from ESIM.utils import train, validate
from sklearn.model_selection import KFold
import config
import pandas as pd

def fit(parameters,train_loader,valid_loader,k,is_classweight):
    '''
    Parameters
    ----------
    parameters : dict
        训练模型需要的参数.
    train_loader : TYPE
        DESCRIPTION.
    valid_loader : TYPE
        DESCRIPTION.
    k : int
        表示第K折.
    is_classweight : bool
        是否使用class_weight.

    Returns
    -------
    None.

    '''
    hidden_size = parameters['hidden_size']
    dropout = parameters['dropout']
    epochs = parameters['epochs']
    lr = parameters['lr']
    patience = parameters['patience']
    max_grad_norm = parameters['max_grad_norm']
    checkpoint = parameters['checkpoint']
    device = parameters['device']
    save_flag = parameters['save_flag']
    target_dir=parameters['target_dir']+str(k)
    embedding_matrix=parameters['embedding_matrix']
    embeddings = torch.tensor(embedding_matrix, dtype=torch.float).to(device)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    random.seed(123)
    model = ESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 padding_idx=1,
                 dropout=dropout,
                 num_classes=2,
                 device=device).to(device)
    
    
    #为了防止过拟合，尝试把embedding层给冻结
    #for param in model._word_embedding.parameters():
     #   param.requires_grad=False
        
    # -------------------- Preparation for training  ------------------- #
    
    if is_classweight:
        class_weight=torch.tensor([1.367,1]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    ##添加了一个权重项，缓解数据不平衡带来的偏差
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",factor=0.5,patience=0)
    best_score = 0.0
    start_epoch = 1
    
    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []
    train_acc=[]
    valid_acc=[]
    
    # Continuing training from a checkpoint if one was given as argument.
    ###中断之后可以接着训练
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
    
        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))
    
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
    
    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, prob = validate(model,valid_loader,criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%".format(valid_loss, (valid_accuracy*100)))
    
    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training ESIM model on device: {}".format(device),
          20 * "=")
    
    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        epochs_count.append(epoch)
    
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model,
                                       train_loader,
                                       optimizer,
                                       criterion,
                                       epoch,
                                       max_grad_norm)
    
        train_losses.append(epoch_loss)
        train_acc.append(epoch_accuracy)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%" .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
    
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, prob = validate(model,valid_loader,criterion)
    
        valid_losses.append(epoch_loss)
        valid_acc.append(epoch_accuracy)
        
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n".format(epoch_time, epoch_loss, (epoch_accuracy*100)))
    
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
    
        # Early stopping on validation accuracy.
        #保留验证集精度最高的epoch
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
    #         Save the best model. The optimizer is not saved to avoid having
    #         a checkpoint file that is too heavy to be shared. To resume
    #         training from the best model, use the 'esim_*.pth.tar'
    #         checkpoints instead.
            if save_flag:
                torch.save({"epoch": epoch,
                            "model": model.state_dict(),
                            "best_score": best_score,
                            "epochs_count": epochs_count,
                            "train_losses": train_losses,
                            "valid_losses": valid_losses},
                           os.path.join(target_dir, "best.pth.tar"))
    
        # Save the model at each epoch.
        if save_flag:
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "optimizer": optimizer.state_dict(),
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))
    
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    
    df=pd.DataFrame()

    df['train_losses']=train_losses
    df['valid_losses']=valid_losses
    df['train_acc']=train_acc
    df['valid_acc']=valid_acc
    df.to_csv(os.path.join(target_dir,"data.csv"),index=None)





def load_data(word2indices_dir,train_dir):
    '''
    Parameters
    ----------
    word2indices_dir : str
        vocab的地址.
    train_dir : TYPE
        训练集的地址.

    Returns
    -------
    sequences_query : list
        由第一个文本组成的列表.
    sequences_title : list
        由第二个文本组成的列表.
    label_list : list
        由label组成的列表.

    '''
    
    with open(word2indices_dir,'rb') as f:
        word_to_indices=pickle.load(f)

    sequences_query=[]
    sequences_title=[]
    label_list=[]
    with open(train_dir,'r') as f:
        for line in f:
            t1,t2,label=line.rstrip().split('\t')
            #将由word组成的句子变成由indice组成的句子
            t1=[word_to_indices[item] if  item in word_to_indices.keys() else word_to_indices['pos'] for item in t1.split()]
            t2=[word_to_indices[item] if  item in word_to_indices.keys() else word_to_indices['pos'] for item in t2.split()]
            sequences_query.append(t1)
            sequences_title.append(t2)
            label_list.append(int(label))
    f.close()
    return sequences_query,sequences_title,label_list

def data_loader(sequences_query,sequences_title,label_list,train_index,test_index):
    '''
    Parameters
    ----------
    sequences_query : list
        由第一个文本组成的列表.
    sequences_title : list
        由第二个文本组成的列表.
    label_list : list
        由label组成的列表.
    train_index : list
        训练数据对应的索引.
    test_index : list
        验证数据对应的索引
    Returns
    -------
    train_loader : TYPE
        DESCRIPTION.
    valid_loader : TYPE
        DESCRIPTION.

    '''
    train_esmim = {
        'ids': train_index ,
        'premises': [sequences_query[i] for i in train_index ],
        'hypotheses': [sequences_title[i] for i in train_index ],
        'labels': [label_list[i] for i in train_index ]
    }

    valid_esmim = {
        'ids': test_index ,
        'premises': [sequences_query[i] for i in test_index  ],
        'hypotheses': [sequences_title[i] for i in test_index  ],
        'labels': [label_list[i] for i in test_index  ]
    }
    print("\t* Loading training data...")
    Ntrain = NLIDataset(train_esmim, padding_idx=1, max_premise_length=32, max_hypothesis_length=32)
    Nvalid = NLIDataset(valid_esmim, padding_idx=1, max_premise_length=32, max_hypothesis_length=32)
    train_loader = DataLoader(Ntrain, shuffle=True, batch_size=32)
    valid_loader = DataLoader(Nvalid, shuffle=False, batch_size=32)
    return train_loader,valid_loader

   
if __name__=='__main__':
    
    
    
    
    sequences_query,sequences_title,label_list=load_data(config.word2indices_dir,config.train_dir)
    
    kf=KFold(n_splits=10, shuffle=True)
    k=0
    for train_index, test_index in kf.split(sequences_query):
        k+=1
        train_loader,valid_loader=data_loader(sequences_query,sequences_title,label_list,train_index,test_index)
        fit(config.parameters,train_loader,valid_loader,k,config.is_classweight)
    print('训练完毕')
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    