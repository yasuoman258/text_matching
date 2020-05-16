from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import numpy as np
import torch
from ESIM.esim.data import NLIDataset
from ESIM.esim.model import ESIM
import pickle
import os
import argparse

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--predict_dir', type=str, default='predict.txt', help='the dir of predict')
FLAGS.add_argument('--weight_dir', type=str,default='weights/best.pth.tar', help='the dir of the model')
FLAGS.add_argument('--word2indices_dir',type=str,default='data/word_to_indices_w2v100.file', help='the dir of word2indices')



def test_loader(word2indices_dir):
    #读取字典
    word_to_indices={} #word_to_indices_glove.file
    with open(word2indices_dir, "rb") as f:
        word_to_indices = pickle.load(f)
    #读取测试文件
    text1=[]
    text2=[]
    with open('data/test.txt','r') as f:
        for line in f:
            t1,t2=line.rstrip().split('\t')
            
            t1=[word_to_indices[item] if  item in word_to_indices.keys() 
                else word_to_indices['pos'] for item in t1.split()]
            t2=[word_to_indices[item] if  item in word_to_indices.keys() 
                else word_to_indices['pos'] for item in t2.split()]

            text1.append(t1)
            text2.append(t2)
    f.close()
    idx=list(range(len(text1)))
    labels=[0]*len(text1) 
    test_esmim = {
    'ids': idx,
    'premises': text1,
    'hypotheses': text2,
    'labels': labels
    }
    test = NLIDataset(test_esmim, padding_idx=1, max_premise_length=32, max_hypothesis_length=32)
    test_loader = DataLoader(test, shuffle=False, batch_size=32)
    return test_loader

def predict(weight_dir,predict_dir,test_loader):
    '''
    Parameters
    ----------
    weight_dir : str
        预测模型的地址.
    predict_dir : str
        存储预测结果的地址.

    Returns
    -------
    None.

    '''
    
    state=torch.load(weight_dir)['model']
    input_size=state['_word_embedding.weight'].size(0)
    output_size=state['_word_embedding.weight'].size(1)
    hidden_size=state['_encoding._encoder.weight_hh_l0_reverse'].size(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = ESIM(input_size,
          output_size,
          hidden_size,
          padding_idx=1,
          num_classes=2,
          device=device).to(device)
    model.load_state_dict(state)

    model.eval()
    
    predict_list=[]

    for batch in tqdm(test_loader,desc='predict processing:'):
        # Move input and output data to the GPU if one is used.
        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)
        logits, probs = model(premises,premises_lengths,hypotheses,hypotheses_lengths)
        scores_1 = probs.data.cpu().numpy()[:,1]
        # scores_1[scores_1 >= threshold] = 1
        # scores_1[scores_1 < threshold] = 0
        predict_list.extend(list(scores_1))
    print(sum(predict_list))
    
    #取中位数作为二分类的阈值
    print('threshold:',sorted(predict_list)[6249])
    threshold=sorted(predict_list)[6249]
    #存储预测结果
    with open(predict_dir,'w') as f:
        for item in predict_list:

            if item>threshold:
                f.write('1'+'\n')
            else:
                f.write('0'+'\n')
    f.close()
    print('预测完毕！')
    
if __name__=='__main__':
    args = FLAGS.parse_args()
    loader=test_loader(args.word2indices_dir)
    predict(args.weight_dir,args.predict_dir,loader)