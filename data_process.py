'''
数据预处理
'''


import pickle
import numpy as np
import argparse

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--word_vector_dir', type=str, default='data/wordvector/w2v.txt', help='the dir of word_vector')
FLAGS.add_argument('--embedding_matrix_dir', type=str,default='data/embedding_matrix_w2v.npy', help='the dir of embedding_matrix.npy')
FLAGS.add_argument('--word2indices_dir', type=str,default='data/word_to_indices_w2v.file', help='the dir of word2indices')


def word_set():
    
    word_list=[]
    with open('data/train.txt') as f:
        for line in f:
            t1,t2,_=line.rstrip().split('\t')
            word_list.extend(t1.split())
            word_list.extend(t2.split())
    f.close()
    with open('data/test.txt') as f:
        for line in f:
            t1,t2=line.rstrip().split('\t')
            word_list.extend(t1.split())
            word_list.extend(t2.split())
    f.close()
    word_set=list(set(word_list))
    return word_set
    
def embeddings_index (data_dir,word_set):
    '''
    Parameters
    ----------
    data_dir : str
        预训练词向量的txt文件的地址.

    Returns
    -------
    None.

    '''
    embeddings_index = {}
    with  open(data_dir) as f:
        for line in f:
            values = line.rstrip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()
    ## embeddings_index 包含所有w2v词向量词汇
    print('Loaded {} word vectors'.format(len(embeddings_index)))
    
    for i in set(embeddings_index.keys()):
        if i not in word_set:
            del embeddings_index[i]
    print('Loaded {} word vectors'.format(len(embeddings_index)))
    return embeddings_index

def word_to_indices(embeddings_index,word2indices_dir):
    word_to_indices = {}
    indices = 0
    controlnum = 50000 ##设定的词典最大数目
    
    word_to_indices['</s>'] = 0  # end
    indices += 1 
    word_to_indices['pad'] = 1   # pad
    indices += 1 
    word_to_indices['pos'] = 2   # unknow
    indices += 1 
    for word in embeddings_index.keys():
        if word in ['</s>', 'pad', 'pos']: continue
        word_to_indices[word] = indices
        indices += 1
        #if indices > controlnum: break
    print(indices, 'words in train and test texts')
    
    with open(word2indices_dir,'wb') as f:
        pickle.dump(word_to_indices,f)
    return word_to_indices
    #word_to_indices
    #embeddings_index



def embedding_matrix(embeddings_index,embedding_matrix_dir,word_to_indices):
    ##build embeddings_matrix
    print('Preparing embeddings matrix...')
    mean_word_vector = np.mean(list(embeddings_index.values()), axis=0)
    embedding_dim = len(list(embeddings_index.values())[0])
    num_words = len(word_to_indices)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    found_words = 0
    lost = 0
    for word in word_to_indices.keys():
        try:
            index = word_to_indices[word]
            embedding_vector = embeddings_index[word]
            embedding_matrix[index] = embedding_vector
            found_words += 1
        except:
            lost += 1
            print(word)
    
    embedding_matrix[word_to_indices['pad']] = np.zeros(embedding_dim)
    embedding_matrix[word_to_indices['pos']] = mean_word_vector   #pos用词向量的平均向量代替
    print('{} words find {} lost in our vocabulary had {} vectors and appear more than the min frequency'.format(
        found_words, lost, 'w2v'))
    print(embedding_matrix.shape)
    np.save(embedding_matrix_dir,embedding_matrix)

if __name__=='__main__':
    
    args = FLAGS.parse_args()
    
    word_set=word_set()
    embeddings_index=embeddings_index(args.word_vector_dir,word_set)
    word2indices=word_to_indices(embeddings_index,args.word2indices_dir)
    embedding_matrix(embeddings_index,args.embedding_matrix_dir,word2indices)







