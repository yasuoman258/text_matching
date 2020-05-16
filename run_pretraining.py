from gensim.models import word2vec
import gensim
import logging
import argparse

def word2vector(iteration,min_word_count,downsampling,output_dir):
    '''
    训练word2vec词向量
    Parameters
    ----------
    iteration : int
        预训练的轮次.
    min_word_count : int
        低于这个频次的词语会被删除掉.
    output_dir :str
        输出的文件保存的路径.
    downsampling: float
        Downsample setting for frequent words
    Returns
    -------
    None.

    '''
    
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    num_features = 300    # Word vector dimensionality
    num_workers = 16       # Number of threads to run in parallel
    context = 8          # Context window size
    sentences = word2vec.Text8Corpus("data/corpus.txt")

    model = word2vec.Word2Vec(sentences, workers=num_workers, \
           size=num_features, min_count = min_word_count, \
          window = context, sg = 0, sample = downsampling,iter=iteration)

    with open(output_dir,'w') as f:
        for key in model.wv.vocab.keys():
            array_list=list(model[key])
            array_list=list(map(str,array_list))
            temp=' '.join(array_list)
            f.write(key+' '+temp+'\n')
        f.close()

def fasttext(iteration,min_word_count,downsampling,output_dir):
    
    '''
    训练fasttext词向量
    Parameters
    ----------
    iteration : int
        预训练的轮次.
    min_word_count : int
        低于这个频次的词语会被删除掉.
    output_dir : TYPE
        输出的文件保存的路径.

    Returns
    -------
    None.

    '''
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    num_features = 300    #词向量的维度
    num_workers = 16       # Number of threads to run in parallel
    context = 8          # Context window size
    sentences = word2vec.Text8Corpus("data/corpus.txt")
    model = fasttext.FastText(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context,  sample = downsampling,iter=iteration)
    
    with open(output_dir,'w') as f:
        for key in model.wv.vocab.keys():
            array_list=list(model[key])
            array_list=list(map(str,array_list))
            temp=' '.join(array_list)
            f.write(key+' '+temp+'\n')
        f.close()

def txt_concat():
    '''
    将训练集，测试集和corpus中的语句结合在一起
    Returns 
    -------
    None.
    '''
    with open('data/corpus.txt','a') as f:
        with open('data/train.txt') as f1:
            for line in f1:
                t1,t2,_=line.rstrip().split('\t')
                f.write(t1+'\n')
                f.write(t2+'\n')
        f1.close()
        with open('data/test.txt') as f2:
            for line in f2:
                t1,t2=line.rstrip().split('\t')
                f.write(t1+'\n')
                f.write(t2+'\n')
        f2.close()        
    f.close()
    print('拼接完成！')
    
if __name__=='__main__':
    txt_concat()
    word2vector(iteration=20,min_word_count=1,downsampling=1e-3,output_dir='data/word_vector/w2v20.txt')
    word2vector(iteration=50,min_word_count=5,downsampling=1e-3,output_dir='data/word_vector/w2v50.txt')
    word2vector(iteration=100,min_word_count=5,downsampling=1e-4,output_dir='data/word_vector/w2v100.txt')
    fasttext(iteration=20,min_word_count=5,downsampling=1e-3,output_dir='data/word_vector/ft.txt')
