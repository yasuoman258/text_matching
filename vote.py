import numpy as np
import argparse
import os

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--input_dir', type=str, default='vote', help='the dir of vote')
FLAGS.add_argument('--output_dir', type=str,default='predict.txt', help='the dir of the output')

def txt(data_dir):
    data=[]
    with open(data_dir) as f:
        for line in f :
            data.append(int(line.rstrip()[0]))
    f.close()
    return np.array(data)

def vote(input_dir,output_dir):
    a=[]
    filenames = os.listdir(input_dir)
    filenames = [os.path.join(input_dir, f) for f in filenames if f.endswith('.txt')]
    for filename in filenames:
        a.append(txt(filename))
    score=list(sum(a))
    with open(output_dir,'w') as f:
        for item in score:
            if item>=len(filenames)/2:
                f.write('1'+'\n')
            else:
                f.write('0'+'\n')
    f.close()   

if __name__=='__main__':
    args = FLAGS.parse_args()
    vote(args.input_dir,args.output_dir)