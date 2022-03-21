"""
Generate trian and test files for galaxc
Input: train.pkl and test.pkl: [List of tuple of JD, binary vector of skills], Y.txt skill list txt file
"""

import pickle
import numpy
from  sklearn.datasets import dump_svmlight_file
import pandas as pd
import argparse
import os

def save_text(data_path,save_path):

    load_files = ['train.pkl','test.pkl']
    save_files = ['trn_X_Y.txt','tst_X_Y.txt']

    for i,file in enumerate(load_files):
        with open(os.path.join(data_path,file), 'rb') as f:
            df = pickle.load(f)
        with open(os.path.join(save_path,save_files[i]),'w') as f:
            for item in df:
                f.write(item[0]+'\n')

def csr_format(data_path,save_path):
    save_files = ['trn_X_Y.txt','tst_X_Y.txt']
    load_files = ['train.pkl','test.pkl']

    for i,file in enumerate(load_files):
        with open(os.path.join(data_path,file), 'rb') as f:
            df = pickle.load(f)
        
        lis =[]
        for item in df:
            a,b = item
            l =[]
            for i,x in enumerate(b):
                if x ==1:
                    l.append(i+1)
            lis.append(l)

        with open(os.path.join(save_path,save_files[i]),'w') as f:
            f.write(str(len(lis))+" "+str(len(df[0][1]))+'\n')
            
            for x in lis:
                st =[]
                for t in x:
                    st.append(str(t)+":"+str(1))
                _st=" ".join(st)
                print(_st)
                f.write(_st)
                f.write('\n')

if '__name__' =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--save_path")
    args = parser.parse_args()

    csr_format(args.data_path,args.save_path)

