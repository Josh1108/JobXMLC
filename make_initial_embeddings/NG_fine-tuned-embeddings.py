import fasttext
import numpy as np
import tables
modelpath="../data/COLING/Fine-FTCondensedData/fine-tune-FTskill.bin"
inputpath='../data/COLING/trn_X.txt'
testpath='../data/COLING/tst_X.txt'
labelpath='../data/COLING/Y.txt'
trainembpath="../data/COLING/Fine-FTCondensedData/trn_point_embs.npy"
testembpath="../data/COLING/Fine-FTCondensedData/tst_point_embs.npy"
labelembpath="../data/COLING/Fine-FTCondensedData/label_embs.npy"

def make_embs(path,vectorfilepath):
    file=open(path,"r")
    jobs=file.readlines()
    vectors=[]
    #print(jobs[0])
    for i in range(len(jobs)):
        words=jobs[i].split()
        print("wordslength",words,len(words))
        vectorlist=[model[word] for word in words]
        jobvector=np.mean(vectorlist, axis=0)
        vectors.append(jobvector)
    np.save(vectorfilepath,vectors)
    print("vectors length saved",len(vectors))
    return
# model = fasttext.train_unsupervised(inputpath, model='skipgram',dim=300,thread=3,minCount=1)
# print(model.words)   # list of words in dictionary
# print(model['capgemini']) # get the vector of the word 'king'
# model.save_model(modelpath)
model = fasttext.load_model(modelpath)
make_embs(labelpath,labelembpath)
