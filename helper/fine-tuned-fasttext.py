import fasttext
import numpy as np
import tqdm
import tables
from scipy.sparse import csr_matrix,load_npz
# the split here b/w test train is different than given by authors.
modelpath="../../data/COLING/Fine-FTCondensedData/fine-tune-FTskill.bin"
trainpath='../../data/COLING-sub-nodes/trn_X_job-category.npz'
testpath='../../data/COLING-sub-nodes/tst_X_job-category.npz'
labelpath='../../data/COLING-sub-nodes/Y.txt'
trainembpath="../../data/COLING-sub-nodes/Fine-FTCondensedData/trn_job-category-point_embs.npy"
testembpath="../../data/COLING-sub-nodes/Fine-FTCondensedData/tst_job-category-point_embs.npy"
labelembpath="../../data/COLING-sub-nodes/Fine-FTCondensedData/label_embs.npy"

def make_embs(path,vectorfilepath, csr = False):
    if csr==True:
        jobs = load_npz(path)
        jobs = csr_matrix.toarray(jobs)
        f = open('../../data/COLING-sub-nodes/trn_X_job-category-list.txt')
        index =[x.split('\n')[0] for x in f.readlines()]
        print(index)
        vectors=[]
        #print(jobs[0])
        for i in tqdm.tqdm(range(len(jobs))):
            words=jobs[i]
            # print("wordslength",words,len(words))
            vectorlist=[model[index[word]] for word in words]
            if vectorlist ==[]:
                jobvector = np.zeros(1)
            jobvector=np.mean(vectorlist, axis=0)
            vectors.append(jobvector)
        np.save(vectorfilepath,vectors)
    else:
        file=open(path,"r")
        jobs=file.readlines()
        vectors=[]
        #print(jobs[0])
        for i in tqdm.tqdm(range(len(jobs))):
            words=jobs[i].split()
            # print("wordslength",words,len(words))
            vectorlist=[model[word] for word in words]
            jobvector=np.mean(vectorlist, axis=0)
            vectors.append(jobvector)
        np.save(vectorfilepath,vectors)
        # print("vectors length saved",len(vectors))
        return

model = fasttext.load_model(modelpath)
make_embs(trainpath,trainembpath,True)
make_embs(testpath,testembpath,True)

