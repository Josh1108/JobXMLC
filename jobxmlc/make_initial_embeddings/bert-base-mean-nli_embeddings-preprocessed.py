from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser(description='Embeddings ')
parser.add_argument('--output')
parser.add_argument("--path", default="./../data/COLING/Y.txt")
parser.add_argument("--model", default="bert")

def word_tokenizer(text):
    return text.split()
def preprocess(corpus):
    """
    remove the 10 most unimportant words in a job based on tf-idf scores
    """
    vectorizer = TfidfVectorizer(tokenizer=word_tokenizer)
    X = vectorizer.fit_transform(corpus)
    Y = vectorizer.get_feature_names()
    full_corpus_new = []
    for i in tqdm(range(len(corpus))):
        dicti = dict(zip(vectorizer.get_feature_names(), X.toarray()[i]))
        lisi=[]
        for text in corpus[i].split():
            lisi.append(dicti[text])
        lisi_index = sorted(range(len(lisi)), key=lambda k: lisi[k])
        lisi_index=lisi_index[:10]
        corpus_split = corpus[i].split()
        corpus_split_pre = []

        for i, item in enumerate(corpus_split):
            if i in lisi_index:
                continue
            else:
                corpus_split_pre.append(item)
        corpus_preprocessed = " ".join(corpus_split_pre)
        full_corpus_new.append(corpus_preprocessed)
    # print(full_corpus_new[:5])
    return full_corpus_new

args = parser.parse_args()

if args.model=='bert': model_name = "bert-base-nli-mean-tokens" 
elif args.model == 'distilbert': model_name = "distilbert-base-nli-mean-tokens"

embedder = SentenceTransformer(model_name)

    
# with open(args.path, 'rb') as f:
#     data = pickle.load(f)

f = open(args.path,'r').readlines()
dataset=[x.split('\n')[0] for x in f]
print(dataset[0])

# for a,b in data:
#     dataset.append(a)
print(len(dataset))
print(dataset[0],len(dataset))
print("Encoding {} sentences using {}.....\n".format(len(dataset), args.model))
# sents = np.array(data)

corpus_embeddings = np.empty((0, 768))
dataset = preprocess(dataset)
for batch in tqdm(np.array_split(dataset, 50)):
    embs = embedder.encode(batch)
    print(embs.shape)
    corpus_embeddings = np.vstack((corpus_embeddings, embs))

print("Saving embeddings to file...")
np.save("./../dumps/BERTMEAN-{}.npy".format(args.output), corpus_embeddings)
