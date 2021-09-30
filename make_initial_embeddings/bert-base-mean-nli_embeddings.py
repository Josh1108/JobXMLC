from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='Embeddings ')
parser.add_argument('--output')
parser.add_argument("--path", default="./../data/COLING/Y.txt")
parser.add_argument("--model", default="bert")

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
dataset = dataset
for batch in tqdm(np.array_split(dataset, 50)):
    embs = embedder.encode(batch)
    print(embs.shape)
    corpus_embeddings = np.vstack((corpus_embeddings, embs))

print("Saving embeddings to file...")
np.save("./../data/COLING/BERTMEANCondensedData/{}.npy".format(args.output), corpus_embeddings)
