from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class initialEmbeddings:
    def __init__(self):
        pass
    def preprocessing_data(self):
        return NotImplementedError
    def tokenizer(self):
        return NotImplementedError
    def create_embeddings(self):
        return NotImplementedError
    def save_embeddings(self):
        return NotImplementedError

class hfEmbeddings(initialEmbeddings):
    def __init__(self,args):
        self.args = args
    def word_tokenizer(self, text):
        """
        tokenize a sentence into words
        """
        return text.split() 
    def preprocessing_data(self,corpus):
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
        return full_corpus_new
    def create_embeddings(self):
        model_name = self.args.model
        embedder = SentenceTransformer(model_name)
        dataset_path = self.args.dataset_path
        f = open(dataset_path,'r').readlines()
        dataset=[x.split('\n')[0] for x in f]
        print("First row of dataset:", dataset[0])
        print("Encoding {} sentences using {}.....\n".format(len(dataset), args.model))

        corpus_embeddings = np.empty((0, 768)) 
        dataset = preprocess(dataset)
        for batch in tqdm(np.array_split(dataset, 50)):
            embs = embedder.encode(batch)
            corpus_embeddings = np.vstack((corpus_embeddings, embs))
    
    def save_embeddings(self):
        print("Saving embeddings to file...")
        save_path = args.save_path

        np.save("./../dumps/BERTMEAN-{}.npy".format(args.output), corpus_embeddings)
    def dataset_runner(self):
        return NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert', help='model name')
    parser.add_argument('--dataset_path', type=str, default='./../data/processed_data/processed_data.csv', help='path to dataset')
    parser.add_argument('--save_path', type=str, default='./../dumps/BERTMEAN-{}.npy'.format(args.output), help='path to save embeddings')
    parser.add_argument('--output', type=str, default='bert', help='output name')
    args = parser.parse_args()
    hfEmbeddings(args).create_embeddings()
    hfEmbeddings(args).save_embeddings()