from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm
import numpy as np
import fasttext
import os
from jobxmlc.registry import ENCODER, DATA_FILTER_REGISTRY, register


class initialEmbeddings:
    def __init__(self):
        pass
    def tokenizer(self):
        return NotImplementedError
    
    def create_embeddings(self):
        return NotImplementedError
    
    def preprocessing_data(self,corpus):
        data_filter = DATA_FILTER_REGISTRY[self.data_filter['name']](self.data_filter['params'])
        data_filter.preprocessing_data(corpus)
    
    def save_embeddings(self,data_name, data_embeddings):
        print(f"Saving {data_name} embeddings to file...")
        save_path = self.params['embeddings_save_dir']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, f"{data_name}_embeddings.npy"), data_embeddings)
    
    def load_file(self,data_path):
        f = open(data_path,'r').readlines()
        dataset=[x.split('\n')[0] for x in f]
        print("First row of dataset:", dataset[0])
        print("Encoding {} sentences using {}.....\n".format(len(dataset), self.params['model_name']))
        return dataset
    
    def embeddings_runner(self):
        """
        run the embeddings pipeline for the dataset"""
        data_directory_path = self.params['dataset_dir']
        train_data_path = os.path.join(data_directory_path,'trn_X.txt')
        test_data_path = os.path.join(data_directory_path,'tst_X.txt')
        labels_path = os.path.join(data_directory_path,'Y.txt')

        for data_name, data_path in zip(['train','test','label'],[train_data_path, test_data_path, labels_path]):
            if not os.path.exists(data_path):
                raise FileNotFoundError("File {} not found".format(data_path))
            data = self.load_file(data_path)
            data_embeddings = self.create_embeddings(data)
            self.save_embeddings(data_name, data_embeddings) 

@register(_name="sentence-transformer-encoder", _type=ENCODER)
class SentenceTransformerEncoder(initialEmbeddings):
    def __init__(self,params,**kwargs):
        self.params = params
        self.data_filter = kwargs.get("data_filter", None)
    def create_embeddings(self,data):
        model_name = self.params['model_name']
        embedder = SentenceTransformer(model_name)
        embedding_size = embedder.get_sentence_embedding_dimension()
        corpus_embeddings = np.empty((0, embedding_size)) 
        if self.data_filter:
            dataset = self.preprocessing_data(data)
        else:
            dataset = data
        for batch in tqdm(np.array_split(dataset, self.params['data_split_number'])):
            embs = embedder.encode(batch)
            corpus_embeddings = np.vstack((corpus_embeddings, embs))
        return corpus_embeddings

@register(_name='fasttext', _type=ENCODER)
class FasttextEncoder(initialEmbeddings):
    def __init__(self,params):
        self.params = params
    def create_embeddings(self,data):
        modelpath = self.params.model_path
        model = fasttext.load_model(modelpath)
        embedding_size = model.get_dimension()
        corpus_embeddings = np.empty((0, embedding_size)) 
        dataset = self.preprocessing_data(data)
        vectors =[]
        for i in range(len(dataset)):
            vectorlist=[model[word] for word in dataset.split()]
            jobvector=np.mean(vectorlist, axis=0)
            corpus_embeddings = np.vstack((corpus_embeddings, jobvector))
        return vectors