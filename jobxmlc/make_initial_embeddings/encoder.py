from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm
import numpy as np

import os
from jobxmlc.registry import ENCODER, DATA_FILTER, register
class initialEmbeddings:
    def __init__(self):
        pass
    def preprocessing_data(self):
        return NotImplementedError
    def tokenizer(self):
        return NotImplementedError
    def create_embeddings(self):
        return NotImplementedError
    def save_embeddings(self,data_name, data_embeddings):
        print(f"Saving {data_name} embeddings to file...")
        save_path = args.embeddings_save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, f"{data_name}_embeddings.npy"), data_embeddings)
    def embeddings_runner(self):
        """
        run the embeddings pipeline for the dataset"""
        data_directory_path = self.args.dataset_path
        train_data_path = os.path.join(data_directory_path,'trn_X.txt')
        test_data_path = os.path.join(data_directory_path,'tst_X.txt')
        labels_path = os.path.join(data_directory_path,'Y.txt')

        for data_name, data in zip(['train','test','label'],[train_data_path, test_data_path, labels_path]):
            if not os.path.exists(data):
                raise FileNotFoundError("File {} not found".format(data))
            data_embeddings = self.create_embeddings(data)
            self.save_embeddings(self,data_name, data_embeddings) 

@register(_name="sentence-encoder", _type=ENCODER)
class SentenceTransformerEmbeddings(initialEmbeddings):
    def __init__(self,args):
        self.args = args
    def preprocessing_data(self,corpus):
        data_filter = DATA_FILTER[self.args.data_filter.name](self.args.data_filter.params)
        data_filter.preprocessing_data(corpus)
    def create_embeddings(self):
        model_name = self.args.model
        embedder = SentenceTransformer(model_name)
        dataset_path = self.args.dataset_path
        f = open(dataset_path,'r').readlines()
        dataset=[x.split('\n')[0] for x in f]
        print("First row of dataset:", dataset[0])
        print("Encoding {} sentences using {}.....\n".format(len(dataset), args.model))
        embedding_size = embedder.get_sentence_embedding_dimension()
        corpus_embeddings = np.empty((0, embedding_size)) 
        dataset = self.preprocessing_data(dataset)
        for batch in tqdm(np.array_split(dataset, self.batch_size)):
            embs = embedder.encode(batch)
            corpus_embeddings = np.vstack((corpus_embeddings, embs))
        return corpus_embeddings





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert', help='model name')
    parser.add_argument('--dataset_path', type=str, default='./../data/processed_data/processed_data.csv', help='path to dataset')
    parser.add_argument('--save_path', type=str, default='./../dumps/BERTMEAN-{}.npy'.format(args.output), help='path to save embeddings')
    parser.add_argument('--output', type=str, default='bert', help='output name')
    args = parser.parse_args()
    hfEmbeddings(args).create_embeddings()
    hfEmbeddings(args).save_embeddings()