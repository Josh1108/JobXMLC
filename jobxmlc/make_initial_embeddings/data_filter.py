from jobxmlc.registry import ENCODER, DATA_FILTER, register
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


@register(_name="tf-idf", _type=DATA_FILTER)
class tfidf10:
    def __init__(self,params):
        self.number_of_words = params['number_of_words']
    def word_tokenizer(self, text):
        """
        tokenize a sentence into words
        """
        return text.split() 
    def preprocessing_data(self,corpus):
        """
        remove the 10 most unimportant words in a job based on tf-idf scores
        """
        vectorizer = TfidfVectorizer(tokenizer=self.word_tokenizer)
        X = vectorizer.fit_transform(corpus)
        Y = vectorizer.get_feature_names()
        full_corpus_new = []
        for i in tqdm(range(len(corpus))):
            dicti = dict(zip(vectorizer.get_feature_names(), X.toarray()[i]))
            lisi=[]
            for text in corpus[i].split():
                lisi.append(dicti[text])
            lisi_index = sorted(range(len(lisi)), key=lambda k: lisi[k])
            lisi_index=lisi_index[:self.number_of_words]
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