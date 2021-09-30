from transformers import BertTokenizer, BertModel
import torch
import pickle
from tqdm import tqdm
import numpy as np
import sys
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')



input_file ='./../data/COLING/job_dataset.test.pkl'
output_file ='./../data/COLING/try1.npy'
output_file2 = './../data/COLING/try2.npy'

with open(input_file, 'rb') as f:
    df_test = pickle.load(f)


embed_mat =[]
embed_mat_2=[]

model.eval()

with torch.no_grad():
    for item in tqdm(df_test[:40]):

        inputs = tokenizer(item[0], padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        pt_t = last_hidden_states.squeeze(0)[0,:].detatch().clone()

        embed_mat.append(pt_t)
        pt_t = last_hidden_states.squeeze(0).mean(0).detach().clone()
        if pt_t.storage().data_ptr() == last_hidden_states.storage().data_ptr():
            break
        embed_mat_2.append(pt_t)
        # print(embed_mat[0].shape,embed_mat_2[0].shape) 
        print(sys.getsizeof(embed_mat))   
    embed_mat = np.array(embed_mat)
    np.save(output_file, embed_mat)

    del(embed_mat)
    embed_mat = np.array(embed_mat_2)
    np.save(output_file2,embed_mat_2)



