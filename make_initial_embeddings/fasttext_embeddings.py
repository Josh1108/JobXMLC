import pandas as pd
import sister
import numpy as np

# file= "../data/COLING/DRIVECondensedData/trn_point_embs.npy"
# filedata=np.load(file)
# print(filedata.shape)

# sentence_embedding = sister.MeanEmbedding(lang="en")
# file1 = open('../data/COLING/trn_X.txt', 'r')
# Lines = file1.readlines()
# print(len(Lines), Lines[0])
# print("making vector embeddings")
# vector = [sentence_embedding(line) for line in Lines]
# print(len(vector), len(vector[0]))
# arr = np.array(vector)
# np.save("../data/COLING/DRIVECondensedData/trn_point_embs.npy",arr)

sentence_embedding = sister.MeanEmbedding(lang="en")
file1 = open('../data/COLING/Y.txt', 'r')
Lines = file1.readlines()
print(len(Lines), Lines[0])
vector = [sentence_embedding(line) for line in Lines]
print(len(vector), len(vector[0]))
arr = np.array(vector)
np.save("../data/COLING/DRIVECondensedData/label_embs.npy",arr)

#sentence = "requirements preferably least 1 year relevant exp providing support director admin manager assisting daily day day activities carry indoor sales network tele marketing exp willing learn also generate quotation invoicing etc sales coordination functions"
#vector = sentence_embedding(sentence)
#print(vector,vector.shape)