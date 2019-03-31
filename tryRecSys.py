# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:53:09 2019

@author: ab computer
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import sklearn

M = pd.read_csv("Book1.csv")
M = pd.pivot_table(M, index = ["user"],columns = ["movie"],values = "rating", fill_value=0, dropna=True)
print(M)

#declaring k as global which can be changed by the user later
global k
k=3

#get cosine similarities for ratings matrix M; pairwise_distances returns the distances between ratings and hence
#similarities are obtained by subtracting distances from 1
#Cosine similarity matrix
cosine_sim = 1-pairwise_distances(M, metric="cosine")
print(pd.DataFrame(cosine_sim))

#This function finds k similar users given the user_id and ratings matrix M
#Note that the similarities are same as obtained via using pairwise_distances
def findksimilarusers(user_id, ratings, k=k):
    similarities=[]
    indices=[]
    
    model_knn = NearestNeighbors(metric = "cosine", algorithm = 'brute')
    print(model_knn)
    model_knn.fit(ratings)
    print(model_knn.fit(ratings))
    
    print(ratings.iloc[user_id-1, :].values)
    distances, indices = model_knn.kneighbors(ratings.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors = k+1)
    print(distances)
    print(indices)
    
    similarities = 1-distances.flatten()
    print(similarities)
    print ('{0} most similar users for User {1}:\n'.format(k,user_id))
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user_id:
            continue;

        else:
            print ('{0}: User {1}, with similarity of {2}'.format(i, indices.flatten()[i]+1, similarities.flatten()[i]))
            
    return similarities,indices

similarities,indices = findksimilarusers(4,M)
#print(similarities.shape)
print(indices.flatten())

def recommendations(user_id, similarities, indices, ratings):
    actual_indices = [x+1 for x in indices.flatten()]
#    print(actual_indices)
    sim_ind = dict(zip(actual_indices, similarities))
    #print(sim_ind)

    del sim_ind[user_id]
#    print(sim_ind)
    max_ind = [key for m in [max(sim_ind.values())] for key,val in sim_ind.items() if val == m]
#    print(max_ind)
    
    for maxSim_row in max_ind:
        similarUser = M.iloc[maxSim_row-1,:]
        currUser = M.iloc[user_id-1,:]   
#        print(currUser)
        currUser_zeros = currUser[currUser==0]
#        print(currUser_zeros)
#        print(similarUser)
        for k in currUser_zeros.keys():
            if similarUser[k] != 0:
                print('Recommendations for User {0}: {1}'.format(user_id, k))

recommendations(4, similarities, indices, M)

M_new = np.asarray([ (M.loc[4]).values, (M.loc[1]).values ])
M_new=pd.DataFrame(M_new)
print(M_new)

print(pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(M_new)))
    