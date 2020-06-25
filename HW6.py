import numpy as np
import random
import networkx as nx
from IPython.display import Image
import matplotlib.pyplot as plt
import csv

# read in the competition graph
G = nx.Graph()

with open('train_edges.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        if row[1]=='1':
            edge= row[0].split('-')
            # print(edge)
            G.add_edge(int(edge[0]),int(edge[1]))

# how many nodes in the network?
G.number_of_nodes()

# how many edges in the network?
G.number_of_edges()

# possible edges from the test file
Gsub = nx.Graph()

with open('sample_submission.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        if row[1]=='1' or row[1]=='0':
            edge= row[0].split('-')
            # print(edge)
            Gsub.add_edge(int(edge[0]),int(edge[1]))

# how many nodes in the network?           
Gsub.number_of_nodes()

# how many edges in the network?           
Gsub.number_of_edges()

# add the nodes from the test data to training network
G.add_nodes_from(Gsub.nodes)

# I have commented out all algorithms for this assignment.
# But you can uncomment each of them and get the prediction or valid/invalid edge in form of 1 or 0
with open('submission_file.csv', 'w') as csvfile:
    fieldnames = ['edge', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    with open('sample_submission.csv') as csvfile2:
        reader = csv.reader(csvfile2, delimiter=',')
        for row in reader:
            if row[1]=='1' or row[1]=='0':
                edge= row[0].split('-')
                
                i=int(edge[0]) # gets the left edge without quotes
                # print(i)
                j=int(edge[1]) # gets the right edge without guotes
                # print(j)
                
                # networkx algorithms (Uncomment each to get the prediction results)
                prediction = nx.resource_allocation_index(G, [(i, j)])
                # prediction = nx.jaccard_coefficient(G, [(i, j)])
                # prediction = nx.adamic_adar_index(G, [(i, j)])
                # prediction = nx.within_inter_cluster(G, [(i, j)])
                # prediction = nx.preferential_attachment(G, [(i, j)])

                for u, v, p in prediction:
                    # print('(%d, %d) -> %.8f' % (u, v, p))
                    if p>0:
                        p=1
                    else:
                        p=0
                
                    edg_out=str(u)+"-"+str(v)
                    writer.writerow({'edge': edg_out, 'label': p})
                    print(edg_out, p)




# GLM Model 
from statsmodels.formula.api import glm

X = sm.add_constant(prediction)
model = sm.glm(i, j, family=sm.families.Gamma())
results = model.fit()

print(results.summary())                   

                                   
# CONCLUSION: Among all of these algorithms, adamic_adar_index performed the best for me  with a score of 0.67368 on Kaggle. 

