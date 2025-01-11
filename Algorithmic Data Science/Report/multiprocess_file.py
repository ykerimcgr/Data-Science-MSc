import pandas as pd
import math

data = pd.read_csv( 'data/data2024.csv', index_col=0 ).values

def dot(v1,v2):
    total=0
    for i in range(0,len(v1)):
        total+=v1[i]*v2[i]
        
    return total

def cosine(v1,v2):
    
    dot_product = dot(v1,v2)
    denominator = math.sqrt(dot(v1,v1)*dot(v2,v2))
    return 0 if denominator == 0 else dot_product / denominator

def work_that_CPU(num_of_iter ):

    
    num_of_docs = 15
    num_of_rows = 200
    
    for i in range(1, num_of_docs):
        for j in range(i + 1, num_of_docs):
            doc1_subset = data[: num_of_rows, i]
            doc2_subset = data[: num_of_rows, j]

            for k in range( num_of_iter ):
                cos_sim = cosine( doc1_subset, doc2_subset )