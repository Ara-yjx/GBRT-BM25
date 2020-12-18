import pickle
import math
import numpy as np
import argparse
from indexer import *
from feature import *
from evaluator import evalUnsorted, verbose

# dataset:
# [ <0>qid(group):string, 
#   <1>features:(tf, df)[],  <2>docLen, <3>avgDocLen, <4>docCount, 
#   <5>relevance:int, 
#   <6>docno:string ] 

def F_tfidf(tf, df, docLen, avgDocLen, docCount):
    if df == 0 or docLen == 0:
        print('!!! Zero division:', df, docLen)
    return tf * math.log(docCount / df) / docLen

def F_bm25(tf, df, docLen, avgDocLen, docCount):
    B = 0.75
    K = 1.2
    idfPart = math.log( (docCount-df+0.5) / (df+0.5) )
    tfPart = tf*(K+1) / (tf + K * (1-B+B*docLen/avgDocLen))
    return idfPart * tfPart

# formula:  <0>tf, <1>df, <2>docLen, <3>avgDocLen, <4>docCount, 
def computeScore(formula, dataset):
    # qid -> (score, relevance, docno)[]
    queryScores = {}

    for datarow in dataset:
        qid = datarow[0]
        score = 0
        for feature in datarow[1]:
            score += formula(*feature, *datarow[2:5])
            
        if qid not in queryScores:
            queryScores[qid] = []
        queryScores[qid].append((score, datarow[5], datarow[6],))

    # Rank and Evaluate
    groupResults = []
    groupWeights = []
    for qid, docScores in queryScores.items():
        # Sort and extract item
        predict = [ x[1] for x in sorted(docScores, key=lambda x:x[0], reverse=True) ]
        truth   = [ x[1] for x in sorted(docScores, key=lambda x:x[1], reverse=True) ]
        # truth   = sorted((x[1] for x in docScores), reverse=True)
        groupResults.append(evalUnsorted(predict, truth))
        groupWeights.append(len(docScores))
    return np.average(groupResults, weights=groupWeights, axis=0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset file from feature.py')
    args = parser.parse_args()
    datasetFile = 'dataset.pickle' if args.dataset is None else args.dataset
    
    with open(datasetFile, 'rb') as f:
        nGramDataset = pickle.load(f)

    for i in range(1,4):
        print(i, 'gram')
        # print('tf-idf')
        # verbose( computeScore(F_tfidf, nGramDataset[i]) )
        # print('bm25')
        verbose( computeScore(F_bm25, nGramDataset[i]) )
        print()


# Result: 
# 1 gram
# tf-idf
# ndcg @ 5/10/20 : [0.36206385 0.32825502 0.30206731]
#  map @ 5/10/20 : [0.36893413 0.2979647  0.22252176]
#    p @ 5/10/20 : [0.27154981 0.1943827  0.13398666]
# bm25
# ndcg @ 5/10/20 : [0.3563609  0.36088165 0.34030086]
#  map @ 5/10/20 : [0.33977975 0.30981853 0.26329994]
#    p @ 5/10/20 : [0.30553647 0.26825621 0.1857547 ]

# 2 gram
# tf-idf
# ndcg @ 5/10/20 : [0.11316327 0.11098704 0.10985822]
#  map @ 5/10/20 : [0.12534775 0.10955492 0.09826392]
#    p @ 5/10/20 : [0.10347871 0.09381523 0.08354938]
# bm25
# ndcg @ 5/10/20 : [0.27174384 0.24711309 0.22783634]
#  map @ 5/10/20 : [0.30674497 0.25645639 0.21504276]
#    p @ 5/10/20 : [0.23066388 0.19492841 0.15609781]

# 3 gram
# tf-idf
# ndcg @ 5/10/20 : [0.13644259 0.12080805 0.11756597]
#  map @ 5/10/20 : [0.1477735  0.12133183 0.10497895]
#    p @ 5/10/20 : [0.11333696 0.09263763 0.09096014]
# bm25
# ndcg @ 5/10/20 : [0.26034796 0.21479248 0.19817659]
#  map @ 5/10/20 : [0.32441909 0.24932233 0.19884574]
#    p @ 5/10/20 : [0.21554023 0.15822365 0.1423286 ]
