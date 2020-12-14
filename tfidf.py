import pickle
import math
from indexer import *
from feature import *

# [ qid(group):string, features:(tf, idf, docLen, docCount)[], score:int, docno:string ] 
# with open('1m.dataset', 'rb') as f:
    # dataset = pickle.load(f)

# qid -> (score, docno,)[]
queryScores = {}

for datarow in dataset:
    qid = datarow[0]
    score = 0
    for feature in datarow[1]:
        # tfidf = tf * log(D/df) / docLen
        if feature[0] != 0:
            print('feature0')

        score += feature[0] * math.log(feature[3] / feature[1]) / feature[2]
    if qid not in queryScores:
        queryScores[qid] = []
    queryScores[qid].append((score, datarow[3],))

# print(queryScores)

# for qid, scores in queryScores.items():
    
