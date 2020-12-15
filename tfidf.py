import pickle
import math
from indexer import *
from feature import *
from eval import dcg, ndcg

# [ <0>qid(group):string, 
#   <1>features:(tf, df)[],  <2>docLen, <3>avgDocLen, <4>docCount, 
#   <5>relevance:int, 
#   <6>docno:string ] 
with open('trec45.ds', 'rb') as f:
    dataset = pickle.load(f)

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
def computeScore(formula):

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
    ndcg5 = []
    ndcg10 = []
    for qid, docScores in queryScores.items():
        # Sort and extract item
        predict = [ x[1] for x in sorted(docScores, key=lambda x:x[0], reverse=True) ]
        truth   = [ x[1] for x in sorted(docScores, key=lambda x:x[1], reverse=True) ]
        # truth   = sorted((x[1] for x in docScores), reverse=True)
        ndcg5.append(ndcg(5, predict, truth))
        ndcg10.append(ndcg(10, predict, truth))

    print('ndcg5 :', np.mean(ndcg5))
    print('ndcg10:', np.mean(ndcg10))
    return (ndcg5, ndcg10)

print('tf-idf')
computeScore(F_tfidf)
print('bm25')
computeScore(F_bm25)

# print(queryScores)

# for qid, scores in queryScores.items():
    
