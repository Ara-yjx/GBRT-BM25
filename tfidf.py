import pickle
import math
from indexer import *
from feature import *
from eval import dcg, ndcg

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
    ndcg5 = []
    ndcg10 = []
    for qid, docScores in queryScores.items():
        # Sort and extract item
        predict = [ x[1] for x in sorted(docScores, key=lambda x:x[0], reverse=True) ]
        truth   = [ x[1] for x in sorted(docScores, key=lambda x:x[1], reverse=True) ]
        # truth   = sorted((x[1] for x in docScores), reverse=True)
        ndcg5.append(ndcg(5, predict, truth))
        ndcg10.append(ndcg(10, predict, truth))

    return (ndcg5, ndcg10)



if __name__ == "__main__":
    
    with open('trec45.ds', 'rb') as f:
        nGramDataset = pickle.load(f)


    for i in range(1,5):
        print(i+1, 'gram')
        print('tf-idf')
        ndcg5, ndcg10 = computeScore(F_tfidf, nGramDataset[i])
        print('ndcg5 :', np.mean(ndcg5))
        print('ndcg10:', np.mean(ndcg10))
        print('bm25')
        ndcg5, ndcg10 = computeScore(F_bm25, nGramDataset[i])
        print('ndcg5 :', np.mean(ndcg5))
        print('ndcg10:', np.mean(ndcg10))
        print()

# Result:
# 1 gram
# tf-idf
# ndcg5 : 0.5173693318161369
# ndcg10: 0.47089861053882215
# bm25
# ndcg5 : 0.5701882838399329
# ndcg10: 0.555435606341711

# 2 gram
# tf-idf
# ndcg5 : 0.15054793396987923
# ndcg10: 0.1590246038705892
# bm25
# ndcg5 : 0.4133191177404347
# ndcg10: 0.4125877334139463

# 3 gram
# tf-idf
# ndcg5 : 0.1704988000671355
# ndcg10: 0.1611449925612211
# bm25
# ndcg5 : 0.45748160427386736
# ndcg10: 0.42326618208766437

# 4 gram
# tf-idf
# ndcg5 : 0.0
# ndcg10: 0.020012383435365922
# bm25
# ndcg5 : 0.4737194277664056
# ndcg10: 0.4494863553003425