from math import log2
import numpy as np

# rel: float[] of relevance score
def dcg(rel):
    if len(rel) == 0:
        return 0
    total = rel[0]
    for i in range(1, len(rel)):
        total += rel[i] / log2(i+1)
    return total


def ndcg(p, predict, truth):
    idcg = dcg(truth[:p])
    if idcg == 0:
        return 0
    return dcg(predict[:p]) / idcg


def evalSort(predict, truth):
    preds_rank = np.argsort(predict)[::-1]
    preds_rel = np.array(truth)[preds_rank] # sorted relevance
    test_rel = sorted(truth, reverse=True)

    print('NDCG5 :', ndcg(5,preds_rel,test_rel))
    print('NDCG10:', ndcg(10,preds_rel,test_rel))
    
    return ndcg(5,preds_rel,test_rel), ndcg(10,preds_rel,test_rel)
