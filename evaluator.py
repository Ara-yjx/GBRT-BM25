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


def avgPrecision(p, rel):
    precision = [ np.mean(rel[:i+1]) for i in range(p)]
    return np.mean(precision)


def evalUnsorted(predict, truth):
    preds_rank = np.argsort(predict)[::-1]
    preds_rel = np.array(truth)[preds_rank] # sorted relevance
    optimal_rel = sorted(truth, reverse=True)
   
    ndcg5  = ndcg(5 , preds_rel, optimal_rel)
    ndcg10 = ndcg(10, preds_rel, optimal_rel)
    ndcg25 = ndcg(25, preds_rel, optimal_rel)
    map5  = avgPrecision(5 , preds_rel)
    map10 = avgPrecision(10, preds_rel)
    map25 = avgPrecision(25, preds_rel)
    return (ndcg5, ndcg10, ndcg25, map5, map10, map25, )
