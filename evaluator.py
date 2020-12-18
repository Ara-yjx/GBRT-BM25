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


def precision(p, rel):
    return np.mean(rel[:p])


def avgPrecision(p, rel):
    precision = [ np.mean(rel[:i+1]) for i in range(p)]
    return np.mean(precision)


def evalUnsorted(predict, truth):
    preds_rank = np.argsort(predict)[::-1]
    preds_rel = np.array(truth)[preds_rank] # sorted relevance
    optimal_rel = sorted(truth, reverse=True)
   
    ndcg5  = ndcg(5 , preds_rel, optimal_rel)
    ndcg10 = ndcg(10, preds_rel, optimal_rel)
    ndcg20 = ndcg(20, preds_rel, optimal_rel)
    map5  = avgPrecision(5 , preds_rel)
    map10 = avgPrecision(10, preds_rel)
    map20 = avgPrecision(20, preds_rel)
    p5  = precision(5 , preds_rel)
    p10 = precision(10, preds_rel)
    p20 = precision(20, preds_rel)
    return (ndcg5, ndcg10, ndcg20, map5, map10, map20, p5, p10, p20)


def verbose(result):
    print('ndcg @ 5/10/20 :', result[0:3])
    print(' map @ 5/10/20 :', result[3:6])
    print('   p @ 5/10/20 :', result[6:9])
