from math import log2

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

