import math
from random import shuffle
import numpy as np
from catboost import Pool, CatBoost
from indexer import *
from feature import *
from evaluator import dcg, ndcg, evalUnsorted
from itertools import permutations

# avgDocLen and docCount is not included as a feature
# since they're the same across docs
# So, 
# relDocLen = docLen / avgDocLen (should be the same as docLen)
# df% = df / docCount
# should yield same result


# nGramDataset: 
# [ <0>qid(group):string, 
#   <1>features:(tf, df)[],  <2>docLen, <3>avgDocLen, <4>docCount, 
#   <5>relevance:int, 
#   <6>docno:string ] 


# switch the order of query terms for more data
def permute(dataset):
    permutedDataset = []
    for datarow in dataset:
        for p in permutations(datarow[1]):
            permutedDatarow = datarow[:]
            permutedDatarow[1] = p
            permutedDataset.append(permutedDatarow)
    return permutedDataset


# Group dataset[] by groupid[]
# Return dict: groupid -> dataset[:]
def group(dataset, groupid):
    result = {}
    for ds, g in zip(dataset, groupid):
        if g not in result:
            result[g] = []
        result[g].append(ds)
    return result


# Split grouped dataset into train and test data of k folds
def kfold(groupDataset, datasetSize, k=3):
    # Split into folds
    folds = [ [] for i in range(k) ] # [ dataset[:] ]
    expectedFoldSize = datasetSize / k
    processedDatarow = 0
    for groupid, ds in groupDataset.items():
        targetFold = math.floor(processedDatarow / expectedFoldSize)
        folds[targetFold] += ds
        processedDatarow += len(ds)
        # print(groupid, '->', targetFold)

    print('fold size:', [ len(f) for f in folds ])
    # Yield test and train
    for testFold in range(k):
        train = []
        test = []
        for i in range(k):
            if i == testFold:
                test = folds[i]
            else:
                train += folds[i]
        yield train, test



# Expand by query length, seperate data & label & groupid
def seperate(dataset):
    groupid = [ d[0] for d in dataset]
    data = []
    label = [ d[5] for d in dataset]
    for datarow in dataset:
        expandedData = []
        for feature in datarow[1]:
            expandedData += list(feature)
        expandedData.append(datarow[2])
        data.append(expandedData)
    return groupid, data, label



if __name__ == "__main__":
    with open('trec45.ds', 'rb') as f:
        nGramDataset = pickle.load(f)
    print('nGram:', list(map(len, nGramDataset)))

    dataset = nGramDataset[2]
    dataset = permute(dataset)
    shuffle(dataset)
    groupDataset = group(dataset, (d[0] for d in dataset))

    foldResults = []

    for train, test in kfold(groupDataset, len(dataset), 4):

        print('len(train) =', len(train))
        print('len(test)  =', len(test))

        train_groupid, train_data, train_label = seperate(train)
        test_groupid, test_data, test_label = seperate(test)
        train_pool = Pool(train_data, train_label, group_id=train_groupid)
        test_pool = Pool(test_data, test_label, group_id=test_groupid)

        param = {
            # 'loss_function': 'StochasticRank:metric=NDCG;top=10',
            'loss_function': 'YetiRank',
            # 'learn_metrics': 'NDCG',
            'custom_metric': ['NDCG:top=10;hints=skip_train~false','MAP:top=10;hints=skip_train~false'],
            'iterations': 300,
            'depth': 4,
            'learning_rate': 0.01
            # 'verbose': False,
            # 'task_type': "GPU",
        }
        model = CatBoost(param)
        model.fit(train_pool)
        predicts = model.predict(test_pool)

        # Evaluate
        groupResult = [] # return values of evalUnsorted()
        groupSize = []

        # Zip, group, unzip : predicts[], rel[]
        grouped_pred_rel = group(zip(predicts, test_label), test_groupid)
        for groupid, pred_rel in grouped_pred_rel.items():
            preds, rel = tuple(zip(*pred_rel))

            groupResult.append( evalUnsorted(preds, rel) )
            groupSize.append( len(groupDataset[groupid]) )

        # print(groupResult)

        foldResult = np.average(groupResult, weights=groupSize, axis=0)
        foldResults.append(foldResult)
        print(foldResult)

        print()

    gramResult = np.average(foldResults, axis=0)
    print(gramResult)

    # [0.56485243 0.55001862]


    # [0.38388007 0.37443232]

