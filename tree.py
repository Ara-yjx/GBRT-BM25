import math
from random import shuffle, randrange, sample
import argparse
import numpy as np
from catboost import Pool, CatBoost
from indexer import *
from feature import *
from evaluator import dcg, ndcg, evalUnsorted, verbose
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

MAX_QUERY_SIZE = 1000


# switch the order of query terms for more data
def permute(dataset):
    permutedDataset = []
    for datarow in dataset:
        for p in permutations(datarow[1]):
            permutedDatarow = datarow[:]
            permutedDatarow[1] = p
            permutedDataset.append(permutedDatarow)
    return permutedDataset


def unpermute(pred, ngram):
    p = math.factorial(ngram) # number of permutation
    return [ np.mean(pred[i:i+p]) for i in range(0, len(pred), p) ]
            

# Group dataset[] by groupid[]
# Return dict: groupid -> dataset[:]
def group(dataset, groupid=None):
    if groupid == None:
        groupid = [ d[0] for d in dataset ]
    result = {}
    for ds, g in zip(dataset, groupid):
        if g not in result:
            result[g] = []
        result[g].append(ds)
    return result


def ungroup(groupDataset):
    result = []
    for ds in groupDataset.values():
        result += ds
    return result


# cannot exceed 1023 queries per group for GPU
# groupid(qid): xxx -> xxx:xx
def subgroup(dataset):
    result = []
    groupDataset = group(dataset)
    for ds in groupDataset.values():
        if len(ds) > MAX_QUERY_SIZE:
            subgroupCount = math.ceil((len(ds) / MAX_QUERY_SIZE)**2)
            for subgroupId in range(subgroupCount):
                subgroup = sample(ds, MAX_QUERY_SIZE)
                for datarow in subgroup:
                    newDatarow = datarow[:]
                    newDatarow[0] += ':' + str(subgroupId)
                    result.append(newDatarow)
        else:
            result += ds
    return result


# Split grouped dataset into train and test data of k folds
def kfold(groupDataset, datasetSize, k=3):
    # Split into folds
    folds = [ {} for i in range(k) ] # [ dataset[:] ]
    expectedFoldSize = datasetSize / k
    processedDatarow = 0
    for groupid, ds in groupDataset.items():
        targetFold = math.floor(processedDatarow / expectedFoldSize)
        folds[targetFold][groupid] = ds
        processedDatarow += len(ds)
        # print(groupid, '->', targetFold)

    print('fold size:', [ len(f) for f in folds ])
    # Yield test and train
    for testFold in range(k):
        train = {}
        test = {}
        for i in range(k):
            if i == testFold:
                test = folds[i]
            else:
                train.update(folds[i])
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset file from feature.py')
    parser.add_argument('--iteration', type=int, help='number of repetition for average result')
    parser.add_argument('--fold', type=int, help='k-fold validation')
    args = parser.parse_args()
    datasetFile = 'dataset.pickle' if args.dataset is None else args.dataset
    ITERATION = 1 if args.iteration is None else args.iteration
    FOLD = 4 if args.fold is None else args.fold

    with open(datasetFile, 'rb') as f:
        nGramDataset = pickle.load(f)
    # print('nGram:', list(map(len, nGramDataset)))

    def run(NGRAM):

        dataset = nGramDataset[NGRAM]
        # dataset = permute(dataset)
        shuffle(dataset)
        groupDataset = group(dataset)

        foldResults = []

        for groupTrain, groupTest in kfold(groupDataset, len(dataset), FOLD):
            if len(groupTrain) == 0 or len(groupTest) == 0:
                continue

            print('Training', NGRAM, 'gram,', 'fold', len(foldResults)+1)

            # Select 20% from each group as eval
            evalDataset = []
            for gid, ds in groupTrain.items():
                for i in range(int(len(ds) * 0.2)):
                    evalDataset.append( ds.pop(randrange(0, len(ds))) )
            # evalDataset = evalDataset[:MAX_QUERY_SIZE]
            eval_groupid, eval_data, eval_label = seperate(evalDataset)
            # print(eval_data)
            eval_pool = Pool(eval_data, eval_label, group_id=eval_groupid)

            train = ungroup(groupTrain)
            # train = subgroup(permute(train))
            train = permute(train)
            train_groupid, train_data, train_label = seperate(train)
            train_pool = Pool(train_data, train_label, group_id=train_groupid)
            print('len(train) =', len(train))        
            # train_permute, train_groupsize = permute(train)

            test = ungroup(groupTest)
            _, _, test_label_unpermute = seperate(test)
            test = permute(test)
            test_groupid, test_data, test_label = seperate(test)
            test_pool = Pool(test_data, test_label, group_id=test_groupid)
            print('len(test)  =', len(test))

            param = {
                'loss_function': 'StochasticRank:metric=NDCG;top=10',
                # 'loss_function': 'YetiRank',
                'eval_metric': 'NDCG:top=10',
                # 'custom_metric': ['NDCG:top=10;hints=skip_train~false','MAP:top=10;hints=skip_train~false'],
                # 'custom_metric': ['NDCG:top=10;hints=skip_train~false'],
                'metric_period': 10,
                'iterations': 1000,
                'depth': 4,
                'learning_rate': 0.01,
                # 'verbose': False,
                # 'task_type': "GPU",
            }
            model = CatBoost(param)
            model.fit(train_pool, eval_set=eval_pool)
            predicts = model.predict(test_pool)
            predicts_unpermute = unpermute(predicts, NGRAM)

            # Evaluate
            groupResult = [] # return values of evalUnsorted()
            groupSize = []

            # Zip, group, unzip : predicts[], rel[]
            grouped_pred_rel = group(zip(predicts_unpermute, test_label_unpermute), test_groupid)
            for groupid, pred_rel in grouped_pred_rel.items():
                preds, rel = tuple(zip(*pred_rel))

                groupResult.append( evalUnsorted(preds, rel) )
                groupSize.append( len(groupDataset[groupid]) )

            foldResult = np.average(groupResult, weights=groupSize, axis=0)
            foldResults.append(foldResult)
            print(foldResult)
            print()

        gramResult = np.average(foldResults, axis=0)
        print(gramResult)
        return gramResult


    allResult = [[],[],[],[],[]]
    for NGRAM in range(1,4):
        for t in range(ITERATION):
            allResult[NGRAM].append( run(NGRAM) )
        allResult[NGRAM] = (np.average(allResult[NGRAM], axis=0))
    for NGRAM in range(1,4):
        print(NGRAM, 'gram')
        verbose(allResult[NGRAM])
        print()
