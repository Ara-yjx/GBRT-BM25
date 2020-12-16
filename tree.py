import math
from random import shuffle
# from itertools import groupby
from catboost import Pool, CatBoost
from indexer import *
from feature import *
from eval import dcg, ndcg, evalSort


# avgDocLen and docCount is not included as a feature
# since they're the same across docs
# So, 
# relDocLen = docLen / avgDocLen (should be the same as docLen)
# df% = df / docCount
# should yield same result


# [ <0>qid(group):string, 
#   <1>features:(tf, df)[],  <2>docLen, <3>avgDocLen, <4>docCount, 
#   <5>relevance:int, 
#   <6>docno:string ] 
with open('trec45.ds', 'rb') as f:
    nGramDataset = pickle.load(f)
print('nGram:', list(map(len, nGramDataset)))

dataset = nGramDataset[1]
shuffle(dataset)


# Group dataset[] by groupid[]
# Return dict: groupid -> dataset[:]
def group(dataset, groupid):
	result = {}
	for ds, g in zip(dataset, groupid):
		if g not in result:
			result[g] = []
		result[g].append(ds)
	return result

groupDataset = group(dataset, (d[0] for d in dataset))
# groupDataset = groupby(dataset, lambda x:x[0])


# Split train and test, by qid
train = []
test = []
expectTrainSize = math.floor(len(dataset) * 0.8)
for qid, ds in groupDataset.items():
	if len(train) < expectTrainSize:
		train += ds
	else:
		test += ds


print('len(train) =', len(train))
print('len(test)  =', len(test))




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

# print(data[:5])



train_groupid, train_data, train_label = seperate(train)
test_groupid, test_data, test_label = seperate(test)

train_pool = Pool(train_data, train_label, group_id=train_groupid)
test_pool = Pool(test_data, test_label, group_id=test_groupid)


param = {
    'loss_function': 'YetiRank',
	'custom_metric': ['NDCG'],
    'iterations': 10,
    'depth': 5,
	'verbose': True,
}
model = CatBoost(param)
model.fit(train_pool)

predicts = model.predict(test_pool)

# print(predicts[:5])

# Evaluate
# [ groupid, #doc-query-pair, ndcg5, ndcg10 ] # eval of the field
foldResult = []

# Zip, group, unzip : predicts[], rel[]
grouped_pred_rel = group(zip(predicts, test_label), test_groupid)
for groupid, pred_rel in grouped_pred_rel.items():
	preds, rel = tuple(zip(*pred_rel))

	result = evalSort(preds, rel)
	foldResult.append([ groupid, len(groupDataset[groupid]), *result])

print(foldResult)





