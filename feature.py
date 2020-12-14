from indexer import *
from query_loader import queries, groundTruthExpand
# [ qid(group):string, queryTerms:string[], docno:string, score:int ] 

# print(groundTruthExpand[0])

ii = InvertedIndex()
with open('1m.ii', 'rb') as f:
    ii = pickle.load(f)

# [ qid(group):string, features:(tf, idf, docLen, docCount)[], score:int, docno:string ] 
dataset = []

for docQueryPair in groundTruthExpand:
    # print(docQueryPair)
    docno = docQueryPair[2]

    if docno in ii.docCollection: # skip unrecorded doc    
        features = []
        docId = ii.docCollection[docno]
        for term in docQueryPair[1]: # for each term
            if term in ii.bodyIndex: # skip term not in dictionary
                postings = ii.bodyIndex[term]
                df = len(postings)
                # Find the TF of doc 
                tf = 0
                for docTf in postings:
                    if docTf[0] == docId:
                        tf = docTf[1]
                    if docTf[0] >= docId:
                        continue
                docLength = ii.docInfo[docId].length
                docCount = len(ii.docInfo)
                features.append((tf, df, docLength, docCount))

        dataset.append((docQueryPair[0], features, docQueryPair[3], docQueryPair[2],))



# with open('1m.dataset', 'wb') as f:
    # pickle.dump(dataset, f)
