from bisect import bisect_left
from indexer import *
from query_loader import queries, groundTruthExpand
# [ qid(group):string, queryTerms:string[], docno:string, relevance:int ] 


# [ <0>qid(group):string, 
#   <1>features:(tf, df)[],  <2>docLen, <3>avgDocLen, <4>docCount, 
#   <5>relevance:int, 
#   <6>docno:string ] 
def generateDataset(ii, groundTruthExpand=groundTruthExpand):
    dataset = []

    docCount = len(ii.docInfo)
    totalDocLen = 0

    for docQueryPair in groundTruthExpand:
        # print(docQueryPair)
        docno = docQueryPair[2]

        if docno in ii.docCollection and len(docQueryPair[1]) != 0: # skip unrecorded doc and empty query
            docId = ii.docCollection[docno]
            docLen = ii.docInfo[docId].length
            if(docLen == 0):
                print('docLen 0', docQueryPair)
            totalDocLen += docLen

            termFeatures = []
            for term in docQueryPair[1]: # for each term
                if term in ii.bodyIndex: # skip term not in dictionary
                    posting = ii.bodyIndex[term] # :(docid[], tf[],)
                    df = len(posting[0])
                    # Find the TF of doc 
                    position = bisect_left(posting[0], docId)
                    if position < df and posting[0][position] == docId:
                        tf = posting[1][position]
                    else:
                        tf = 0
                    termFeatures.append((tf, df,))

            dataset.append([docQueryPair[0], termFeatures, docLen, 0, docCount, docQueryPair[3], docQueryPair[2],])
            if len(dataset) % 10000 == 0:
                print(len(dataset) / 1000, 'k...')

    avgDocLen = totalDocLen / len(dataset)
    for i in dataset:
        i[3] = avgDocLen

    return dataset



if __name__ == "__main__":
    print('Loading inverted index...')
    ii = InvertedIndex().load('trec45.ii')
    dataset = generateDataset(ii)

    with open('trec45.ds', 'wb') as f:
        pickle.dump(dataset, f)
