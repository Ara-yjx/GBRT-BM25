from bisect import bisect_left
from indexer import *
from query_loader import queries, groundTruthExpand
# [ qid(group):string, queryTerms:string[], docno:string, relevance:int ] 

# print(groundTruthExpand[0])

# ii = InvertedIndex().load('trec45.ii')

# [ qid(group):string, features:(tf, df, docLen, docCount)[], relevance:int, docno:string ] 
def generateDataset(ii, groundTruthExpand=groundTruthExpand):
    dataset = []

    docCount = len(ii.docInfo)
    for docQueryPair in groundTruthExpand:
        # print(docQueryPair)
        docno = docQueryPair[2]

        if docno in ii.docCollection and len(docQueryPair[1]) != 0: # skip unrecorded doc and empty query
            features = []
            docId = ii.docCollection[docno]
            docLength = ii.docInfo[docId].length
            if(docLength == 0):
                print('doclength 0', docQueryPair)
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
                    # tf = 0
                    # for docTf in postings:
                    #     if docTf[0] == docId:
                    #         tf = docTf[1]
                    #     if docTf[0] >= docId:
                    #         continue
                    features.append((tf, df, docLength, docCount))

            dataset.append((docQueryPair[0], features, docQueryPair[3], docQueryPair[2],))
            if len(dataset) % 10000 == 0:
                print(len(dataset))

    return dataset



if __name__ == "__main__":
    print('Loading inverted index...')
    ii = InvertedIndex().load('trec45.ii')
    dataset = generateDataset(ii)

    with open('trec45.ds', 'wb') as f:
        pickle.dump(dataset, f)
