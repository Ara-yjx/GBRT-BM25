from bisect import bisect_left
import argparse
from indexer import *
from query_loader import loadGroundtruth
# [ qid(group):string, queryTerms:string[], docno:string, relevance:int ] 


# [ <0>qid(group):string, 
#   <1>features:(tf, df)[],  <2>docLen, <3>avgDocLen, <4>docCount, 
#   <5>relevance:int, 
#   <6>docno:string ] 
def generateDataset(ii, groundtruth):
    dataset = []

    docCount = len(ii.docInfo)
    totalDocLen = 0

    for docQueryPair in groundtruth:
        # print(docQueryPair)eval
        docno = docQueryPair[2]

        if docno in ii.docCollection and len(docQueryPair[1]) != 0: # skip unrecorded doc and empty query
            docId = ii.docCollection[docno]
            docLen = ii.docInfo[docId].length
            if(docLen == 0):
                print('docLen 0', docQueryPair)
            totalDocLen += docLen

            # Retrieve tf and df
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

    # Add avgDocLen
    avgDocLen = totalDocLen / len(dataset)
    for i in dataset:
        i[3] = avgDocLen

    # Group by n-gram
    nGramDataset = [[],[],[],[],[]] # [0] is oversize gram (>4 gram)
    for datarow in dataset:
        if len(datarow[1]) <= 4:
            nGramDataset[ len(datarow[1]) ].append(datarow)
        else:
            nGramDataset[ 0 ].append(datarow)
            
    return nGramDataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--invertedindex', help='inverted index file from indexer.py')
    parser.add_argument('--query', help='queries title file')
    parser.add_argument('--relevance', help='relevance file of doc-query pair')
    parser.add_argument('--dataset', help='dataset file to output')
    args = parser.parse_args()
    iiFile = 'invertedindex.pickle' if args.invertedindex is None else args.invertedindex
    queryFile = 'title-queries.301-450' if args.query is None else args.query
    groundtruthFile = 'qrels.trec6-8.nocr' if args.relevance is None else args.relevance
    datasetFile = 'dataset.pickle' if args.dataset is None else args.dataset

    print('Loading inverted index...')
    ii = InvertedIndex().load(iiFile)
    groundtruth = loadGroundtruth(queryFile, groundtruthFile)
    dataset = generateDataset(ii, groundtruth)

    with open(datasetFile, 'wb') as f:
        pickle.dump(dataset, f)
