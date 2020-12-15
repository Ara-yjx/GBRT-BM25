from indexer import regularizeText
# Load query and ground truth. Read from 'qrels.trec6-8.nocr' 


# [ qid:string, docno:string, score:int ]
groundTruthRaw = []

# qid:string -> terms:string[]
queries = {} 
# # [ queryTerms:string[], (docid:int, score:int,)[] ] per query
# groundTruthAggregate = []

# [ qid(group):string, queryTerms:string[], docno:string, score:int ] per doc-query pair
groundTruthExpand = []

# Load query
with open('../title-queries.301-450', 'r') as f:
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        spacePosition = line.find(' ')
        qid = line[:spacePosition]
        terms = line[spacePosition+1:-1]
        queries[qid] = regularizeText(terms)


# Load ground truth
with open('../qrels.trec6-8.nocr', 'r') as f:
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        segments = line.split(' ')
        # qid, 0, docno, score
        groundTruthRaw.append((segments[0], segments[2], segments[3],))
        groundTruthExpand.append((segments[0], queries[segments[0]], segments[2], int(segments[3]),))
