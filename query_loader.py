# Load query and ground truth. Read from 'qrels.trec6-8.nocr' and 

# [ qid:string, terms:string[] ]
queriesRaw = []
# [ qid:string, docno:string, score:int ]
groundTruthRaw = []

# qid:string -> terms:string[]
queries = {} 
# # [ queryTerms:string[], (docid:int, score:int,)[] ] per query
# groundTruth = []

# [ queryTerms:string[], docno:string, score:int ] per doc-query pair
groundTruthExpand = []

# Load query
with open('../title-queries.301-450', 'r') as f:
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        segments = line.split()
        queriesRaw.append(segments)
        queries[segments[0]] = tuple(segments[1:])


# Load ground truth
with open('../qrels.trec6-8.nocr', 'r') as f:
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        segments = line.split(' ')
        # qid, 0, docno, score
        groundTruthRaw.append((segments[0], segments[2], segments[3],))
        groundTruthExpand.append((queries[segments[0]], segments[2], int(segments[3]),))
