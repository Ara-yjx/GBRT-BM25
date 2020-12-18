from indexer import regularizeText
# Load query and ground truth. Read from 'qrels.trec6-8.nocr' 

def loadGroundtruth(queryFile, groundtruthFile):

    # qid:string -> terms:string[]
    queries = {} 

    # [ qid(group):string, queryTerms:string[], docno:string, score:int ] per doc-query pair
    groundtruth = []

    # Load query
    with open(queryFile, 'r') as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            spacePosition = line.find(' ')
            qid = line[:spacePosition]
            terms = line[spacePosition+1:-1]
            queries[qid] = regularizeText(terms)

    # Load ground truth
    with open(groundtruthFile, 'r') as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            segments = line.split(' ')
            # qid, 0, docno, score
            groundtruth.append((segments[0], queries[segments[0]], segments[2], int(segments[3]),))

    return groundtruth