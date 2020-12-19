# Installation

Recommand using Python 3.6.9.   
Also available with Python 3.8.6 (as on csil).


Install dependencies  
`$ pip install catboost==0.24.3 krovetz==1.0.2`

Copy necessary data file into project directory
1.  TREC disk4-5 (must be in valid xml format)  
    Unzip csil file `/cs/sandbox/faculty/tyang/290N/trec-disk4-5_processed.zip`

2.  Query title  
    From hw1-starter: `title-queries.301-450`

3.  Relevance score  
    From hw1-starter: `qrels.trec6-8.nocr`


# Architecture and Commands

1.  Generate inverted index  
    Trec45 data -> indexer.py -> inverted index      
    `$ python indexer.py --trec trec-disk4-5_processed.xml`

2.  Generate feature data for training  
    inverted index, query title, relevance score -> feature.py -> feature dataset  
    `$ python feature.py --query title-queries.301-450 --relevance qrels.trec6-8.nocr`

3.  Train models and evaluate  
    feature dataset -> tree.py  
    `$ python tree.py` 

    Compute BM25 as baseline  
    feature dataset -> tfidf.py  
    `$ python tfidf.py`


# Command Line Arguments

| Argument | Description | Default |
| -- | -- | -- |
| indexer.py | |
| --trec          | TREC disk 4-5 file | `trec-disk4-5_processed.xml` |
| --invertedindex | inverted index file to output | `invertedindex.pickle` |
| feature.py | |
| --invertedindex | inverted index file from indexer.py | `invertedindex.pickle` 
| --query         | queries title file | `title-queries.301-450` |
| --relevance     | relevance file of doc-query pair | `qrels.trec6-8.nocr` |
| --dataset       | dataset file to output | `dataset.pickle` |
| tfidf.py | |
| --dataset       | dataset file from feature.py | `dataset.pickle` |
| tree.py | |
| --dataset       | dataset file from feature.py | `dataset.pickle` |
| --iteration     | (int) number of repetition for average result | `1` |
| --fold          | (int) k-fold validation | `4` |


Format:
`$ python xxx.py --arg1name arg1value --arg2name arg2value ...`
