import numpy as np
import string
import argparse
from collections import Counter
import pickle
import xml.sax
import krovetz
ks = krovetz.PyKrovetzStemmer()
# STOPWORDS = {"a", "about", "above", "according", "across", "after", "afterwards", "again", "against", "albeit", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anywhere", "apart", "are", "around", "as", "at", "av", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "canst", "certain", "cf", "choose", "contrariwise", "cos", "could", "cu", "day", "do", "does", "doesn't", "doing", "dost", "doth", "double", "down", "dual", "during", "each", "either", "else", "elsewhere", "enough", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "except", "excepted", "excepting", "exception", "exclude", "excluding", "exclusive", "far", "farther", "farthest", "few", "ff", "first", "for", "formerly", "forth", "forward", "from", "front", "further", "furthermore", "furthest", "get", "go", "had", "halves", "hardly", "has", "hast", "hath", "have", "he", "hence", "henceforth", "her", "here", "hereabouts", "hereafter", "hereby", "herein", "hereto", "hereupon", "hers", "herself", "him", "himself", "hindmost", "his", "hither", "hitherto", "how", "however", "howsoever", "i", "ie", "if", "in", "inasmuch", "inc", "include", "included", "including", "indeed", "indoors", "inside", "insomuch", "instead", "into", "inward", "inwards", "is", "it", "its", "itself", "just", "kind", "kg", "km", "last", "latter", "latterly", "less", "lest", "let", "like", "little", "ltd", "many", "may", "maybe", "me", "meantime", "meanwhile", "might", "moreover", "most", "mostly", "more", "mr", "mrs", "ms", "much", "must", "my", "myself", "namely", "need", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "nonetheless", "noone", "nope", "nor", "not", "nothing", "notwithstanding", "now", "nowadays", "nowhere", "of", "off", "often", "ok", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "own", "per", "perhaps", "plenty", "provide", "quite", "rather", "really", "round", "said", "sake", "same", "sang", "save", "saw", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "seldom", "selves", "sent", "several", "shalt", "she", "should", "shown", "sideways", "since", "slept", "slew", "slung", "slunk", "smote", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "spake", "spat", "spoke", "spoken", "sprang", "sprung", "stave", "staves", "still", "such", "supposing", "than", "that", "the", "thee", "their", "them", "themselves", "then", "thence", "thenceforth", "there", "thereabout", "thereabouts", "thereafter", "thereby", "therefore", "therein", "thereof", "thereon", "thereto", "thereupon", "these", "they", "this", "those", "thou", "though", "thrice", "through", "throughout", "thru", "thus", "thy", "thyself", "till", "to", "together", "too", "toward", "towards", "ugh", "unable", "under", "underneath", "unless", "unlike", "until", "up", "upon", "upward", "upwards", "us", "use", "used", "using", "very", "via", "vs", "want", "was", "we", "week", "well", "were", "what", "whatever", "whatsoever", "when", "whence", "whenever", "whensoever", "where", "whereabouts", "whereafter", "whereas", "whereat", "whereby", "wherefore", "wherefrom", "wherein", "whereinto", "whereof", "whereon", "wheresoever", "whereto", "whereunto", "whereupon", "wherever", "wherewith", "whether", "whew", "which", "whichever", "whichsoever", "while", "whilst", "whither", "who", "whoa", "whoever", "whole", "whom", "whomever", "whomsoever", "whose", "whosoever", "why", "will", "wilt", "with", "within", "without", "worse", "worst", "would", "wow", "ye", "yet", "year", "yippee", "you", "your", "yours", "yourself", "yourselves"}
STOPWORDS = {'', 'doesn', 't', 's', 'a', 'about', 'above', 'according', 'across', 'after', 'afterwards', 'again', 'against', 'albeit', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'apart', 'are', 'around', 'as', 'at', 'av', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'canst', 'certain', 'cf', 'choose', 'contrariwise', 'cos', 'could', 'cu', 'day', 'do', 'does', 'doing', 'dost', 'doth', 'double', 'down', 'dual', 'during', 'each', 'either', 'else', 'elsewhere', 'enough', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'except', 'excepted', 'excepting', 'exception', 'exclude', 'excluding', 'exclusive', 'far', 'farther', 'farthest', 'few', 'ff', 'first', 'for', 'formerly', 'forth', 'forward', 'from', 'front', 'further', 'furthermore', 'furthest', 'get', 'go', 'had', 'halves', 'hardly', 'has', 'hast', 'hath', 'have', 'he', 'hence', 'henceforth', 'her', 'here', 'hereabouts', 'hereafter', 'hereby', 'herein', 'hereto', 'hereupon', 'hers', 'herself', 'him', 'himself', 'hindmost', 'his', 'hither', 'hitherto', 'how', 'however', 'howsoever', 'i', 'ie', 'if', 'in', 'inasmuch', 'inc', 'include', 'included', 'including', 'indeed', 'indoors', 'inside', 'insomuch', 'instead', 'into', 'inward', 'inwards', 'is', 'it', 'its', 'itself', 'just', 'kind', 'kg', 'km', 'last', 'latter', 'latterly', 'less', 'lest', 'let', 'like', 'little', 'ltd', 'many', 'may', 'maybe', 'me', 'meantime', 'meanwhile', 'might', 'moreover', 'most', 'mostly', 'more', 'mr', 'mrs', 'ms', 'much', 'must', 'my', 'myself', 'namely', 'need', 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'nonetheless', 'noone', 'nope', 'nor', 'not', 'nothing', 'notwithstanding', 'now', 'nowadays', 'nowhere', 'of', 'off', 'often', 'ok', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'own', 'per', 'perhaps', 'plenty', 'provide', 'quite', 'rather', 'really', 'round', 'said', 'sake', 'same', 'sang', 'save', 'saw', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'seldom', 'selves', 'sent', 'several', 'shalt', 'she', 'should', 'shown', 'sideways', 'since', 'slept', 'slew', 'slung', 'slunk', 'smote', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'spake', 'spat', 'spoke', 'spoken', 'sprang', 'sprung', 'stave', 'staves', 'still', 'such', 'supposing', 'than', 'that', 'the', 'thee', 'their', 'them', 'themselves', 'then', 'thence', 'thenceforth', 'there', 'thereabout', 'thereabouts', 'thereafter', 'thereby', 'therefore', 'therein', 'thereof', 'thereon', 'thereto', 'thereupon', 'these', 'they', 'this', 'those', 'thou', 'though', 'thrice', 'through', 'throughout', 'thru', 'thus', 'thy', 'thyself', 'till', 'to', 'together', 'too', 'toward', 'towards', 'ugh', 'unable', 'under', 'underneath', 'unless', 'unlike', 'until', 'up', 'upon', 'upward', 'upwards', 'us', 'use', 'used', 'using', 'very', 'via', 'vs', 'want', 'was', 'we', 'week', 'well', 'were', 'what', 'whatever', 'whatsoever', 'when', 'whence', 'whenever', 'whensoever', 'where', 'whereabouts', 'whereafter', 'whereas', 'whereat', 'whereby', 'wherefore', 'wherefrom', 'wherein', 'whereinto', 'whereof', 'whereon', 'wheresoever', 'whereto', 'whereunto', 'whereupon', 'wherever', 'wherewith', 'whether', 'whew', 'which', 'whichever', 'whichsoever', 'while', 'whilst', 'whither', 'who', 'whoa', 'whoever', 'whole', 'whom', 'whomever', 'whomsoever', 'whose', 'whosoever', 'why', 'will', 'wilt', 'with', 'within', 'without', 'worse', 'worst', 'would', 'wow', 'ye', 'yet', 'year', 'yippee', 'you', 'your', 'yours', 'yourself', 'yourselves'}

# DOC: 
#   DOCNO, PROFILE, DATE,
#   HEADLINE, BYLINE, DATELINE, 
#   TEXT: string || P: string
#       
#   (XX, **)^

REMOVE_PUNCTUATION = str.maketrans(string.punctuation, ' '*len(string.punctuation))
# @param text: string
# @return : list(iterable) of terms
def regularizeText(text): 
    text = text.lower()
    text = text.translate(REMOVE_PUNCTUATION)
    terms = text.split(' ')
    return list(filter(lambda t: t not in STOPWORDS, map(ks.stem, terms)))
    # return map(ks.stem, terms)

# DocInfo
#     length: int
#     docno: string
# InvertedIndex
#     index: OrderedDict
#         "term" -> (docid:np.int32[], tf:int16[],)
#     docInfo: DocInfo[]
#     docCollection: docno:str -> docid:int
# Handler
#     ii: InvertedIndex


class DocInfo():
    def __init__(self, length=0, docno=''):
        self.length = length
        self.docno = docno


# Lifecycle: __init__ -> addToDictionary... -> finalizeDictionary 
#            -> addBody|addTitle... -> finalizeBody
class InvertedIndex():
    def __init__(self):
        # self.titleIndex = {}
        self.dictionary = set()
        self.bodyIndex = {}
        self.docInfo = [] 
        self.docCollection = {} # docno -> docid

    def addToDictionary(self, text):
        terms = regularizeText(text)
        self.dictionary.update(terms)
    
    def finalizeDictionary(self):
        self.bodyIndex = dict.fromkeys(self.dictionary)
        for k in self.bodyIndex.keys():
            self.bodyIndex[k] = ([],[],)
    
    def addBody(self, terms, docid):
        for term, freq in Counter(terms).items(): # count terms
            self.bodyIndex[term][0].append(docid) # add to index
            self.bodyIndex[term][1].append(freq)

    def addDocInfo(self, info):
        self.docCollection[info.docno] = len(self.docInfo)
        self.docInfo.append(info)
        
    def finalizeBody(self):
        # to numpy
        for term in self.dictionary:
            self.bodyIndex[term] = np.array(self.bodyIndex[term], dtype=np.uint32)
            # self.bodyIndex[term] = np.swapaxes(np.array(self.bodyIndex[term], dtype=np.uint32),0,1)
    
    def save(self, filename):
        for term in self.dictionary:
            self.bodyIndex[term] = self.bodyIndex[term].tolist()
            # self.bodyIndex[term] = np.swapaxes(self.bodyIndex[term],0,1).tolist()
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        for term in self.dictionary:
            self.bodyIndex[term] = np.array(self.bodyIndex[term], dtype=np.uint32)
            # self.bodyIndex[term] = np.swapaxes(np.array(self.bodyIndex[term], dtype=np.uint32),0,1)
        return self
        


# DocCounter must be sync with InvertedIndex.docInfo
class DocCounter():
    def __init__(self):
        self.count = 0

    def add(self):
        self.count += 1
        if self.count % 10000 == 0:
            print(self.count // 1000, 'k')



class Handler( xml.sax.ContentHandler ):
    def __init__(self):
        self.recordContent = True # :boolean. Ignore tag content if False
        self.currentContents = ''
        self.docCounter = DocCounter()
        super().__init__()

    def characters(self, content):
        if self.recordContent and content != '\n':
            self.currentContents += content



class DictionaryHandler(Handler):
    def __init__(self, invertedIndex):
        self.ii = invertedIndex
        super().__init__()
    
    def startElement(self, tag, attributes):
        self.currentContents = ''
        self.recordContent = (tag in ('HEADLINE', 'TEXT',))

    def endElement(self, tag):
        if tag == 'TEXT':
            self.ii.addToDictionary(self.currentContents)
        elif tag == 'DOC':
            self.docCounter.add()

    def endDocument(self):
        self.ii.finalizeDictionary()



class IndexHandler(Handler):
    def __init__(self, invertedIndex):
        self.ii = invertedIndex
        self.currentText = ''
        self.currentDocno = ''
        super().__init__()

    def startElement(self, tag, attributes):
        self.currentContents = ''
        self.recordContent = (tag in ('DOCNO', 'HEADLINE', 'TEXT',))
    
    def endElement(self, tag):
        if tag == 'DOCNO':
            self.currentDocno = self.currentContents.replace(' ', '')
        # elif tag == 'HEADLINE':
        #     self.ii.addTitle(self.currentContents, self.currentDocid)
        elif tag == 'TEXT':
            self.currentText = self.currentContents
        elif tag == 'DOC':
            terms = regularizeText(self.currentText)
            if len(terms) != 0: # only add non-empty docs
                self.ii.addBody(terms, self.docCounter.count)   
                self.ii.addDocInfo(DocInfo(len(terms), self.currentDocno))
                self.docCounter.add()

    def endDocument(self):
        self.ii.finalizeBody()



def readXml(filename):
    print('Reading XML file', filename)
    ii = InvertedIndex()
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0) # turn off namepsaces

    parser.setContentHandler( DictionaryHandler(ii) )    
    parser.parse(filename)
    print('Dictionary size:', len(ii.dictionary), 'terms')

    parser.setContentHandler( IndexHandler(ii) )    
    parser.parse(filename)
    print('Document count:', len(ii.docInfo))
    return ii



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trec', help='TREC disk 4-5 file')
    parser.add_argument('--invertedindex', help='inverted index file to output')
    args = parser.parse_args()
    trecFile = 'trec-disk4-5_processed.xml' if args.trec is None else args.trec
    iiFile = 'invertedindex.pickle' if args.invertedindex is None else args.invertedindex

    ii = readXml(trecFile)
    ii.save(iiFile)

