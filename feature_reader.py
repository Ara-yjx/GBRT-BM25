import numpy as np

features = []
values = []
comments = [] # qid

with open("../trec45-ranklib-top2k.txt.shuffled", "r") as f:

    while True:
        line = f.readline()
        if len(line) == 0:
            break
        segments = line.split(' ')

        values.append(int(segments[0]))
        
        # feature = list(map(lambda x: float(x.split(':')[1]), segments[2:2+36]))
        feature = [ float(seg.split(':')[1]) for seg in segments[2:2+36] ]
        features.append(feature)

        comments.append(segments[1].split(':')[1])
    
features = np.array(features)
values = np.array(values)


if __name__ == "__main__":
    print(features.shape)
    print(features)
    print(values.shape)
    print(values)
    print(comments)