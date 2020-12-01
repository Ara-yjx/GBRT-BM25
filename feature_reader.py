import numpy as np

features = []
values = []

with open("../trec45-ranklib-top2k.txt.shuffled", "r") as f:

    while True:
        line = f.readline()
        if len(line) == 0:
            break
        segments = line.split(' ')

        values.append(int(segments[0]))
        
        feature = list(map(lambda x: float(x.split(':')[1]), segments[2:2+36]))
        features.append(feature)
    
features = np.array(features)
values = np.array(values)



if __name__ == "__main__":
    print(features)
    print(values)
