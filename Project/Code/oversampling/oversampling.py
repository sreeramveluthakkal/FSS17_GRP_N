import re
import time
import sys
# import random
# import numpy as np
# from random import randrange, choice
# from sklearn.neighbors import NearestNeighbors
# from sklearn.datasets import make_classification
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

# Helper functions:====================
# https://stackoverflow.com/a/15357477
def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b
#========================================

def clean (str):
    str = re.sub('#.+$', '', str) # remove comments
    str = str.replace('\n', '') # remove new lines
    str = str.replace('\r', '') # remove new lines
    str = str.replace(' ', '') # remove spaces
    return str

def cleanHeader (str):
    str = clean(str)
    str = re.sub(r'[<>$?]+', '', str)
    return str

def parse (filename):
    lineNumber = 0 # line count
    headers = [] # list of dictionaries to keep track of flags
    data = [] # Adding data here
    nCol = 0

    with open(filename, 'r') as f:
        for line in f:

            line = clean(line)
            if line == '': continue

            while line[-1] == ',':
                line += clean(f.next())
                lineNumber += 1

            lineList = line.split(',')
            
            # Getting the headers
            if lineNumber == 0: 
                nCol = len(lineList)            
                headers = [{} for _ in range(nCol)] # making list of empty dicts             

                for h in xrange(nCol):

                    # Checking ignore
                    if lineList[h][0] == '?':
                        headers[h]["ignore"] = True
                        headers[h]["typeof"] = None
                    else: 
                        headers[h]["ignore"] = False

                        # Checking NUM/SYM --> We only check if not '?'
                        if lineList[h][0] == '>' or lineList[h][0] == '<' or lineList[h][0] == '$' or lineList[h][0] == '<$' or lineList[h][0] == '>$':
                            headers[h]["typeof"] = 'NUM'  
                        else: # includes '!', right?
                            headers[h]["typeof"] = 'SYM'
                    
                    headers[h]["name"] = cleanHeader(lineList[h])

                lineNumber += 1
                continue # Not adding to data list

            if len(lineList) != nCol: # Checking number of columns
                print 'err: bad line:', lineNumber #, lineList
            else:
                failed = False
                for h in xrange(len(headers)):
                    if headers[h]['typeof'] == 'NUM' and not (isfloat(lineList[h]) or isint(lineList[h])):
                        print 'err: unexpected data found in line:', lineNumber #, lineList
                        failed = True
                if not failed:
                    data.append(lineList)
            lineNumber += 1
            
    return {'headers': headers, 'data': data, 'fileLineCount': lineNumber}

def smote(T, N, K):
    print len(T)
    print len(T[0])
    T = np.asarray(T, dtype = np.float)
    nsamples = T.shape[0]
    nfeatures = T.shape[1]
    if nsamples < nfeatures:
        warnings.warn("Make sure the features are in the columns.")
    
    if N < 100:
        N = 100

    N = int(N) / 100

    nn = NearestNeighbors(K)
    
    nn.fit(T)

    synthetic = np.zeros([N * nsamples, nfeatures])
    
    for sample in xrange(nsamples):
        nn_minority = nn.kneighbors(T[sample], return_distance = False)[0]
        N_next = N
        
        newindex = 0
        while N_next != 0:
            k_chosen = random.randint(0, K - 1)
            while nn_minority[k_chosen] == sample: # don't pick itself
                k_chosen = random.randint(0, K - 1)                
            
            for feature in xrange(nfeatures):
                diff = T[nn_minority[k_chosen], feature] - T[sample, feature]
                gap = random.uniform(0, 1)
                synthetic[N*sample + newindex, feature] = T[sample, feature] + gap * diff

            newindex += 1
            N_next -= 1

    return synthetic

if len(sys.argv) < 2:
    print 'Usage: python oversampling.py velocity.csv'
    exit(1)

else:
    # Running the parser
    start_time = time.time()

    results = parse(sys.argv[1])
    data = results['data']
    # del
    data = map(list, zip(*data))
    
    print len(data[0:len(data)-1])
    print data[0:len(data)-1]
    
    X, y = data[0:len(data)-1],data[-1]
    # make_classification(n_classes=2, class_sep=2,weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    sm = SMOTE()
    print len(X), len(y)
    X_res, y_res = sm.fit_sample(X, y)
    print len(X_res), len(y_res)

    # mutator = SMOTE(data,500,5)
    # headers = results['headers']
    # lineCount = len(data)
    # print 'Number of lines of valid data:', lineCount


    # print("--- %s seconds ---" % (time.time() - start_time))

    # f = open('velocity_m.csv', 'w')
    # for header in headers:
    #     f.write(header["name"] + ',')
    # for row in mutator:
    #     f.write(str(row) + '\n')
    # f.close()
    # print 'Please see output.txt in current directory for the valid read data.'

