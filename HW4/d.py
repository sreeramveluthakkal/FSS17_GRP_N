import re
import time
import sys
import math
import numpy as np

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

def calculateSD(header, x):
    mu = header["mu"]
    m2 = header["m2"]
    n = header["count"]
    sd = header["sd"]
    delta = x - mu
    mu = mu + delta/n
    m2 = m2 + delta*(x- mu)
    if n > 1:
        sd = (m2/(n-1))**0.5
    return sd, mu, m2

def updateHeaders(lineList, headers):
    for idx, val in enumerate(lineList):
        if "count" in headers[idx]:
                headers[idx]["count"] = headers[idx]["count"]+1
        else:
            headers[idx]["count"] = 1
        if headers[idx]["typeof"] == "NUM":
            val = float(val)
            if "min" in headers[idx]:
                headers[idx]["min"] = min(headers[idx]["min"], val)
            else:
                headers[idx]["min"] = val
            if "max" in headers[idx]:
                headers[idx]["max"] = max(headers[idx]["max"], val)
            else:
                headers[idx]["max"] = val
            if "sd" in headers[idx]:
                headers[idx]["sd"],headers[idx]["mu"],headers[idx]["m2"] = calculateSD(headers[idx], float(val))
            else:
                headers[idx]["sd"] = 0
                headers[idx]["mu"] = 0
                headers[idx]["m2"] = 0
                headers[idx]["sd"],headers[idx]["mu"],headers[idx]["m2"]  = calculateSD(headers[idx], float(val))
        else:
            if "fmap" in headers[idx]:
                headers[idx]["fmap"][val] = headers[idx]["fmap"].get(val,0)+1
                #print "freq for "+val+" is "+str(headers[idx]["fmap"][val])
            else:
                headers[idx]["fmap"] = {}
                headers[idx]["fmap"][val] = 1
            if "most" in headers[idx]:
                seen = headers[idx]["fmap"][val]
                if(seen>headers[idx]["most"]):
                    headers[idx]["most"] = seen
                    headers[idx]["mode"] = val
            else:
                headers[idx]["most"] = headers[idx]["fmap"].get(val,0)

def norm(currval, minval, maxval):
    try:
        val = (currval - minval)/(maxval - minval)
    except ZeroDivisionError:
        val = (currval - minval)/(10**-2)
    return val

def dominate(i, j, headers, n):
    sum1,sum2,e = 0.0,0.0,2.71828
    index = 0
    while index < len(headers):
        if(headers[index]["goal"]==True):
            weight = headers[index]["weight"]
            #print "Weight=",weight,",I=",i[index],",J=",j[index],",MIN=",float(headers[index]["min"]),",MAX=",float(headers[index]["max"])
            x = norm(float(i[index]),float(headers[index]["min"]),float(headers[index]["max"]))
            y = norm(float(j[index]),float(headers[index]["min"]),float(headers[index]["max"]))
            sum1 = sum1 - e**(weight * (x - y)/n)
            sum2 = sum2 - e**(weight * (y - x)/n)
        index += 1
    return sum1/n < sum2/n

def dom(index, row, data, headers, numgoals):
    rowrank = 1
    for i, otherrow in enumerate(data):
        if i != index:
            if dominate(row, otherrow, headers, numgoals):
                rowrank += 1
    return rowrank

def parse (filename):
    lineNumber = 0 # line count
    headers = [] # list of dictionaries to keep track of flags
    data = [] # Adding data here
    nCol = 0
    numgoals = 0

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
                    headers[h]["goal"] = False #initialise goal as False
                    headers[h]["weight"] = 0 #initialise weight as 0
                    if lineList[h][0] == '?':
                        headers[h]["ignore"] = True
                        headers[h]["typeof"] = None
                    else: 
                        headers[h]["ignore"] = False

                        # Checking NUM/SYM --> We only check if not '?'
                        if lineList[h][0] == '>' or lineList[h][0] == '<' or  lineList[h][0] == '$' or lineList[h][0] == '<$' or lineList[h][0] == '>$':
                            headers[h]["typeof"] = 'NUM'
                            if lineList[h][0] != '$':
                                headers[h]["goal"] = True
                                numgoals += 1
                                if lineList[h][0] == '>':
                                    headers[h]["weight"] = 1
                                else:
                                    headers[h]["weight"] = -1

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
                        print 'err: unexpected data found in line:', lineNumber + 1 #, lineList
                        failed = True
                if not failed:
                    data.append(lineList)
                    updateHeaders(lineList, headers)
            lineNumber += 1
    print headers
    for i,row in enumerate(data):
        data[i].append(dom(i,row,data,headers,numgoals)) #append domination rank to row
    data.sort(key=lambda x: x[len(data)-1],reverse=True) #sort by last column
    return {'headers': headers, 'data': data, 'fileLineCount': lineNumber}

## Part 3

# data: [[]]
# i: index to sort on
def sortData(data, i):
    return sorted(data, key=lambda x: float(x[i]))


def combineBins(bins):
    # calculating epsilon based on SD of dependent variable
    depValues = []
    for r in bins:
        depValues += r.get('values')
    epsilon = 0.2* np.std(depValues)
    supeviseddRanges = []
    label = 1
    most = bins[0].get('hi')
    i = 0 
    binCount = len(bins)
    while(i < binCount-1):
        most = bins[i].get('hi')
        j = i + 1
        currentMedian = bins[i].get('median')
        binValues = bins[i].get('values')
        while(j<binCount):
            if abs(currentMedian - bins[j].get('median')) < epsilon:
                most = bins[j].get('hi')
                binValues += bins[j].get('values')
                # get median of the combined ranges and update the new median and contents of the range
                currentMedian = np.median(binValues)
                bins[j]["median"] = currentMedian
                bins[j]["values"] = binValues
                j = j + 1
            else:
                break
        supeviseddRanges += [{"label": label, "most":most}]
        i = j
        label = label + 1
    if i == binCount-1:
        supeviseddRanges += [{"label": label, "most":bins[i].get('hi')}]
    return supeviseddRanges


def unsupervisedDiscretization(data, headers, i):
    data = sortData(data, i)
    lineNumber = len(data)
    binSize = int(math.floor(math.sqrt(lineNumber)))
    bins = []
    small_value = 0.2
    if len(sys.argv)>3:
        small_value = float(sys.argv[3])    

    epsilon = small_value* headers[i]["sd"]
    print 'bin size (i.e. sqrt(n)):', binSize
    print 'epsilon:', epsilon

    counter = 0
    dom_index = len(data[counter])-1
    while counter < lineNumber:
        n = 1
        bin = {"lo": float(data[counter][i]), "hi": float(data[counter][i])}
        #Getting list of dependent variable/domination value for each row in a bin
        tmp_list = [float(data[counter][dom_index])]
        while (counter+1 < lineNumber) and \
                (((bin["hi"] - bin["lo"]) < epsilon) or \
                (n < binSize)):
            bin["hi"] = float(data[counter + 1][i])
            #Getting list of dependent variable/domination value for each row in a bin
            tmp_list.append(float(data[counter + 1][dom_index]))
            n += 1
            counter += 1
        #Each bin now contains the median value of domination index
        bins += [{"lo": bin["lo"], "hi": bin["hi"], "span": bin["hi"]-bin["lo"], "n": n, "median":np.median(tmp_list),"values":tmp_list}]
        counter += 1

    
    # checking last bin
    if len(bins) > 1 and bins[-1]["hi"] - bins[-1]["lo"] < epsilon:
        bins[-2]["hi"] = bins[-1]["hi"]
        bins[-2]["n"] += bins[-1]["n"]
        del bins[-1]

    return {"bins": bins, "sortedData": data}



## Running the script
if len(sys.argv) < 3:
    print 'Usage: python b.py <filename> <column index>'
    exit(1)

else:
    # Running the parser
    start_time = time.time()

    results = parse(sys.argv[1])
    data = results['data']
    headers = results['headers']
    lineCount = len(data)
    print 'Number of lines of valid data:', lineCount
    print("--- %s seconds ---" % (time.time() - start_time))

    #write data to terminal
    print 'Printing the top and bottom ten rows, as sorted by their dom score, with the top 5 and the bottom 5 domination scores:'
    for header in headers:
        print (header["name"] + ','),
    print 'Rank' 
    print 'TOP 5 DATA RANKED BY DOMINATION SCORE (ASC)'
    index = 0
    while index < min(5,len(data)):
        print str(data[index])
        index += 1
    print 'BOTTOM 5 DATA RANKED BY DOMINATION SCORE (DESC)'
    index = len(data)-1
    bottomFive = ""
    while index > len(data)-min(6,len(data)):
        bottomFive = str(data[index]) + '\n' + bottomFive
        index -= 1
    print bottomFive

    print 'We have many unsupervised ranges.'
    # Note: Change the third column to run unsupervised discretization on other columns
    ud = unsupervisedDiscretization(data, headers, int(sys.argv[2]))
    sortedData = ud["sortedData"]
    bins = ud["bins"]
    for i,r in enumerate(bins):
        print 'x    ',i+1,'{ span = ', r.get('span'),', lo= ',r.get('lo'),' n= ',r.get('n'),' hi= ',r.get('hi'),'} median: ',r.get('median')
    supervisedBins = combineBins(bins)
    if len(supervisedBins) < len(bins):
        print 'We have fewer supervised ranges :)'
    else:
        print 'We have the same number of supervised ranges :('
    for i,r in enumerate(supervisedBins):
        print 'super    ',i+1,'  {label= ',r.get('label'),', most= ',r.get('most'),'}'

#write data to file
f = open('output.txt', 'w')
for header in headers:
    f.write(header["name"] + ',')
f.write('Rank\n')
for row in sortedData:
    f.write(str(row) + '\n')
f.close()
print '\nPlease see output.txt in current directory for the sorted valid read data sorted by domination rank.'
