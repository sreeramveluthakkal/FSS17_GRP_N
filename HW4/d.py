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
    # print headers
    for i,row in enumerate(data):
        data[i].append(dom(i,row,data,headers,numgoals)) #append domination rank to row
    data.sort(key=lambda x: x[len(data)-1],reverse=True) #sort by last column
    return {'headers': headers, 'data': data, 'fileLineCount': lineNumber}

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
        sub_set = bins[i].get('subSet')
        while(j<binCount):
            if abs(currentMedian - bins[j].get('median')) < epsilon:
                most = bins[j].get('hi')
                binValues += bins[j].get('values')
                sub_set += bins[j].get('subSet')
                # get median of the combined ranges and update the new median and contents of the range
                bins[j]["median"] = currentMedian
                bins[j]["values"] = binValues
                bins[j]["subSet"] = sub_set
                j = j + 1
            else:
                break
        supeviseddRanges += [{"label": label, "most":most, "median":bins[i].get('median'), "subSet":sub_set}]
        i = j
        label = label + 1
    if i == binCount-1:
        supeviseddRanges += [{"label": label, "most":bins[i].get('hi'), "median":bins[i].get('median'), "subSet":bins[i].get('subSet')}]
    return supeviseddRanges

def unsupervisedDiscretization(data, headers, i, cohen, useDom):
    data = sortData(data, i)
    lineNumber = len(data)
    binSize = int(math.floor(math.sqrt(lineNumber)))
    bins = []
    #small_value = 0.2
    small_value = cohen    

    epsilon = small_value* headers[i]["sd"]
    # print 'bin size (i.e. sqrt(n)):', binSize
    # print 'epsilon:', epsilon

    counter = 0
    dom_index = len(data[counter]) - 1
    if useDom != 1:
        dom_index-=1
    while counter < lineNumber:
        n = 1
        bin = {"lo": float(data[counter][i]), "hi": float(data[counter][i])}
        #Getting list of dependent variable/domination value for each row in a bin
        tmp_list = [float(data[counter][dom_index])]
        sub_set = [data[counter]]
        while (counter+1 < lineNumber) and \
                (((bin["hi"] - bin["lo"]) < epsilon) or \
                (n < binSize)):
            bin["hi"] = float(data[counter + 1][i])
            #Getting list of dependent variable/domination value for each row in a bin
            tmp_list.append(float(data[counter + 1][dom_index]))
            sub_set.append(data[counter+1])
            n += 1
            counter += 1
        #Each bin now contains the median value of domination index
        bins += [{"lo": bin["lo"], "hi": bin["hi"], "span": bin["hi"]-bin["lo"], "n": n, "median":np.median(tmp_list),"values":tmp_list, "subSet":sub_set}]
        counter += 1

    
    # checking last bin
    if len(bins) > 1 and bins[-1]["hi"] - bins[-1]["lo"] < epsilon:
        bins[-2]["hi"] = bins[-1]["hi"]
        bins[-2]["n"] += bins[-1]["n"]
        del bins[-1]

    return {"bins": bins, "sortedData": data}

def binVarianceNUM(bin, index):
    data = [float(row[index]) for row in bin]
    # print '>>>DATA: ', data
    return np.var(data)
    

def getVariance(bins, index):
    total_count = 0
    product = 0
    for _, bin in enumerate(bins):
        bin_count = len(bin['subSet'])
        total_count += bin_count
        bin_v = binVarianceNUM(bin['subSet'], index)
        product += bin_count*bin_v
    return product/total_count


def findColumnToSplit(data,splitColumns,tooFew):
    index = 0
    minColVariance = float('inf')
    minIndex = 0
    superBins = []
    sortedData = []
    # print '^',len(data)
    while index < len(headers):
        if(len(data)>tooFew and headers[index]['goal']==False and headers[index]['typeof']=='NUM' and headers[index]['ignore']==False and index not in splitColumns):
            ud = unsupervisedDiscretization(data, headers, index, float(sys.argv[2]), int(sys.argv[3]))
            sortedData = ud["sortedData"]
            bins = ud["bins"]
            supervisedBins = combineBins(bins)
            # print '*',headers[index]["name"],len(supervisedBins)
            # for _,r in enumerate(supervisedBins):
            #         print len(r.get('subSet'))
            colVariance = getVariance(supervisedBins, index)
            print '>>variance for ',index,' is ',colVariance
            if(colVariance<minColVariance):
                minColVariance = colVariance
                minIndex=index
                del superBins[:]
                for bindid,r in enumerate(supervisedBins):
                    if(len(r.get('subSet'))>tooFew):
                        temp = r.get("subSet")
                        temp.append(bindid+1)
                        superBins.append(temp)
        index += 1
    return minIndex,superBins,sortedData

def datastats(data):
    dataTranspose = zip(*data)
    ranks = dataTranspose[len(dataTranspose)-1]
    [float(i) for i in ranks]
    lineCount = float(len(data))
    mu = float(reduce(lambda x, y: x + y, ranks) / float(len(ranks)))
    stddev = float(np.std(ranks,ddof=1))
    return lineCount, mu, stddev

def createRegressionTree(data, headers, treelevel, splitColumns, lastSupScore = 0):
    index = 0
    superBins = []
    tooFew = int(sys.argv[4])
    maxDepth = int(sys.argv[5])
    if (len(data)<tooFew):
        linec, mu, stddev = datastats(data)
        print "n=%d mu=%-.2f sd=%-.2f"%(linec, mu, stddev)
        return
    #find initial split for the tree
    index, superBins, sortedData = findColumnToSplit(data,splitColumns,tooFew)
    
    domScore = sys.maxint
    ###############################################################
    # Comment this section for ignoring my changes    
    ###############################################################
    # sortedData = sortData(sortedData, -1)
    # if len(sortedData) > 0: domScore = sortedData[-1][-1]
    # if domScore < lastSupScore:
    #     linec, mu, stddev = datastats(data)
    #     print "n=%d mu=%-.2f sd=%-.2f"%(linec, mu, stddev)
    #     return
    ###############################################################
    if not superBins:
        linec, mu, stddev = datastats(data)
        print "n=%d mu=%-.2f sd=%-.2f"%(linec, mu, stddev)
        return
    print '\n',
    if (treelevel>0):
        superBins.sort(key = len,reverse=True)
    splitColumns.append(index)
    treelevel+=1
    if (treelevel>maxDepth):
        return " "
    temp = splitColumns[:]
    for i,currBin in enumerate(superBins):
        splitColumns = temp[:]
        print '|'*(treelevel-1)+headers[index]["name"]+'='+str(currBin[-1])+'\t\t:\t\t',
        # leafstats = 
        createRegressionTree(currBin[:len(currBin)-1], headers, treelevel, splitColumns, domScore)
        # if not leafstats:
        #     print leafstats
        # for p in splitColumns: print '##',p
        

## Running the script
if len(sys.argv) < 6:
    print 'Usage: python d.py <inputfile> <small value> <useDom> <tooFew> <maxDepth>'
    exit(1)

else:
    # Running the parser
    start_time = time.time()

    results = parse(sys.argv[1])
    data = results['data']
    headers = results['headers']
    lineCount = float(len(data))

    print '\n\n\n############# REGRESSION TREE #############'
    linecount, mu, stddev = datastats(data)
    print "in=%d mu=%-.2f sd=%-.2f"%(lineCount, mu, stddev),
    # print "in=%s mu=%-.2f sd=%-.2f"%(lineCount, (lineCount-1)/2, math.sqrt(((lineCount**2)-1)/12)) #TODO check stddev
    createRegressionTree(data, headers, 0, [])

    print '\n\n\n############# SOME STATS #############'
    print 'Number of lines of valid data:', lineCount
    print ("Total execution time: %s seconds ---" % (time.time() - start_time))
#write data to file
f = open('output.txt', 'w')
for header in headers:
    f.write(header["name"]+'-'+header["typeof"] + ',')
f.write('Rank\n')
for row in data:
    f.write(str(row) + '\n')
f.close()
print 'NOTE: See output.txt in current directory for the valid read data sorted by domination rank.'
