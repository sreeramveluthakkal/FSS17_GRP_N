import re
import time
import sys

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
        val = float(val)
        if "count" in headers[idx]:
                headers[idx]["count"] = headers[idx]["count"]+1
        else:
            headers[idx]["count"] = 1
        if headers[idx]["typeof"] == "NUM":
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
    #print headers
    for i,row in enumerate(data):
        data[i].append(dom(i,row,data,headers,numgoals)) #append domination rank to row
    data.sort(key=lambda x: x[len(data)-1]) #sort by last column
    return {'headers': headers, 'data': data, 'fileLineCount': lineNumber}

if len(sys.argv) < 2:
    print 'Usage: python b.py <filename>'
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
    while index > len(data)-min(6,len(data)):
        print str(data[index])
        index -= 1

    #write data to file
    f = open('output.txt', 'w')
    for header in headers:
        f.write(header["name"] + ',')
    f.write('Rank\n')
    for row in data:
        f.write(str(row) + '\n')
    f.close()
    print 'Please see output.txt in current directory for the valid read data sorted by domination rank.'
