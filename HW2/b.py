import re
import time

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

def updateHeaders(line_list, headers):
    for idx, val in enumerate(line_list):
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
                headers[idx]["max"] = max(headers[idx]["min"], val)
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
                print "freq for "+val+" is "+str(headers[idx]["fmap"][val])
            else:
                headers[idx]["fmap"] = {}
                headers[idx]["fmap"][val] = 1;
            if "most" in headers[idx]:
                seen = headers[idx]["fmap"][val]
                if(seen>headers[idx]["most"]):
                    headers[idx]["most"] = seen;
                    headers[idx]["mode"] = val;
            else:
                headers[idx]["most"] = headers[idx]["fmap"].get(val,0);


                

            


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
                        if lineList[h][0] == '>' or lineList[h][0] == '<' or  lineList[h][0] == '$' or lineList[h][0] == '<$' or lineList[h][0] == '>$':
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
                    updateHeaders(lineList, headers)
            lineNumber += 1
    print headers        
    return {'headers': headers, 'data': data, 'fileLineCount': lineNumber}


# Running the parser
start_time = time.time()

lineCount = len(parse('auto.csv')['data'])
print 'Number of lines of valid data:', lineCount
# print parse('test.csv')['data']

print("--- %s seconds ---" % (time.time() - start_time))

