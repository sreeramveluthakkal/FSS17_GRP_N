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
            lineNumber += 1
            
    return {'headers': headers, 'data': data, 'fileLineCount': lineNumber}

if len(sys.argv) < 2:
    print 'Usage: python a.py <filename>'
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

    f = open('output.txt', 'w')
    for header in headers:
        f.write(header["name"] + ',')
    for row in data:
        f.write(str(row) + '\n')
    f.close()
    print 'Please see output.txt in current directory for the valid read data.'

