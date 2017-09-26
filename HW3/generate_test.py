import random

# sample size
N = 50

def klass(x):
    global N
    if x < 0.2:
        return 0.2 + random.random()/N
    elif x < 0.6:
        return 0.6 + random.random()/N
    else:
        return 0.9 + random.random()/N

if __name__ == '__main__':
    random.seed(random.randint(0,1000))
    arr = []
    with open("test_sample.csv","w+") as fp:
        fp.write("$column,<goal\n")
        for i in range(N):
            x = random.random()
            y = klass(x)
            fp.write("%f,%f\n"%(x,y))