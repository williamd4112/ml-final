import sys, os
import csv

def load(filename):
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        data = { row[0]:row[1] for row in spamreader }
    return data

myans = load(sys.argv[1])
ans = load(sys.argv[2])

n = 0
cnt = 0
for key in myans:
    if key in ans:
        n +=1
        if myans[key] == ans[key]:
            cnt += 1
acc = float(cnt) / float(n)
print acc

      

