# -*- coding: utf-8 -*-
import os
file1 = open('1.txt','r')
a = []
for line in file1.readlines():
    a.append(float(line.strip()))


b = []
file2 = open('3.txt','r')
for line2 in file2.readlines():
    b.append(int(line2.strip()))

for c in b:
    d = c-1
    a[d] = a[d]+4

e =[]
for aa in a:
    e.append(str(aa)+'\n')
# e = [str(f)+os.linesep for f in a]
file3 = open('2.txt','w')
file3.writelines(e)