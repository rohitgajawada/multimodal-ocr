import os
fil = open('malWords','r')
max = 0
name = ''
for line in fil:
    x = len(line)
    if x>max:
        max = x
        name = line

print (max)
print (name)
f = open('tstword.txt','w')
f.write(name)
f.close()
fil.close()
