import os
import random
import shutil
presdir = os.getcwd()

x = os.listdir('tmp')
out = {}
for i in x:
    out[i] = os.listdir('tmp' + '/' + i)

for i in x:
    temp = out[i]
    num = len(temp)
    num = int(num / 15)
    samp = random.sample(temp, num)
    for j in samp:
        shutil.move(presdir + "/tmp/" + i + "/" + j, presdir + "/tmp2/" + i + "/" + j)
