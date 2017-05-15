import os

cwd = os.getcwd()
list = os.listdir(cwd)
max = -1
count = 0
for i in list:
    tmp = cwd + "/" + i
    newli = os.listdir(tmp)
    print(newli)
    for j in newli:
            # file = open(tmp + "/" + j, "r")
            # count = sum(1 for _ in file)
        with open(tmp + "/" + j, "r") as f:
            count = 0
            fil = f.read()
            for lines in fil:
                count += 1
            if count > max:
                max = count


print(max)
