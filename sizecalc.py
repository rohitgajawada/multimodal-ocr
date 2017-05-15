import math

dims = input("Height Width:")
dims = dims.split(' ')
height = int(dims[0])
width = int(dims[1])

print("Types of layers:\n 1)Conv2d InC OutC Ksize Stride PaddingW PaddingH\n 2)Pool Ksize Stride\n")

while True:
    layer = input("Layer:")
    params = layer.split(' ')
    if params[0] == "Conv2d":
        inchan = int(params[1])
        outchan = int(params[2])
        ksize = int(params[3])
        stride = int(params[4])
        padw = int(params[5])
        padh = int(params[6])

        width  = math.floor((width  + 2*padw - ksize) / stride + 1)
        height  = math.floor((height  + 2*padh - ksize) / stride + 1)

        print("New Width, New Height, No. of channels:", width, height, outchan)

    elif params[0] == "Pool":
        ksize = int(params[1])
        stride = int(params[2])
        pad = 0

        width  = math.floor((width  + 2*pad - ksize) / stride + 1)
        height = math.floor((height + 2*pad - ksize) / stride + 1)

        print("New Width, New Height, No. of channels:", width, height, outchan)
