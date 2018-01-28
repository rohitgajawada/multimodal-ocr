import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
from resizeimage import resizeimage
import PIL
import numpy as np
import cv2
from rdp import rdp
import pickle
import random


def invert(image):
    inverted_image = PIL.ImageOps.invert(image)
    return inverted_image


def make_black_go(img):
	img = img.convert("RGBA")
	datas = img.getdata()

	newData = []
	for item in datas:
	    if item[0] == 0 and item[1] == 0 and item[2] == 0:
	        newData.append((0, 0, 0, 0))
	    else:
	        newData.append(item)

	img.putdata(newData)
	return img

def get_correct_size(img):
    left,up,right,down = img.getbbox()
    width = abs(right - left)
    height = abs(up - down)
    x = 10000/height
    img = img.resize((x,x),Image.ANTIALIAS)

    left,up,right,down = img.getbbox()
    width = abs(right - left)
    height = abs(up - down)
    img  = img.crop((left,up,right,down))
    return img

def extractdata(filename):
    # print filename
    f = open(filename,'r+')
    onlinedata = []
    for line in f:
        if '=' in line:
            continue
        y,x = line.split(' ')
        x = float(x)
        y = float(y)
        temp_tuple = [x,y]
        onlinedata.append(temp_tuple)
    return onlinedata

def normalize(data,lower_x,lower_y,magnification = 1,height=32,rdp_flg = False):
    data = np.array(data)
    min_x = np.min(data[:,0])
    max_x = np.max(data[:,0])
    min_y = np.min(data[:,1])
    max_y = np.max(data[:,1])

    y = float(height)/4
    factor_y = float(max_y - min_y)
    factor_x = float(max_x - min_x)
    x = (magnification*y*2*factor_x)/factor_y
    # print factor_y/factor_x == y*2*magnification/x, factor_y/factor_x, y*2*magnification/x
    # assert factor_y/factor_x == y*2*magnification/x

    upper_x = lower_x + x
    upper_y = lower_y + 2*y*magnification

    data[:,0] = ((upper_x-lower_x)*(data[:,0]-min_x)/factor_x) + lower_x
    data[:,1] = ((upper_y-lower_y)*(data[:,1]-min_y)/factor_y) + lower_y
    # if rdp_flg:
    #     data = rdp(data,0.5)
    return data,x

def rdp_iso(data):
    return rdp(data,0.85)

def transform_online(data,size):
    # print len(data)
    if len(data) < size:
        val = size/len(data)
        new_data = []
        for i in range(len(data)):
            for j in range(val):
                new_data.append(data[i])
        while len(new_data) < size:
            ind = random.randint(0,len(data)-1)
            new_ind = new_data.index(data[ind])
            new_data.insert(new_ind,data[ind])

        # print len(new_data)
    else:
        return -1
    return new_data





target_lis = []
styles = []
for i in range(1,51):
    styles.append(str(i)+'.stk')

SC_triple = {"u'\\u0d15'u'\\u0d4d'u'\\u0d15'":'69',"u'\\u0d15'u'\\u0d4d'u'\\u0d24'":'70',"u'\\u0d15'u'\\u0d4d'u'\\u0d37'":'71',"u'\\u0d17'u'\\u0d4d'u'\\u0d28'":'72',"u'\\u0d17'u'\\u0d4d'u'\\u0d2e'":'73',"u'\\u0d19'u'\\u0d4d'u'\\u0d15'":'74',"u'\\u0d19'u'\\u0d4d'u'\\u0d19'":'75',"u'\\u0d1a'u'\\u0d4d'u'\\u0d1a'":'76',"u'\\u0d1c'u'\\u0d4d'u'\\u0d1c'":'77',"u'\\u0d1c'u'\\u0d4d'u'\\u0d1e'":'78',"u'\\u0d1e'u'\\u0d4d'u'\\u0d1a'":'79',"u'\\u0d1e'u'\\u0d4d'u'\\u0d1e'":'80',"u'\\u0d1f'u'\\u0d4d'u'\\u0d1f'":'81',"u'\\u0d23'u'\\u0d4d'u'\\u0d1f'":'82',"u'\\u0d23'u'\\u0d4d'u'\\u0d21'":'83',"u'\\u0d23'u'\\u0d4d'u'\\u0d2e'":'84',"u'\\u0d24'u'\\u0d4d'u'\\u0d24'":'85',"u'\\u0d24'u'\\u0d4d'u'\\u0d25'":'86',"u'\\u0d24'u'\\u0d4d'u'\\u0d28'":'87',"u'\\u0d24'u'\\u0d4d'u'\\u0d2d'":'88',"u'\\u0d24'u'\\u0d4d'u'\\u0d2e'":'89',"u'\\u0d24'u'\\u0d4d'u'\\u0d38'":'90',"u'\\u0d26'u'\\u0d4d'u'\\u0d26'":'91',"u'\\u0d26'u'\\u0d4d'u'\\u0d27'":'92',"u'\\u0d28'u'\\u0d4d'u'\\u0d24'":'93',"u'\\u0d28'u'\\u0d4d'u'\\u0d25'":'94',"u'\\u0d28'u'\\u0d4d'u'\\u0d26'":'95',"u'\\u0d28'u'\\u0d4d'u'\\u0d27'":'96',"u'\\u0d28'u'\\u0d4d'u'\\u0d28'":'97',"u'\\u0d2e'u'\\u0d4d'u'\\u0d2a'":'98',"u'\\u0d28'u'\\u0d4d'u'\\u0d2e'":'99',"u'\\u0d2c'u'\\u0d4d'u'\\u0d2c'":'101',"u'\\u0d2e'u'\\u0d4d'u'\\u0d2e'":'102',"u'\\u0d2f'u'\\u0d4d'u'\\u0d2f'":'103',"u'\\u0d35'u'\\u0d4d'u'\\u0d35'":'104',"u'\\u0d36'u'\\u0d4d'u'\\u0d1a'":'105',"u'\\u0d39'u'\\u0d4d'u'\\u0d28'":'106',"u'\\u0d39'u'\\u0d4d'u'\\u0d2e'":'107',"u'\\u0d17'u'\\u0d4d'u'\\u0d26'":'109',"u'\\u0d1e'u'\\u0d4d'u'\\u0d1c'":'110',"u'\\u0d23'u'\\u0d4d'u'\\u0d23'":"u'\u0d23'","u'\\u0d2a'u'\\u0d4d'u'\\u0d2a'":"u'\u0d2a'"}

SC_single = {"u'\u0d15'":'20',"u'\u0d16'":'21',"u'\u0d17'":'22',"u'\u0d18'":'23',"u'\u0d19'":'24',"u'\u0d1a'":'25',"u'\u0d1b'":'26',"u'\u0d1c'":'27',"u'\u0d1d'":'28',"u'\u0d1e'":'29',"u'\u0d1f'":'30',"u'\u0d20'":'31',
"u'\u0d21'":'32',"u'\u0d22'":'33',"u'\u0d23'":'34',"u'\u0d24'":'35',"u'\u0d25'":'36',"u'\u0d26'":'37',"u'\u0d27'":'38',"u'\u0d28'":'39',"u'\u0d2a'":'40',"u'\u0d2b'":'41',"u'\u0d2c'":'42',"u'\u0d2d'":'43',
"u'\u0d2e'":'44',"u'\u0d2f'":'45',"u'\u0d30'":'46',"u'\u0d32'":'47',"u'\u0d35'":'48',"u'\u0d36'":'49',"u'\u0d37'":'50',"u'\u0d38'":'51',"u'\u0d39'":'52',"u'\u0d33'":'53',"u'\u0d34'":'54',"u'\u0d31'":'55',
"u'\u0d7b'":'56',"u'\u0d7c'":'57',"u'\u0d7d'":'58',"u'\u0d7e'":'59',"u'\u0d7a'":'60'}


lis_tosplit = ["u'\u0d4a'","u'\u0d4b'"]# special cases to be checked

mtrs = {"u'\u0d3f'":"13","u'\u0d40'":"14","u'\u0d41'":"15","u'\u0d42'":"16","u'\u0d43'":"17"}

SV = {"u'\u0d05'":'1',"u'\u0d06'":'2',"u'\u0d07'":'3',"u'\u0d09'":'4',"u'\u0d0b'":'5',"u'\u0d0e'":'7',"u'\u0d0f'":'8',"u'\u0d12'":'9',"u'\u0d57'":'10',"u'\u0d3e'":'12',"u'\u0d46'":'18',"u'\u0d47'":'19'}

lis_down_scale_down = {"u'\u0d02'":'11'}
lis_down_scale_up = {"u'\u0d4d'":'65'}
targets = {"u'\\u0d23'u'\\u0d4d'u'\\u0d23'":61,"u'\\u0d2a'u'\\u0d4d'u'\\u0d2a'":62,"u'\u0d03'":63}

word_count = 0
count = 0
stroke_space = 4 # spcae between every character
str_delim = -1 # its the delimiter between strokes
delim_repetitions = 3
img_height = 32
pen_lift = False
save_interval = 20000 # After how many words irrespective of styles
#fil = open('/home/rohit/Documents/sreekarfiles/Targetstmp.txt','w+')

# Path to the images
path = './FinalStrokes1/'
num_styles = 1
f = open('./malWords','r+')
for line in f:
    # print(line)
    # print("---")
    lis = []
    word_count += 1
    print("word", word_count)
    word_unicode = line.decode('utf-8')
    for i in range(len(word_unicode)):
        char_uni = repr(word_unicode[i])
        if char_uni != "u'\\n'":
            lis.append(char_uni)
            count += 1
    if "u'\u0d0c'" in lis:
        print 'Symbol not available ',lis
        continue

    # if word_count == 20000:
    #     break




    while "u'\u200c'" in lis:
        pntr = lis.index("u'\u200c'")
        del lis[pntr]
    while "u'\u200d'" in lis:
        pntr = lis.index("u'\u200d'")
        if lis[pntr - 2] == "u'\u0d28'":
            lis[pntr - 2] = "u'\u0d7b'"
        elif lis[pntr - 2] == "u'\u0d30'":
            lis[pntr - 2] = "u'\u0d7c'"
        elif lis[pntr - 2] == "u'\u0d32'":
            lis[pntr - 2] = "u'\u0d7d'"
        elif lis[pntr - 2] == "u'\u0d33'":
            lis[pntr - 2] = "u'\u0d7e'"
        elif lis[pntr - 2] == "u'\u0d23'":
            lis[pntr - 2] = "u'\u0d7a'"
        del lis[pntr]
        del lis[pntr - 1]
    while "u'\u0d10'" in lis:
        pntr = lis.index("u'\u0d10'")
        lis[pntr] = "u'\u0d30'"
        lis.insert(pntr+1,"u'\u0d1e'")
    while "u'\u0d13'" in lis:
        pntr = lis.index("u'\u0d13'")
        lis[pntr] = "u'\u0d12'"
        lis.insert(pntr+1,"u'\u0d3e'")
    while "u'\u0d14'" in lis:
        pntr = lis.index("u'\u0d14'")
        lis[pntr] = "u'\u0d12'"
        lis.insert(pntr+1,"u'\u0d57'")
    while "u'\u0d0a'" in lis:
        pntr = lis.index("u'\u0d0a'")
        lis[pntr] = "u'\u0d09'"
        lis.insert(pntr+1,"u'\u0d57'")
    while "u'\u0d08'" in lis:
        pntr = lis.index("u'\u0d08'")
        lis[pntr] = "u'\u0d07'"
        lis.insert(pntr+1,"u'\u0d57'")
    pntrs = []
    cnt = 0
    for elem in range(len(lis)):
        if lis[elem] == "u'\u0d4d'":
            if elem + 1 < len(lis):
                check = lis[elem-1]+lis[elem]+lis[elem+1]
                if check in SC_triple:
                    pntrs.append(elem)
                    lis[elem] = check
                else:
                    continue
    if len(pntrs) > 0:
        for i in pntrs:
            del lis[i-(2*cnt)+1]
            del lis[i-(2*cnt)-1]
            cnt += 1

    lis_46 = []
    for it in range(len(lis)):
        if lis[it] == "u'\u0d46'": # Handling matra ____
            lis_46.append(it)
    if len(lis_46) > 0:
        for pntr in lis_46:
            lis[pntr] = lis[pntr-1]
            lis[pntr-1] = "u'\u0d46'"

    lis_47 = []
    for it1 in range(len(lis)):
        if lis[it1] == "u'\u0d47'": # Handling matra ____
            lis_47.append(it1)

    if len(lis_47) > 0:
        for pntr in lis_47:
            lis[pntr] = lis[pntr-1]
            lis[pntr-1] = "u'\u0d47'"


    while "u'\u0d4a'" in lis: # Handling matra ____ matra
        pntr = lis.index("u'\u0d4a'")
        if lis[pntr-2] == "u'\u0d4d'":
            lis[pntr] = "u'\u0d46'"
            lis.insert(pntr+1,"u'\u0d3e'")
        lis[pntr] = lis[pntr-1]
        lis[pntr - 1] = "u'\u0d46'"
        lis.insert(pntr+1,"u'\u0d3e'")
    while "u'\u0d4b'" in lis: # Handling matra ____ matra
        pntr = lis.index("u'\u0d4b'")
        if lis[pntr-2] == "u'\u0d4d'":
            lis[pntr] = "u'\u0d47'"
            lis.insert(pntr+1,"u'\u0d3e'")
        else:
            lis[pntr] = lis[pntr-1]
            lis[pntr - 1] = "u'\u0d47'"
            lis.insert(pntr+1,"u'\u0d3e'")
    while "u'\u0d4c'" in lis: # Handling matra ____ matra
        pntr = lis.index("u'\u0d4c'")
        if lis[pntr-2] == "u'\u0d4d'":
            lis[pntr] = "u'\u0d46'"
            lis.insert(pntr+1,"u'\u0d57'")
        else:
            lis[pntr] = lis[pntr-1]
            lis[pntr - 1] = "u'\u0d46'"
            lis.insert(pntr+1,"u'\u0d57'")
    while "u'\u0d48'" in lis: # Handling matra ____ matra
        pntr = lis.index("u'\u0d48'")
        if lis[pntr-2] == "u'\u0d4d'":
            lis[pntr] = "u'\u0d46'"
            lis.insert(pntr+1,"u'\u0d46'")
        else:
            temp = lis[pntr-1]
            lis[pntr] = "u'\u0d46'"
            lis[pntr - 1] = "u'\u0d46'"
            lis.insert(pntr+1,temp)









    for num in range(num_styles):  ### Implies that we are using just 3 different styles of writing to generate the words
        image_end = styles[random.randint(0,len(styles)-1)]
        sty = image_end[:image_end.index('.')]
        # image_end = str(style)+'.stk'
        target = []
        # word = [[]]

        count = 0

        # Left and right
        l_x = 1

        #Now start creating the image of the word from stroke images
        # print (lis)
        for stroke in lis:
            if stroke == "u'\u0d03'":
                target.append(int(targets[stroke]))
                filename = path+lis_down_scale_down["u'\u0d02'"]+'/'+image_end
                data = extractdata(filename)
                onlinedata,delta = normalize(data,l_x,8,0.4,img_height)
                onlinedata1,delta1 = normalize(data,l_x,17,0.4,img_height)
                temp_ = onlinedata.tolist()
                for _ in range(delim_repetitions):
                    temp_.append([str_delim,str_delim])
                temp_ += onlinedata1.tolist()
                onlinedata = np.array(temp_)
                l_x = l_x + delta+stroke_space
            if stroke in SV:
                target.append(int(SV[stroke]))
                filename = path+SV[stroke]+'/'+image_end
                data = extractdata(filename)
                if stroke == "u'\u0d07'" or stroke == "u'\u0d09'":
                    onlinedata,delta = normalize(data,l_x,0,1.5,img_height)
                else:
                    onlinedata,delta = normalize(data,l_x,8,1,img_height)
                l_x = l_x + delta+stroke_space
            elif stroke in SC_single:
                target.append(int(SC_single[stroke]))
                filename = path+SC_single[stroke]+'/'+image_end
                data = extractdata(filename)
                if stroke == "u'\u0d33'":
                    onlinedata,delta = normalize(data,l_x,0,1.5,img_height)
                if stroke == "u'\u0d7a'" or stroke == "u'\u0d7b'" or stroke == "u'\u0d7c'" or stroke == "u'\u0d7d'" or stroke == "u'\u0d7e'" or stroke == "u'\u0d7f'":
                    onlinedata,delta = normalize(data,l_x,8,1.5,img_height)
                else:
                    onlinedata,delta = normalize(data,l_x,8,1,img_height)
                l_x = l_x + delta+stroke_space
            elif stroke in SC_triple:
                if SC_triple[stroke] == "u'\u0d23'" or SC_triple[stroke] == "u'\u0d2a'":
                    target.append(int(targets[stroke]))
                    filename = path+SC_single[SC_triple[stroke]]+'/'+image_end
                    data = extractdata(filename)
                    onlinedata,delta = normalize(data,l_x,8,1,img_height)
                    onlinedata1,delta1 = normalize(data,l_x+delta/4.0,0,0.65,img_height)
                    temp_ = onlinedata.tolist()
                    for _ in range(delim_repetitions):
                        temp_.append([str_delim,str_delim])
                    temp_ += onlinedata1.tolist()
                    onlinedata = np.array(temp_)
                    l_x = l_x + delta+stroke_space
                else:
                    target.append(int(SC_triple[stroke]))
                    filename = path+SC_triple[stroke]+'/'+image_end
                    data = extractdata(filename)
                    onlinedata,delta = normalize(data,l_x,8,1,img_height)
                    l_x = l_x + delta+stroke_space
            elif stroke in mtrs: # Handle the matras seperately
                target.append(int(mtrs[stroke]))
                filename = path+mtrs[stroke]+'/'+image_end
                data = extractdata(filename)
                l_x -= int(1.8*stroke_space)
                if stroke == "u'\u0d3f'":
                    onlinedata,delta = normalize(data,l_x,8,1.5,img_height)
                    l_x = l_x + delta+stroke_space
                elif stroke == "u'\u0d40'":
                    onlinedata,delta = normalize(data,l_x,8,1.5,img_height)
                    l_x = l_x + delta+stroke_space
                elif stroke == "u'\u0d41'":
                    onlinedata,delta = normalize(data,l_x,0,1.5,img_height)
                    l_x = l_x + delta+stroke_space
                elif stroke == "u'\u0d42'":
                    onlinedata,delta = normalize(data,l_x,0,1.5,img_height)
                    l_x = l_x + delta+stroke_space
                elif stroke == "u'\u0d43'":
                    onlinedata,delta = normalize(data,l_x,0,1.5,img_height)
                    l_x = l_x + delta+stroke_space
            elif stroke == "u'\u0d02'":
                target.append(int(lis_down_scale_down[stroke]))
                filename = path+lis_down_scale_down[stroke]+'/'+image_end
                data = extractdata(filename)
                onlinedata,delta = normalize(data,l_x,8,0.25,img_height,False)
                l_x = l_x + delta+stroke_space
            elif stroke == "u'\u0d4d'":
                target.append(int(lis_down_scale_up[stroke]))
                filename = path+lis_down_scale_up[stroke]+'/'+image_end
                data = extractdata(filename)
                onlinedata,delta = normalize(data,l_x-stroke_space,24,0.25,img_height,False)
                l_x = l_x + delta+stroke_space-stroke_space


            if count == 0:
                word = onlinedata.tolist()
                od = onlinedata.tolist()
                od = rdp_iso(od)
                od_ = np.ones((len(od),3))
                od_[:,:-1] = od

                od_[-1,-1] = 0.0
                if pen_lift:
                    word_stroke = od_.tolist()
                else:
                    word_stroke = od
                count += 1
                for _ in range(delim_repetitions):
                    word.append([str_delim,str_delim])

            else:
                od = onlinedata.tolist()
                word += od
                od = onlinedata.tolist()
                od = rdp_iso(od)
                od_ = np.ones((len(od),3))

                od_[:,:-1] = od
                od_[-1,-1] = 0.0
                # print od_
                if pen_lift:
                    word_stroke += od_.tolist()
                else:
                    word_stroke += od
                for _ in range(delim_repetitions):
                    word.append([str_delim,str_delim])



        word = np.array(word)
        begin_pntr = 0
        cnt = 0
        temp_ = np.max(word[:,0])
        plt.figure(figsize=(512/100.0, img_height/100.0), dpi=100)
        for i in range(word.shape[0]):
            if word[i,0] == str_delim and cnt%delim_repetitions == 0:
                plt.plot(word[begin_pntr:i,0],word[begin_pntr:i,1],color = 'black')
                begin_pntr = i+delim_repetitions
                cnt += 1

            elif word[i,0] and cnt%delim_repetitions != 0:
                cnt += 1
                continue

        # print np.array(word_stroke).shape
        final_online = transform_online(word_stroke,544)
        # print final_online
        if final_online == -1:
            print 'here'
            continue

        f_ = open('words/testword'+str(word_count)+'_'+sty+'.p','wb')
        pickle.dump(final_online,f_)
        f_.close()

        plt.axis('off')
        plt.savefig('words/testword'+str(word_count)+'_'+sty+'.png')
        plt.close('all')

        t_ = pickle.dumps(target)
        target_lis.append(t_)

    if word_count == 100000:
        break

    # if word_count%save_interval == 0:
    #     val = word_count/save_interval
    #     pickle.dump(target_lis,open('words//t'+str(val)+'.p','wb'))
    #     target_lis = []



pickle.dump(target_lis,open('words/t.p','wb'))
f.close()
