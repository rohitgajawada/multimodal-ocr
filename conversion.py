# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 12:09:27 2016

@author: sreekar
"""

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
from resizeimage import resizeimage
import PIL

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






SC_triple = {"u'\\u0d15'u'\\u0d4d'u'\\u0d15'":'69',"u'\\u0d15'u'\\u0d4d'u'\\u0d24'":'70',"u'\\u0d15'u'\\u0d4d'u'\\u0d37'":'71',"u'\\u0d17'u'\\u0d4d'u'\\u0d28'":'72',"u'\\u0d17'u'\\u0d4d'u'\\u0d2e'":'73',"u'\\u0d19'u'\\u0d4d'u'\\u0d15'":'74',"u'\\u0d19'u'\\u0d4d'u'\\u0d19'":'75',"u'\\u0d1a'u'\\u0d4d'u'\\u0d1a'":'76',"u'\\u0d1c'u'\\u0d4d'u'\\u0d1c'":'77',"u'\\u0d1c'u'\\u0d4d'u'\\u0d1e'":'78',"u'\\u0d1e'u'\\u0d4d'u'\\u0d1a'":'79',"u'\\u0d1e'u'\\u0d4d'u'\\u0d1e'":'80',"u'\\u0d1f'u'\\u0d4d'u'\\u0d1f'":'81',"u'\\u0d23'u'\\u0d4d'u'\\u0d1f'":'82',"u'\\u0d23'u'\\u0d4d'u'\\u0d21'":'83',"u'\\u0d23'u'\\u0d4d'u'\\u0d2e'":'84',"u'\\u0d24'u'\\u0d4d'u'\\u0d24'":'85',"u'\\u0d24'u'\\u0d4d'u'\\u0d25'":'86',"u'\\u0d24'u'\\u0d4d'u'\\u0d28'":'87',"u'\\u0d24'u'\\u0d4d'u'\\u0d2d'":'88',"u'\\u0d24'u'\\u0d4d'u'\\u0d2e'":'89',"u'\\u0d24'u'\\u0d4d'u'\\u0d38'":'90',"u'\\u0d26'u'\\u0d4d'u'\\u0d26'":'91',"u'\\u0d26'u'\\u0d4d'u'\\u0d27'":'92',"u'\\u0d28'u'\\u0d4d'u'\\u0d24'":'93',"u'\\u0d28'u'\\u0d4d'u'\\u0d25'":'94',"u'\\u0d28'u'\\u0d4d'u'\\u0d26'":'95',"u'\\u0d28'u'\\u0d4d'u'\\u0d27'":'96',"u'\\u0d28'u'\\u0d4d'u'\\u0d28'":'97',"u'\\u0d2e'u'\\u0d4d'u'\\u0d2a'":'98',"u'\\u0d28'u'\\u0d4d'u'\\u0d2e'":'99',"u'\\u0d2c'u'\\u0d4d'u'\\u0d2c'":'101',"u'\\u0d2e'u'\\u0d4d'u'\\u0d2e'":'102',"u'\\u0d2f'u'\\u0d4d'u'\\u0d2f'":'103',"u'\\u0d35'u'\\u0d4d'u'\\u0d35'":'104',"u'\\u0d36'u'\\u0d4d'u'\\u0d1a'":'105',"u'\\u0d39'u'\\u0d4d'u'\\u0d28'":'106',"u'\\u0d39'u'\\u0d4d'u'\\u0d2e'":'107',"u'\\u0d17'u'\\u0d4d'u'\\u0d26'":'109',"u'\\u0d1e'u'\\u0d4d'u'\\u0d1c'":'110'}

SC_single = {"u'\u0d15'":'20',"u'\u0d16'":'21',"u'\u0d17'":'22',"u'\u0d18'":'23',"u'\u0d19'":'24',"u'\u0d1a'":'25',"u'\u0d1b'":'26',"u'\u0d1c'":'27',"u'\u0d1d'":'28',"u'\u0d1e'":'29',"u'\u0d1f'":'30',"u'\u0d20'":'31',"u'\u0d21'":'32',"u'\u0d22'":'33',"u'\u0d23'":'34',"u'\u0d24'":'35',"u'\u0d25'":'36',"u'\u0d26'":'37',"u'\u0d27'":'38',"u'\u0d28'":'39',"u'\u0d2a'":'40',"u'\u0d2b'":'41',"u'\u0d2c'":'42',"u'\u0d2d'":'43',"u'\u0d2e'":'44',"u'\u0d2f'":'45',"u'\u0d30'":'46',"u'\u0d32'":'47',"u'\u0d35'":'48',"u'\u0d36'":'49',"u'\u0d37'":'50',"u'\u0d38'":'51',"u'\u0d39'":'52',"u'\u0d33'":'53',"u'\u0d34'":'54',"u'\u0d31'":'55',"u'\u0d7b'":'56',"u'\u0d7c'":'57',"u'\u0d7d'":'58',"u'\u0d7e'":'59',"u'\u0d7a'":'60',"u'\\u0d4d'":'65'}


lis_tosplit = ["u'\u0d4a'","u'\u0d4b'"]# special cases to be checked

mtrs = {"u'\u0d3f'":"13","u'\u0d40'":"14","u'\u0d41'":"15","u'\u0d42'":"16","u'\u0d43'":"17"}

SV = {"u'\u0d05'":'1',"u'\u0d06'":'2',"u'\u0d07'":'3',"u'\u0d09'":'4',"u'\u0d0b'":'5',"u'\u0d0e'":'7',"u'\u0d0f'":'8',"u'\u0d12'":'9',"u'\u0d57'":'10',"u'\u0d3e'":'12',"u'\u0d46'":'18',"u'\u0d47'":'19'}

lis_down_scale_down = {"u'\u0d02'":'11'}
lis_down_scale_up = {"u'\u0d4d'":'65'}

word_count = 0
file_count = 0
count = 0
fil = open('Targets.txt','w+')
f = open('malWords','r+')
for line in f:
    print(line)
#     lis = []
#     word_count += 1
#     print("word", word_count)
#     word_unicode = line.encode('latin-1').decode('utf-8')
#     for i in range(len(word_unicode)):
#         char_uni = repr(word_unicode[i])
#         if char_uni != "u'\\n'":
#             lis.append(char_uni)
#             count += 1
#         if "u'\0d03'" in lis:
#             continue
#
# 	# number of words to be generated
#     if word_count == 30000:
#         break
#
# 	# Saving the targets
#     if word_count%10 == 0:
#         fil.close()
#         file_count += 1
#         fil = open('Targets' + str(file_count) +'.txt','w+')
#
#     if "u'\u200c'" in lis:
#         pntr = lis.index("u'\u200c'")
#         del lis[pntr]
#     while "u'\u200d'" in lis:
#         pntr = lis.index("u'\u200d'")
#         if lis[pntr - 2] == "u'\u0d28'":
#             lis[pntr - 2] = "u'\u0d7b'"
#         elif lis[pntr - 2] == "u'\u0d30'":
#             lis[pntr - 2] = "u'\u0d7c'"
#         elif lis[pntr - 2] == "u'\u0d32'":
#             lis[pntr - 2] = "u'\u0d7d'"
#         elif lis[pntr - 2] == "u'\u0d33'":
#             lis[pntr - 2] = "u'\u0d7e'"
#         elif lis[pntr - 2] == "u'\u0d23'":
#             lis[pntr - 2] = "u'\u0d7a'"
#         del lis[pntr]
#         del lis[pntr - 1]
#
#
#
#
#
#
#     path = '/home/rohit/Documents/sreekarfiles/train'  #### change the path according to the files in your file system
#     print(lis)
#     for i in range(1,3):  ### Implies that we are using just 3 different styles of writing to generate the words
#         style = i
#         image_end = str(style)+'.png'
#
#
#         string = ''
#         for l in lis:
#             string += l +"-"
#         string += "EOL"
#         string += '\n'
#         fil.write(string)
#
#
#         #lis_scaled_up = {"u'\u0d41'":'15',"u'\u0d42'":'16',"u'\u0d43'":'17'}
#         #lis_scaled_down = {"u'\u0d3f'":'13',"u'\u0d40'":'14'}
#
#
#         reached_flag = 0
#
#         img = Image.new('RGBA',(200*22,200),"black") #Creating the required backgroudfor the word
#         count = 0
#         elem = 0
#         l,r = 0,0
#
#
#         while elem < len(lis):
#
#             # print("hello")
#             reached_flag = 0
#             if elem < len(lis) - 2:
#                 if lis[elem + 1] == "u'\\u0d4d'":
#                     wrd = lis[elem] + lis[elem + 1] + lis[elem + 2]
#                     if wrd in SC_triple:
#                         elem += 3
#                         if elem < len(lis):
#                             if lis[elem] == "u'\u0d4a'":
#                                 wrd1 =  "u'\u0d46'"
#                                 im = Image.open(path+SV[wrd1]+'/'+image_end)
#                                 #im = invert(im)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 wrd2 =  wrd
#                                 im = Image.open(path+SC_triple[wrd2]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 wrd3 = "u'\u0d3e'"
#                                 im = Image.open(path+SV[wrd3]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 elem += 1
#                                 continue
#                             elif lis[elem] == "u'\u0d4b'":
#                                 wrd1 =  "u'\u0d47'"
#                                 im = Image.open(path+SV[wrd1]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 wrd2 =  wrd
#                                 im = Image.open(path+SC_triple[wrd2]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 wrd3 = "u'\u0d3e'"
#                                 im = Image.open(path+SV[wrd3]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 elem += 1
#                                 continue
#                             elif lis[elem] == "u'\u0d4c'":
#                                 wrd1 =  "u'\u0d46'"
#                                 im = Image.open(path+SV[wrd1]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 wrd2 =  wrd
#                                 im = Image.open(path+SC_triple[wrd2]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 wrd3 = "u'\u0d57'"
#                                 im = Image.open(path+SV[wrd3]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 elem += 1
#                                 continue
#                             elif lis[elem] == "u'\u0d46'":
#                                 wrd1 = "u'\u0d46'"
#                                 im = Image.open(path+SV[wrd1]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 wrd2 = wrd
#                                 im = Image.open(path+SC_triple[wrd2]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 elem += 1
#                                 continue
#
#                             elif lis[elem] == "u'\u0d47'":
#                                 wrd1 = "u'\u0d47'"
#                                 im = Image.open(path+SV[wrd1]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 wrd2 = wrd
#                                 im = Image.open(path+SC_triple[wrd2]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50))
#                                 l += right - left
#                                 r += right - left
#                                 elem += 1
#                                 continue
#
#                             elif lis[elem] == "u'\u0d48'":
#                                 wrd1 = "u'\u0d46'"
#                                 im = Image.open(path+SV[wrd1]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 wrd1 = "u'\u0d46'"
#                                 im = Image.open(path+SV[wrd1]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 wrd2 = wrd
#                                 im = Image.open(path+SC_triple[wrd2]+'/'+image_end)
#                                 im = im.convert("RGBA")
#                                 im = make_black_go(im)
#                                 im = get_correct_size(im)
#                                 left,up,right,down = im.getbbox()
#                                 img.paste(im,(l,50),im)
#                                 l += right - left
#                                 r += right - left
#                                 elem += 1
#                                 continue
#                         im = Image.open(path+SC_triple[wrd]+'/'+image_end)
#                         #im = invert(im)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#
#
#                         continue
#                     else:
#                         reached_flag = 1
#
#             if elem + 1 < len(lis):
#                 if elem+3 < len(lis):
#                     if lis[elem + 1] == "u'\u0d4a'" or (reached_flag == 1 and lis[elem + 3] == "u'\u0d4a'" ):
#                         wrd1 =  "u'\u0d46'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         #im = invert(im)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         if reached_flag == 1:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             wrd2 =  lis[elem + 1]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             im = im.resize((40,40),Image.ANTIALIAS)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l, 20),im)
#                             l += right-left - 20
#                             wrd2 =  lis[elem + 2]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 3
#                         else:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 1
#                         wrd3 = "u'\u0d3e'"
#                         im = Image.open(path+SV[wrd3]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         elem += 1
#                         continue
#                     if lis[elem + 1] == "u'\u0d4b'" or (reached_flag == 1 and lis[elem + 3] == "u'\u0d4b'" ):
#                         wrd1 =  "u'\u0d47'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         if reached_flag == 1:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             wrd2 =  lis[elem + 1]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             im = im.resize((40,40),Image.ANTIALIAS)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l, 20),im)
#                             l += right-left - 20
#                             wrd2 =  lis[elem + 2]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 3
#                         else:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 1
#                         wrd3 = "u'\u0d3e'"
#                         im = Image.open(path+SV[wrd3]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         elem += 1
#                         continue
#                     if lis[elem + 1] == "u'\u0d4c'" or (reached_flag == 1 and lis[elem + 3] == "u'\u0d4c'" ):
#                         wrd1 =  "u'\u0d46'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         if reached_flag == 1:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             wrd2 =  lis[elem + 1]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             im = im.resize((40,40),Image.ANTIALIAS)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l, 20),im)
#                             l += right-left - 20
#                             wrd2 =  lis[elem + 2]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 3
#                         else:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 1
#                         wrd3 = "u'\u0d57'"
#                         im = Image.open(path+SV[wrd3]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         elem += 1
#                         continue
#                     if lis[elem + 1] == "u'\u0d46'" or (reached_flag == 1 and lis[elem + 3] == "u'\u0d46'" ):
#                         wrd1 = "u'\u0d46'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         if reached_flag == 1:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             wrd2 =  lis[elem + 1]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             im = im.resize((40,40),Image.ANTIALIAS)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l, 20),im)
#                             l += right-left - 20
#                             wrd2 =  lis[elem + 2]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 3
#                         else:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 1
#                         elem += 1
#                         continue
#                     if lis[elem + 1] == "u'\u0d47'" or (reached_flag == 1 and lis[elem + 3] == "u'\u0d47'" ):
#                         wrd1 = "u'\u0d47'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         if reached_flag == 1:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             wrd2 =  lis[elem + 1]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             im = im.resize((40,40),Image.ANTIALIAS)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l, 20),im)
#                             l += right-left - 20
#                             wrd2 =  lis[elem + 2]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 3
#                         else:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 1
#                         elem += 1
#                         continue
#                     if lis[elem + 1] == "u'\u0d48'" or (reached_flag == 1 and lis[elem + 3] == "u'\u0d48'" ):
#                         wrd1 = "u'\u0d46'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         wrd1 = "u'\u0d46'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         if reached_flag == 1:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             wrd2 =  lis[elem + 1]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             im = im.resize((40,40),Image.ANTIALIAS)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l, 20),im)
#                             l += right-left - 20
#                             wrd2 =  lis[elem + 2]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 3
#                         else:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 1
#                         elem += 1
#                         continue
#                 else:
#                     if lis[elem + 1] == "u'\u0d4a'" :
#                         wrd1 =  "u'\u0d46'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         #im = invert(im)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         if reached_flag == 1:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             wrd2 =  lis[elem + 1]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             im = im.resize((40,40),Image.ANTIALIAS)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l, 20),im)
#                             l += right-left - 20
#                             wrd2 =  lis[elem + 2]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 3
#                         else:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 1
#                         wrd3 = "u'\u0d3e'"
#                         im = Image.open(path+SV[wrd3]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         elem += 1
#                         continue
#                     if lis[elem + 1] == "u'\u0d4b'" :
#                         wrd1 =  "u'\u0d47'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         if reached_flag == 1:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             wrd2 =  lis[elem + 1]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             im = im.resize((40,40),Image.ANTIALIAS)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l, 20),im)
#                             l += right-left - 20
#                             wrd2 =  lis[elem + 2]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 3
#                         else:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 1
#                         wrd3 = "u'\u0d3e'"
#                         im = Image.open(path+SV[wrd3]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         elem += 1
#                         continue
#                     if lis[elem + 1] == "u'\u0d4c'" :
#                         wrd1 =  "u'\u0d46'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         if reached_flag == 1:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             wrd2 =  lis[elem + 1]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             im = im.resize((40,40),Image.ANTIALIAS)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l, 20),im)
#                             l += right-left - 20
#                             wrd2 =  lis[elem + 2]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 3
#                         else:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 1
#                         wrd3 = "u'\u0d57'"
#                         im = Image.open(path+SV[wrd3]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         elem += 1
#                         continue
#
#                     if lis[elem + 1] == "u'\u0d46'" :
#                         wrd1 = "u'\u0d46'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         if reached_flag == 1:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             wrd2 =  lis[elem + 1]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             im = im.resize((40,40),Image.ANTIALIAS)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l, 20),im)
#                             l += right-left - 20
#                             wrd2 =  lis[elem + 2]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 3
#                         else:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 2
#                         elem += 1
#                         continue
#                     if lis[elem + 1] == "u'\u0d47'" :
#                         wrd1 = "u'\u0d47'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         if reached_flag == 1:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             wrd2 =  lis[elem + 1]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             im = im.resize((40,40),Image.ANTIALIAS)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l, 20),im)
#                             l += right-left - 20
#                             wrd2 =  lis[elem + 2]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 3
#                         else:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 1
#                         elem += 1
#                         continue
#                     if lis[elem + 1] == "u'\u0d48'" :
#                         wrd1 = "u'\u0d46'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         wrd1 = "u'\u0d46'"
#                         im = Image.open(path+SV[wrd1]+'/'+image_end)
#                         im = im.convert("RGBA")
#                         im = make_black_go(im)
#                         im = get_correct_size(im)
#                         left,up,right,down = im.getbbox()
#                         img.paste(im,(l,50),im)
#                         l += right - left
#                         r += right - left
#                         if reached_flag == 1:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             wrd2 =  lis[elem + 1]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             im = im.resize((40,40),Image.ANTIALIAS)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l, 20),im)
#                             l += right-left - 20
#                             wrd2 =  lis[elem + 2]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 3
#                         else:
#                             wrd2 =  lis[elem]
#                             if wrd2 in SC_single:
#                                 im = Image.open(path+SC_single[wrd2]+'/'+image_end)
#                             else:
#                                 im = Image.open(path+SV[wrd2]+'/'+image_end)
#                             im = im.convert("RGBA")
#                             im = make_black_go(im)
#                             im = get_correct_size(im)
#                             left,up,right,down = im.getbbox()
#                             img.paste(im,(l,50),im)
#                             l += right - left
#                             r += right - left
#                             elem += 2
#                         elem += 1
#                         continue
#
#
#
#
#             if lis[elem] in SC_single:
#                 if lis[elem] == "u'\\u0d4d'":
#                     im = Image.open(path+SC_single[lis[elem]]+'/'+image_end)
#                     #im = invert(im)
#                     im = im.convert("RGBA")
#                     im = make_black_go(im)
#                     im = get_correct_size(im)
#                     im = im.resize((40,40),Image.ANTIALIAS)
#                     left,up,right,down = im.getbbox()
#                     img.paste(im,(l, 20),im)
#                     l += right-left - 20
#                 else:
#                     im = Image.open(path+SC_single[lis[elem]]+'/'+image_end)
#                     #im = invert(im)
#                     im = im.convert("RGBA")
#                     im = make_black_go(im)
#                     im = get_correct_size(im)
#                     left,up,right,down = im.getbbox()
#                     img.paste(im,(l,50),im)
#                     l += right - left
#                     r += right - left
#                 elem += 1
#                 continue
#
#             if lis[elem] == "u'\u0d02'":
#                 im = Image.open(path+lis_down_scale_down[lis[elem]]+'/'+image_end)
#                 #im = invert(im)
#                 im = im.convert("RGBA")
#                 im = make_black_go(im)
#                 im = get_correct_size(im)
#                 im = im.resize((40,40),Image.ANTIALIAS)
#                 left,up,right,down = im.getbbox()
#                 img.paste(im,(l, 110),im)
#                 l += right-left + 5
#                 elem += 1
#                 continue
#             if lis[elem] in SV and lis[elem] != "u'\u0d46'" and lis[elem] != "u'\u0d47'":
#                 im = Image.open(path+SV[lis[elem]]+'/'+image_end)
#                 #im = invert(im)
#                 im = im.convert("RGBA")
#                 im = make_black_go(im)
#                 im = get_correct_size(im)
#                 left,up,right,down = im.getbbox()
#                 img.paste(im,(l,50),im)
#                 if lis[elem] == "u'\u0d3e'":
#                     l+= right-left +20
#                 else:
#                     l += right - left
#                     r += right - left
#                 elem += 1
#                 continue
#             if lis[elem] == "u'\u0d3f'":
#                 im = Image.open(path+mtrs[lis[elem]]+'/'+image_end)
#                 #im = invert(im)
#                 im = im.convert("RGBA")
#                 im = make_black_go(im)
#                 im = get_correct_size(im)
#                 im = im.resize((im.size[0],150),Image.ANTIALIAS)
#                 left,up,right,down = im.getbbox()
#                 img.paste(im,(l - 30,0),im)
#                 l += right - left
#                 elem += 1
#                 continue
#             if lis[elem] == "u'\u0d40'":
#                 im = Image.open(path+mtrs[lis[elem]]+'/'+image_end)
#                 im = im.convert("RGBA")
#                 im = make_black_go(im)
#                 im = get_correct_size(im)
#                 im = im.resize((im.size[0],150),Image.ANTIALIAS)
#                 left,up,right,down = im.getbbox()
#                 img.paste(im,(l - 30,0),im)
#                 l += right - left
#                 elem += 1
#                 continue
#             if lis[elem] == "u'\u0d41'":
#                 im = Image.open(path+mtrs[lis[elem]]+'/'+image_end)
#                 im = im.convert("RGBA")
#                 im = make_black_go(im)
#                 im = get_correct_size(im)
#                 im = im.resize((im.size[0],150),Image.ANTIALIAS)
#                 left,up,right,down = im.getbbox()
#                 img.paste(im,(l-15,50),im)
#                 l += right - left + 15
#                 elem += 1
#                 continue
#             if lis[elem] == "u'\u0d42'":
#                 im = Image.open(path+mtrs[lis[elem]]+'/'+image_end)
#                 im = im.convert("RGBA")
#                 im = make_black_go(im)
#                 im = get_correct_size(im)
#                 im = im.resize((im.size[0],150),Image.ANTIALIAS)
#                 left,up,right,down = im.getbbox()
#                 img.paste(im,(l,50),im)
#                 l += right - left
#                 elem += 1
#                 continue
#             if lis[elem] == "u'\u0d43'":
#                 im = Image.open(path+mtrs[lis[elem]]+'/'+image_end)
#                 im = im.convert("RGBA")
#                 im = make_black_go(im)
#                 im = get_correct_size(im)
#                 im = im.resize((im.size[0],150),Image.ANTIALIAS)
#                 left,up,right,down = im.getbbox()
#                 img.paste(im,(l - 30,50),im)
#                 l += 30
#                 elem += 1
#                 continue
#
#
#
#
#         #left,up,right,down = img.getbbox()
#         #plt.imshow(img,cmap = "gray")
#         print("namaste")
#         img.save('word'+str(word_count)+'_'+str(style)+'.png',quality = 150)
#
# f.close()
# fil.close()
