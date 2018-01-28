import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import pickle

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        # print (label)
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

def run_func(lmdb_path,data_folder,target_file,num_words,TrainValSplit):
    imglst = []
    for i in range(1,num_words+1):
        st1 = data_folder+'/testword%d'%i+'.png'
        #st2 = data_folder+'/testword%d'%i+'_2.png'
        #st3 = data_folder+'/testword%d'%i+'_3.png'
        imglst.append(st1)
        #imglst.append(st2)
        #imglst.append(st3)


    lablst = pickle.load(open(target_file,'rb'))
    lablst = lablst[: 120000]
    imglst = imglst[: 120000]
    assert len(imglst) == len(lablst)
    #split = int(TrainValSplit*len(imglst))
    #trn_inp = imglst[:split]
    #trn_lab = lablst[:split]
    #val_inp = imglst[split:]
    #val_lab = lablst[split:]

    trn_inp = imglst
    trn_lab = lablst

    assert len(trn_inp) == len(trn_lab)
    #assert len(val_inp) == len(val_lab)
    print 'Total Data',len(imglst)
    print 'Train Data',len(trn_inp)
    #print 'Val Data',len(val_inp)

    cwd = os.getcwd()
    print cwd
    createDataset(lmdb_path+'/train_150k',trn_inp,trn_lab)

run_func('./', 'words', './words/t.p', 141321, 1.0)
