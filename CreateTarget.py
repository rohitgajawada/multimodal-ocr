def createtarget(inputfile):

    SC_triple = {"u'\\u0d15'u'\\u0d4d'u'\\u0d15'":'69',"u'\\u0d15'u'\\u0d4d'u'\\u0d24'":'70',"u'\\u0d15'u'\\u0d4d'u'\\u0d37'":'71',"u'\\u0d17'u'\\u0d4d'u'\\u0d28'":'72',"u'\\u0d17'u'\\u0d4d'u'\\u0d2e'":'73',"u'\\u0d19'u'\\u0d4d'u'\\u0d15'":'74',"u'\\u0d19'u'\\u0d4d'u'\\u0d19'":'75',"u'\\u0d1a'u'\\u0d4d'u'\\u0d1a'":'76',"u'\\u0d1c'u'\\u0d4d'u'\\u0d1c'":'77',"u'\\u0d1c'u'\\u0d4d'u'\\u0d1e'":'78',"u'\\u0d1e'u'\\u0d4d'u'\\u0d1a'":'79',"u'\\u0d1e'u'\\u0d4d'u'\\u0d1e'":'80',"u'\\u0d1f'u'\\u0d4d'u'\\u0d1f'":'81',"u'\\u0d23'u'\\u0d4d'u'\\u0d1f'":'82',"u'\\u0d23'u'\\u0d4d'u'\\u0d21'":'83',"u'\\u0d23'u'\\u0d4d'u'\\u0d2e'":'84',"u'\\u0d24'u'\\u0d4d'u'\\u0d24'":'85',"u'\\u0d24'u'\\u0d4d'u'\\u0d25'":'86',"u'\\u0d24'u'\\u0d4d'u'\\u0d28'":'87',"u'\\u0d24'u'\\u0d4d'u'\\u0d2d'":'88',"u'\\u0d24'u'\\u0d4d'u'\\u0d2e'":'89',"u'\\u0d24'u'\\u0d4d'u'\\u0d38'":'90',"u'\\u0d26'u'\\u0d4d'u'\\u0d26'":'91',"u'\\u0d26'u'\\u0d4d'u'\\u0d27'":'92',"u'\\u0d28'u'\\u0d4d'u'\\u0d24'":'93',"u'\\u0d28'u'\\u0d4d'u'\\u0d25'":'94',"u'\\u0d28'u'\\u0d4d'u'\\u0d26'":'95',"u'\\u0d28'u'\\u0d4d'u'\\u0d27'":'96',"u'\\u0d28'u'\\u0d4d'u'\\u0d28'":'97',"u'\\u0d2e'u'\\u0d4d'u'\\u0d2a'":'98',"u'\\u0d28'u'\\u0d4d'u'\\u0d2e'":'99',"u'\\u0d2c'u'\\u0d4d'u'\\u0d2c'":'101',"u'\\u0d2e'u'\\u0d4d'u'\\u0d2e'":'102',"u'\\u0d2f'u'\\u0d4d'u'\\u0d2f'":'103',"u'\\u0d35'u'\\u0d4d'u'\\u0d35'":'104',"u'\\u0d36'u'\\u0d4d'u'\\u0d1a'":'105',"u'\\u0d39'u'\\u0d4d'u'\\u0d28'":'106',"u'\\u0d39'u'\\u0d4d'u'\\u0d2e'":'107',"u'\\u0d17'u'\\u0d4d'u'\\u0d26'":'109',"u'\\u0d1e'u'\\u0d4d'u'\\u0d1c'":'110'}

    SC_single = {"u'\u0d15'":'20',"u'\u0d16'":'21',"u'\u0d17'":'22',"u'\u0d18'":'23',"u'\u0d19'":'24',"u'\u0d1a'":'25',"u'\u0d1b'":'26',"u'\u0d1c'":'27',"u'\u0d1d'":'28',"u'\u0d1e'":'29',"u'\u0d1f'":'30',"u'\u0d20'":'31',
    "u'\u0d21'":'32',"u'\u0d22'":'33',"u'\u0d23'":'34',"u'\u0d24'":'35',"u'\u0d25'":'36',"u'\u0d26'":'37',"u'\u0d27'":'38',"u'\u0d28'":'39',"u'\u0d2a'":'40',"u'\u0d2b'":'41',"u'\u0d2c'":'42',"u'\u0d2d'":'43',"u'\u0d2e'":'44',
    "u'\u0d2f'":'45',"u'\u0d30'":'46',"u'\u0d32'":'47',"u'\u0d35'":'48',"u'\u0d36'":'49',"u'\u0d37'":'50',"u'\u0d38'":'51',"u'\u0d39'":'52',"u'\u0d33'":'53',"u'\u0d34'":'54',"u'\u0d31'":'55',"u'\u0d7b'":'56',"u'\u0d7c'":'57',"u'\u0d7d'":'58',"u'\u0d7e'":'59',"u'\u0d7a'":'60',"u'\\u0d4d'":'65'}


    lis_tosplit = {"u'\u0d4a'":'18-12',"u'\u0d4b'":"19-12","u'\u0d4c'":"10-12","u'\u0d46'":"19-12"}# special cases to be checked

    mtrs = {"u'\u0d3f'":"13","u'\u0d40'":"14","u'\u0d41'":"15","u'\u0d42'":"16","u'\u0d43'":"17"}

    SV = {"u'\u0d05'":'1',"u'\u0d06'":'2',"u'\u0d07'":'3',"u'\u0d09'":'4',"u'\u0d0b'":'5',"u'\u0d0e'":'7',"u'\u0d0f'":'8',"u'\u0d12'":'9',"u'\u0d57'":'10',"u'\u0d3e'":'12'}
    SV_left = {"u'\u0d46'":'18',"u'\u0d47'":'19'} # check this

    lis_down_scale_down = {"u'\u0d02'":'11'}
    lis_down_scale_up = {"u'\u0d4d'":'65'}

    f = open(inputfile)
    for i in f:
        #print (i.split('-'))
        targetlis = i.split('-')
        targ = ''
        flg = 0
        flg_1 = 0
        for i in range(len(targetlis)):
            #print (targetlis[i])
            #print (targetlis[i] in SC_single)
            #print (targ)
            if flg != 0:
                flg -= 1
                continue
            if targetlis[i] in lis_tosplit:
                spl = lis_tosplit[targetlis[i]].split('-')
                l = targ.split('-')
                targ = targ[:len(targ)-len(l[-1])]+spl[0]+'-'+targ[len(l[-1]):]+'-'+spl[1]+'-'
            if i+1<len(targetlis):
                if targetlis[i+1] == "u'\u0d4d'":
                    w = targetlis[i]+targetlis[i+1]+targetlis[i+2]
                    if w in SC_triple:
                        targ += SC_triple[w]+'-'
                        flg = 2
            if targetlis[i] in SC_single:
                #print ('here')
                #print(SC_single[targetlis[i]])
                targ += SC_single[targetlis[i]]+'-'
            if targetlis[i] in SV:
                targ += SV[targetlis[i]]+'-'
            if targetlis[i] in lis_down_scale_down:
                targ += lis_down_scale_down[targetlis[i]]+'-'
            if targetlis[i] in mtrs:
                targ += mtrs[targetlis[i]]+'-'
            if targetlis[i] in SV_left:
                l = targ.split('-')
                targ = targ[:len(targ)-len(l[-1])]+SV_left[targetlis[i]]+'-'+targ[len(l[-1]):]
            if targetlis[i] == "u'\u0d48'":
                l = targ.split('-')
                targ = targ[:len(targ)-len(l[-1])]+SV_left["u'\u0d46'"]+'-'+SV_left["u'\u0d46'"]+'-'+targ[len(l[-1]):]
            if targetlis[i] == "EOL\n":
                targ += 'EOL'  #End of line token

    print(targ)
    f.close()
