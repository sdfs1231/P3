# 我们发现做水平翻转的时候 loss不降反升

import os
import re
def del_imgs():
    folder = [os.path.join('data','I'),os.path.join('data','II')]
    for f in folder:
        imgs = os.listdir(f)
        for img in imgs:
            if re.match(r'.*flip.jpg',img) or re.match(r'.*cc.jpg',img) or re.match(r'.*bc.jpg',img):
                # print(img)
                os.remove(os.path.join(f,img))
def del_txts():
    folder = 'data'
    files = os.listdir(folder)
    for f in files:
        if os.path.isfile(os.path.join(folder,f)) and re.match(r'.*\.txt',f):
            os.remove(os.path.join(folder,f))

if __name__=='__main__':
    del_txts()