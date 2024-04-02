import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
import random
import os
from keras import *

facedetect = cv2.CascadeClassifier('reference 1/haarcascade_frontalface_default.xml')
# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to your dataset
relative_path = 'images/train'

# Construct the full path using os.path.join()
full_path = os.path.join(current_dir, relative_path)
p = full_path

def localize_resize(path_image,facedetect):
    image=cv2.imread(path_image)
    
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    classifier= facedetect
    faces=classifier.detectMultiScale(gray,1.1,6)
    if len(faces) != 1:#condition if we dont have any faces or cant be detected y haar cascade we will skip those
        return -1
    
    x,y,w,h=faces.squeeze()
    crop=image[y:y+h,x:x+w]
    image=cv2.resize(crop,(96,96))
    image=np.transpose(image,(2,0,1))
    image=image.astype('float32')/255.0
    return image

def data_gen(batch_size=32):
    while True:
        i=0
        positive=[]
        anchor=[]
        negative=[]    
        

        while(i<batch_size):
            # r=random.choice(os.listdir(PATH))
            # p=PATH+'/'+ r
            id=os.listdir(p)
            ra=random.sample(id,2)
            pos_dir=p+'/'+ra[0]
            neg_dir=p+'/'+ra[1]
            pos=pos_dir+'/'+random.choice(os.listdir(pos_dir))
            anc=pos_dir+'/'+random.choice([x for x in os.listdir(pos_dir) if 'script' in x])
            neg=neg_dir+'/'+random.choice(os.listdir(neg_dir))
            pos_img=localize_resize(pos,facedetect)
                    #print(pos+anc+neg)
            if pos_img is -1:
                continue
            neg_img=localize_resize(neg,facedetect)
            if neg_img is -1:
                continue
            anc_img=localize_resize(anc,facedetect)
            if anc_img is -1:
                continue
            positive.append(list(pos_img))
                #print('positive{0}'.format(i))
            negative.append(list(neg_img))
                #print('negative{0}'.format(i))
            anchor.append(list(anc_img))
                #print('anchor{0}'.format(i))
            i=i+1
        #return anchor,positive,negative
        yield ([np.array(anchor),np.array(positive),np.array(negative)],np.zeros((batch_size,1)).astype("float32"))

next(data_gen())