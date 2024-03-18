import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np


facedetect = cv2.CascadeClassifier('reference 1/haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX

from tensorflow import keras
from keras import *
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
target_size = (224, 224)
inception_model = MobileNetV2(weights='imagenet', input_shape = target_size + (3,), include_top=False)

for layer in inception_model.layers:
    layer.trainable = False

flat_layer = layers.Flatten()(inception_model.output)
dense_layer_1 = layers.Dense(512, activation='relu')(flat_layer)
dense_layer_1 = layers.BatchNormalization()(dense_layer_1)
dense_layer_2 = layers.Dense(256, activation='relu')(dense_layer_1)
dense_layer_2 = layers.BatchNormalization()(dense_layer_2)
dense_layer_3 = layers.Dense(256, activation='relu')(dense_layer_2)

transfer_inception_model = Model(inputs = inception_model.inputs, outputs = dense_layer_3)

class SimilarityLayer(layers.Layer):
    # compute and return the two distances:
    # d(anchor,positive) 
    # d(anchor,negative)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, anchor, positive, negative):
        d1 = tf.reduce_sum(tf.square(anchor-positive), -1)
        d2 = tf.reduce_sum(tf.square(anchor-negative), -1)
        return(d1,d2)
    
anchor = layers.Input(name='anchor', shape = target_size + (3,))
positive = layers.Input(name='positive', shape = target_size + (3,))
negative = layers.Input(name='negative', shape = target_size + (3,))

sim_layer_output = SimilarityLayer().call(
    transfer_inception_model(inputs = mobilenet_v2.preprocess_input(anchor)),
    transfer_inception_model(inputs = mobilenet_v2.preprocess_input(positive)),
    transfer_inception_model(inputs = mobilenet_v2.preprocess_input(negative))
)

siamese_model = Model(inputs=[anchor, positive,negative], outputs=sim_layer_output)

class SiameseModelClass(Model):
    def __init__(self, siamese_model, margin = 0.5):
        super(SiameseModelClass, self).__init__()
        
        self.siamese_model = siamese_model
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        
    def call(self, inputs):
        return self.siamese_model(inputs)
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.custom_loss(data)
            
        trainable_vars = self.siamese_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.custom_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def custom_loss(self, data):
        d1, d2 = self.siamese_model(data)
        loss = tf.maximum(d1 - d2 + self.margin, 0)
        
        return loss
    
    @property
    def metrics(self):
        return [self.loss_tracker]



model = load_model('reference 1/keras_model.h5')


def get_className(classNo):
	if classNo==0:
		return "Aik Lok"
	elif classNo==1:
		return "idk"

count = 0
while True:
	sucess, imgOrignal=cap.read()
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:
		crop_img=imgOrignal[y:y+h,x:x+h]
		img=cv2.resize(crop_img, (224,224))
		img=img.reshape(1, 224, 224, 3)
		prediction=model.predict(img)
		classIndex = [x[0] for x in prediction].index(max([x[0] for x in prediction]))
		probabilityValue=np.amax(prediction)

		if classIndex==0:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			count += 1
			if count > 20:
				count = 0
				print('index 0')
				import sys
				sys.exit()
		elif classIndex==1:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)

		cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv2.LINE_AA)
	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()