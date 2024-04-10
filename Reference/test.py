import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
from keras import *
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2



facedetect = cv2.CascadeClassifier('reference 1/haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX


target_size = (224, 224)

mobile_model = MobileNetV2(weights='imagenet', input_shape = target_size + (3,), include_top=False)
flat_layer = layers.Flatten()(mobile_model.output)
dense_layer_1 = layers.Dense(512, activation='relu')(flat_layer)
dense_layer_1 = layers.BatchNormalization()(dense_layer_1)
dense_layer_2 = layers.Dense(256, activation='relu')(dense_layer_1)
dense_layer_2 = layers.BatchNormalization()(dense_layer_2)
dense_layer_3 = layers.Dense(3, activation='relu')(dense_layer_2)

for layer in mobile_model.layers:
    layer.trainable = False

transfer_mobile_model = Model(inputs = mobile_model.inputs, outputs = dense_layer_3)

train = tf.keras.utils.image_dataset_from_directory('images/train', labels = 'inferred', image_size=target_size, batch_size= 100)
val = tf.keras.utils.image_dataset_from_directory('images/val', labels = 'inferred', image_size=target_size, batch_size= 100)


transfer_mobile_model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

transferhistory = transfer_mobile_model.fit(train, validation_data=val, epochs=10)



# model = load_model('reference 1/keras_model.h5')
model = transfer_mobile_model

   
count = 0
while True:
	success, imgOrignal=cap.read()
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:
		crop_img=imgOrignal[y:y+h,x:x+h]
		img=cv2.resize(crop_img, (224,224))
		img=img.reshape(1, 224, 224, 3)
		prediction=model.predict(img)
		classIndex = np.argmax(prediction)

		if classIndex==0:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, 'Aik Lok',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			count += 1
			if count > 200000000: # TODO: CHANGE TO 20
				count = 0
				print('index 0')
				import sys
				sys.exit()
		elif classIndex==1:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, 'idk',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
		else:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, 'yew',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			

	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()