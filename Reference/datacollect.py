import cv2
import os

video=cv2.VideoCapture(0)

# For this, copy the absolute path and replace it with your own
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if facedetect.empty():
    print("Error: Cascade Classifier not loaded")
else:
    print("Cascade Classifier loaded successfully")


count=0
test = False
anchor = True

nameID=str(input("Enter Your Name: ")).lower()
while True:
	path='images/train/'+nameID
	path2='images/val/'+nameID
	isExist = os.path.exists(path)

	# If name is already taken, ask for another one
	if isExist:
		print("Name Already Taken")
		nameID=str(input("Enter Your Name Again: ")).lower()
	else:
		os.makedirs(path)
		os.makedirs(path2)
		break


while True:
	# Start reading input from webcam, shows webcam video capture on device
	ret,frame=video.read()
	faces=facedetect.detectMultiScale(frame,1.3, 5)
	for x,y,w,h in faces:
		if not test and anchor:
			name='images/train/'+nameID+'/script.jpg'
			anchor = False
		elif not test and not anchor:
			name='images/train/'+nameID+'/'+ str(count) + '.jpg'
		else:
			name='images/val/'+nameID+'/'+ str(count) + '.jpg'
		count=count+1
		print("Creating Images..." +name)
		cv2.imwrite(name, frame[y:y+h,x:x+w])
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
	cv2.imshow("WindowFrame", frame)
	cv2.waitKey(1)
	if not test and count>=5:
		test = True
		count = 0
		continue
	elif test and count>=3:
		break

# Stops reading input from webcam, removes webcam video capture on device
video.release()
cv2.destroyAllWindows()