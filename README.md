# CS5131 Project

Aim:
This duo project intends to provide an alternative to the fingerprint scanners at our boarding school using facial recognition. The project only intends to cover the backend, and there will be no prototype. Hardware used for demonstration will be camera from computer.

Execution:
- Use the input method from the youtube reference in order to find a way to capture face inputs to register new students.
- Use the test method from the youtube refernce in order to identify that faces are in the camera
- We will be using 2 models:
    1. stupid CNN that we developd ourselves using transfer learning from mobilenet
    2. model from the kaggle reference
- Train the model based on the list of students given
- For testing: Run through the face with every student in the model and 
- Compare how accurate both models are and conclude

For models:
https://www.youtube.com/watch?v=lH01BgsIPuE
https://www.kaggle.com/code/amankumarmallik/one-shot-learning-for-face-verification


Train the models in each section
in the comaparing section give 2 code blocks to test using the camera

compare the accuracy, precision and actually using the camera


```py
from keras import backend as K

def custom_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',custom_precision])
```
