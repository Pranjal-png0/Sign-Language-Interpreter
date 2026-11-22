import tensorflow as tf
import numpy as np
import cv2
import os

model = tf.keras.models.load_model("sign_model.h5")

img = cv2.imread("archive/asl_alphabet_train/asl_alphabet_train/A/A1.jpg")

img = cv2.resize(img,(224,224))
img = img.astype(np.float32)/255.0
img = np.expand_dims(img,0)

pred = model.predict(img,verbose=0)

print(np.argmax(pred))
print(np.max(pred))