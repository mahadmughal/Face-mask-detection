import cv2
import os
import numpy as np
from keras.models import load_model

results={0:'mask',1:'without mask'}
GR_dict={0:(0,0,255),1:(0,255,0)}

model=load_model("model.h5")

rect_size = 4
cap = cv2.VideoCapture(0)

while True:
    (rval, im) = cap.read()
    im=cv2.flip(im,1,1) 
    rerect_sized=cv2.resize(im,(224,224))
    normalized=rerect_sized/255.0
    reshaped=np.reshape(normalized,(1,224,224,3))
    reshaped = np.vstack([reshaped])
    result=model.predict(reshaped)
    label=np.argmax(result,axis=1)[0]
    print(results[label])
    image = cv2.putText(im, results[label], (50, 50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('LIVE', image)
    key = cv2.waitKey(10)
    if key == 27: 
        break
cap.release() 