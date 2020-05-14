import cv2

from django.shortcuts import render
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.models import model_from_json

global graph,model

#initializing the graph

#loading our trained model
# load model
model = model_from_json(open("AIModelv1.json", "r").read())
# load weights
model.load_weights('AIModelv1.h5')


#creating a dictionary of classes
class_dict = {'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'sad': 4,
            'surprise': 5,
            'neutral': 6
              }

class_names = list(class_dict.keys())

def prediction(request):
    if request.method == 'POST' and request.FILES['myfile']:
        post = request.method == 'POST'
        myfile = request.FILES['myfile']
        img = image.load_img(myfile, target_size=(48, 48), grayscale=True)
        #img = cv2.resize(img, (48, 48))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        preds = model.predict(img)
        preds = preds.flatten()
        m = max(preds)
        for index, item in enumerate(preds):
            if item == m:
                result = class_names[index]
        return render(request, "prediction.html", {
            'result': result})
    else:
        return render(request, "prediction.html")