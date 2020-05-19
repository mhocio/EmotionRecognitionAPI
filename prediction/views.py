import os
import pathlib

import cv2
import matplotlib
from PIL import Image
from django.http import HttpResponse

from django.shortcuts import render
from imageio import imsave
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from rest_framework.decorators import parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView
from tensorflow_core.python.keras.models import model_from_json

global graph, model

from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView

# initializing the graph

# loading our trained model
# load model
model = model_from_json(open("AIModelv1.json", "r").read())
# load weights
model.load_weights('AIModelv1.h5')

# creating a dictionary of classes
class_dict = {'angry': 0,
              'disgust': 1,
              'fear': 2,
              'happy': 3,
              'sad': 4,
              'surprise': 5,
              'neutral': 6
              }

class_names = list(class_dict.keys())
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(face_haar_cascade.empty())


#
# def prediction(request):
#     if request.method == 'POST' and request.FILES['myfile']:
#         post = request.method == 'POST'
#         myfile = request.FILES['myfile']
#
#         uploaded_file = request.FILES['myfile']
#         test_img = image.img_to_array(image.load_img(uploaded_file))
#         test_img = np.array(test_img, dtype='uint8')
#
#
#         #        im = np.fromstring(m, np.uint8)
#         gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
#         face_haar_cascade = cv2.CascadeClassifier(
#             '/Users/jorginaarrescardona/PycharmProjects/deployML/prediction/haarcascade_frontalface_default.xml')
#         faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
#
#         for (x, y, w, h) in faces_detected:
#             cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 255, 0), thickness=5)
#             roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
#             roi_gray = cv2.resize(roi_gray, (48, 48))
#             img_pixels = image.img_to_array(roi_gray)
#             img_pixels = np.expand_dims(img_pixels, axis=0)
#             # img_pixels /= 255
#             # uncoment if expected predict are normalized pixels
#
#             predictions = model.predict(img_pixels)
#
#             # find max indexed array
#             max_index = np.argmax(predictions[0])
#
#             emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
#             predicted_emotion = emotions[max_index]
#
#             cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
#
#         # preds = model.predict(img)
#         # preds = preds.flatten()
#         # m = max(preds)
#         # for index, item in enumerate(preds):
#         #     if item == m:
#         #         result = class_names[index]
#
#         # resized_img = cv2.resize(m, (1000, 700))
#         # predictedimg= cv2.imshow('Facial emotion analysis ', m)
#         return render(request, "prediction.html", {
#             'result': predicted_emotion})
#     else:
#         return render(request, "prediction.html")
#
@parser_classes((MultiPartParser,))
class PredictionsView(APIView):

    def post(self, request):
        uploaded_file = request.FILES['myfile']
        print(uploaded_file)
        test_img = image.img_to_array(image.load_img(uploaded_file))
        test_img = np.array(test_img, dtype='uint8')

        #        im = np.fromstring(m, np.uint8)
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        face_haar_cascade = cv2.CascadeClassifier(
            'prediction/haarcascade_frontalface_default.xml')
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 255, 0), thickness=5)
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            # img_pixels /= 255
            # uncoment if expected predict are normalized pixels

            predictions = model.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        new_image = Image.fromarray(test_img)
        imsave('images/File_name.png',new_image)
        dt = open('images/File_name.png', 'rb')
        return HttpResponse(dt, content_type="image/png")