import numpy as np
import tensorflow as tf
import os
import cv2 as cv
import json
from utils import preprocessing, calculate_similarity
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
from collections import defaultdict
from keras_vggface.vggface import VGGFace


gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num of GPUS: ", len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = VGGFace(model='resnet50', 
                include_top=False, 
                input_shape=(250, 250, 3),
                pooling='avg')

url = "input"

faceDetection = MTCNN()
cap = cv.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    faces = faceDetection.detect_faces(rgb)
    
    for idx, face in enumerate(faces):
        x, y, w, h = face['box']
        cut_frame = frame[y: y + h + 1, x: x + w + 1, :].copy()
        cv.rectangle(frame, (x - 1, y - 1), (x + w + 1, y + h + 1), (255, 255, 0), 1)
        path = "input\\input_" + str(idx) + ".jpg"
        cv.imwrite(path, cut_frame)
        similarity = calculate_similarity(path, "features.json", model)
        new_name = max(similarity, key=similarity.get)
        cv.putText(frame, new_name, (x, y - 10), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1)
        cv.putText(frame, str(similarity[new_name] * 100) + "%", (x, y + h + 20), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1)
        cv.imshow("Face Recognition", frame)
    cv.imshow("Face Recognition", frame)
    
    for img in os.listdir(url):
        os.remove(os.path.join(url, img))
    
    if cv.waitKey(1) & 0XFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()

for img in os.listdir(url):
	os.remove(os.path.join(url, img))