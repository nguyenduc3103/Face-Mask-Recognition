import numpy as np
import os
import cv2 as cv

from mtcnn import MTCNN

model = MTCNN()
for person in os.listdir("data"):
    path = os.path.join("data", person)
    for img_url in os.listdir(path):
        path_img = os.path.join(path, img_url)
        img = cv.imread(path_img)
        
        scale_percent = 50
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        faces = model.detect_faces(rgb)
        
        for face in faces:
            x, y, w, h = face['box']
            croping = img[y: y + h + 1, x: x + w + 1, :].copy()
            cv.imwrite(path_img, croping)