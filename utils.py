import numpy as np
import os
import cv2 as cv
import json
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing import image
from keras_vggface import utils


def preprocessing(path):
    img = image.load_img(path, target_size=(250, 250))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img, version=2)
    return img

def calculate_features(folder_url, model):
    feat_per = dict()
    for person in os.listdir(folder_url):
        features = []
        path = os.path.join(folder_url, person)
        for img_url in os.listdir(path):
            img = preprocessing(os.path.join("data", person, img_url))
            feature = model.predict(img)
            features.append(feature[0])
        avg_feature = np.array(features).sum(axis=0) / len(features)
        feat_per[person] = avg_feature.tolist()
    with open("features.json", "w+") as file:
        json.dump(feat_per, file)
        
def calculate_similarity(inp_url, feat_url, model):
    img = preprocessing(inp_url)
    inp_feature = model.predict(img).reshape(1, -1)
    similarity = dict()
    similarity["Unknown"] = 0.5
    with open(feat_url, "r") as file:
        features = json.load(file)
    for name, feature in features.items():
        feature = np.array(feature).reshape(1, -1)
        similarity[name] =  round(cosine_similarity(feature, inp_feature)[0][0], 3)
    return similarity
