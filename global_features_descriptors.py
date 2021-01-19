import cv2
import mahotas
import cv2
import numpy as np


def hu_moments(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(img)).flatten()
    return feature


def heralick_textures(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = mahotas.features.haralick(img).mean(axis=0)
    return feature


def get_features(df):
    features = []
    labels = []
    for _, data in df.iterrows():
        img = cv2.imread(data.filename)
        hu = hu_moments(img)
        heralick = heralick_textures(img)
        img_features = np.hstack([hu, heralick])
        labels.append(data.category)
        features.append(img_features)
    return features, labels
